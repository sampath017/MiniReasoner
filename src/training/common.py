"""Shared configuration, batch streaming, optimization, evaluation, and session lifecycle."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import signal

from dotenv import load_dotenv
import numpy as np
import torch
from torch.nn import functional as F

from ..core.tokenizer import TOKEN_DTYPE
from ..infra.checkpoint_store import CheckpointStore
from ..infra.metrics import track_run
from ..infra.tracking import checkpoint_uploads, publish_run_log, tee_to_log
from ..settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT, PROJECT_ROOT


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    steps: int
    warmup_steps: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    eval_every: int
    eval_token_fraction: float
    seed: int

    def __post_init__(self):
        if min(self.batch_size, self.steps, self.eval_every) < 1:
            raise ValueError("Batch, step, and evaluation counts must be positive")
        if not 0 < self.eval_token_fraction <= 1:
            raise ValueError("eval_token_fraction must be greater than zero and at most one")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("Warmup must be shorter than the run")
        if not 0 <= self.min_learning_rate <= self.learning_rate or self.learning_rate == 0:
            raise ValueError("Require 0 <= min_learning_rate <= learning_rate and a positive peak")
        if self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative")

    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        decay_intervals = max(1, self.steps - self.warmup_steps - 1)
        progress = min(1.0, max(0.0, (step - self.warmup_steps - 1) / decay_intervals))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_learning_rate + cosine * (self.learning_rate - self.min_learning_rate)


class TokenStream:
    def __init__(self, path: Path, size: int, context: int):
        with Path(path).open("rb") as stream:
            self.tokens = np.memmap(stream, dtype=TOKEN_DTYPE, mode="r", shape=(size,))
        self.offsets = np.arange(context + 1)
        self.context_length = context
        self.target_count = size - 1
        self.windows, self.tail_targets = divmod(self.target_count, context)
        if self.target_count < 1:
            raise ValueError(f"{path} needs at least two tokens for next-token prediction")

    def batch(self, starts: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = starts[:, None] + self.offsets
        if indices[:, -1].max() >= len(self.tokens):
            # Pad the last short window and omit padded targets from cross-entropy.
            valid = indices < len(self.tokens)
            batch = self.tokens[np.minimum(indices, len(self.tokens) - 1)].astype(np.int64)
            targets = batch[:, 1:].copy()
            targets[~valid[:, 1:]] = -100
            return torch.from_numpy(batch[:, :-1].copy()).to(device), torch.from_numpy(targets).to(device)
        ids = torch.from_numpy(self.tokens[indices].astype(np.int64)).to(device)
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


class EpochBatchSampler:
    def __init__(self, data: TokenStream, batch_size: int, seed: int):
        self.data = data
        self.batch_size = batch_size
        self.seed = seed
        self.window_count = data.windows + int(data.tail_targets > 0)
        self.steps_per_epoch = math.ceil(self.window_count / batch_size)
        self.epoch = -1
        self.order = np.empty(0, dtype=np.int64)

    def starts(self, step: int) -> np.ndarray:
        epoch, batch_index = divmod(step - 1, self.steps_per_epoch)
        if epoch != self.epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self.order = rng.permutation(self.data.windows)
            if self.data.tail_targets:
                self.order = np.append(self.order, self.data.windows)
            self.epoch = epoch
        start = batch_index * self.batch_size
        return self.order[start:start + self.batch_size] * self.data.context_length

    def targets_through(self, step: int) -> int:
        epochs, batches = divmod(step, self.steps_per_epoch)
        return epochs * self.data.target_count + batches * self.batch_size * self.data.context_length


def sample_validation_windows(data: TokenStream, config: TrainConfig, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Return random window starts and the number of targets to score in each."""
    budget = math.ceil(data.target_count * config.eval_token_fraction)
    rng = np.random.default_rng([config.seed, step])
    window_ids = rng.permutation(math.ceil(data.target_count / data.context_length))
    starts = window_ids * data.context_length
    lengths = np.minimum(data.context_length, data.target_count - starts)
    cumulative = lengths.cumsum()
    last = int(cumulative.searchsorted(budget))
    lengths[last] -= cumulative[last] - budget
    return starts[:last + 1], lengths[:last + 1]


def require_training_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU. For CPU runs, pass an explicit cpu device.")
    return torch.device("cuda")


def resolve_precision(request: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32" if request in ("auto", "fp32") else _raise_cuda_precision_error()
    if request == "fp32":
        return "fp32"
    if request not in ("auto", "bf16"):
        raise ValueError("Training precision must be auto, fp32, or bf16")
    native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    if request == "auto":
        return "bf16" if native_bf16 else "fp32"
    if not native_bf16:
        raise ValueError("This CUDA device does not have native BF16 support")
    return "bf16"


def _raise_cuda_precision_error() -> str:
    raise ValueError("BF16 precision requires a CUDA device.")


def configure_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.AdamW:
    """Create fused AdamW separating 2D+ weight-decayed parameters from 1D biases/norms."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )


def loss_for(model, inputs: torch.Tensor, targets: torch.Tensor, precision: str) -> torch.Tensor:
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(inputs)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


@torch.inference_mode()
def evaluate(model, data: TokenStream, config: TrainConfig, device: torch.device, precision: str, *, step: int) -> float:
    starts, lengths = sample_validation_windows(data, config, step)
    model.eval()
    total = torch.zeros((), device=device, dtype=torch.float64)
    for first in range(0, len(starts), config.batch_size):
        stop = first + config.batch_size
        x, y = data.batch(starts[first:stop], device)
        if stop >= len(starts) and lengths[-1] < data.context_length:
            y = y.clone()  # A single input and target window can share storage.
            y[-1, int(lengths[-1]):] = -100
        total += loss_for(model, x, y, precision).double() * int(lengths[first:stop].sum())
    model.train()
    return total.item() / int(lengths.sum())


@dataclass
class RunSession:
    """Mutable handle providing MLflow run, Azure store, checkpoint uploader, and stop flag."""

    run: object
    store: CheckpointStore
    log_path: Path
    log_lock: object
    save_checkpoint: object
    stop_requested: bool = False


@contextmanager
def open_run_log(log_label: str, *, logs_dir: Path | None = None):
    """Initialize `.env`, single per-run log teeing, and graceful SIGINT/SIGTERM capture."""
    load_dotenv(PROJECT_ROOT / ".env")
    resolved_logs_dir = Path(logs_dir) if logs_dir is not None else PROJECT_ROOT / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = resolved_logs_dir / f"train_{timestamp}_{log_label}.log"

    state = {"stop_requested": False}
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_signal(sig, frame):
        state["stop_requested"] = True
        print(f"\nReceived termination signal ({sig}). Finishing current step and saving checkpoint...", flush=True)

    with ExitStack() as stack:
        stack.callback(signal.signal, signal.SIGTERM, prev_sigterm)
        stack.callback(signal.signal, signal.SIGINT, prev_sigint)
        log_lock = stack.enter_context(tee_to_log(log_path))
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        print(f"Logging run to {log_path}", flush=True)
        yield log_path, log_lock, lambda: state["stop_requested"]


@contextmanager
def tracked_training_session(
    *,
    run_params: dict,
    log_path: Path,
    log_lock,
    is_stop_requested,
    azure_container: str | None = None,
    resume_run_id: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Open Databricks MLflow run, prepare Azure store, and upload checkpoints/logs reliably."""
    if not all(os.getenv(key, "").strip() for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")):
        raise ValueError("Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.")

    container = azure_container or AZURE_CONTAINER
    with track_run(
        "databricks",
        MLFLOW_EXPERIMENT,
        run_params,
        run_id=resume_run_id,
    ) as run:
        store = CheckpointStore(container, run.id)
        store.prepare()
        run.set_tag("checkpoint_location", store.location)
        for key, value in (tags or {}).items():
            run.set_tag(key, str(value))
        print(f"MLflow run ID: {run.id} | Azure checkpoints: {store.location}", flush=True)

        run_failed = False
        try:
            with checkpoint_uploads(store, run, log_path=log_path, log_lock=log_lock) as save_checkpoint:
                yield RunSession(
                    run=run,
                    store=store,
                    log_path=log_path,
                    log_lock=log_lock,
                    save_checkpoint=save_checkpoint,
                ), is_stop_requested
        except BaseException as error:
            import traceback

            run_failed = True
            print(f"\nTraining aborted due to {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            raise
        finally:
            run.flush()
            azure_log_uri = f"{store.location}/logs/{log_path.name}"
            print(f"Uploading run log to Azure ({azure_log_uri}) and Databricks MLflow...", flush=True)
            try:
                publish_run_log(store, log_path, run, lock=log_lock)
            except Exception as error:
                if run_failed:
                    print(f"Warning: could not upload run log during failure cleanup ({error}).", flush=True)
                else:
                    raise
