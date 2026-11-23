from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .tokenizer import TOKEN_DTYPE


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
            raise ValueError(
                "Batch, step, and evaluation counts must be positive")
        if not 0 < self.eval_token_fraction <= 1:
            raise ValueError("eval_token_fraction must be greater than zero and at most one")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("Warmup must be shorter than the run")
        if not 0 <= self.min_learning_rate <= self.learning_rate or self.learning_rate == 0:
            raise ValueError(
                "Require 0 <= min_learning_rate <= learning_rate and a positive peak")
        if self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative")

    def lr_at(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        decay_intervals = max(1, self.steps - self.warmup_steps - 1)
        progress = (step - self.warmup_steps - 1) / decay_intervals
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_learning_rate + cosine * (self.learning_rate - self.min_learning_rate)


class TokenStream:
    def __init__(self, path: Path, size: int, context: int):
        with path.open("rb") as stream:
            self.tokens = np.memmap(
                stream, dtype=TOKEN_DTYPE, mode="r", shape=(size,))
        self.offsets = np.arange(context + 1)
        self.context_length = context
        self.target_count = size - 1
        self.windows, self.tail_targets = divmod(self.target_count, context)
        if self.target_count < 1:
            raise ValueError(f"{path} needs at least two tokens for next-token prediction")

    def batch(self, starts: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = starts[:, None] + self.offsets
        if indices[:, -1].max() >= len(self.tokens):
            # Pad the last short window and omit the padded targets from cross-entropy.
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
        raise RuntimeError("Training requires a CUDA GPU. For CPU inference, set train_model=False.")
    return torch.device("cuda")


def resolve_precision(request: str, device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError("Training precision requires a CUDA device.")
    if request == "fp32":
        return "fp32"
    if request not in ("auto", "bf16"):
        raise ValueError("Training precision must be auto, fp32, or bf16")
    native_bf16 = torch.cuda.get_device_capability(
        device)[0] >= 8 and torch.cuda.is_bf16_supported()
    if request == "auto":
        return "bf16" if native_bf16 else "fp32"
    if not native_bf16:
        raise ValueError("This CUDA device does not have native BF16 support")
    return "bf16"


def loss_for(model, inputs, targets, precision):
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(inputs)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.inference_mode()
def evaluate(model, data, config, device, precision, *, step: int) -> float:
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


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    for old_name, new_name in (
        ("context", "context_length"),
        ("block_size", "context_length"),
        ("layers", "n_layer"),
        ("heads", "n_head"),
        ("width", "n_embd"),
        ("embedding_dim", "n_embd"),
    ):
        if old_name in config:
            config.setdefault(new_name, config.pop(old_name))

    # Keep old model weights and optimizer parameter order compatible after renaming.
    aliases = {
        "tokens.weight": "transformer.wte.weight",
        "embedding_table.weight": "transformer.wte.weight",
        "positions.weight": "transformer.wpe.weight",
        "position_embedding_table.weight": "transformer.wpe.weight",
        "norm.weight": "transformer.ln_f.weight",
        "norm.bias": "transformer.ln_f.bias",
    }
    state = checkpoint["model"]
    for name in list(state):
        renamed = aliases.get(name, name)
        if name.startswith("blocks."):
            renamed = "transformer.h." + name.removeprefix("blocks.")
            for old_name, new_name in (
                (".attention_norm.", ".ln_1."),
                (".attention.qkv.", ".attn.c_attn."),
                (".attention.out.", ".attn.c_proj."),
                (".mlp_norm.", ".ln_2."),
                (".mlp.0.", ".mlp.c_fc."),
                (".mlp.2.", ".mlp.c_proj."),
            ):
                renamed = renamed.replace(old_name, new_name)
        if renamed != name:
            state[renamed] = state.pop(name)
    state.setdefault("lm_head.weight", state["transformer.wte.weight"])
    return checkpoint


def ensure_tokenized_corpus(
    *,
    dataset_name: str = "SimpleStories/SimpleStories",
    source_splits: dict[str, str] | None = None,
    seed: int = 42,
    validation_fraction: float | None = None,
    target_vocab_size: int | None = None,
    cache_dir: Path | None = None,
    tokenized_path: Path | None = None,
    drive_data_root: str | None = None,
    tokenizer_device: str = "auto",
) -> dict:
    import json
    import os
    from hashlib import blake2b
    from tempfile import TemporaryDirectory
    from tokenizers import Tokenizer

    from .drive_corpus import load_raw_training_stories, publish_tokenized, reuse_tokenized
    from .settings import (
        DATASET_CACHE_DIR, DRIVE_DATA_ROOT, SIMPLESTORIES_TRAIN_FILES,
        TOKENIZED_DATA_DIR, VALIDATION_FRACTION, VOCAB_SIZE,
    )
    from .tokenizer import (
        EOS, TOKEN_DTYPE, pack_documents, publish_prepared_corpus, sha256,
        story_texts, train_tokenizer, verify_corpus_files, verify_packed_eos,
    )

    if source_splits is None:
        source_splits = {"train": "train", "val": "train"}
    if validation_fraction is None:
        validation_fraction = VALIDATION_FRACTION
    if target_vocab_size is None:
        target_vocab_size = VOCAB_SIZE
    if cache_dir is None:
        cache_dir = DATASET_CACHE_DIR
    if tokenized_path is None:
        tokenized_path = TOKENIZED_DATA_DIR
    if drive_data_root is None:
        drive_data_root = DRIVE_DATA_ROOT

    corpus_settings = {
        "dataset": dataset_name,
        "source_splits": source_splits,
        "seed": seed,
        "split_method": "story_hash_v1",
        "validation_fraction": validation_fraction,
        "target_vocab_size": target_vocab_size,
    }

    manifest_path = tokenized_path / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_corpus_files(tokenized_path, manifest)
            print(f"Reusing verified packed data on local disk at {tokenized_path}.")
            return manifest
        except Exception as e:
            print(f"Local tokenized corpus verification failed ({e}); checking Google Drive...")

    manifest = reuse_tokenized(drive_data_root, tokenized_path, corpus_settings)
    if manifest is not None:
        print(f"Restored and verified packed data from Google Drive to {tokenized_path}.")
        return manifest

    print("No prepared corpus found. Preparing SimpleStories dataset from scratch...")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    dataset = load_raw_training_stories(
        drive_data_root, cache_dir, dataset_name, SIMPLESTORIES_TRAIN_FILES,
        token=os.getenv("HF_TOKEN", "").strip() or None,
    )

    split_indices = {"train": [], "val": []}
    validation_threshold = int(validation_fraction * (1 << 64))
    hash_key = str(seed).encode("ascii")
    row_index = 0
    for batch in dataset.iter(batch_size=2048):
        for text in batch["story"]:
            story = (text or "").strip()
            if story and EOS not in story:
                digest = blake2b(story.encode("utf-8"), digest_size=8, key=hash_key).digest()
                split_name = "val" if int.from_bytes(digest, "big") < validation_threshold else "train"
                split_indices[split_name].append(row_index)
            row_index += 1

    splits = {}
    for name, indices in split_indices.items():
        if not indices:
            raise ValueError(f"No usable {name} stories. Adjust VALIDATION_FRACTION.")
        splits[name] = dataset.select(indices)

    print(f"Training tokenizer with target vocab size {target_vocab_size:,}...")
    tokenizer = train_tokenizer(story_texts(splits["train"]), target_vocab_size, device=tokenizer_device)

    tokenized_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=tokenized_path.parent) as temporary:
        staging = Path(temporary) / tokenized_path.name
        staging.mkdir()
        tokenizer_path = staging / "tokenizer.json"
        tokenizer_path.write_text(tokenizer.to_str(pretty=True), encoding="utf-8")
        saved_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        eos_id = saved_tokenizer.token_to_id(EOS)

        manifest = {
            "version": 1,
            "dtype": TOKEN_DTYPE.str,
            **corpus_settings,
            "source_rows": len(dataset),
            "vocab_size": saved_tokenizer.get_vocab_size(),
            "eos_id": eos_id,
            "tokenizer_sha256": sha256(tokenizer_path),
        }
        for name, rows in splits.items():
            print(f"Packing {len(rows):,} {name} stories...", flush=True)
            split_info, story_ends = pack_documents(
                staging / f"{name}.bin", saved_tokenizer, story_texts(rows)
            )
            verify_packed_eos(staging / f"{name}.bin", split_info, story_ends, eos_id)
            manifest[name] = split_info

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        publish_prepared_corpus(staging, tokenized_path)

    publish_tokenized(drive_data_root, tokenized_path, corpus_settings)
    return manifest


class TeeLogger:
    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def run_training(
    *,
    preset: str = "base",
    resume: bool = False,
    mlflow_run_id: str | None = None,
    stop_after: int | None = None,
    precision_request: str = "auto",
    seed: int = 42,
    dataset_name: str = "SimpleStories/SimpleStories",
    source_splits: dict[str, str] | None = None,
    azure_container: str | None = None,
    tokenizer_device: str = "auto",
    loss_decimals: int = 4,
    device: torch.device | None = None,
) -> dict:
    from dataclasses import asdict
    from datetime import datetime
    import os
    import shutil
    import signal
    import sys
    from time import perf_counter
    from dotenv import load_dotenv

    from .checkpoint_store import CheckpointStore
    from .metrics import track_run
    from .model import GPT, GPTConfig
    from .settings import (
        AZURE_CONTAINER, DATASET_CACHE_DIR, DRIVE_DATA_ROOT,
        MLFLOW_EXPERIMENT, PRESETS, TOKENIZED_DATA_DIR,
        VALIDATION_FRACTION, VOCAB_SIZE,
    )
    from .tokenizer import TOKEN_DTYPE, verify_corpus_files
    from .tracking import checkpoint_uploads, download_checkpoint

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    # Setup per-run logging to logs/
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{timestamp}_{preset}.log"
    log_path = logs_dir / log_filename
    latest_path = logs_dir / "latest.log"
    log_file = log_path.open("a", encoding="utf-8")

    # Point latest.log to the active run log immediately so `tail -f logs/latest.log` works live.
    using_latest_symlink = False
    try:
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(log_filename)
        using_latest_symlink = True
    except OSError:
        pass

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = TeeLogger(orig_stdout, log_file)
    sys.stderr = TeeLogger(orig_stderr, log_file)

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    run_id_log_path: Path | None = None
    using_run_id_symlink = False

    try:
        print(f"Logging run to {log_path} (latest: {latest_path})", flush=True)

        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(PRESETS.keys())}")
        settings = PRESETS[preset]

        if azure_container is None:
            azure_container = AZURE_CONTAINER

        if device is None:
            device = require_training_gpu()
        precision = resolve_precision(precision_request, device) if device.type == "cuda" else "fp32"

        tracking_uri = "databricks"
        if not all(os.getenv(key, "").strip() for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")):
            raise ValueError("Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.")
        if resume and not mlflow_run_id:
            raise ValueError("Set mlflow_run_id when resuming a previous run.")

        tokenized_path = TOKENIZED_DATA_DIR
        manifest = ensure_tokenized_corpus(
            dataset_name=dataset_name,
            source_splits=source_splits,
            seed=seed,
            validation_fraction=VALIDATION_FRACTION,
            target_vocab_size=VOCAB_SIZE,
            cache_dir=DATASET_CACHE_DIR,
            tokenized_path=tokenized_path,
            drive_data_root=DRIVE_DATA_ROOT,
            tokenizer_device=tokenizer_device,
        )

        verify_corpus_files(tokenized_path, manifest)
        tokenizer_path = tokenized_path / "tokenizer.json"
        tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
        vocab_size = manifest["vocab_size"]

        model_config = GPTConfig(vocab_size=vocab_size, **settings["model"])
        head_dim = model_config.n_embd // model_config.n_head
        print(
            f"MiniReasoner {preset} model: vocab={model_config.vocab_size:,}, "
            f"layers={model_config.n_layer}, heads={model_config.n_head}, "
            f"n_embd={model_config.n_embd:,}, head_dim={head_dim}, "
            f"context_length={model_config.context_length:,}, pos_encoding={model_config.pos_encoding}",
            flush=True,
        )

        streams = {}
        for split in ("train", "val"):
            path = tokenized_path / f"{split}.bin"
            size = manifest[split]["tokens"]
            streams[split] = TokenStream(path, size, model_config.context_length)

        training_settings = settings["training"].copy()
        epochs = training_settings.pop("epochs", None)
        sampler = EpochBatchSampler(streams["train"], training_settings["batch_size"], training_settings["seed"])
        if epochs is not None:
            if not isinstance(epochs, int) or epochs < 1:
                raise ValueError("epochs must be a positive integer")
            training_settings["steps"] = epochs * sampler.steps_per_epoch

        config = TrainConfig(**training_settings)
        val_data = streams["val"]
        val_targets_per_check = math.ceil(val_data.target_count * config.eval_token_fraction)
        stop = config.steps if stop_after is None else stop_after
        if not 1 <= stop <= config.steps:
            raise ValueError("stop_after must be between 1 and the configured total steps")

        train_data = streams["train"]
        scheduled_targets = sampler.targets_through(stop)
        print(
            f"Run schedule: batch_size={config.batch_size:,}, steps_per_epoch={sampler.steps_per_epoch:,}, "
            f"steps={stop:,} ({scheduled_targets / train_data.target_count:.4f} epochs)",
            flush=True,
        )

        torch.manual_seed(config.seed)
        model = GPT(model_config).to(device)
        decay, no_decay = [], []
        for parameter in model.parameters():
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)

        optimizer = torch.optim.AdamW(
            [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            fused=(device.type == "cuda"),
        )

        run_identity = {
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "sampling": "shuffled_windows_v1",
            "data_manifest": manifest,
        }

        start = 0
        if resume:
            print(f"Resuming run {mlflow_run_id} from Azure container {azure_container}...", flush=True)
            with download_checkpoint(CheckpointStore(azure_container, mlflow_run_id)) as checkpoint_path:
                checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint["version"] != 1 or any(checkpoint.get(k) != v for k, v in run_identity.items()):
                raise ValueError("Resume requires matching training settings and data. Set resume=False for a fresh run.")
            start = checkpoint["step"]
            if stop <= start:
                raise ValueError(f"stop_after ({stop}) must exceed completed step ({start})")
            model.load_state_dict(checkpoint["model"])
            for group in checkpoint["optimizer"]["param_groups"]:
                group["fused"] = (device.type == "cuda")
            optimizer.load_state_dict(checkpoint["optimizer"])
            history = checkpoint["metrics"]
            del checkpoint
            print(f"Successfully resumed model and optimizer state from step {start:,}.", flush=True)
        else:
            history = []

        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        device_desc = f"{torch.cuda.get_device_name(device)} | {precision}" if device.type == "cuda" else f"{device} | {precision}"
        print(f"Parameters: {num_parameters:,} | Runtime: {device_desc} | Steps {start + 1:,} to {stop:,}", flush=True)

        # Graceful stop handling (SIGINT/SIGTERM)
        stop_requested = False
        def handle_signal(sig, frame):
            nonlocal stop_requested
            print(f"\nReceived termination signal ({sig}). Finishing current step and saving checkpoint...", flush=True)
            stop_requested = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        train_loss = float("nan")
        val_loss = float("nan")
        last_step = start
        with track_run(
            tracking_uri, MLFLOW_EXPERIMENT, {"preset": preset, "device": device.type, "precision": precision, **run_identity},
            run_id=mlflow_run_id if resume else None,
        ) as run:
            active_run_id = run.id
            run_id_log_path = logs_dir / f"train_{active_run_id}.log"
            try:
                if run_id_log_path.is_symlink() or run_id_log_path.exists():
                    run_id_log_path.unlink()
                run_id_log_path.symlink_to(log_filename)
                using_run_id_symlink = True
            except OSError:
                pass
            store = CheckpointStore(azure_container, active_run_id)
            store.prepare()
            run.set_tag("checkpoint_location", store.location)
            print(f"MLflow run ID: {active_run_id} | Azure checkpoints: {store.location}", flush=True)

            if resume:
                store.prune()
                for row in history:
                    run.log(row, step=row["step"])
                if history:
                    val_loss = history[-1].get("val_loss", val_loss)
            else:
                print("Evaluating validation loss at step 0 before first training batch...", flush=True)
                initial_loss = evaluate(model, val_data, config, device, precision, step=0)
                val_loss = initial_loss
                history = [{"step": 0, "tokens": 0, "val_loss": initial_loss, "val_targets": val_targets_per_check}]
                print(f"Step 0 | mean train loss N/A | validation loss {initial_loss:.{loss_decimals}f}", flush=True)
                run.log(history[0], step=0)

            with checkpoint_uploads(store, run) as save_checkpoint:
                interval_start = perf_counter()
                interval_start_targets = sampler.targets_through(start)
                interval_loss_sum = 0.0

                for step in range(start + 1, stop + 1):
                    if stop_requested:
                        print(f"Stopping early as requested at step {last_step:,}.", flush=True)
                        break

                    lr = config.lr_at(step)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)

                    starts = sampler.starts(step)
                    x, y = streams["train"].batch(starts, device)
                    loss = loss_for(model, x, y, precision)
                    tokens_seen = sampler.targets_through(step)
                    batch_targets = tokens_seen - sampler.targets_through(step - 1)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    last_step = step

                    train_loss = loss.detach().item()
                    if not math.isfinite(train_loss):
                        raise FloatingPointError(f"Non-finite training loss at step {step}.")
                    print(f"Step {step:,} | train loss {train_loss:.{loss_decimals}f}", flush=True)
                    interval_loss_sum += train_loss * batch_targets

                    metrics = {"step": step, "train_loss": train_loss}
                    is_eval_step = (step % config.eval_every == 0) or (step == stop) or stop_requested
                    if not is_eval_step:
                        run.log(metrics, step=step)
                        continue

                    interval_targets = tokens_seen - interval_start_targets
                    interval_mean = interval_loss_sum / interval_targets
                    throughput = interval_targets / max(1e-6, perf_counter() - interval_start)
                    val_loss = evaluate(model, val_data, config, device, precision, step=step)
                    if not math.isfinite(val_loss):
                        raise FloatingPointError(f"Non-finite validation loss at step {step}.")

                    print(
                        f"Step {step:,} | mean train loss {interval_mean:.{loss_decimals}f} | "
                        f"validation loss {val_loss:.{loss_decimals}f} | {round(throughput):,} tok/s",
                        flush=True,
                    )
                    metrics.update({
                        "tokens": tokens_seen,
                        "epoch": tokens_seen / streams["train"].target_count,
                        "train_loss_interval_mean": interval_mean,
                        "val_loss": val_loss,
                        "val_targets": val_targets_per_check,
                        "lr": lr,
                        "train_tokens_per_second": round(throughput),
                    })
                    run.log(metrics, step=step)
                    history.append(metrics)

                    print(f"Saving checkpoint snapshot at step {step:,}...", flush=True)
                    save_checkpoint(
                        {
                            "version": 1,
                            **run_identity,
                            "step": step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "tokenizer_json": tokenizer_json,
                            "metrics": history,
                        },
                    )
                    interval_loss_sum = 0.0
                    interval_start_targets = tokens_seen
                    interval_start = perf_counter()

                    if stop_requested:
                        print(f"Stopping early after saving checkpoint at step {last_step:,}.", flush=True)
                        break

        status = "Completed" if not stop_requested and last_step >= config.steps else "Stopped"
        print(f"Training {status} at step {last_step:,}. Final train loss: {train_loss:.{loss_decimals}f}, val loss: {val_loss:.{loss_decimals}f}", flush=True)
        return {
            "status": status,
            "step": last_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mlflow_run_id": active_run_id,
        }
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()
        if not using_latest_symlink:
            try:
                shutil.copyfile(log_path, latest_path)
            except Exception:
                pass
        if run_id_log_path is not None and not using_run_id_symlink:
            try:
                shutil.copyfile(log_path, run_id_log_path)
            except Exception:
                pass
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MiniReasoner Training CLI")
    parser.add_argument("--preset", default="base", choices=["base", "small"], help="Model preset (default: base)")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing MLflow run")
    parser.add_argument("--run-id", default=None, help="Databricks MLflow run ID to resume")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop training at absolute optimizer step")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"], help="Precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer-device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for tokenizer training")

    args = parser.parse_args(argv)
    run_training(
        preset=args.preset,
        resume=args.resume,
        mlflow_run_id=args.run_id,
        stop_after=args.stop_after,
        precision_request=args.precision,
        seed=args.seed,
        tokenizer_device=args.tokenizer_device,
    )

