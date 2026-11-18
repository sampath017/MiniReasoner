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
        raise RuntimeError(
            "Training requires a CUDA GPU. Enable a GPU or set train_model=False for inference."
        )
    return torch.device("cuda")


def resolve_precision(request: str, device: torch.device) -> str:
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
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
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
