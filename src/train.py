from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory

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
    eval_windows: int
    seed: int

    def __post_init__(self):
        if min(self.batch_size, self.steps, self.eval_every, self.eval_windows) < 1:
            raise ValueError(
                "Batch, step, and evaluation counts must be positive")
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
        self.windows = (size - 1) // context
        if self.windows < 1:
            raise ValueError(f"{path} needs at least {context + 1} tokens")

    def batch(self, starts: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokens[starts[:, None] + self.offsets]
        ids = torch.from_numpy(batch.astype(np.int64)).to(device)
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


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
def evaluate(model, data, config, device, precision) -> float:
    model.eval()
    count = min(data.windows, config.eval_windows)
    windows = np.linspace(0, data.windows - 1, num=count, dtype=np.int64)
    starts = windows * model.config.context_length
    total = torch.zeros((), device=device)
    for chunk in range(0, count, config.batch_size):
        offsets = starts[chunk: chunk + config.batch_size]
        x, y = data.batch(offsets, device)
        total += loss_for(model, x, y, precision) * len(offsets)
    model.train()
    return total.item() / count


def save_checkpoint(path: Path, state: dict) -> None:
    with TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as temporary:
        staging = Path(temporary) / path.name
        with staging.open("wb") as stream:
            torch.save(state, stream)
        staging.replace(path)


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
