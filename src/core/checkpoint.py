"""Checkpoint serialization, normalization, and Azure/local resolution."""

from dataclasses import asdict
import os
from pathlib import Path
import re

import torch

from .model import GPTConfig
from ..settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT, PROJECT_ROOT


def load_checkpoint(path: Path) -> dict:
    """Load a serialized MiniReasoner checkpoint and normalize legacy field names."""
    with Path(path).open("rb") as stream:
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


def build_model_config(checkpoint: dict, *, context_length: int | None = None) -> GPTConfig:
    """Instantiate `GPTConfig` from a checkpoint with an optional context extension (e.g. 2048)."""
    cfg_dict = dict(checkpoint["model_config"])
    if context_length is not None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        cfg_dict["context_length"] = context_length
    return GPTConfig(**cfg_dict)


def _discover_active_base_run(container: str, *, experiment: str) -> str:
    """Pick the newest active base pre-training run from Azure Blob Storage + Databricks MLflow."""
    from ..infra.tracking import discover_azure_runs

    runs = discover_azure_runs(
        container,
        experiment=experiment,
        preferred_stages=("pretrain", "base", ""),
    )
    if not runs:
        raise FileNotFoundError(f"No checkpoints found under {experiment} in Azure container {container!r}.")
    return runs[0]["run_id"]


def resolve_base_checkpoint(
    *,
    checkpoint_path: Path | None = None,
    run_id: str | None = None,
    alias: str = "latest",
    azure_container: str = AZURE_CONTAINER,
    experiment: str = MLFLOW_EXPERIMENT,
    checkpoints_dir: Path | None = None,
) -> tuple[Path, dict]:
    """Locate, optionally cache from Azure, load, and return `(local_path, checkpoint_dict)`."""
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Base checkpoint file not found: {resolved}")
        ckpt = load_checkpoint(resolved)
        return resolved, ckpt

    from ..infra.checkpoint_store import CheckpointStore

    target_run_id = run_id or _discover_active_base_run(azure_container, experiment=experiment)
    store = CheckpointStore(azure_container, target_run_id, experiment=experiment)
    versions = store._versions()
    if not versions:
        raise FileNotFoundError(f"No valid checkpoints found for run {target_run_id!r} in {store.location}.")

    if alias == "latest":
        selected = max(versions, key=lambda item: item.step)
    elif alias == "best":
        selected = versions[0]
    elif alias_match := re.fullmatch(r"step-(\d+)", alias):
        step_num = int(alias_match[1])
        selected = next((item for item in versions if item.step == step_num), None)
        if selected is None:
            raise FileNotFoundError(f"Step {step_num} not found in run {target_run_id}.")
    else:
        raise ValueError("checkpoint alias must be 'latest', 'best', or 'step-N'.")

    cache_root = Path(checkpoints_dir) if checkpoints_dir is not None else PROJECT_ROOT / "checkpoints"
    cached_path = cache_root / target_run_id / f"{alias}_step_{selected.step}_{selected.sha256[:8]}.pt"
    if not cached_path.is_file():
        print(f"Downloading base checkpoint ({alias}, step {selected.step:,}) from Azure to {cached_path}...", flush=True)
        store.download(alias, cached_path)
    else:
        print(f"Using cached base checkpoint: {cached_path}", flush=True)

    ckpt = load_checkpoint(cached_path)
    ckpt.setdefault("parent_run_id", target_run_id)
    ckpt.setdefault("model_config_dict", asdict(build_model_config(ckpt)))
    return cached_path, ckpt
