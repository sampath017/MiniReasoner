"""Shared environment, Azure checkpoint discovery, model loading, and plotting helpers for notebooks."""

from pathlib import Path
import re
import sys

from dotenv import load_dotenv
import torch
from tokenizers import Tokenizer

from ..core.checkpoint import build_model_config, load_checkpoint, resolve_base_checkpoint
from ..core.model import GPT
from ..core.tokenizer import EOS
from ..settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT, PROJECT_ROOT
from .checkpoint_store import CheckpointStore
from .tracking import discover_azure_runs


def setup_notebook_environment(start_dir: Path | None = None) -> Path:
    """Locate repository root, load `.env`, and add project root to `sys.path`."""
    base = (start_dir or Path.cwd()).resolve()
    root = next(
        (p for p in (base, *base.parents) if (p / "src" / "main.py").is_file() and (p / "pyproject.toml").is_file()),
        PROJECT_ROOT,
    )
    load_dotenv(root / ".env")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print(f"Project root located at: {root}")
    return root


def configure_eval_device(preference: str = "auto") -> tuple[torch.device, torch.dtype]:
    """Select CPU/CUDA device and BF16/FP32 inference dtype cleanly with hardware printout."""
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preference in ("cuda", "cpu"):
        if preference == "cuda" and not torch.cuda.is_available():
            print("Notice: CUDA requested but unavailable on this system. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(preference)
    else:
        raise ValueError(f"Unknown device preference: {preference!r}")

    print("=" * 60)
    print(f"ACTIVE EXECUTION DEVICE: {device.type.upper()}")
    print("=" * 60)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cap = torch.cuda.get_device_capability(0)
        bf16 = torch.cuda.is_bf16_supported() and cap[0] >= 8
        dtype = torch.bfloat16 if bf16 else torch.float32
        print(f"GPU Hardware:       {gpu_name}")
        print(f"Total VRAM:         {vram_gb:.2f} GB")
        print(f"Compute Capability: {cap[0]}.{cap[1]}")
        print(f"Inference Dtype:    {dtype}")
    else:
        dtype = torch.float32
        print(f"CPU Threads:        {torch.get_num_threads()}")
        print("Inference Dtype:    torch.float32")
        print("Mode:               Portable CPU Inference")
    print("=" * 60)
    return device, dtype


def select_run_and_checkpoint(
    *,
    run_id: str | None = None,
    checkpoint_alias: str = "latest",
    local_checkpoint: Path | str | None = None,
    preferred_stages: tuple[str, ...] | None = None,
    preferred_arith_format: str | None = None,
    azure_container: str = AZURE_CONTAINER,
    experiment: str = MLFLOW_EXPERIMENT,
    checkpoints_dir: Path | None = None,
) -> tuple[str, Path, object | None]:
    """Discover Azure runs, pick target checkpoint snapshot, cache locally, and return `(run_id, path, snapshot)`."""
    ckpt_root = Path(checkpoints_dir) if checkpoints_dir is not None else PROJECT_ROOT / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if local_checkpoint is not None:
        resolved = Path(local_checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Local checkpoint not found: {resolved}")
        print(f"Using offline local checkpoint: {resolved}")
        return resolved.parent.name, resolved, None

    if run_id is not None:
        selected_run_id = run_id
    else:
        discovered = discover_azure_runs(
            azure_container,
            experiment=experiment,
            preferred_stages=preferred_stages,
            preferred_arith_format=preferred_arith_format,
        )
        if not discovered:
            raise FileNotFoundError(f"No checkpoints found under {experiment} in Azure container '{azure_container}'.")
        print(f"Discovered {len(discovered)} run(s) in Azure container '{azure_container}':\n")
        print(f"{'Run ID':<36}{'Stage':<10}{'Format':<12}{'MLflow State':<14}{'Snapshots':<11}{'Last Updated (UTC)'}")
        print("-" * 104)
        for row in discovered:
            ts = row["last_modified"].strftime("%Y-%m-%d %H:%M:%S") if row["last_modified"] else "N/A"
            fmt = row.get("arith_format") or "-"
            print(
                f"{row['run_id']:<36}{row['mlflow_stage']:<10}{fmt:<12}"
                f"{row['mlflow_state']:<14}{row['blob_count']:<11}{ts}"
            )
        selected_run_id = discovered[0]["run_id"]
        print(
            f"\nAuto-selected run: {selected_run_id} "
            f"(stage={discovered[0]['mlflow_stage']}, format={discovered[0].get('arith_format') or '-'})"
        )

    run_cache_dir = ckpt_root / selected_run_id
    try:
        store = CheckpointStore(azure_container, selected_run_id, experiment=experiment)
        versions = store._versions()
    except Exception as exc:
        local_matches = sorted(run_cache_dir.glob(f"{checkpoint_alias}_step_*_*.pt"))
        if local_matches:
            fallback = local_matches[-1]
            size_mb = fallback.stat().st_size / (1024 * 1024)
            print(f"Azure lookup unavailable ({type(exc).__name__}); using local cached checkpoint: {fallback} ({size_mb:.2f} MB)")
            return selected_run_id, fallback, None
        raise
    if not versions:
        raise FileNotFoundError(f"No valid checkpoints found for run {selected_run_id!r} in {store.location}.")

    print(f"\nTop checkpoints by validation loss for {selected_run_id} ({len(versions)} total):\n")
    print(f"{'Rank':<6}{'Step':<8}{'Val Loss':<12}{'Train Mean':<12}{'Filename'}")
    print("-" * 85)
    for rank, snap in enumerate(versions[:5], start=1):
        print(f"#{rank:<5}{snap.step:<8}{snap.val_loss:<12.4f}{snap.train_loss_interval_mean:<12.4f}{snap.name[:45]}...")

    if checkpoint_alias == "best":
        target_snap = versions[0]
    elif checkpoint_alias == "latest":
        target_snap = max(versions, key=lambda item: item.step)
    elif alias_match := re.fullmatch(r"step-(\d+)", checkpoint_alias):
        step_num = int(alias_match[1])
        target_snap = next((item for item in versions if item.step == step_num), None)
        if target_snap is None:
            raise FileNotFoundError(f"Step {step_num} not found in run {selected_run_id}.")
    else:
        raise ValueError("checkpoint_alias must be 'best', 'latest', or 'step-N'.")

    latest_snap = max(versions, key=lambda item: item.step)
    print(
        f"\n--> Selected ({checkpoint_alias}): Step {target_snap.step:,} | "
        f"Val Loss: {target_snap.val_loss:.4f} | Train Mean: {target_snap.train_loss_interval_mean:.4f}"
    )
    if latest_snap.step != target_snap.step:
        print(
            f"    Latest step in run: Step {latest_snap.step:,} | "
            f"Val Loss: {latest_snap.val_loss:.4f} | Train Mean: {latest_snap.train_loss_interval_mean:.4f}"
        )

    run_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = run_cache_dir / f"{checkpoint_alias}_step_{target_snap.step}_{target_snap.sha256[:8]}.pt"
    if not cached_path.is_file():
        print(f"Downloading checkpoint from Azure to {cached_path}...")
        store.download(checkpoint_alias, cached_path)
        print("Download complete and verified via SHA-256 checksum.")
    else:
        size_mb = cached_path.stat().st_size / (1024 * 1024)
        print(f"Using locally cached checkpoint: {cached_path} ({size_mb:.2f} MB)")

    return selected_run_id, cached_path, target_snap


def load_checkpoint_bundle(
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    snapshot=None,
    compare_with_base: bool = False,
    azure_container: str = AZURE_CONTAINER,
    checkpoints_dir: Path | None = None,
) -> dict:
    """Load model, tokenizer, architecture/SFT summary, metrics history, and optional parent base model."""
    print(f"Loading checkpoint weights from: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path)
    ckpt.pop("optimizer", None)

    tokenizer = Tokenizer.from_str(ckpt["tokenizer_json"])
    model_config = build_model_config(ckpt)
    model = GPT(model_config).to(device).eval()
    model.load_state_dict(ckpt["model"])
    ckpt.pop("model", None)
    if device.type == "cuda" and dtype == torch.bfloat16:
        model = model.to(dtype=torch.bfloat16)

    step = int(ckpt.get("step", snapshot.step if snapshot is not None else 0))
    stage = ckpt.get("stage", "sft" if "arith_manifest" in ckpt else "pretrain")
    arith_manifest = ckpt.get("arith_manifest") or {}
    parent_run_id = ckpt.get("parent_run_id")
    parent_step = ckpt.get("parent_checkpoint_step")
    retention_ratio = ckpt.get("retention_ratio")
    metrics_history = ckpt.get("metrics", [])

    total_params = sum(p.numel() for p in model.parameters())
    wte_params = model.transformer.wte.weight.numel()
    blocks_params = sum(p.numel() for p in model.transformer.h.parameters())
    ln_f_params = sum(p.numel() for p in model.transformer.ln_f.parameters())
    head_dim = model_config.n_embd // model_config.n_head
    pos_desc = (
        f"RoPE (theta={model_config.rope_theta:g}, head_dim={head_dim})"
        if model_config.pos_encoding == "rope"
        else model_config.pos_encoding
    )

    print("\n" + "=" * 62)
    print("MINIREASONER CHECKPOINT & ARCHITECTURE SUMMARY")
    print("=" * 62)
    print(f"Stage / Step:                {stage} (step {step:,})")
    print(f"Positional Encoding:         {pos_desc}")
    print(f"Architecture:                {model_config.n_layer}L / {model_config.n_head}H ({head_dim} dim/head) / {model_config.n_embd}D")
    print(f"Context Window Length:       {model_config.context_length:,} tokens")
    print(f"Vocabulary Size:             {model_config.vocab_size:,} tokens")
    print(f"Total Parameters:            {total_params:,} (~{total_params / 1e6:.2f}M)")
    print(f"Breakdown (wte/blocks/ln_f): {wte_params:,} / {blocks_params:,} / {ln_f_params:,}")
    print(f"EOS Token:                   '{EOS}' (ID: {tokenizer.token_to_id(EOS)})")
    print(f"Active Device / Dtype:       {device.type.upper()} / {dtype}")
    if arith_manifest:
        print("-" * 62)
        print(f"Parent Base Lineage:         {parent_run_id} (step {parent_step})")
        print(
            f"Arithmetic Config:           {arith_manifest.get('generator_version')} | "
            f"format={arith_manifest.get('answer_format', 'direct')} | "
            f"multistep={arith_manifest.get('include_multistep', False)}"
        )
        print(
            f"Ops / Digits / Negatives:    {arith_manifest.get('ops')} | "
            f"max_digits={arith_manifest.get('max_digits')} | "
            f"allow_negative={arith_manifest.get('allow_negative')}"
        )
        if retention_ratio is not None:
            print(f"Story Retention Ratio:       {retention_ratio:.2f}")
    print("=" * 62)

    base_model = None
    base_tokenizer = None
    if compare_with_base and parent_run_id:
        try:
            base_alias = f"step-{int(parent_step)}" if parent_step else "latest"
            base_path, base_ckpt = resolve_base_checkpoint(
                run_id=str(parent_run_id),
                alias=base_alias,
                azure_container=azure_container,
                checkpoints_dir=checkpoints_dir or (PROJECT_ROOT / "checkpoints"),
            )
            base_ckpt.pop("optimizer", None)
            base_tokenizer = Tokenizer.from_str(base_ckpt["tokenizer_json"])
            base_cfg = build_model_config(base_ckpt)
            base_model = GPT(base_cfg).to(device).eval()
            base_model.load_state_dict(base_ckpt["model"])
            del base_ckpt
            if device.type == "cuda" and dtype == torch.bfloat16:
                base_model = base_model.to(dtype=torch.bfloat16)
            print(f"Loaded parent Base model for comparison: {base_path.name}")
        except Exception as exc:
            print(f"Skipping Base comparison ({type(exc).__name__}: {exc})")
            base_model = None

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_config": model_config,
        "checkpoint": ckpt,
        "step": step,
        "stage": stage,
        "arith_manifest": arith_manifest,
        "metrics_history": metrics_history,
        "base_model": base_model,
        "base_tokenizer": base_tokenizer,
    }


def plot_run_history(
    metrics_history: list[dict],
    *,
    run_id: str,
    model_config,
    stage: str = "pretrain",
    snapshot=None,
) -> None:
    """Plot pre-training or SFT loss, exact-match/per-op accuracy, and LR curves cleanly."""
    if not metrics_history:
        if snapshot is not None:
            print(
                f"Step {snapshot.step:,}: Validation Loss = {snapshot.val_loss:.4f}, "
                f"Train Mean = {snapshot.train_loss_interval_mean:.4f}"
            )
        else:
            print("No logged metrics history stored in this checkpoint.")
        return

    val_pts = [(m["step"], m["val_loss"]) for m in metrics_history if m.get("val_loss") is not None]
    train_pts = [(m["step"], m["train_loss_interval_mean"]) for m in metrics_history if m.get("train_loss_interval_mean") is not None]
    story_pts = [(m["step"], m["story_val_loss"]) for m in metrics_history if m.get("story_val_loss") is not None]
    strict_pts = [(m["step"], m["val_strict_pass"] * 100.0) for m in metrics_history if m.get("val_strict_pass") is not None]
    acc_pts = [(m["step"], m["val_exact_match"] * 100.0) for m in metrics_history if m.get("val_exact_match") is not None]
    fmt_pts = [(m["step"], m["val_valid_format"] * 100.0) for m in metrics_history if m.get("val_valid_format") is not None]
    lr_pts = [(m["step"], m["lr"]) for m in metrics_history if m.get("lr") is not None]
    tps_vals = [m["train_tokens_per_second"] for m in metrics_history if m.get("train_tokens_per_second") is not None]
    op_keys = sorted({k for m in metrics_history for k in m if k.startswith("val_acc_op_")})
    is_post_train = bool(acc_pts or strict_pts) or stage in ("sft", "rl")

    try:
        import matplotlib.pyplot as plt

        n_cols = 3 if (is_post_train or op_keys) else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5.3 * n_cols, 4.3), dpi=100)
        ax1 = axes[0]
        if train_pts:
            pts = [pt for pt in train_pts if not is_post_train or pt[0] > 0] or train_pts
            lbl = "Train Error Rate (1 - reward)" if stage == "rl" else (
                "Train Interval Mean (90% arith + 10% story)" if story_pts else "Train Loss (Interval Mean)"
            )
            ax1.plot([s for s, _ in pts], [v for _, v in pts], label=lbl, color="#2563eb", linewidth=2)
        if val_pts:
            pts = [pt for pt in val_pts if not is_post_train or pt[0] > 0] or val_pts
            lbl = "Val Error Rate (1 - strict)" if stage == "rl" else ("Val Arith Loss" if is_post_train else "Validation Loss")
            ax1.plot([s for s, _ in pts], [v for _, v in pts], label=lbl, color="#dc2626", linewidth=2, linestyle="--")
        if story_pts:
            ax1.plot([s for s, _ in story_pts], [v for _, v in story_pts], label="Story Val Loss", color="#9333ea", linewidth=1.8, linestyle=":")
        max_step = max((m["step"] for m in metrics_history), default=0)
        ax1.set_title(f"Loss Convergence (0 to {max_step:,})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Error / Loss" if stage == "rl" else "Cross-Entropy Loss")
        later = [v for _, v in val_pts[1:]]
        if is_post_train:
            top = max([v for _, v in story_pts] + later, default=2.0) + 0.15
            ax1.set_ylim(bottom=0.0, top=min(3.0, top))
        else:
            ax1.set_ylim(bottom=0.9, top=min(3.6, max(later, default=3.6) + 0.15))
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=8)

        if acc_pts or strict_pts:
            ax2 = axes[1]
            if strict_pts:
                ax2.plot([s for s, _ in strict_pts], [v for _, v in strict_pts], label="Strict Trace Pass %", color="#059669", linewidth=2.2, marker="s", markersize=3.5)
            if acc_pts:
                ax2.plot([s for s, _ in acc_pts], [v for _, v in acc_pts], label="Exact Match %", color="#0d9488", linewidth=2.0, marker="o", markersize=3.5)
            if fmt_pts:
                ax2.plot([s for s, _ in fmt_pts], [v for _, v in fmt_pts], label="Valid Format %", color="#7c3aed", linewidth=1.6, linestyle="--")
            ax2.set_title("Held-Out Val Accuracy & Format", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_ylim(-2, 104)
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(fontsize=8)

        ax_right = axes[-1]
        if op_keys:
            for k in op_keys:
                pts = [(m["step"], m[k] * 100.0) for m in metrics_history if m.get(k) is not None]
                if pts:
                    ax_right.plot([s for s, _ in pts], [v for _, v in pts], label=k.removeprefix("val_acc_op_"), linewidth=1.9, marker="o", markersize=3)
            ax_right.set_title("Per-Operation Exact Match (%)", fontsize=11, fontweight="bold")
            ax_right.set_xlabel("Step")
            ax_right.set_ylabel("Accuracy (%)")
            ax_right.set_ylim(-2, 104)
            ax_right.grid(True, linestyle=":", alpha=0.6)
            ax_right.legend(fontsize=8)
        elif lr_pts:
            ax_right.plot([s for s, _ in lr_pts], [v * 1e4 for _, v in lr_pts], color="#059669", linewidth=2)
            ax_right.set_title("Learning Rate Schedule", fontsize=11, fontweight="bold")
            ax_right.set_xlabel("Step")
            ax_right.set_ylabel("LR (x1e-4)")
            ax_right.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle(
            f"MiniReasoner ({stage.upper()}) Run {run_id[:8]} ({model_config.pos_encoding.upper()}, ctx={model_config.context_length:,})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed; skipping charts.")

    if val_pts:
        best_step, best_val = min(val_pts, key=lambda x: x[1])
        print(f"Initial Val Loss:        {val_pts[0][1]:.4f} (Step {val_pts[0][0]:,})")
        print(f"Latest Logged Val Loss:  {val_pts[-1][1]:.4f} (Step {val_pts[-1][0]:,})")
        print(f"Best Val Loss:           {best_val:.4f} (Step {best_step:,})")
    if acc_pts:
        peak_step, peak_acc = max(acc_pts, key=lambda x: x[1])
        print(f"Peak Val Exact Match:    {peak_acc:.1f}% (Step {peak_step:,})")
        print(f"Final Val Exact Match:   {acc_pts[-1][1]:.1f}% (Step {acc_pts[-1][0]:,})")
    if story_pts:
        print(f"Story Val Loss:          {story_pts[0][1]:.4f} (Step {story_pts[0][0]:,}) -> {story_pts[-1][1]:.4f} (Step {story_pts[-1][0]:,})")
    if tps_vals:
        print(f"Mean Train Throughput:   {sum(tps_vals) / len(tps_vals):,.0f} tok/s")


def summarize_release_provenance(
    candidates: list[dict],
    *,
    arith_dir: Path | None = None,
    arith_manifest: dict | None = None,
    tokenized_dir: Path | None = None,
) -> list[dict]:
    """Compute SHA-256 checksums and print release artifact URIs across candidate checkpoints and datasets."""
    from ..core.tokenizer import sha256
    from ..settings import TOKENIZED_DATA_DIR
    import json

    rows: list[dict] = []
    print("=" * 100)
    print("PHASE 6 RELEASE ARTIFACT & LINEAGE PROVENANCE")
    print("=" * 100)
    print(f"{'Model Label':<26}{'Stage / Step':<18}{'Run ID':<34}{'Size (MB)':<11}{'SHA-256 (16)'}")
    print("-" * 100)
    for item in candidates:
        label = item["label"]
        run_id = item["run_id"]
        path = Path(item["checkpoint_path"])
        bundle = item["bundle"]
        digest = sha256(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        stage_step = f"{bundle['stage']} @ {bundle['step']:,}"
        print(f"{label:<26}{stage_step:<18}{run_id:<34}{size_mb:<11.2f}{digest[:16]}")
        rows.append({
            "label": label,
            "run_id": run_id,
            "checkpoint_path": str(path),
            "sha256": digest,
            "size_mb": round(size_mb, 2),
            "stage": bundle["stage"],
            "step": bundle["step"],
            "parent_run_id": bundle["checkpoint"].get("parent_run_id"),
            "parent_checkpoint_step": bundle["checkpoint"].get("parent_checkpoint_step"),
            "azure_uri": f"https://minireasoner.blob.core.windows.net/{AZURE_CONTAINER}/checkpoints/Shared/MiniReasoner/{run_id}",
        })

    if arith_dir is not None and arith_manifest is not None:
        test_meta = (arith_manifest.get("files") or {}).get("test.jsonl") or {}
        print("-" * 100)
        print(
            f"Sealed Arith Test: {arith_dir / 'test.jsonl'} | rows={test_meta.get('rows', '?'):,} | "
            f"sha256={str(test_meta.get('sha256', '?'))[:16]}..."
        )
    tok_root = Path(tokenized_dir) if tokenized_dir is not None else TOKENIZED_DATA_DIR
    tok_manifest_path = tok_root / "manifest.json"
    if tok_manifest_path.is_file():
        tok_manifest = json.loads(tok_manifest_path.read_text(encoding="utf-8"))
        print(
            f"Corpus Manifest:   {tok_manifest_path} | vocab={tok_manifest.get('vocab_size')} | "
            f"tok_sha256={str(tok_manifest.get('tokenizer_sha256', '?'))[:16]}..."
        )
    print("=" * 100)
    return rows
