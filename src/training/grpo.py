"""Phase 5 Verifiable-Reward RL (GRPO) trainer targeting failed CoT arithmetic problems."""

from contextlib import ExitStack
from dataclasses import asdict
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .common import (
    TokenStream,
    TrainConfig,
    configure_optimizer,
    evaluate,
    open_run_log,
    require_training_gpu,
    resolve_precision,
    tracked_training_session,
)
from ..core.checkpoint import build_model_config, load_checkpoint, resolve_base_checkpoint
from ..core.generate import generate_batch_greedy_with_meta, generate_batch_rollouts
from ..core.model import GPT
from ..core.tokenizer import EOS, verify_corpus_files
from ..data.arithmetic import (
    SCRATCHPAD_GENERATOR_VERSION,
    ArithmeticExample,
    dataset_slug,
    load_or_restore_arithmetic_dataset,
    load_rl_hard_dataset,
    normalize_answer_format,
    normalize_ops,
    save_rl_hard_dataset,
    score_strict_scratchpad_rollout,
    select_hard_rl_candidates,
    select_stratified_eval_problems,
)
from ..data.corpus import ensure_tokenized_corpus
from ..infra.checkpoint_store import CheckpointStore
from ..infra.tracking import download_checkpoint
from ..settings import AZURE_CONTAINER, RL_DATA_DIR, RL_PRESETS, TOKENIZED_DATA_DIR


def rl_hard_dataset_dir(
    arith_manifest: dict,
    *,
    parent_run_id: str,
    parent_step: int,
    exclude_easy_categories: bool = True,
    scan_limit: int | None = None,
    expand_failed_variants: bool = True,
) -> Path:
    """Return canonical directory path caching hard RL examples mined from `train` failures."""
    base_slug = dataset_slug(
        ops=tuple(arith_manifest["ops"]),
        max_digits=int(arith_manifest["max_digits"]),
        allow_negative=bool(arith_manifest["allow_negative"]),
        samples_per_op=int(arith_manifest["samples_per_op"]),
        seed=int(arith_manifest["seed"]),
        answer_format=normalize_answer_format(arith_manifest.get("answer_format", "scratchpad")),
        include_multistep=bool(arith_manifest.get("include_multistep", True)),
    )
    filter_tag = "nontrivial" if exclude_easy_categories else "allcats"
    if expand_failed_variants:
        filter_tag += "_variants"
    if scan_limit is not None:
        filter_tag += f"_scan{int(scan_limit)}"
    safe_parent = str(parent_run_id).replace("/", "_")[:12]
    return RL_DATA_DIR / f"rl_hard_{base_slug}_{safe_parent}_s{int(parent_step)}_{filter_tag}"


def compute_group_advantages(rewards: list[float], group_size: int) -> list[float]:
    """Compute per-prompt GRPO group-normalized advantages `(r - mean) / (std + eps)`."""
    if group_size < 1 or len(rewards) % group_size != 0:
        raise ValueError("len(rewards) must be divisible by positive group_size")
    advantages: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        mean_r = sum(group) / len(group)
        var_r = sum((r - mean_r) ** 2 for r in group) / len(group)
        std_r = math.sqrt(var_r)
        if std_r <= 1e-8:
            advantages.extend([0.0] * len(group))
        else:
            advantages.extend((r - mean_r) / (std_r + 1e-8) for r in group)
    return advantages


def _encode_response_ids(
    tokenizer,
    prompt_ids: list[int],
    raw_rollout_ids: list[int],
    *,
    eos_id: int,
    context_length: int,
) -> tuple[list[int], list[int]]:
    """Build `(full_x, full_y_masked)` alignments where only generated rollout tokens have non-`-100` labels."""
    resp = list(raw_rollout_ids) if raw_rollout_ids else [eos_id]
    p_ids = list(prompt_ids) if prompt_ids else [eos_id]
    max_total = context_length + 1
    if len(resp) > max_total - 1:
        resp = resp[: max_total - 1]
    if len(p_ids) + len(resp) > max_total:
        p_ids = p_ids[-(max_total - len(resp)) :]
    full = p_ids + resp
    return full[:-1], ([-100] * (len(p_ids) - 1)) + resp


def _collate_logprob_batch(
    pairs: list[tuple[list[int], list[int]]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _ in pairs)
    x_t = torch.full((len(pairs), max_len), pad_id, dtype=torch.long)
    y_t = torch.full((len(pairs), max_len), -100, dtype=torch.long)
    for row_i, (x_seq, y_seq) in enumerate(pairs):
        n = len(x_seq)
        x_t[row_i, :n] = torch.tensor(x_seq, dtype=torch.long)
        y_t[row_i, :n] = torch.tensor(y_seq, dtype=torch.long)
    x_dev = x_t.to(device)
    y_dev = y_t.to(device)
    mask = y_dev != -100
    return x_dev, y_dev, mask


def _forward_token_logprobs(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    """Return exact float32 log-probabilities per position (0.0 where target is `-100`)."""
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(x)
    nll = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="none")
    return (-nll).view_as(y)


@torch.inference_mode()
def _collect_reference_logprobs(
    model: GPT,
    pairs: list[tuple[list[int], list[int]]],
    *,
    mini_batch_size: int,
    pad_id: int,
    device: torch.device,
    precision: str,
) -> list[torch.Tensor]:
    """Compute detached log-prob tensors `(B_chunk, L_chunk)` matching `_collate_logprob_batch` slices."""
    was_training = model.training
    model.eval()
    try:
        chunks: list[torch.Tensor] = []
        for first in range(0, len(pairs), max(1, mini_batch_size)):
            chunk = pairs[first : first + max(1, mini_batch_size)]
            x, y, _ = _collate_logprob_batch(chunk, pad_id=pad_id, device=device)
            chunks.append(_forward_token_logprobs(model, x, y, precision).detach())
        return chunks
    finally:
        if was_training:
            model.train()


def _mine_batch_failures(
    model: GPT,
    tokenizer,
    examples: list[ArithmeticExample],
    *,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    label: str,
) -> list[ArithmeticExample]:
    if not examples:
        return []
    t0 = perf_counter()
    prompts = [ex.prompt for ex in examples]
    total = len(prompts)
    next_report = max(500, total // 5)

    def _progress(done: int, tot: int) -> None:
        nonlocal next_report
        if done >= next_report or done == tot:
            el = perf_counter() - t0
            rate = done / max(1e-6, el)
            print(f"  [Mine:{label}] {done:,}/{tot:,} ({done / tot * 100:.1f}%) | {rate:.1f} ex/s", flush=True)
            next_report += max(500, tot // 5)

    decoded = generate_batch_greedy_with_meta(
        model, tokenizer, prompts, max_new_tokens, device, batch_size=batch_size, progress_callback=_progress,
    )
    failed: list[ArithmeticExample] = []
    for ex, res in zip(examples, decoded):
        verdict = score_strict_scratchpad_rollout(res["text"], ex.answer, ended_with_eos=res["ended_with_eos"])
        if not verdict["strict_pass"]:
            failed.append(ex)
    return failed


def prepare_hard_rl_dataset(
    *,
    preset: str = "grpo",
    sft_run_id: str | None = None,
    sft_checkpoint_alias: str | None = None,
    sft_checkpoint: Path | None = None,
    data_dir: Path | None = None,
    rl_data_dir: Path | None = None,
    ops: tuple[str, ...] | list[str] | str | None = None,
    max_digits: int | None = None,
    allow_negative: bool | None = None,
    samples_per_op: int | None = None,
    scan_limit: int | None = None,
    scan_batch_size: int | None = None,
    exclude_easy_categories: bool | None = None,
    expand_failed_variants: bool | None = None,
    no_drive: bool = False,
    overwrite: bool = False,
    azure_container: str | None = None,
    device: torch.device | None = None,
    precision_request: str = "auto",
    checkpoints_dir: Path | None = None,
    allow_empty_fallback: bool = False,
) -> tuple[Path, list[ArithmeticExample], dict]:
    """Mine and SHA-256 cache ONLY `train` examples where the starting CoT SFT model fails the strict trace verifier."""
    from tokenizers import Tokenizer

    if preset not in RL_PRESETS:
        raise ValueError(f"Unknown RL preset {preset!r}. Choose from: {list(RL_PRESETS.keys())}")
    cfg = dict(RL_PRESETS[preset])

    op_list = normalize_ops(ops if ops is not None else cfg["ops"])
    n_digits = max_digits if max_digits is not None else cfg["max_digits"]
    neg_allowed = allow_negative if allow_negative is not None else cfg["allow_negative"]
    n_samples = samples_per_op if samples_per_op is not None else cfg["samples_per_op"]
    alias = sft_checkpoint_alias if sft_checkpoint_alias is not None else cfg["sft_checkpoint_alias"]
    excl_easy = exclude_easy_categories if exclude_easy_categories is not None else bool(cfg["exclude_easy_categories"])
    expand_vars = expand_failed_variants if expand_failed_variants is not None else bool(cfg["expand_failed_variants"])
    s_batch = scan_batch_size if scan_batch_size is not None else int(cfg["scan_batch_size"])
    container = azure_container or AZURE_CONTAINER

    src_dir, arith_splits, arith_manifest = load_or_restore_arithmetic_dataset(
        dataset_dir=data_dir,
        ops=op_list,
        max_digits=n_digits,
        allow_negative=neg_allowed,
        samples_per_op=n_samples,
        seed=int(cfg["seed"]),
        answer_format=cfg["answer_format"],
        include_multistep=bool(cfg["include_multistep"]),
        use_drive=not no_drive,
    )
    fmt = normalize_answer_format(arith_manifest.get("answer_format", "scratchpad"))
    if fmt != "scratchpad":
        raise ValueError("Phase 5 strict-trace GRPO requires a CoT scratchpad dataset (`--format cot`).")

    sft_path, sft_ckpt = resolve_base_checkpoint(
        checkpoint_path=sft_checkpoint,
        run_id=sft_run_id,
        alias=alias,
        preferred_stages=("sft",),
        preferred_arith_format="scratchpad",
        azure_container=container,
        checkpoints_dir=checkpoints_dir,
    )
    parent_run_id = str(sft_run_id or sft_ckpt.get("resolved_run_id") or sft_path.parent.name)
    parent_step = int(sft_ckpt.get("step", 0))

    target_dir = Path(rl_data_dir) if rl_data_dir is not None else rl_hard_dataset_dir(
        arith_manifest,
        parent_run_id=parent_run_id,
        parent_step=parent_step,
        exclude_easy_categories=excl_easy,
        scan_limit=scan_limit,
        expand_failed_variants=expand_vars,
    )
    if (target_dir / "manifest.json").is_file() and not overwrite:
        hard_items, hard_manifest = load_rl_hard_dataset(target_dir)
        print(
            f"Loaded verified hard RL failure dataset from {target_dir}: "
            f"{len(hard_items):,} failed `train` problems (ops={hard_manifest.get('op_counts')})",
            flush=True,
        )
        return target_dir, hard_items, hard_manifest

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(precision_request, device)

    tokenizer = Tokenizer.from_str(sft_ckpt["tokenizer_json"])
    eos_id = tokenizer.token_to_id(EOS)
    if eos_id is None:
        raise ValueError(f"SFT checkpoint tokenizer is missing {EOS!r}.")
    model_cfg = build_model_config(sft_ckpt, context_length=cfg["context_length"])
    model = GPT(model_cfg).to(device).eval()
    model.load_state_dict(sft_ckpt["model"])
    if device.type == "cuda" and precision == "bf16":
        model = model.to(dtype=torch.bfloat16)

    max_answer_tokens = max(int(cfg["max_new_tokens"]), int(arith_manifest.get("max_digits", n_digits)) * 2 + 64)
    train_only = list(arith_splits["train"])
    canonical_pool, by_canonical = select_hard_rl_candidates(
        train_only, exclude_easy_categories=excl_easy, canonical_only_first=True,
    )
    if scan_limit is not None and scan_limit > 0 and len(canonical_pool) > scan_limit:
        canonical_pool = select_stratified_eval_problems(canonical_pool, scan_limit, seed=int(cfg["seed"]))

    print(
        f"Mining hard RL training dataset strictly from `train` ({len(canonical_pool):,} canonical candidates; "
        f"`val` and `sealed_test` remain untouched)...",
        flush=True,
    )
    failed_canonicals = _mine_batch_failures(
        model, tokenizer, canonical_pool,
        max_new_tokens=max_answer_tokens, batch_size=s_batch, device=device, label="canonical",
    )

    if expand_vars and failed_canonicals:
        failed_cids = {ex.canonical_id for ex in failed_canonicals}
        variant_candidates = [
            vex
            for cid in sorted(failed_cids)
            for vex in by_canonical[cid]
            if vex.variant != "canonical"
        ]
        failed_variants = _mine_batch_failures(
            model, tokenizer, variant_candidates,
            max_new_tokens=max_answer_tokens, batch_size=s_batch, device=device, label="variants",
        )
        seen_keys: set[tuple[str, str, str]] = set()
        hard_examples: list[ArithmeticExample] = []
        for ex in failed_canonicals + failed_variants:
            k = (ex.canonical_id, ex.variant, ex.prompt)
            if k not in seen_keys:
                seen_keys.add(k)
                hard_examples.append(ex)
        scanned_variants = len(variant_candidates)
    else:
        hard_examples = list(failed_canonicals)
        scanned_variants = 0

    if not hard_examples:
        if allow_empty_fallback:
            hard_examples = list(canonical_pool[: max(1, min(len(canonical_pool), 16))])
            print("Warning: zero failures found on scan; using fallback non-trivial train slice for caller test.", flush=True)
        else:
            raise RuntimeError(
                "Zero strict-verifier failures mined from candidate train problems. "
                "Try --scan-limit None or --include-trivial."
            )

    manifest = {
        "version": 1,
        "source_dataset_dir": str(src_dir),
        "source_generator_version": arith_manifest.get("generator_version", SCRATCHPAD_GENERATOR_VERSION),
        "ops": list(op_list),
        "max_digits": n_digits,
        "allow_negative": neg_allowed,
        "answer_format": fmt,
        "include_multistep": bool(arith_manifest.get("include_multistep", True)),
        "parent_run_id": parent_run_id,
        "parent_checkpoint_step": parent_step,
        "exclude_easy_categories": excl_easy,
        "expand_failed_variants": expand_vars,
        "scanned_canonical": len(canonical_pool),
        "failed_canonical": len(failed_canonicals),
        "scanned_variants": scanned_variants,
        "filter_mode": "strict_verifier_failures_only",
    }
    save_rl_hard_dataset(target_dir, hard_examples, manifest)
    _, saved_manifest = load_rl_hard_dataset(target_dir)
    print(
        f"Saved hard RL failure dataset to {target_dir}: "
        f"{len(hard_examples):,} failed examples ({len(failed_canonicals):,}/{len(canonical_pool):,} canonicals failed) | "
        f"ops={saved_manifest['op_counts']}",
        flush=True,
    )
    return target_dir, hard_examples, saved_manifest


@torch.inference_mode()
def evaluate_grpo(
    model: GPT,
    tokenizer,
    val_eval_items: list[ArithmeticExample],
    *,
    batch_size: int,
    device: torch.device,
    max_new_tokens: int,
    sft_baseline_outcomes: list[bool] | None = None,
    story_val_stream: TokenStream | None = None,
    retention_config: TrainConfig | None = None,
    precision: str = "fp32",
    step: int = 0,
    max_failures_to_log: int = 5,
) -> dict:
    """Evaluate policy greedily on held-out `val` with strict `{0,1}` verifier + paired SFT wins/regressions."""
    was_training = model.training
    model.eval()
    try:
        decoded = generate_batch_greedy_with_meta(
            model, tokenizer, [ex.prompt for ex in val_eval_items], max_new_tokens, device, batch_size=batch_size,
        )
        strict_hits = 0
        exact_hits = 0
        format_hits = 0
        work_hits = 0
        eos_hits = 0
        total_toks = 0
        outcomes: list[bool] = []
        per_op: dict[str, list[int]] = {}
        per_cat: dict[str, list[int]] = {}
        failures: list[tuple[ArithmeticExample, str, dict]] = []

        for ex, res in zip(val_eval_items, decoded):
            verdict = score_strict_scratchpad_rollout(res["text"], ex.answer, ended_with_eos=res["ended_with_eos"])
            sp = bool(verdict["strict_pass"])
            outcomes.append(sp)
            strict_hits += int(sp)
            exact_hits += int(verdict["exact_match"])
            format_hits += int(verdict["valid_format"])
            work_hits += int(verdict["valid_work"])
            eos_hits += int(verdict["ended_with_eos"])
            total_toks += len(res["generated_ids"])

            op_s = per_op.setdefault(ex.op, [0, 0])
            op_s[1] += 1
            op_s[0] += int(sp)
            cat_s = per_cat.setdefault(ex.category, [0, 0])
            cat_s[1] += 1
            cat_s[0] += int(sp)
            if not sp and len(failures) < max_failures_to_log:
                failures.append((ex, res["text"], verdict))

        n_items = max(1, len(val_eval_items))
        rl_fixed = 0
        rl_broke = 0
        if sft_baseline_outcomes is not None and len(sft_baseline_outcomes) == len(outcomes):
            rl_fixed = sum(1 for base_ok, rl_ok in zip(sft_baseline_outcomes, outcomes) if not base_ok and rl_ok)
            rl_broke = sum(1 for base_ok, rl_ok in zip(sft_baseline_outcomes, outcomes) if base_ok and not rl_ok)

        story_val_loss = None
        if story_val_stream is not None and retention_config is not None:
            story_val_loss = evaluate(model, story_val_stream, retention_config, device, precision, step=step)

        return {
            "val_strict_pass": strict_hits / n_items,
            "val_exact_match": exact_hits / n_items,
            "val_valid_format": format_hits / n_items,
            "val_valid_work": work_hits / n_items,
            "val_eos_rate": eos_hits / n_items,
            "val_mean_tokens": total_toks / n_items,
            "val_strict_correct": strict_hits,
            "val_total": len(val_eval_items),
            "outcomes": outcomes,
            "rl_fixed": rl_fixed,
            "rl_broke": rl_broke,
            "op_accuracy": {op: ok / cnt for op, (ok, cnt) in sorted(per_op.items())},
            "op_counts": {op: (ok, cnt) for op, (ok, cnt) in sorted(per_op.items())},
            "category_accuracy": {cat: ok / cnt for cat, (ok, cnt) in sorted(per_cat.items())},
            "story_val_loss": story_val_loss,
            "sample_failures": failures,
        }
    finally:
        if was_training:
            model.train()


def _log_grpo_eval(eval_res: dict, *, step: int, loss_decimals: int) -> None:
    strict_pct = eval_res["val_strict_pass"] * 100.0
    exact_pct = eval_res["val_exact_match"] * 100.0
    paired = (
        f" | fixed={eval_res['rl_fixed']} broke={eval_res['rl_broke']} (net={eval_res['rl_fixed'] - eval_res['rl_broke']:+d})"
        if step > 0
        else ""
    )
    story = (
        f" | story val loss {eval_res['story_val_loss']:.{loss_decimals}f}"
        if eval_res["story_val_loss"] is not None
        else ""
    )
    print(
        f"Step {step:,} | strict pass {eval_res['val_strict_correct']}/{eval_res['val_total']} ({strict_pct:.1f}%) | "
        f"exact {exact_pct:.1f}% | work_ok {eval_res['val_valid_work'] * 100:.1f}% | "
        f"eos {eval_res['val_eos_rate'] * 100:.1f}% | mean_tok {eval_res['val_mean_tokens']:.1f}"
        f"{paired}{story}",
        flush=True,
    )
    if eval_res["op_counts"]:
        print("  Ops: " + ", ".join(f"{op}={ok}/{cnt}" for op, (ok, cnt) in eval_res["op_counts"].items()), flush=True)


def run_grpo(
    *,
    preset: str = "grpo",
    sft_run_id: str | None = None,
    sft_checkpoint_alias: str | None = None,
    sft_checkpoint: Path | None = None,
    data_dir: Path | None = None,
    rl_data_dir: Path | None = None,
    ops: tuple[str, ...] | list[str] | str | None = None,
    max_digits: int | None = None,
    allow_negative: bool | None = None,
    samples_per_op: int | None = None,
    scan_limit: int | None = None,
    overwrite_hard_data: bool = False,
    no_drive: bool = False,
    context_length: int | None = None,
    steps: int | None = None,
    prompts_per_step: int | None = None,
    group_size: int | None = None,
    mini_batch_size: int | None = None,
    ppo_epochs: int | None = None,
    kl_coef: float | None = None,
    clip_eps: float | None = None,
    temperature: float | None = None,
    learning_rate: float | None = None,
    eval_every: int | None = None,
    disable_retention_eval: bool = False,
    resume: bool = False,
    mlflow_run_id: str | None = None,
    stop_after: int | None = None,
    precision_request: str = "auto",
    seed: int | None = None,
    azure_container: str | None = None,
    loss_decimals: int = 4,
    device: torch.device | None = None,
    logs_dir: Path | None = None,
    retention_tokenized_dir: Path | None = None,
    checkpoints_dir: Path | None = None,
    allow_empty_fallback: bool = False,
) -> dict:
    """Run Phase 5 Group Relative Policy Optimization (GRPO) on failed hard arithmetic traces."""
    from tokenizers import Tokenizer

    if preset not in RL_PRESETS:
        raise ValueError(f"Unknown RL preset {preset!r}. Choose from: {list(RL_PRESETS.keys())}")
    cfg = dict(RL_PRESETS[preset])

    alias = sft_checkpoint_alias if sft_checkpoint_alias is not None else cfg["sft_checkpoint_alias"]
    target_ctx = context_length if context_length is not None else cfg["context_length"]
    op_list = normalize_ops(ops if ops is not None else cfg["ops"])
    n_digits = max_digits if max_digits is not None else cfg["max_digits"]
    neg_allowed = allow_negative if allow_negative is not None else cfg["allow_negative"]
    n_samples = samples_per_op if samples_per_op is not None else cfg["samples_per_op"]

    n_steps = steps if steps is not None else cfg["steps"]
    n_prompts = prompts_per_step if prompts_per_step is not None else cfg["prompts_per_step"]
    g_size = group_size if group_size is not None else cfg["group_size"]
    m_batch = mini_batch_size if mini_batch_size is not None else cfg["mini_batch_size"]
    n_ppo = ppo_epochs if ppo_epochs is not None else cfg["ppo_epochs"]
    beta_kl = kl_coef if kl_coef is not None else cfg["kl_coef"]
    eps_clip = clip_eps if clip_eps is not None else cfg["clip_eps"]
    temp = temperature if temperature is not None else cfg["temperature"]
    top_k = int(cfg["top_k"])
    top_p = float(cfg["top_p"])
    run_seed = seed if seed is not None else cfg["seed"]
    peak_lr = learning_rate if learning_rate is not None else cfg["learning_rate"]
    min_lr = cfg["min_learning_rate"] if learning_rate is None else peak_lr * 0.2
    eval_interval = eval_every if eval_every is not None else cfg["eval_every"]
    warmup = min(cfg["warmup_steps"], max(0, n_steps - 1))

    if g_size < 2:
        raise ValueError("GRPO group_size must be at least 2 so rewards have within-group variance.")
    if n_ppo < 1 or m_batch < 1 or n_prompts < 1:
        raise ValueError("ppo_epochs, mini_batch_size, and prompts_per_step must be positive.")

    train_cfg = TrainConfig(
        batch_size=n_prompts,
        steps=n_steps,
        warmup_steps=warmup,
        learning_rate=peak_lr,
        min_learning_rate=min_lr,
        weight_decay=cfg["weight_decay"],
        eval_every=eval_interval,
        eval_token_fraction=cfg["eval_token_fraction"],
        seed=run_seed,
    )
    stop = train_cfg.steps if stop_after is None else stop_after
    if not 1 <= stop <= train_cfg.steps:
        raise ValueError("stop_after must be between 1 and total steps.")

    with open_run_log(preset, logs_dir=logs_dir) as (log_path, log_lock, is_stop_requested), ExitStack() as resources:
        if resume and not mlflow_run_id:
            raise ValueError("Set mlflow_run_id when resuming an RL run.")
        container = azure_container or AZURE_CONTAINER
        if device is None:
            device = require_training_gpu()
        precision = resolve_precision(precision_request, device)

        sft_path, sft_ckpt = resolve_base_checkpoint(
            checkpoint_path=sft_checkpoint,
            run_id=sft_run_id,
            alias=alias,
            preferred_stages=("sft",),
            preferred_arith_format="scratchpad",
            azure_container=container,
            checkpoints_dir=checkpoints_dir,
        )
        parent_run_id = str(sft_run_id or sft_ckpt.get("resolved_run_id") or sft_path.parent.name)
        parent_step = int(sft_ckpt.get("step", 0))
        tokenizer_json = sft_ckpt["tokenizer_json"]
        tokenizer = Tokenizer.from_str(tokenizer_json)
        eos_id = tokenizer.token_to_id(EOS)
        if eos_id is None:
            raise ValueError(f"SFT checkpoint tokenizer is missing {EOS!r}.")

        model_config = build_model_config(sft_ckpt, context_length=target_ctx)

        _, arith_splits, arith_manifest = load_or_restore_arithmetic_dataset(
            dataset_dir=data_dir,
            ops=op_list,
            max_digits=n_digits,
            allow_negative=neg_allowed,
            samples_per_op=n_samples,
            seed=int(cfg["seed"]),
            answer_format="scratchpad",
            include_multistep=bool(cfg["include_multistep"]),
            use_drive=not no_drive,
        )
        resolved_rl_dir, hard_train_examples, hard_manifest = prepare_hard_rl_dataset(
            preset=preset,
            sft_run_id=parent_run_id,
            sft_checkpoint_alias=alias,
            sft_checkpoint=sft_path,
            data_dir=data_dir,
            rl_data_dir=rl_data_dir,
            ops=op_list,
            max_digits=n_digits,
            allow_negative=neg_allowed,
            samples_per_op=n_samples,
            scan_limit=scan_limit,
            overwrite=overwrite_hard_data,
            no_drive=no_drive,
            azure_container=container,
            device=device,
            precision_request=precision_request,
            checkpoints_dir=checkpoints_dir,
            allow_empty_fallback=allow_empty_fallback,
        )

        val_eval_items = select_stratified_eval_problems(
            arith_splits["val"], cfg["val_eval_problems"], seed=run_seed,
        )
        max_answer_tokens = max(int(cfg["max_new_tokens"]), int(arith_manifest.get("max_digits", n_digits)) * 2 + 64)

        story_val_stream = None
        retention_eval_cfg = None
        if not disable_retention_eval:
            try:
                tok_dir = retention_tokenized_dir or TOKENIZED_DATA_DIR
                corpus_manifest = ensure_tokenized_corpus(tokenized_path=tok_dir, sync_drive=not no_drive)
                verify_corpus_files(tok_dir, corpus_manifest)
                story_val_stream = TokenStream(
                    tok_dir / "val.bin", corpus_manifest["val"]["tokens"], model_config.context_length,
                )
                resources.callback(story_val_stream.close)
                val_frac = min(
                    1.0,
                    (cfg["retention_eval_batches"] * max(8, n_prompts) * model_config.context_length)
                    / max(1, story_val_stream.target_count),
                )
                retention_eval_cfg = TrainConfig(
                    batch_size=max(8, n_prompts),
                    steps=train_cfg.steps,
                    warmup_steps=train_cfg.warmup_steps,
                    learning_rate=train_cfg.learning_rate,
                    min_learning_rate=train_cfg.min_learning_rate,
                    weight_decay=train_cfg.weight_decay,
                    eval_every=train_cfg.eval_every,
                    eval_token_fraction=max(1e-6, val_frac),
                    seed=run_seed,
                )
            except Exception as exc:
                print(f"Skipping SimpleStories retention check ({type(exc).__name__}: {exc}).", flush=True)

        torch.manual_seed(run_seed)
        policy_model = GPT(model_config).to(device)
        policy_model.load_state_dict(sft_ckpt["model"])

        ref_model = GPT(model_config).to(device).eval()
        ref_model.load_state_dict(sft_ckpt["model"])
        for param in ref_model.parameters():
            param.requires_grad_(False)
        if device.type == "cuda" and precision == "bf16":
            ref_model = ref_model.to(dtype=torch.bfloat16)
        del sft_ckpt

        optimizer = configure_optimizer(
            policy_model,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            device=device,
        )

        run_identity = {
            "stage": "rl",
            "algorithm": "grpo",
            "parent_run_id": parent_run_id,
            "parent_checkpoint_step": parent_step,
            "model_config": asdict(model_config),
            "train_config": asdict(train_cfg),
            "arith_manifest": arith_manifest,
            "rl_hard_manifest": hard_manifest,
            "group_size": g_size,
            "kl_coef": beta_kl,
            "clip_eps": eps_clip,
            "temperature": temp,
        }

        start = 0
        history: list[dict] = []
        if resume:
            resume_store = CheckpointStore(container, mlflow_run_id)
            with download_checkpoint(resume_store, alias="latest") as ckpt_path:
                saved = load_checkpoint(ckpt_path)
            if saved.get("version") != 1:
                raise ValueError(f"Unsupported checkpoint version: {saved.get('version')}")
            if any(saved.get(k) != v for k, v in run_identity.items()):
                raise ValueError("GRPO checkpoint identity disagrees with current settings.")
            policy_model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start = int(saved["step"])
            history = saved["metrics"]
            del saved
            print(f"Restored GRPO checkpoint from step {start:,}. Resuming to step {stop:,}.", flush=True)

        print(
            f"Initialized Phase 5 GRPO from {sft_path.name} (parent_run={parent_run_id}, parent_step={parent_step:,}) | "
            f"hard_train={len(hard_train_examples):,} failed problems | G={g_size}, prompts/step={n_prompts}, "
            f"kl_coef={beta_kl}, clip_eps={eps_clip}, T={temp}",
            flush=True,
        )

        sft_baseline = evaluate_grpo(
            ref_model, tokenizer, val_eval_items,
            batch_size=max(32, m_batch), device=device, max_new_tokens=max_answer_tokens,
            story_val_stream=story_val_stream, retention_config=retention_eval_cfg,
            precision=precision, step=0,
        )
        sft_outcomes = sft_baseline["outcomes"]

        last_eval = sft_baseline
        val_error_rate = 1.0 - last_eval["val_strict_pass"]
        train_error_rate = 1.0
        last_step = start

        with tracked_training_session(
            run_params={
                "preset": preset,
                "algorithm": "grpo",
                "device": device.type,
                "precision": precision,
                "parent_run_id": parent_run_id,
                "parent_checkpoint_step": parent_step,
                "model_config": asdict(model_config),
                "train_config": asdict(train_cfg),
                "group_size": g_size,
                "prompts_per_step": n_prompts,
                "mini_batch_size": m_batch,
                "ppo_epochs": n_ppo,
                "kl_coef": beta_kl,
                "clip_eps": eps_clip,
                "temperature": temp,
                "hard_train_count": len(hard_train_examples),
                "rl_dataset_dir": str(resolved_rl_dir),
            },
            log_path=log_path,
            log_lock=log_lock,
            is_stop_requested=is_stop_requested,
            azure_container=container,
            resume_run_id=mlflow_run_id if resume else None,
            tags={
                "stage": "rl",
                "algorithm": "grpo",
                "parent_run_id": parent_run_id,
                "parent_checkpoint_step": str(parent_step),
                "arith_generator_version": str(arith_manifest.get("generator_version", SCRATCHPAD_GENERATOR_VERSION)),
                "arith_format": "scratchpad",
                "reward_spec": "strict_binary_eos_work_answer_v1",
                "context_length": str(model_config.context_length),
            },
        ) as (session, check_stop):
            run = session.run
            active_run_id = run.id

            if resume:
                session.store.prune()
                for row in history:
                    run.log(row, step=row["step"])
                if history:
                    val_error_rate = history[-1].get("val_loss", val_error_rate)
            else:
                print("Evaluating Step 0 frozen CoT SFT baseline strictly on held-out val set...", flush=True)
                _log_grpo_eval(sft_baseline, step=0, loss_decimals=loss_decimals)
                step0_row = {
                    "step": 0,
                    "val_loss": val_error_rate,
                    "val_strict_pass": sft_baseline["val_strict_pass"],
                    "val_exact_match": sft_baseline["val_exact_match"],
                    "val_valid_format": sft_baseline["val_valid_format"],
                    "val_valid_work": sft_baseline["val_valid_work"],
                    "val_eos_rate": sft_baseline["val_eos_rate"],
                    "val_mean_tokens": sft_baseline["val_mean_tokens"],
                    "train_loss_interval_mean": val_error_rate,
                }
                if sft_baseline["story_val_loss"] is not None:
                    step0_row["story_val_loss"] = sft_baseline["story_val_loss"]
                history = [step0_row]
                run.log(step0_row, step=0)

            interval_start = perf_counter()
            interval_rewards: list[float] = []
            batch_rng = np.random.default_rng(run_seed)

            for step in range(start + 1, stop + 1):
                if check_stop():
                    print(f"Stopping GRPO early as requested at step {last_step:,}.", flush=True)
                    break

                lr = train_cfg.lr_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                chosen_idxs = batch_rng.integers(0, len(hard_train_examples), size=n_prompts)
                step_examples = [hard_train_examples[int(i)] for i in chosen_idxs]

                policy_model.eval()
                rollouts = generate_batch_rollouts(
                    policy_model,
                    tokenizer,
                    [ex.prompt for ex in step_examples],
                    max_answer_tokens,
                    device,
                    num_return_sequences=g_size,
                    batch_size=max(g_size, m_batch),
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                )

                rewards: list[float] = []
                eos_flags: list[int] = []
                fmt_flags: list[int] = []
                work_flags: list[int] = []
                exact_flags: list[int] = []
                tok_lens: list[int] = []
                encoded_pairs: list[tuple[list[int], list[int]]] = []

                for r in rollouts:
                    ex = step_examples[r["prompt_idx"]]
                    verdict = score_strict_scratchpad_rollout(r["text"], ex.answer, ended_with_eos=r["ended_with_eos"])
                    rewards.append(verdict["reward"])
                    eos_flags.append(int(verdict["ended_with_eos"]))
                    fmt_flags.append(int(verdict["valid_format"]))
                    work_flags.append(int(verdict["valid_work"]))
                    exact_flags.append(int(verdict["exact_match"]))
                    tok_lens.append(len(r["rollout_ids"]))
                    encoded_pairs.append(
                        _encode_response_ids(
                            tokenizer, r["prompt_ids"], r["rollout_ids"],
                            eos_id=eos_id, context_length=model_config.context_length,
                        )
                    )

                advantages = compute_group_advantages(rewards, g_size)
                group_means = [
                    sum(rewards[i : i + g_size]) / g_size for i in range(0, len(rewards), g_size)
                ]
                mixed_frac = sum(1 for m in group_means if 0.0 < m < 1.0) / max(1, len(group_means))
                mean_reward = sum(rewards) / max(1, len(rewards))

                old_logprob_chunks = _collect_reference_logprobs(
                    policy_model, encoded_pairs, mini_batch_size=m_batch, pad_id=eos_id, device=device, precision=precision,
                )
                ref_logprob_chunks = _collect_reference_logprobs(
                    ref_model, encoded_pairs, mini_batch_size=m_batch, pad_id=eos_id, device=device, precision=precision,
                )

                policy_model.train()
                total_rollouts = len(encoded_pairs)
                step_policy_loss = 0.0
                step_kl = 0.0
                step_clip_frac = 0.0

                for _epoch in range(n_ppo):
                    optimizer.zero_grad(set_to_none=True)
                    for chunk_idx, first in enumerate(range(0, total_rollouts, m_batch)):
                        chunk_pairs = encoded_pairs[first : first + m_batch]
                        chunk_adv = torch.tensor(advantages[first : first + m_batch], dtype=torch.float32, device=device)
                        x, y, mask = _collate_logprob_batch(chunk_pairs, pad_id=eos_id, device=device)
                        old_lp = old_logprob_chunks[chunk_idx]
                        ref_lp = ref_logprob_chunks[chunk_idx]

                        cur_lp = _forward_token_logprobs(policy_model, x, y, precision)
                        log_ratio = torch.clamp(cur_lp - old_lp, min=-20.0, max=20.0)
                        ratio = torch.exp(log_ratio)
                        adv_col = chunk_adv[:, None]

                        surr1 = ratio * adv_col
                        surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * adv_col
                        pg_tok = -torch.minimum(surr1, surr2)

                        log_ref_diff = torch.clamp(ref_lp - cur_lp, min=-20.0, max=20.0)
                        kl_tok = torch.exp(log_ref_diff) - log_ref_diff - 1.0

                        tok_loss = pg_tok + beta_kl * kl_tok
                        mask_f = mask.float()
                        seq_lengths = mask_f.sum(dim=1).clamp(min=1.0)
                        seq_losses = (tok_loss * mask_f).sum(dim=1) / seq_lengths
                        chunk_loss = seq_losses.sum() / total_rollouts
                        chunk_loss.backward()

                        with torch.no_grad():
                            n_valid = mask_f.sum().clamp(min=1.0)
                            seq_pg = ((pg_tok * mask_f).sum(dim=1) / seq_lengths).sum().item() / total_rollouts
                            mean_kl_c = (kl_tok * mask_f).sum().item() / float(n_valid.item()) * (len(chunk_pairs) / total_rollouts)
                            clipped_c = (((ratio - 1.0).abs() > eps_clip) & mask).float().sum().item() / float(n_valid.item()) * (len(chunk_pairs) / total_rollouts)
                            step_policy_loss += seq_pg
                            step_kl += mean_kl_c
                            step_clip_frac += clipped_c

                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()

                step_policy_loss /= n_ppo
                step_kl /= n_ppo
                step_clip_frac /= n_ppo
                last_step = step
                interval_rewards.append(mean_reward)
                train_error_rate = max(0.0, 1.0 - mean_reward)

                if not math.isfinite(step_policy_loss) or not math.isfinite(step_kl):
                    raise FloatingPointError(f"Non-finite GRPO objective at step {step}.")

                print(
                    f"Step {step:,} | mean_reward {mean_reward:.{loss_decimals}f} | "
                    f"mixed_groups {mixed_frac * 100:.1f}% | kl {step_kl:.{loss_decimals}f} | "
                    f"pg_loss {step_policy_loss:.{loss_decimals}f} | mean_tok {sum(tok_lens) / len(tok_lens):.1f}",
                    flush=True,
                )

                metrics: dict = {
                    "step": step,
                    "train_loss": train_error_rate,
                    "mean_reward": mean_reward,
                    "strict_pass_rate": mean_reward,
                    "rollout_eos_rate": sum(eos_flags) / len(eos_flags),
                    "rollout_format_rate": sum(fmt_flags) / len(fmt_flags),
                    "rollout_work_rate": sum(work_flags) / len(work_flags),
                    "rollout_answer_match_rate": sum(exact_flags) / len(exact_flags),
                    "frac_mixed_groups": mixed_frac,
                    "mean_completion_tokens": sum(tok_lens) / len(tok_lens),
                    "grpo_policy_loss": step_policy_loss,
                    "kl_div": step_kl,
                    "clip_frac": step_clip_frac,
                }

                is_eval_step = (step % train_cfg.eval_every == 0) or (step == stop) or check_stop()
                if not is_eval_step:
                    run.log(metrics, step=step)
                    continue

                interval_mean_err = max(0.0, 1.0 - (sum(interval_rewards) / max(1, len(interval_rewards))))
                last_eval = evaluate_grpo(
                    policy_model, tokenizer, val_eval_items,
                    batch_size=max(32, m_batch), device=device, max_new_tokens=max_answer_tokens,
                    sft_baseline_outcomes=sft_outcomes,
                    story_val_stream=story_val_stream, retention_config=retention_eval_cfg,
                    precision=precision, step=step,
                )
                val_error_rate = max(0.0, 1.0 - last_eval["val_strict_pass"])
                elapsed = max(1e-6, perf_counter() - interval_start)
                _log_grpo_eval(last_eval, step=step, loss_decimals=loss_decimals)

                metrics.update({
                    "train_loss_interval_mean": interval_mean_err,
                    "val_loss": val_error_rate,
                    "val_strict_pass": last_eval["val_strict_pass"],
                    "val_exact_match": last_eval["val_exact_match"],
                    "val_valid_format": last_eval["val_valid_format"],
                    "val_valid_work": last_eval["val_valid_work"],
                    "val_eos_rate": last_eval["val_eos_rate"],
                    "val_mean_tokens": last_eval["val_mean_tokens"],
                    "rl_fixed": last_eval["rl_fixed"],
                    "rl_broke": last_eval["rl_broke"],
                    "lr": lr,
                    "steps_per_second": len(interval_rewards) / elapsed,
                })
                if last_eval["story_val_loss"] is not None:
                    metrics["story_val_loss"] = last_eval["story_val_loss"]
                for op_name, acc in last_eval["op_accuracy"].items():
                    metrics[f"val_acc_op_{op_name}"] = acc
                for cat, acc in last_eval["category_accuracy"].items():
                    metrics[f"val_acc_{cat}"] = acc

                run.log(metrics, step=step)
                history.append(metrics)

                print(f"Saving GRPO checkpoint snapshot at step {step:,}...", flush=True)
                session.save_checkpoint(
                    {
                        "version": 1,
                        **run_identity,
                        "step": step,
                        "model": policy_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "tokenizer_json": tokenizer_json,
                        "metrics": history,
                    },
                )
                interval_rewards.clear()
                interval_start = perf_counter()

                if check_stop():
                    print(f"Stopping GRPO early after saving checkpoint at step {last_step:,}.", flush=True)
                    break

            status = "Completed" if not check_stop() and last_step >= train_cfg.steps else "Stopped"
            strict_pct = last_eval.get("val_strict_pass", 0.0) * 100.0
            print(
                f"GRPO {status} at step {last_step:,}. Final mean reward: {1.0 - train_error_rate:.{loss_decimals}f}, "
                f"val strict pass: {strict_pct:.1f}% (fixed={last_eval.get('rl_fixed', 0)}, broke={last_eval.get('rl_broke', 0)})",
                flush=True,
            )

        if story_val_stream is not None:
            story_val_stream.close()

        return {
            "status": status,
            "step": last_step,
            "val_loss": val_error_rate,
            "val_strict_pass": last_eval.get("val_strict_pass", 0.0),
            "val_exact_match": last_eval.get("val_exact_match", 0.0),
            "val_valid_work": last_eval.get("val_valid_work", 0.0),
            "val_eos_rate": last_eval.get("val_eos_rate", 0.0),
            "rl_fixed": last_eval.get("rl_fixed", 0),
            "rl_broke": last_eval.get("rl_broke", 0),
            "story_val_loss": last_eval.get("story_val_loss"),
            "op_accuracy": last_eval.get("op_accuracy", {}),
            "mlflow_run_id": active_run_id,
            "log_path": str(log_path),
            "rl_dataset_dir": str(resolved_rl_dir),
            "hard_train_count": len(hard_train_examples),
            "parent_run_id": parent_run_id,
            "parent_checkpoint_step": parent_step,
        }
