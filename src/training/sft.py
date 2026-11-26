"""Phase 4 Arithmetic Supervised Fine-Tuning (SFT) trainer."""

from dataclasses import asdict
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .common import (
    TokenStream,
    TrainConfig,
    configure_optimizer,
    evaluate,
    loss_for,
    open_run_log,
    require_training_gpu,
    resolve_precision,
    tracked_training_session,
)
from ..core.checkpoint import build_model_config, load_checkpoint, resolve_base_checkpoint
from ..core.generate import generate_text
from ..core.model import GPT
from ..core.tokenizer import EOS, verify_corpus_files
from ..data.arithmetic import (
    GENERATOR_VERSION,
    ArithmeticExample,
    collate_sft_batch,
    encode_sft_split,
    load_or_restore_arithmetic_dataset,
    normalize_answer_format,
    normalize_ops,
    score_arithmetic_completion,
    select_stratified_eval_problems,
)
from ..data.corpus import ensure_tokenized_corpus
from ..infra.checkpoint_store import CheckpointStore
from ..infra.tracking import download_checkpoint
from ..settings import AZURE_CONTAINER, SFT_PRESETS, TOKENIZED_DATA_DIR


def _sample_retention_windows(
    stream: TokenStream,
    batch_size: int,
    step: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng([seed, 900_000 + step])
    n_windows = max(1, stream.windows)
    window_ids = rng.integers(0, n_windows, size=batch_size, dtype=np.int64)
    return stream.batch(window_ids * stream.context_length, device)


def _sample_arith_batch(
    encoded: list[tuple[list[int], list[int]]],
    batch_size: int,
    step: int,
    seed: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(encoded)
    batches_per_epoch = max(1, math.ceil(n / batch_size))
    epoch, batch_idx = divmod(step - 1, batches_per_epoch)
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(n)
    start = batch_idx * batch_size
    chosen = order[start:start + batch_size]
    if len(chosen) < batch_size and n > 0:
        extra = rng.integers(0, n, size=batch_size - len(chosen))
        chosen = np.concatenate([chosen, extra])
    return collate_sft_batch(encoded, chosen, pad_id=pad_id, device=device)


@torch.inference_mode()
def evaluate_sft(
    model: GPT,
    tokenizer,
    val_encoded: list[tuple[list[int], list[int]]],
    val_eval_items: list[ArithmeticExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device,
    precision: str,
    story_val_stream: TokenStream | None = None,
    retention_config: TrainConfig | None = None,
    step: int = 0,
    max_new_tokens: int = 24,
    max_failures_to_log: int = 5,
    answer_format: str = "direct",
) -> dict:
    """Compute masked val loss, greedy exact-match/op/category breakdown, and story retention loss."""
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        total_targets = 0
        for first in range(0, len(val_encoded), batch_size):
            indices = list(range(first, min(first + batch_size, len(val_encoded))))
            x, y = collate_sft_batch(val_encoded, indices, pad_id=pad_id, device=device)
            unmasked = int((y != -100).sum().item())
            if unmasked > 0:
                total_loss += float(loss_for(model, x, y, precision).item()) * unmasked
                total_targets += unmasked
        val_arith_loss = total_loss / max(1, total_targets)

        correct = 0
        formatted = 0
        per_cat: dict[str, list[int]] = {}
        per_op: dict[str, list[int]] = {}
        failures: list[tuple[ArithmeticExample, str]] = []
        for item in val_eval_items:
            raw = generate_text(
                model, tokenizer, item.prompt, max_new_tokens=max_new_tokens, device=device, temperature=0.0,
            )
            verdict = score_arithmetic_completion(raw, item.answer, answer_format=answer_format)
            cat_stats = per_cat.setdefault(item.category, [0, 0])
            cat_stats[1] += 1
            op_stats = per_op.setdefault(item.op, [0, 0])
            op_stats[1] += 1
            if verdict["valid_format"]:
                formatted += 1
            if verdict["exact_match"]:
                correct += 1
                cat_stats[0] += 1
                op_stats[0] += 1
            elif len(failures) < max_failures_to_log:
                failures.append((item, raw))

        total_items = max(1, len(val_eval_items))
        story_val_loss = None
        if story_val_stream is not None and retention_config is not None:
            story_val_loss = evaluate(model, story_val_stream, retention_config, device, precision, step=step)

        return {
            "val_arith_loss": val_arith_loss,
            "val_exact_match": correct / total_items,
            "val_valid_format": formatted / total_items,
            "val_correct": correct,
            "val_total": len(val_eval_items),
            "op_accuracy": {op: ok / cnt for op, (ok, cnt) in sorted(per_op.items())},
            "op_counts": {op: (ok, cnt) for op, (ok, cnt) in sorted(per_op.items())},
            "category_accuracy": {cat: ok / cnt for cat, (ok, cnt) in sorted(per_cat.items())},
            "category_counts": {cat: (ok, cnt) for cat, (ok, cnt) in sorted(per_cat.items())},
            "story_val_loss": story_val_loss,
            "sample_failures": failures,
        }
    finally:
        if was_training:
            model.train()


def _log_val_summary(eval_res: dict, *, step: int, loss_decimals: int) -> None:
    acc_pct = eval_res["val_exact_match"] * 100.0
    fmt_pct = eval_res["val_valid_format"] * 100.0
    story_part = (
        f" | story val loss {eval_res['story_val_loss']:.{loss_decimals}f}"
        if eval_res["story_val_loss"] is not None else ""
    )
    print(
        f"Step {step:,} | val arith loss {eval_res['val_arith_loss']:.{loss_decimals}f} | "
        f"exact match {eval_res['val_correct']}/{eval_res['val_total']} ({acc_pct:.1f}%) | "
        f"valid format {fmt_pct:.1f}%{story_part}",
        flush=True,
    )
    if eval_res["op_counts"]:
        op_parts = [f"{op}={ok}/{cnt}" for op, (ok, cnt) in eval_res["op_counts"].items()]
        print("  Ops:        " + ", ".join(op_parts), flush=True)
    if eval_res["category_counts"]:
        parts = [f"{c}={ok}/{cnt}" for c, (ok, cnt) in eval_res["category_counts"].items()]
        print("  Categories: " + ", ".join(parts), flush=True)
    if eval_res["sample_failures"]:
        for item, raw in eval_res["sample_failures"]:
            preview = raw.strip().replace("\n", "\\n")[:140]
            prompt_one_line = item.prompt.replace("\n", " ").strip()
            print(f"  [MISS] {item.category} | {prompt_one_line} -> {preview!r} (expected {item.answer})", flush=True)


def run_sft(
    *,
    preset: str = "sft",
    data_dir: Path | None = None,
    ops: tuple[str, ...] | list[str] | str | None = None,
    max_digits: int | None = None,
    allow_negative: bool | None = None,
    samples_per_op: int | None = None,
    answer_format: str | None = None,
    include_multistep: bool | None = None,
    no_drive: bool = False,
    base_run_id: str | None = None,
    base_checkpoint_alias: str | None = None,
    base_checkpoint: Path | None = None,
    context_length: int | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    retention_ratio: float | None = None,
    disable_retention: bool = False,
    resume: bool = False,
    mlflow_run_id: str | None = None,
    stop_after: int | None = None,
    precision_request: str = "auto",
    seed: int | None = None,
    eval_every: int | None = None,
    azure_container: str | None = None,
    loss_decimals: int = 4,
    device: torch.device | None = None,
    logs_dir: Path | None = None,
    retention_tokenized_dir: Path | None = None,
    checkpoints_dir: Path | None = None,
) -> dict:
    """Run Phase 4 masked arithmetic SFT using a prepared dataset (disk or Google Drive)."""
    from tokenizers import Tokenizer

    if preset not in SFT_PRESETS:
        raise ValueError(f"Unknown SFT preset {preset!r}. Choose from: {list(SFT_PRESETS.keys())}")
    cfg_defaults = dict(SFT_PRESETS[preset])

    alias = base_checkpoint_alias if base_checkpoint_alias is not None else cfg_defaults["base_checkpoint_alias"]
    target_ctx = context_length if context_length is not None else cfg_defaults["context_length"]
    op_list = normalize_ops(ops if ops is not None else cfg_defaults["ops"])
    n_digits = max_digits if max_digits is not None else cfg_defaults["max_digits"]
    neg_allowed = allow_negative if allow_negative is not None else cfg_defaults["allow_negative"]
    n_samples = samples_per_op if samples_per_op is not None else cfg_defaults["samples_per_op"]
    fmt = normalize_answer_format(answer_format if answer_format is not None else cfg_defaults["answer_format"])
    with_multistep = include_multistep if include_multistep is not None else bool(cfg_defaults["include_multistep"])
    ret_ratio = 0.0 if disable_retention else (
        retention_ratio if retention_ratio is not None else cfg_defaults["retention_ratio"]
    )
    if not 0.0 <= ret_ratio < 1.0:
        raise ValueError("retention_ratio must be in [0.0, 1.0).")

    run_seed = seed if seed is not None else cfg_defaults["seed"]
    n_steps = steps if steps is not None else cfg_defaults["steps"]
    n_batch = batch_size if batch_size is not None else cfg_defaults["batch_size"]
    peak_lr = learning_rate if learning_rate is not None else cfg_defaults["learning_rate"]
    min_lr = cfg_defaults["min_learning_rate"] if learning_rate is None else peak_lr * 0.1
    eval_interval = eval_every if eval_every is not None else cfg_defaults["eval_every"]
    warmup = min(cfg_defaults["warmup_steps"], max(0, n_steps - 1))

    train_cfg = TrainConfig(
        batch_size=n_batch,
        steps=n_steps,
        warmup_steps=warmup,
        learning_rate=peak_lr,
        min_learning_rate=min_lr,
        weight_decay=cfg_defaults["weight_decay"],
        eval_every=eval_interval,
        eval_token_fraction=cfg_defaults["eval_token_fraction"],
        seed=run_seed,
    )
    stop = train_cfg.steps if stop_after is None else stop_after
    if not 1 <= stop <= train_cfg.steps:
        raise ValueError("stop_after must be between 1 and total steps.")

    with open_run_log(preset, logs_dir=logs_dir) as (log_path, log_lock, is_stop_requested):
        if resume and not mlflow_run_id:
            raise ValueError("Set mlflow_run_id when resuming an SFT run.")
        container = azure_container or AZURE_CONTAINER
        if device is None:
            device = require_training_gpu()
        precision = resolve_precision(precision_request, device)

        # Load verified prepared arithmetic dataset from disk or Google Drive (never touches sealed test)
        resolved_data_dir, arith_splits, arith_manifest = load_or_restore_arithmetic_dataset(
            dataset_dir=data_dir,
            ops=op_list,
            max_digits=n_digits,
            allow_negative=neg_allowed,
            samples_per_op=n_samples,
            seed=run_seed,
            answer_format=fmt,
            include_multistep=with_multistep,
            use_drive=not no_drive,
        )
        effective_digits = int(arith_manifest.get("max_digits", n_digits))
        effective_format = normalize_answer_format(arith_manifest.get("answer_format", fmt))
        effective_multistep = bool(arith_manifest.get("include_multistep", with_multistep))
        print(
            f"Loaded prepared dataset ({arith_manifest.get('generator_version', GENERATOR_VERSION)}) from {resolved_data_dir}: "
            f"ops={arith_manifest.get('ops')}, max_digits={effective_digits}, "
            f"negatives={arith_manifest.get('allow_negative')}, format={effective_format}, multistep={effective_multistep}, "
            f"train={len(arith_splits['train']):,} examples, val={len(arith_splits['val']):,}, "
            f"sealed_test={len(arith_splits['test']):,} (reserved for Phase 6)",
            flush=True,
        )

        # Load base checkpoint weights & tokenizer
        base_path, base_ckpt = resolve_base_checkpoint(
            checkpoint_path=base_checkpoint,
            run_id=base_run_id,
            alias=alias,
            azure_container=container,
            checkpoints_dir=checkpoints_dir,
        )
        parent_run_id = base_ckpt.get("parent_run_id") or base_path.parent.name
        parent_step = int(base_ckpt.get("step", 0))
        tokenizer_json = base_ckpt["tokenizer_json"]
        tokenizer = Tokenizer.from_str(tokenizer_json)
        eos_id = tokenizer.token_to_id(EOS)
        if eos_id is None:
            raise ValueError(f"Checkpoint tokenizer is missing {EOS!r}.")

        model_config = build_model_config(base_ckpt, context_length=target_ctx)
        print(
            f"Initialized SFT from {base_path.name} (parent_run={parent_run_id}, parent_step={parent_step:,}, "
            f"context_length={model_config.context_length:,}, pos_encoding={model_config.pos_encoding})",
            flush=True,
        )

        train_encoded = encode_sft_split(
            arith_splits["train"], tokenizer, context_length=model_config.context_length, eos_id=eos_id,
        )
        val_encoded = encode_sft_split(
            arith_splits["val"], tokenizer, context_length=model_config.context_length, eos_id=eos_id,
        )
        val_eval_items = select_stratified_eval_problems(
            arith_splits["val"], cfg_defaults["val_eval_problems"], seed=run_seed,
        )
        max_val_target_tokens = max((sum(1 for t in y if t != -100) for _, y in val_encoded), default=0)
        base_budget = 24 if effective_format == "direct" else 96
        max_answer_tokens = max(base_budget, effective_digits * 2 + 6, max_val_target_tokens + 24)

        # Optional SimpleStories retention streams
        story_train_stream = None
        story_val_stream = None
        retention_eval_cfg = None
        if ret_ratio > 0.0:
            tok_dir = retention_tokenized_dir or TOKENIZED_DATA_DIR
            corpus_manifest = ensure_tokenized_corpus(tokenized_path=tok_dir)
            verify_corpus_files(tok_dir, corpus_manifest)
            story_train_stream = TokenStream(
                tok_dir / "train.bin", corpus_manifest["train"]["tokens"], model_config.context_length,
            )
            story_val_stream = TokenStream(
                tok_dir / "val.bin", corpus_manifest["val"]["tokens"], model_config.context_length,
            )
            val_frac = min(
                1.0,
                (cfg_defaults["retention_eval_batches"] * train_cfg.batch_size * model_config.context_length)
                / max(1, story_val_stream.target_count),
            )
            retention_eval_cfg = TrainConfig(
                batch_size=train_cfg.batch_size,
                steps=train_cfg.steps,
                warmup_steps=train_cfg.warmup_steps,
                learning_rate=train_cfg.learning_rate,
                min_learning_rate=train_cfg.min_learning_rate,
                weight_decay=train_cfg.weight_decay,
                eval_every=train_cfg.eval_every,
                eval_token_fraction=max(1e-6, val_frac),
                seed=run_seed,
            )
            print(f"SimpleStories retention active: ratio={ret_ratio:.2f} from {tok_dir}", flush=True)

        torch.manual_seed(run_seed)
        model = GPT(model_config).to(device)
        model.load_state_dict(base_ckpt["model"])
        del base_ckpt
        optimizer = configure_optimizer(
            model,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            device=device,
        )

        run_identity = {
            "stage": "sft",
            "parent_run_id": parent_run_id,
            "parent_checkpoint_step": parent_step,
            "model_config": asdict(model_config),
            "train_config": asdict(train_cfg),
            "arith_manifest": arith_manifest,
            "retention_ratio": ret_ratio,
        }

        start = 0
        history: list[dict] = []
        if resume:
            print(f"Resuming SFT run {mlflow_run_id} from Azure container {container}...", flush=True)
            resume_store = CheckpointStore(container, mlflow_run_id)
            with download_checkpoint(resume_store, alias="latest") as ckpt_path:
                saved = load_checkpoint(ckpt_path)
            if saved.get("version") != 1:
                raise ValueError(f"Unsupported checkpoint version: {saved.get('version')}")
            if any(saved.get(key) != value for key, value in run_identity.items()):
                raise ValueError("SFT checkpoint identity disagrees with current SFT settings.")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start = int(saved["step"])
            history = saved["metrics"]
            del saved
            print(f"Restored SFT checkpoint from step {start:,}. Resuming to step {stop:,}.", flush=True)

        train_loss = float("nan")
        val_loss = float("nan")
        last_eval: dict = {}
        last_step = start

        with tracked_training_session(
            run_params={
                "preset": preset,
                "device": device.type,
                "precision": precision,
                "parent_run_id": parent_run_id,
                "parent_checkpoint_step": parent_step,
                "model_config": asdict(model_config),
                "train_config": asdict(train_cfg),
                "retention_ratio": ret_ratio,
                "ops": ",".join(arith_manifest.get("ops", op_list)),
                "max_digits": effective_digits,
                "allow_negative": arith_manifest.get("allow_negative", neg_allowed),
                "answer_format": effective_format,
                "include_multistep": effective_multistep,
                "dataset_dir": str(resolved_data_dir),
            },
            log_path=log_path,
            log_lock=log_lock,
            is_stop_requested=is_stop_requested,
            azure_container=container,
            resume_run_id=mlflow_run_id if resume else None,
            tags={
                "stage": "sft",
                "parent_run_id": str(parent_run_id),
                "parent_checkpoint_step": str(parent_step),
                "base_checkpoint_alias": alias,
                "arith_generator_version": str(arith_manifest.get("generator_version", GENERATOR_VERSION)),
                "arith_format": effective_format,
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
                    val_loss = history[-1].get("val_loss", val_loss)
            else:
                print("Evaluating Step 0 pre-SFT baseline on held-out val set...", flush=True)
                last_eval = evaluate_sft(
                    model, tokenizer, val_encoded, val_eval_items,
                    batch_size=train_cfg.batch_size, pad_id=eos_id, device=device, precision=precision,
                    story_val_stream=story_val_stream, retention_config=retention_eval_cfg, step=0,
                    max_new_tokens=max_answer_tokens, answer_format=effective_format,
                )
                val_loss = last_eval["val_arith_loss"]
                _log_val_summary(last_eval, step=0, loss_decimals=loss_decimals)
                step0_row = {
                    "step": 0,
                    "val_loss": val_loss,
                    "val_exact_match": last_eval["val_exact_match"],
                    "val_valid_format": last_eval["val_valid_format"],
                    "train_loss_interval_mean": val_loss,
                }
                if last_eval["story_val_loss"] is not None:
                    step0_row["story_val_loss"] = last_eval["story_val_loss"]
                history = [step0_row]
                run.log(step0_row, step=0)

            interval_start = perf_counter()
            interval_losses: list[float] = []
            step_rng = np.random.default_rng(run_seed)
            for _ in range(start):
                step_rng.random()

            for step in range(start + 1, stop + 1):
                if check_stop():
                    print(f"Stopping SFT early as requested at step {last_step:,}.", flush=True)
                    break

                lr = train_cfg.lr_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)

                use_story = (story_train_stream is not None) and (step_rng.random() < ret_ratio)
                if use_story:
                    x, y = _sample_retention_windows(
                        story_train_stream, train_cfg.batch_size, step, run_seed, device,
                    )
                    batch_kind = "story"
                else:
                    if story_train_stream is None:
                        step_rng.random()
                    x, y = _sample_arith_batch(
                        train_encoded, train_cfg.batch_size, step, run_seed, eos_id, device,
                    )
                    batch_kind = "arith"

                loss = loss_for(model, x, y, precision)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last_step = step

                train_loss = float(loss.detach().item())
                if not math.isfinite(train_loss):
                    raise FloatingPointError(f"Non-finite SFT loss at step {step}.")
                interval_losses.append(train_loss)
                print(f"Step {step:,} ({batch_kind}) | train loss {train_loss:.{loss_decimals}f}", flush=True)

                metrics: dict = {"step": step, "train_loss": train_loss, f"{batch_kind}_train_loss": train_loss}
                is_eval_step = (step % train_cfg.eval_every == 0) or (step == stop) or check_stop()
                if not is_eval_step:
                    run.log(metrics, step=step)
                    continue

                interval_mean = sum(interval_losses) / max(1, len(interval_losses))
                last_eval = evaluate_sft(
                    model, tokenizer, val_encoded, val_eval_items,
                    batch_size=train_cfg.batch_size, pad_id=eos_id, device=device, precision=precision,
                    story_val_stream=story_val_stream, retention_config=retention_eval_cfg, step=step,
                    max_new_tokens=max_answer_tokens, answer_format=effective_format,
                )
                val_loss = last_eval["val_arith_loss"]
                elapsed = max(1e-6, perf_counter() - interval_start)
                _log_val_summary(last_eval, step=step, loss_decimals=loss_decimals)

                metrics.update({
                    "train_loss_interval_mean": interval_mean,
                    "val_loss": val_loss,
                    "val_exact_match": last_eval["val_exact_match"],
                    "val_valid_format": last_eval["val_valid_format"],
                    "lr": lr,
                    "steps_per_second": len(interval_losses) / elapsed,
                })
                if last_eval["story_val_loss"] is not None:
                    metrics["story_val_loss"] = last_eval["story_val_loss"]
                for op_name, acc in last_eval["op_accuracy"].items():
                    metrics[f"val_acc_op_{op_name}"] = acc
                for cat, acc in last_eval["category_accuracy"].items():
                    metrics[f"val_acc_{cat}"] = acc

                run.log(metrics, step=step)
                history.append(metrics)

                print(f"Saving SFT checkpoint snapshot at step {step:,}...", flush=True)
                session.save_checkpoint(
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
                interval_losses.clear()
                interval_start = perf_counter()

                if check_stop():
                    print(f"Stopping SFT early after saving checkpoint at step {last_step:,}.", flush=True)
                    break

            status = "Completed" if not check_stop() and last_step >= train_cfg.steps else "Stopped"
            acc_pct = last_eval.get("val_exact_match", 0.0) * 100.0
            print(
                f"SFT {status} at step {last_step:,}. Final train loss: {train_loss:.{loss_decimals}f}, "
                f"val arith loss: {val_loss:.{loss_decimals}f}, val exact match: {acc_pct:.1f}%",
                flush=True,
            )

        return {
            "status": status,
            "step": last_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_exact_match": last_eval.get("val_exact_match", 0.0),
            "val_valid_format": last_eval.get("val_valid_format", 0.0),
            "story_val_loss": last_eval.get("story_val_loss"),
            "op_accuracy": last_eval.get("op_accuracy", {}),
            "category_accuracy": last_eval.get("category_accuracy", {}),
            "mlflow_run_id": active_run_id,
            "log_path": str(log_path),
            "dataset_dir": str(resolved_data_dir),
            "answer_format": effective_format,
            "include_multistep": effective_multistep,
            "parent_run_id": parent_run_id,
            "parent_checkpoint_step": parent_step,
        }
