#!/usr/bin/env python3
"""MiniReasoner main entrypoint and CLI."""

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path and load .env
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.data.arithmetic import ANSWER_FORMAT_CHOICES, SUPPORTED_OPS, prepare_arithmetic_dataset
from src.data.corpus import ensure_tokenized_corpus
from src.infra.tracking import sync_deleted_runs
from src.settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT, PRESETS, RL_PRESETS, SFT_DATA_DEFAULTS, SFT_PRESETS
from src.training.grpo import prepare_hard_rl_dataset, run_grpo
from src.training.pretrain import run_training
from src.training.sft import run_sft


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main [train]",
        description="MiniReasoner CLI - Pre-train, prepare data, run SFT/GRPO RL, or sync Azure checkpoints.",
        epilog=(
            "Subcommands:\n"
            "  train                Pretrain a MiniReasoner base model (default command)\n"
            "  prepare-data         Generate, preview, save, and cache Phase 1 corpus, Phase 4 SFT, or Phase 5 failed RL data\n"
            "  sft                  Phase 4 Arithmetic Supervised Fine-Tuning from a prepared dataset\n"
            "  grpo (rl)            Phase 5 Verifiable-Reward GRPO RL on failed hard CoT traces\n"
            "  cleanup (sync-azure) Preview or delete Azure checkpoints for MLflow runs deleted in Databricks\n\n"
            "Run `python src/main.py <subcommand> --help` for options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        default="base",
        choices=list(PRESETS.keys()),
        help="Model preset architecture and hyperparameters (default: base).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing training run from Azure Blob Storage.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Databricks MLflow run ID to resume (required when --resume is set).",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Stop at absolute optimizer step (default: None for full 1-epoch run).",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "bf16", "fp32"],
        help="Precision (default: auto -> BF16 on Ampere/A30+, FP32 otherwise).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--loss-decimals",
        type=int,
        default=4,
        help="Number of decimal digits for loss formatting (default: 4).",
    )
    parser.add_argument(
        "--tokenizer-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training fresh tokenizer (default: auto).",
    )
    return parser


def build_prepare_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main prepare-data",
        description="Generate, inspect, save to disk, and cache Phase 1 corpus, Phase 4 SFT splits (`direct` vs `cot`), or Phase 5 failed RL prompts.",
    )
    parser.add_argument(
        "--stage",
        default="sft",
        choices=["sft", "corpus", "rl"],
        help="Dataset stage to prepare: 'sft' arithmetic splits (default), 'corpus' SimpleStories tokens, or 'rl' mined failed prompts.",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=list(SFT_DATA_DEFAULTS["ops"]),
        help=f"Operations to include (choices: {', '.join(SUPPORTED_OPS)}; default: add sub mul div).",
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=SFT_DATA_DEFAULTS["max_digits"],
        help="Maximum operand digits (supports 3, 4, 5+ digits; default: 3).",
    )
    parser.add_argument(
        "--nonnegative-only",
        action="store_true",
        help="Restrict operands and results to nonnegative integers (default: negative numbers enabled).",
    )
    parser.add_argument(
        "--samples-per-op",
        type=int,
        default=SFT_DATA_DEFAULTS["samples_per_op"],
        help=f"Target canonical problems per operation (default: {SFT_DATA_DEFAULTS['samples_per_op']:,}).",
    )
    parser.add_argument(
        "--format",
        default=SFT_DATA_DEFAULTS["answer_format"],
        choices=list(ANSWER_FORMAT_CHOICES),
        help="SFT target format: 'direct' (normal `Answer: 55`) or 'cot' / 'scratchpad' (`Work: ...\\nAnswer: 55`).",
    )
    parser.add_argument(
        "--multistep",
        action=argparse.BooleanOptionalAction,
        default=SFT_DATA_DEFAULTS["include_multistep"],
        help="Include multi-step chained expressions and conversational math prompts.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=SFT_DATA_DEFAULTS["val_fraction"],
        help="Fraction of canonical problems reserved for val evaluation (default: 0.1).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=SFT_DATA_DEFAULTS["test_fraction"],
        help="Fraction of canonical problems reserved for sealed final test (default: 0.1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom directory for generated splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SFT_DATA_DEFAULTS["seed"],
        help="Random seed for stratified generation and splitting (default: 42).",
    )
    parser.add_argument(
        "--tokenizer-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training BPE tokenizer when --stage corpus runs from scratch (default: auto).",
    )
    parser.add_argument("--sft-run-id", default=None, help="Source CoT SFT run ID for --stage rl mining.")
    parser.add_argument("--checkpoint-alias", default=None, help="SFT checkpoint alias (`latest`, `best`, `step-N`) for --stage rl.")
    parser.add_argument("--sft-checkpoint", type=Path, default=None, help="Optional local CoT SFT .pt checkpoint for --stage rl.")
    parser.add_argument("--scan-limit", type=int, default=None, help="Optional cap on canonical train problems scanned when --stage rl mines failures.")
    parser.add_argument("--scan-batch-size", type=int, default=None, help="Batch size for --stage rl greedy failure scan.")
    parser.add_argument("--include-trivial", action="store_true", help="Include 100%%-solved trivial categories (`add`, `*_zero`, `*_unit`, `*_1d`) during --stage rl mining.")
    parser.add_argument("--no-expand-variants", action="store_true", help="Keep only canonical failed train prompts instead of scanning failed wording variants.")
    parser.add_argument(
        "--no-drive",
        action="store_true",
        help="Skip uploading the generated SFT dataset to Google Drive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate and overwrite local files if target directory already exists.",
    )
    return parser


def build_sft_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main sft",
        description="Run Phase 4 masked arithmetic Supervised Fine-Tuning (SFT).",
    )
    parser.add_argument(
        "--preset",
        default="sft",
        choices=list(SFT_PRESETS.keys()),
        help="SFT hyperparameter preset (default: sft; use sft_pilot for quick validation).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to prepared arithmetic SFT directory (default: matches --ops / --max-digits / --samples-per-op).",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=None,
        help="Operations selector matching prepared dataset (default: add sub mul div).",
    )
    parser.add_argument("--max-digits", type=int, default=None, help="Maximum operand digits (default: 3).")
    parser.add_argument(
        "--nonnegative-only",
        action="store_true",
        default=None,
        help="Match a nonnegative-only dataset instead of default signed arithmetic.",
    )
    parser.add_argument("--samples-per-op", type=int, default=None, help="Canonical problems per op (default: 12000).")
    parser.add_argument(
        "--format",
        default=None,
        choices=list(ANSWER_FORMAT_CHOICES),
        help="Override target format ('direct' vs 'cot' / 'scratchpad').",
    )
    parser.add_argument(
        "--multistep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include multi-step chained expressions and conversational math prompts.",
    )
    parser.add_argument(
        "--no-drive",
        action="store_true",
        help="Disable Google Drive restore fallback for SFT dataset loading.",
    )
    parser.add_argument(
        "--base-run-id",
        default=None,
        help="Azure/MLflow run ID of the Phase 3 base model (default: newest active base run).",
    )
    parser.add_argument(
        "--checkpoint-alias",
        default=None,
        help="Base checkpoint alias to load: latest, best, or step-N (default: latest).",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Optional local .pt base checkpoint file for offline initialization.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Optional context extension for SFT, e.g. 2048 (default: inherit base checkpoint's 1,024).",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override total SFT optimizer steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override SFT batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override SFT peak learning rate.")
    parser.add_argument("--eval-every", type=int, default=None, help="Override val evaluation / Azure save interval.")
    parser.add_argument(
        "--retention-ratio",
        type=float,
        default=None,
        help="Fraction of batches sampled from SimpleStories to prevent forgetting (default: 0.1).",
    )
    parser.add_argument(
        "--no-retention",
        action="store_true",
        help="Disable SimpleStories retention mixing during SFT.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing SFT run from Azure Blob Storage.")
    parser.add_argument("--run-id", default=None, help="Databricks MLflow run ID to resume when --resume is set.")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop SFT early at absolute step.")
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "bf16", "fp32"],
        help="Precision (default: auto -> BF16 on Ampere+, FP32 otherwise).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override SFT random seed (default: 42).")
    parser.add_argument("--loss-decimals", type=int, default=4, help="Decimal digits for losses (default: 4).")
    return parser


def build_grpo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main grpo",
        description="Run Phase 5 Group Relative Policy Optimization (GRPO) on failed hard arithmetic traces.",
    )
    parser.add_argument(
        "--preset",
        default="grpo",
        choices=list(RL_PRESETS.keys()),
        help="GRPO hyperparameter preset (default: grpo; use grpo_pilot for fast smoke runs).",
    )
    parser.add_argument("--sft-run-id", default=None, help="Azure/MLflow run ID of the Phase 4 CoT SFT checkpoint (default: newest active scratchpad SFT run).")
    parser.add_argument("--checkpoint-alias", default=None, help="Source SFT checkpoint alias: latest, best, or step-N (default: latest).")
    parser.add_argument("--sft-checkpoint", type=Path, default=None, help="Optional local CoT SFT .pt file for offline initialization.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to source CoT arithmetic SFT directory (arith_v3).")
    parser.add_argument("--rl-data-dir", type=Path, default=None, help="Optional directory to read/write mined hard failed RL examples.")
    parser.add_argument("--ops", nargs="+", default=None, help="Source operations selector (default: add sub mul div).")
    parser.add_argument("--max-digits", type=int, default=None, help="Maximum operand digits (default: 3).")
    parser.add_argument("--nonnegative-only", action="store_true", default=None, help="Match nonnegative-only arithmetic dataset.")
    parser.add_argument("--samples-per-op", type=int, default=None, help="Source canonical problems per op (default: 12000).")
    parser.add_argument("--scan-limit", type=int, default=None, help="Optional cap on canonical train items scanned if mining hard failures on the fly.")
    parser.add_argument("--overwrite-hard-data", action="store_true", help="Re-mine failed hard RL examples instead of reusing cached hard_train.jsonl.")
    parser.add_argument("--no-drive", action="store_true", help="Disable Google Drive restore fallback for dataset loading.")
    parser.add_argument("--context-length", type=int, default=None, help="Optional context length override.")
    parser.add_argument("--steps", type=int, default=None, help="Override total GRPO optimizer steps.")
    parser.add_argument("--prompts-per-step", type=int, default=None, help="Number of hard failed prompts sampled per GRPO step.")
    parser.add_argument("--group-size", type=int, default=None, help="Number of rollouts G sampled per prompt (must be >= 2).")
    parser.add_argument("--mini-batch-size", type=int, default=None, help="Chunk size for log-prob/backward accumulation.")
    parser.add_argument("--ppo-epochs", type=int, default=None, help="Gradient passes per sampled rollout batch (default: 1).")
    parser.add_argument("--kl-coef", type=float, default=None, help="Frozen CoT SFT reference KL penalty coefficient beta.")
    parser.add_argument("--clip-eps", type=float, default=None, help="PPO/GRPO probability ratio clipping epsilon.")
    parser.add_argument("--temperature", type=float, default=None, help="Rollout sampling temperature.")
    parser.add_argument("--lr", type=float, default=None, help="Override GRPO peak learning rate.")
    parser.add_argument("--eval-every", type=int, default=None, help="Override val evaluation / Azure checkpoint interval.")
    parser.add_argument("--no-retention-eval", action="store_true", help="Skip SimpleStories validation loss monitoring during GRPO.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing GRPO run from Azure Blob Storage.")
    parser.add_argument("--run-id", default=None, help="Databricks MLflow run ID to resume when --resume is set.")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop GRPO early at absolute step.")
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "bf16", "fp32"],
        help="Precision (default: auto -> BF16 on Ampere+, FP32 otherwise).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override GRPO random seed (default: 42).")
    parser.add_argument("--loss-decimals", type=int, default=4, help="Decimal digits for printed metrics (default: 4).")
    return parser


def build_cleanup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main cleanup",
        description="Delete Azure Blob Storage checkpoints for runs deleted in Databricks MLflow (dry-run by default).",
    )
    parser.add_argument(
        "--container",
        default=AZURE_CONTAINER,
        help=f"Azure Blob Storage container name (default: {AZURE_CONTAINER}).",
    )
    parser.add_argument(
        "--experiment",
        default=MLFLOW_EXPERIMENT,
        help=f"MLflow experiment path used as Azure folder prefix (default: {MLFLOW_EXPERIMENT}).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete candidate Azure blobs (without this flag, only a dry-run preview is shown).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation when --delete is specified.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Also target Azure run folders that return RESOURCE_DOES_NOT_EXIST in Databricks MLflow.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in ("cleanup", "sync-azure"):
        args = build_cleanup_parser().parse_args(raw_argv[1:])
        sync_deleted_runs(
            container=args.container,
            experiment=args.experiment,
            delete=args.delete,
            yes=args.yes,
            include_missing=args.include_missing,
        )
        return

    if raw_argv and raw_argv[0] in ("prepare-data", "prepare-sft-data"):
        args = build_prepare_data_parser().parse_args(raw_argv[1:])
        if args.stage == "corpus":
            ensure_tokenized_corpus(
                seed=args.seed,
                tokenized_path=args.output_dir,
                tokenizer_device=args.tokenizer_device,
                sync_drive=not args.no_drive,
            )
        elif args.stage == "rl":
            prepare_hard_rl_dataset(
                sft_run_id=args.sft_run_id,
                sft_checkpoint_alias=args.checkpoint_alias,
                sft_checkpoint=args.sft_checkpoint,
                rl_data_dir=args.output_dir,
                ops=args.ops,
                max_digits=args.max_digits,
                allow_negative=not args.nonnegative_only,
                samples_per_op=args.samples_per_op,
                scan_limit=args.scan_limit,
                scan_batch_size=args.scan_batch_size,
                exclude_easy_categories=not args.include_trivial,
                expand_failed_variants=not args.no_expand_variants,
                no_drive=args.no_drive,
                overwrite=args.overwrite,
            )
        else:
            prepare_arithmetic_dataset(
                output_dir=args.output_dir,
                ops=args.ops,
                max_digits=args.max_digits,
                allow_negative=not args.nonnegative_only,
                samples_per_op=args.samples_per_op,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
                answer_format=args.format,
                include_multistep=args.multistep,
                sync_drive=not args.no_drive,
                overwrite=args.overwrite,
            )
        return

    if raw_argv and raw_argv[0] in ("grpo", "rl"):
        args = build_grpo_parser().parse_args(raw_argv[1:])
        allow_negative = False if args.nonnegative_only else None
        run_grpo(
            preset=args.preset,
            sft_run_id=args.sft_run_id,
            sft_checkpoint_alias=args.checkpoint_alias,
            sft_checkpoint=args.sft_checkpoint,
            data_dir=args.data_dir,
            rl_data_dir=args.rl_data_dir,
            ops=args.ops,
            max_digits=args.max_digits,
            allow_negative=allow_negative,
            samples_per_op=args.samples_per_op,
            scan_limit=args.scan_limit,
            overwrite_hard_data=args.overwrite_hard_data,
            no_drive=args.no_drive,
            context_length=args.context_length,
            steps=args.steps,
            prompts_per_step=args.prompts_per_step,
            group_size=args.group_size,
            mini_batch_size=args.mini_batch_size,
            ppo_epochs=args.ppo_epochs,
            kl_coef=args.kl_coef,
            clip_eps=args.clip_eps,
            temperature=args.temperature,
            learning_rate=args.lr,
            eval_every=args.eval_every,
            disable_retention_eval=args.no_retention_eval,
            resume=args.resume,
            mlflow_run_id=args.run_id,
            stop_after=args.stop_after,
            precision_request=args.precision,
            seed=args.seed,
            loss_decimals=args.loss_decimals,
        )
        return

    if raw_argv and raw_argv[0] == "sft":
        args = build_sft_parser().parse_args(raw_argv[1:])
        allow_negative = False if args.nonnegative_only else None
        run_sft(
            preset=args.preset,
            data_dir=args.data_dir,
            ops=args.ops,
            max_digits=args.max_digits,
            allow_negative=allow_negative,
            samples_per_op=args.samples_per_op,
            answer_format=args.format,
            include_multistep=args.multistep,
            no_drive=args.no_drive,
            base_run_id=args.base_run_id,
            base_checkpoint_alias=args.checkpoint_alias,
            base_checkpoint=args.base_checkpoint,
            context_length=args.context_length,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            retention_ratio=args.retention_ratio,
            disable_retention=args.no_retention,
            resume=args.resume,
            mlflow_run_id=args.run_id,
            stop_after=args.stop_after,
            precision_request=args.precision,
            seed=args.seed,
            eval_every=args.eval_every,
            loss_decimals=args.loss_decimals,
        )
        return

    if raw_argv and raw_argv[0] == "train":
        raw_argv = raw_argv[1:]

    args = build_train_parser().parse_args(raw_argv)
    run_training(
        preset=args.preset,
        resume=args.resume,
        mlflow_run_id=args.run_id,
        stop_after=args.stop_after,
        precision_request=args.precision,
        seed=args.seed,
        loss_decimals=args.loss_decimals,
        tokenizer_device=args.tokenizer_device,
    )


if __name__ == "__main__":
    main()
