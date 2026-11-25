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

from src.data.arithmetic import SUPPORTED_OPS, prepare_arithmetic_dataset
from src.infra.tracking import sync_deleted_runs
from src.settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT, PRESETS, SFT_DATA_DEFAULTS, SFT_PRESETS
from src.training.pretrain import run_training
from src.training.sft import run_sft


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main [train]",
        description="MiniReasoner CLI - Pre-train, prepare SFT data, run SFT, or sync Azure checkpoints.",
        epilog=(
            "Subcommands:\n"
            "  train                Pretrain a MiniReasoner base model (default command)\n"
            "  prepare-sft-data     Generate, preview, save, and cache Phase 4 arithmetic SFT data\n"
            "  sft                  Phase 4 Arithmetic Supervised Fine-Tuning from a prepared dataset\n"
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


def build_prepare_sft_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main prepare-sft-data",
        description="Generate, inspect, save to disk, and cache Phase 4 arithmetic SFT splits to Google Drive.",
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
        help="Optional custom directory for generated train/val/test JSONL splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SFT_DATA_DEFAULTS["seed"],
        help="Random seed for stratified generation and splitting (default: 42).",
    )
    parser.add_argument(
        "--no-drive",
        action="store_true",
        help="Skip uploading the generated dataset to Google Drive.",
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

    if raw_argv and raw_argv[0] in ("prepare-sft-data", "prepare-data"):
        args = build_prepare_sft_data_parser().parse_args(raw_argv[1:])
        prepare_arithmetic_dataset(
            output_dir=args.output_dir,
            ops=args.ops,
            max_digits=args.max_digits,
            allow_negative=not args.nonnegative_only,
            samples_per_op=args.samples_per_op,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            sync_drive=not args.no_drive,
            overwrite=args.overwrite,
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
