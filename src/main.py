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

from src.settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT
from src.tracking import sync_deleted_runs
from src.train import run_training


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main [train]",
        description="MiniReasoner CLI - Train models or sync Azure checkpoints with Databricks MLflow.",
        epilog=(
            "Subcommands:\n"
            "  train               Pretrain a MiniReasoner model (default command)\n"
            "  cleanup (sync-azure) Preview or delete Azure checkpoints for MLflow runs deleted in Databricks\n\n"
            "Run `python src/main.py cleanup --help` for cleanup options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        default="base",
        choices=["base", "small"],
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
