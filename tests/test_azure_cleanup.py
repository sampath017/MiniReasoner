"""Unit tests for Azure Blob Storage <-> Databricks MLflow cleanup sync."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from mlflow.exceptions import MlflowException, RestException
import numpy as np
import torch

from src.infra import azure_blob, checkpoint_store, tracking
from src.main import build_cleanup_parser, build_prepare_sft_data_parser, build_sft_parser, build_train_parser
from src.settings import PRESETS
from src.training import pretrain


class _FakeBlob:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.last_modified = datetime(2026, 11, 24, 18, 0, tzinfo=timezone.utc)


class _FakeBlobClient:
    def __init__(self, container, blob_name: str):
        self.container = container
        self.blob_name = blob_name

    def upload_blob(self, stream, *, overwrite: bool = False, metadata: dict | None = None):
        if self.blob_name in self.container.data and not overwrite:
            raise ResourceExistsError(self.blob_name)
        payload = stream.read()
        self.container.data[self.blob_name] = payload
        self.container.blobs[self.blob_name] = len(payload)
        self.container.metadata[self.blob_name] = dict(metadata or {})

    def get_blob_properties(self):
        if self.blob_name not in self.container.data:
            raise ResourceNotFoundError(self.blob_name)
        return SimpleNamespace(
            size=len(self.container.data[self.blob_name]),
            metadata=self.container.metadata.get(self.blob_name, {}),
            etag='"etag"',
        )


class _FakeContainer:
    url = "https://example.blob.core.windows.net/minireasoner"

    def __init__(self, blobs: dict[str, int]):
        self.blobs = dict(blobs)
        self.data = {name: b"x" * size for name, size in blobs.items()}
        self.metadata: dict[str, dict] = {}

    def list_blobs(self, *, name_starts_with: str = ""):
        return [
            _FakeBlob(name, size)
            for name, size in sorted(self.blobs.items())
            if name.startswith(name_starts_with)
        ]

    def list_blob_names(self, *, name_starts_with: str = "", results_per_page=None):
        return [name for name in sorted(self.blobs) if name.startswith(name_starts_with)]

    def get_blob_client(self, blob_name: str):
        return _FakeBlobClient(self, blob_name)

    def delete_blob(self, name: str):
        if name not in self.blobs:
            raise ResourceNotFoundError(name)
        del self.blobs[name]
        self.data.pop(name, None)
        self.metadata.pop(name, None)


class _FakeMlflowClient:
    def __init__(self, states: dict[str, str | Exception]):
        self.states = states

    def get_run(self, run_id: str):
        val = self.states[run_id]
        if isinstance(val, Exception):
            raise val
        return SimpleNamespace(info=SimpleNamespace(lifecycle_stage=val))


def test_sync_deleted_runs_dry_run_and_delete():
    """Verify dry-run safety, confirmation gate, active preservation, and include_missing."""
    prefix = "checkpoints/Shared/MiniReasoner/"
    container = _FakeContainer({
        f"{prefix}run_active/step-500_val-1.5_train-1.6_{'a' * 64}.pt": 1_000_000,
        f"{prefix}run_deleted/step-500_val-2.1_train-2.2_{'b' * 64}.pt": 2_000_000,
        f"{prefix}run_deleted/step-1000_val-1.9_train-2.0_{'c' * 64}.pt": 2_000_000,
        f"{prefix}run_missing/step-100_val-3.0_train-3.1_{'d' * 64}.pt": 500_000,
    })
    mlflow_client = _FakeMlflowClient({
        "run_active": "active",
        "run_deleted": "deleted",
        "run_missing": RestException({"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Not found"}),
    })

    with patch.object(checkpoint_store, "container_client", return_value=container):
        # 1. Dry-run leaves all blobs untouched
        res = tracking.sync_deleted_runs(delete=False, client=mlflow_client)
        assert res["deleted_blobs"] == 0
        assert [c["run_id"] for c in res["candidates"]] == ["run_deleted"]
        assert len(container.blobs) == 4

        # 2. Aborted confirmation leaves all blobs untouched
        res_abort = tracking.sync_deleted_runs(
            delete=True, client=mlflow_client, confirm_fn=lambda _: "no",
        )
        assert res_abort["deleted_blobs"] == 0
        assert len(container.blobs) == 4

        # 3. Confirmed deletion removes only run_deleted blobs
        res_del = tracking.sync_deleted_runs(
            delete=True, client=mlflow_client, confirm_fn=lambda _: "delete",
        )
        assert res_del["deleted_blobs"] == 2
        assert res_del["reclaimed_bytes"] == 4_000_000
        assert f"{prefix}run_active/step-500_val-1.5_train-1.6_{'a' * 64}.pt" in container.blobs
        assert f"{prefix}run_missing/step-100_val-3.0_train-3.1_{'d' * 64}.pt" in container.blobs

        # 4. --include-missing --yes removes purged run_missing while preserving run_active
        res_missing = tracking.sync_deleted_runs(
            delete=True, yes=True, include_missing=True, client=mlflow_client,
        )
        assert res_missing["deleted_blobs"] == 1
        assert list(container.blobs.keys()) == [
            f"{prefix}run_active/step-500_val-1.5_train-1.6_{'a' * 64}.pt"
        ]
    print("  [PASSED] Dry-run, confirmation guard, and deleted/missing sync")


def test_transient_mlflow_error_aborts_cleanly():
    """Verify transient MLflow API failures never classify runs as missing or delete blobs."""
    prefix = "checkpoints/Shared/MiniReasoner/"
    container = _FakeContainer({
        f"{prefix}run_unknown/step-500_val-1.5_train-1.6_{'a' * 64}.pt": 1_000_000,
    })
    mlflow_client = _FakeMlflowClient({
        "run_unknown": RestException({"error_code": "TEMPORARILY_UNAVAILABLE", "message": "Temporarily unavailable"}),
    })

    with patch.object(checkpoint_store, "container_client", return_value=container):
        try:
            tracking.sync_deleted_runs(delete=True, yes=True, include_missing=True, client=mlflow_client)
            raise AssertionError("Expected MlflowException on transient error")
        except MlflowException:
            pass
        assert len(container.blobs) == 1
    print("  [PASSED] Transient MLflow error aborts without deleting blobs")


def test_cli_parsers():
    """Verify train, prepare-sft-data, sft, and cleanup argument parsers remain clean and compatible."""
    train_args = build_train_parser().parse_args(["--preset", "small", "--stop-after", "5"])
    assert train_args.preset == "small" and train_args.stop_after == 5

    prep_args = build_prepare_sft_data_parser().parse_args([
        "--ops", "add", "sub", "mul", "div", "mod", "--max-digits", "4", "--val-fraction", "0.1", "--nonnegative-only", "--no-drive",
    ])
    assert prep_args.max_digits == 4 and prep_args.val_fraction == 0.1 and prep_args.nonnegative_only and prep_args.no_drive

    sft_args = build_sft_parser().parse_args([
        "--preset", "sft_pilot", "--max-digits", "4", "--nonnegative-only", "--context-length", "2048", "--no-retention",
    ])
    assert sft_args.preset == "sft_pilot" and sft_args.max_digits == 4 and sft_args.context_length == 2048 and sft_args.no_retention

    cleanup_args = build_cleanup_parser().parse_args(["--delete", "--yes", "--include-missing"])
    assert cleanup_args.delete and cleanup_args.yes and cleanup_args.include_missing
    print("  [PASSED] CLI train, prepare-sft-data, sft & cleanup argument parsers")


class _FakeRunHandle:
    def __init__(self, run_id: str, *, fail_artifact: bool = False):
        self.id = run_id
        self.tags: dict[str, str] = {}
        self.metrics: list[tuple[int, dict]] = []
        self.artifacts: dict[str, bytes] = {}
        self.fail_artifact = fail_artifact

    def log(self, metrics: dict, *, step: int) -> None:
        self.metrics.append((step, dict(metrics)))

    def flush(self) -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_artifact(self, path, artifact_path: str = "logs") -> None:
        if self.fail_artifact:
            raise RuntimeError("Artifact upload disabled")
        p = Path(path)
        self.artifacts[f"{artifact_path}/{p.name}"] = p.read_bytes()


def test_run_log_azure_mlflow_and_cleanup():
    """Verify logs save to Azure & MLflow, support overwrite, never break checkpoints, and clean up."""
    container = _FakeContainer({})
    mlflow_client = _FakeMlflowClient({"run_log_del": "deleted"})

    with (
        patch.object(azure_blob, "container_client", return_value=container),
        patch.object(checkpoint_store, "container_client", return_value=container),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        store = checkpoint_store.CheckpointStore("minireasoner", "run_log_del")
        log_file = Path(tmpdir) / "train_20261125_110000_small.log"
        log_file.write_text("Step 0 | validation loss 7.0000\n", encoding="utf-8")

        run_ok = _FakeRunHandle("run_log_del")
        res1 = tracking.publish_run_log(store, log_file, run_ok)
        assert res1["mlflow_saved"] is True
        blob_key = "checkpoints/Shared/MiniReasoner/run_log_del/logs/train_20261125_110000_small.log"
        assert blob_key in container.data
        assert "logs/train_20261125_110000_small.log" in run_ok.artifacts
        assert run_ok.tags["log_location"].endswith("/logs/train_20261125_110000_small.log")
        assert store._versions() == []

        # Updating the log later in the same run overwrites cleanly
        log_file.write_text("Step 0 | validation loss 7.0000\nStep 1 | train loss 6.5000\n", encoding="utf-8")
        tracking.publish_run_log(store, log_file, run_ok)
        assert b"Step 1 | train loss 6.5000" in container.data[blob_key]
        assert b"Step 1 | train loss 6.5000" in run_ok.artifacts["logs/train_20261125_110000_small.log"]

        # MLflow artifact failure does not block Azure log upload
        run_no_mlflow = _FakeRunHandle("run_log_del", fail_artifact=True)
        res2 = tracking.publish_run_log(store, log_file, run_no_mlflow)
        assert res2["mlflow_saved"] is False
        assert blob_key in container.data

        # Deleted run cleanup removes both checkpoint and log blobs
        del_res = tracking.sync_deleted_runs(delete=True, yes=True, client=mlflow_client)
        assert del_res["deleted_blobs"] == 1
        assert container.blobs == {}
    print("  [PASSED] Azure + MLflow run log upload, overwrite, and cleanup sync")


def test_single_local_log_and_remote_upload_in_run_training():
    """Verify run_training creates strictly one local log file (no aliases) and uploads it to Azure & MLflow."""
    container = _FakeContainer({})
    run_handle = _FakeRunHandle("run_single_log")

    @contextmanager
    def fake_track_run(*args, **kwargs):
        yield run_handle

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tok_dir = tmp / "tokenized"
        tok_dir.mkdir()
        logs_dir = tmp / "logs"

        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        vocab = {bytes([i]).decode("latin-1"): i for i in range(256)}
        tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.save(str(tok_dir / "tokenizer.json"))

        tokens = np.arange(512, dtype=np.uint16) % 256
        tokens.tofile(tok_dir / "train.bin")
        tokens.tofile(tok_dir / "val.bin")
        manifest = {
            "version": 1,
            "vocab_size": 256,
            "tokenizer_sha256": checkpoint_store.sha256(tok_dir / "tokenizer.json"),
            "train": {"tokens": len(tokens), "sha256": checkpoint_store.sha256(tok_dir / "train.bin")},
            "val": {"tokens": len(tokens), "sha256": checkpoint_store.sha256(tok_dir / "val.bin")},
        }

        small_preset = {
            "model": dict(PRESETS["small"]["model"]),
            "training": {**PRESETS["small"]["training"], "batch_size": 2, "steps": 2, "warmup_steps": 1, "eval_every": 2},
        }

        with (
            patch.dict("os.environ", {"DATABRICKS_HOST": "https://example.cloud.databricks.com", "DATABRICKS_TOKEN": "tok"}),
            patch.object(azure_blob, "container_client", return_value=container),
            patch.object(checkpoint_store, "container_client", return_value=container),
            patch("src.training.pretrain.TOKENIZED_DATA_DIR", tok_dir),
            patch("src.settings.TOKENIZED_DATA_DIR", tok_dir),
            patch("src.settings.VOCAB_SIZE", 256),
            patch("src.training.pretrain.PRESETS", {"small": small_preset}),
            patch("src.training.pretrain.ensure_tokenized_corpus", return_value=manifest),
            patch("src.training.common.track_run", fake_track_run),
        ):
            result = pretrain.run_training(
                preset="small",
                device=torch.device("cpu"),
                logs_dir=logs_dir,
            )

        log_files = list(logs_dir.iterdir())
        assert len(log_files) == 1, f"Expected strictly 1 local log file, got: {[p.name for p in log_files]}"
        log_path = log_files[0]
        assert not (logs_dir / "latest.log").exists()
        assert not (logs_dir / f"train_{run_handle.id}.log").exists()
        assert result["log_path"] == str(log_path)

        azure_log_key = f"checkpoints/Shared/MiniReasoner/{run_handle.id}/logs/{log_path.name}"
        mlflow_log_key = f"logs/{log_path.name}"
        assert azure_log_key in container.data
        assert mlflow_log_key in run_handle.artifacts
        assert b"Training Completed" in container.data[azure_log_key]
        assert b"Training Completed" in run_handle.artifacts[mlflow_log_key]
        assert run_handle.tags["log_location"].endswith(f"/logs/{log_path.name}")
    print("  [PASSED] Single local log file per run and remote upload in run_training")


if __name__ == "__main__":
    print("Running Azure cleanup sync test suite...")
    test_sync_deleted_runs_dry_run_and_delete()
    test_transient_mlflow_error_aborts_cleanly()
    test_cli_parsers()
    test_run_log_azure_mlflow_and_cleanup()
    test_single_local_log_and_remote_upload_in_run_training()
    print("All tests in test_azure_cleanup.py passed successfully!")
