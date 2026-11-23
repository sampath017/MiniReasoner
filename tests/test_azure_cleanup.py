"""Unit tests for Azure Blob Storage <-> Databricks MLflow cleanup sync."""

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mlflow.exceptions import MlflowException, RestException

from src import checkpoint_store, tracking
from src.main import build_cleanup_parser, build_train_parser


class _FakeBlob:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.last_modified = datetime(2026, 11, 24, 18, 0, tzinfo=timezone.utc)


class _FakeContainer:
    def __init__(self, blobs: dict[str, int]):
        self.blobs = dict(blobs)

    def list_blobs(self, *, name_starts_with: str = ""):
        return [
            _FakeBlob(name, size)
            for name, size in sorted(self.blobs.items())
            if name.startswith(name_starts_with)
        ]

    def delete_blob(self, name: str):
        if name not in self.blobs:
            from azure.core.exceptions import ResourceNotFoundError
            raise ResourceNotFoundError(name)
        del self.blobs[name]


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
    """Verify train and cleanup argument parsers remain clean and compatible."""
    train_args = build_train_parser().parse_args(["--preset", "small", "--stop-after", "5"])
    assert train_args.preset == "small" and train_args.stop_after == 5

    cleanup_args = build_cleanup_parser().parse_args(["--delete", "--yes", "--include-missing"])
    assert cleanup_args.delete and cleanup_args.yes and cleanup_args.include_missing
    print("  [PASSED] CLI train & cleanup argument parsers")


if __name__ == "__main__":
    print("Running Azure cleanup sync test suite...")
    test_sync_deleted_runs_dry_run_and_delete()
    test_transient_mlflow_error_aborts_cleanly()
    test_cli_parsers()
    print("All tests in test_azure_cleanup.py passed successfully!")
