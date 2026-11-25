from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp
import threading

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import torch

from .azure_blob import container_client
from .checkpoint_store import _FILENAME, CheckpointStore, delete_azure_blobs, experiment_checkpoint_prefix, list_azure_run_blobs
from .metrics import databricks_host
from ..settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT


class TeeLogger:
    """Thread-safe stream wrapper mirroring writes identically to console and one log file."""

    def __init__(self, stream, file, lock=None):
        self.stream = stream
        self.file = file
        self.lock = lock if lock is not None else threading.RLock()

    def write(self, data):
        with self.lock:
            self.stream.write(data)
            self.file.write(data)
            self.file.flush()

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


@contextmanager
def tee_to_log(log_path: Path | None, lock=None):
    """Mirror stdout and stderr identically into `log_path` unless already teeing."""
    if log_path is None or isinstance(sys.stdout, TeeLogger):
        yield getattr(sys.stdout, "lock", lock)
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_lock = lock if lock is not None else threading.RLock()
    log_file = path.open("a", encoding="utf-8")
    sys.stdout = TeeLogger(orig_stdout, log_file, lock=log_lock)
    sys.stderr = TeeLogger(orig_stderr, log_file, lock=log_lock)
    try:
        yield log_lock
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.close()


def _copy_log_snapshot(log_path: Path, directory: Path, *, lock=None) -> Path:
    """Copy `log_path` into `directory` preserving filename for race-free upload."""
    guard = lock if lock is not None else nullcontext()
    snapshot = directory / log_path.name
    with guard:
        sys.stdout.flush()
        sys.stderr.flush()
        shutil.copyfile(log_path, snapshot)
    return snapshot


def _upload_log_snapshot(store: CheckpointStore, snapshot_path: Path, run) -> dict:
    """Upload a snapshotted log file to Azure Blob Storage and best-effort to MLflow."""
    blob_name = store.save_log(snapshot_path, name=snapshot_path.name)
    log_uri = f"{store.location}/{blob_name}"
    run.set_tag("log_location", log_uri)
    mlflow_saved = False
    if hasattr(run, "log_artifact"):
        try:
            run.log_artifact(snapshot_path, artifact_path="logs")
            mlflow_saved = True
        except Exception as error:
            print(f"Warning: could not upload log artifact to MLflow ({error}).", flush=True)
    return {"azure_uri": log_uri, "mlflow_saved": mlflow_saved}


def publish_run_log(store: CheckpointStore, log_path: Path, run, *, lock=None) -> dict:
    """Snapshot `log_path` safely and upload it to Azure Blob Storage and MLflow."""
    with TemporaryDirectory(prefix="minireasoner-log-") as directory:
        snapshot = _copy_log_snapshot(log_path, Path(directory), lock=lock)
        return _upload_log_snapshot(store, snapshot, run)


def publish_checkpoint(
    store: CheckpointStore, path: Path, metrics: dict, run, *, log_snapshot: Path | None = None,
) -> None:
    step = metrics["step"]
    store.save(
        path, step=step, val_loss=metrics["val_loss"],
        train_loss_interval_mean=metrics["train_loss_interval_mean"],
    )
    run.set_tag("checkpoint_step", str(step))
    if log_snapshot is not None:
        _upload_log_snapshot(store, log_snapshot, run)
    print(f"Saved Azure checkpoint at step {step:,}.", flush=True)


@contextmanager
def checkpoint_uploads(store: CheckpointStore, run, *, log_path: Path | None = None, log_lock=None):
    """Queue independent snapshots (and run logs) for ordered upload while training continues."""
    def upload(path: Path, metrics: dict, log_dir: Path | None, log_snapshot: Path | None) -> None:
        try:
            publish_checkpoint(store, path, metrics, run, log_snapshot=log_snapshot)
        except Exception as error:
            raise RuntimeError(
                f"Could not publish checkpoint at step {metrics['step']:,}. "
                f"The complete snapshot is available at {path}."
            ) from error
        finally:
            if log_dir is not None:
                shutil.rmtree(log_dir, ignore_errors=True)
        path.unlink()

    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = deque()

        def save(checkpoint: dict) -> None:
            while pending and pending[0].done():
                pending.popleft().result()
            metrics = checkpoint["metrics"][-1].copy()
            snapshot = NamedTemporaryFile(
                prefix=f"minireasoner-step-{metrics['step']}-", suffix=".pt", delete=False,
            )
            path = Path(snapshot.name)
            log_dir: Path | None = None
            log_snapshot: Path | None = None
            try:
                with snapshot:
                    # The model and optimizer tensors still reference live weights.
                    torch.save(checkpoint, snapshot)
                if log_path is not None and log_path.is_file():
                    log_dir = Path(mkdtemp(prefix="minireasoner-log-"))
                    log_snapshot = _copy_log_snapshot(log_path, log_dir, lock=log_lock)
            except BaseException:
                path.unlink(missing_ok=True)
                if log_dir is not None:
                    shutil.rmtree(log_dir, ignore_errors=True)
                raise
            pending.append(worker.submit(upload, path, metrics, log_dir, log_snapshot))
            print(f"Queued Azure checkpoint at step {metrics['step']:,}.", flush=True)

        try:
            yield save
        finally:
            for future in pending:
                future.result()


@contextmanager
def download_checkpoint(store: CheckpointStore, alias: str = "latest"):
    with TemporaryDirectory(prefix="minireasoner-checkpoint-") as directory:
        path = Path(directory) / "checkpoint.pt"
        store.download(alias, path)
        yield path


def discover_azure_runs(
    container: str = AZURE_CONTAINER,
    *,
    experiment: str = MLFLOW_EXPERIMENT,
    preferred_stages: tuple[str, ...] | None = None,
) -> list[dict]:
    """List checkpoint runs in Azure prioritized by active MLflow state, optional stage, and recency."""
    exp_prefix = experiment_checkpoint_prefix(experiment)
    run_pattern = re.compile(rf"^{re.escape(exp_prefix)}([A-Za-z0-9_-]+)/([^/]+)$")
    runs_info: dict[str, dict] = {}
    client = container_client(container)
    for blob in client.list_blobs(name_starts_with=exp_prefix):
        match = run_pattern.match(blob.name)
        if not match:
            continue
        rid, fname = match.groups()
        if not _FILENAME.fullmatch(fname):
            continue
        entry = runs_info.setdefault(rid, {"count": 0, "last_modified": blob.last_modified})
        entry["count"] += 1
        if blob.last_modified and (entry["last_modified"] is None or blob.last_modified > entry["last_modified"]):
            entry["last_modified"] = blob.last_modified

    mlflow_states: dict[str, str] = {}
    mlflow_stages: dict[str, str] = {}
    if os.getenv("DATABRICKS_HOST", "").strip() and os.getenv("DATABRICKS_TOKEN", "").strip():
        try:
            mlflow.set_tracking_uri("databricks")
            ml_client = MlflowClient(tracking_uri="databricks")
            for rid in runs_info:
                try:
                    run = ml_client.get_run(rid)
                    mlflow_states[rid] = str(run.info.lifecycle_stage).lower()
                    mlflow_stages[rid] = str(run.data.tags.get("stage", "pretrain")).lower()
                except Exception:
                    mlflow_states[rid] = "unknown"
        except Exception:
            pass

    def sort_key(item):
        rid, info = item
        is_active = 1 if mlflow_states.get(rid, "active") == "active" else 0
        if preferred_stages is not None:
            stage_rank = 1 if mlflow_stages.get(rid, "pretrain") in preferred_stages else 0
        else:
            stage_rank = 1
        ts = info["last_modified"].timestamp() if info["last_modified"] else 0.0
        return (is_active, stage_rank, ts)

    results = []
    for rid, info in sorted(runs_info.items(), key=sort_key, reverse=True):
        results.append({
            "run_id": rid,
            "mlflow_state": mlflow_states.get(rid, "unverified"),
            "mlflow_stage": mlflow_stages.get(rid, "pretrain"),
            "blob_count": info["count"],
            "last_modified": info["last_modified"],
        })
    return results


def inspect_azure_runs(
    container: str = AZURE_CONTAINER,
    *,
    experiment: str = MLFLOW_EXPERIMENT,
    client=None,
) -> list[dict]:
    """Compare run checkpoint folders in Azure against Databricks MLflow lifecycle states."""
    if client is None:
        databricks_host()
        if not os.getenv("DATABRICKS_TOKEN", "").strip():
            raise ValueError("Set DATABRICKS_TOKEN in .env to inspect Databricks MLflow runs.")
        mlflow.set_tracking_uri("databricks")
        client = MlflowClient(tracking_uri="databricks")

    run_blobs = list_azure_run_blobs(container, experiment=experiment)
    records = []
    for run_id, blobs in sorted(run_blobs.items()):
        try:
            run = client.get_run(run_id)
            state = str(run.info.lifecycle_stage).lower()
        except MlflowException as error:
            if getattr(error, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
                state = "missing"
            else:
                raise
        records.append({
            "run_id": run_id,
            "mlflow_state": state,
            "blob_count": len(blobs),
            "total_bytes": sum(item["size"] for item in blobs),
            "blob_names": [item["name"] for item in blobs],
        })
    return records


def sync_deleted_runs(
    *,
    container: str = AZURE_CONTAINER,
    experiment: str = MLFLOW_EXPERIMENT,
    delete: bool = False,
    yes: bool = False,
    include_missing: bool = False,
    client=None,
    confirm_fn=None,
) -> dict:
    """Preview or delete Azure checkpoint blobs for runs deleted in Databricks MLflow."""
    records = inspect_azure_runs(container, experiment=experiment, client=client)
    target_states = {"deleted", "missing"} if include_missing else {"deleted"}

    candidates = [rec for rec in records if rec["mlflow_state"] in target_states]
    print(f"Azure container '{container}' under experiment '{experiment}':")
    if not records:
        print("  No checkpoint run folders found in Azure.")
        return {"runs": records, "candidates": [], "deleted_blobs": 0, "reclaimed_bytes": 0}

    print(f"  {'Run ID':<36} {'MLflow State':<12} {'Blobs':<7} {'Size (MB)':<11} {'Action'}")
    print("  " + "-" * 80)
    for rec in records:
        is_target = rec["mlflow_state"] in target_states
        action = ("DELETE" if delete else "WOULD DELETE") if is_target else "KEEP"
        size_mb = rec["total_bytes"] / (1024 * 1024)
        print(f"  {rec['run_id']:<36} {rec['mlflow_state']:<12} {rec['blob_count']:<7} {size_mb:<11.2f} {action}")

    total_blobs = sum(rec["blob_count"] for rec in candidates)
    total_bytes = sum(rec["total_bytes"] for rec in candidates)
    total_mb = total_bytes / (1024 * 1024)

    if not candidates:
        print("\nNo Azure checkpoint blobs belong to deleted Databricks MLflow runs.")
        return {"runs": records, "candidates": [], "deleted_blobs": 0, "reclaimed_bytes": 0}

    if not delete:
        print(
            f"\nDry run: {len(candidates)} run(s), {total_blobs} blob(s) ({total_mb:.2f} MB) can be removed."
        )
        print("Run `python src/main.py cleanup --delete` to delete them from Azure.")
        return {"runs": records, "candidates": candidates, "deleted_blobs": 0, "reclaimed_bytes": 0}

    if not yes:
        if confirm_fn is None:
            if not sys.stdin.isatty():
                raise RuntimeError("Refusing to delete Azure blobs non-interactively without --yes.")
            confirm_fn = input
        answer = confirm_fn(
            f"\nType 'delete' to permanently remove {total_blobs} blob(s) ({total_mb:.2f} MB) "
            f"from Azure container '{container}': "
        )
        if answer.strip().lower() != "delete":
            print("Aborted. No Azure blobs were deleted.")
            return {"runs": records, "candidates": candidates, "deleted_blobs": 0, "reclaimed_bytes": 0}

    deleted_blobs = 0
    for rec in candidates:
        count = delete_azure_blobs(container, rec["blob_names"])
        deleted_blobs += count
        size_mb = rec["total_bytes"] / (1024 * 1024)
        print(f"Deleted {count} blob(s) ({size_mb:.2f} MB) for deleted run {rec['run_id']}.")

    print(f"\nCleanup complete: removed {deleted_blobs} blob(s) ({total_mb:.2f} MB) from Azure.")
    return {
        "runs": records,
        "candidates": candidates,
        "deleted_blobs": deleted_blobs,
        "reclaimed_bytes": total_bytes,
    }
