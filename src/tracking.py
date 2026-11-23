from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import torch

from .checkpoint_store import CheckpointStore, delete_azure_blobs, list_azure_run_blobs
from .metrics import databricks_host
from .settings import AZURE_CONTAINER, MLFLOW_EXPERIMENT


def publish_checkpoint(store: CheckpointStore, path: Path, metrics: dict, run) -> None:
    step = metrics["step"]
    store.save(
        path, step=step, val_loss=metrics["val_loss"],
        train_loss_interval_mean=metrics["train_loss_interval_mean"],
    )
    run.set_tag("checkpoint_step", str(step))
    print(f"Saved Azure checkpoint at step {step:,}.", flush=True)


@contextmanager
def checkpoint_uploads(store: CheckpointStore, run):
    """Queue independent snapshots for ordered upload while training continues."""
    def upload(path: Path, metrics: dict) -> None:
        try:
            publish_checkpoint(store, path, metrics, run)
        except Exception as error:
            raise RuntimeError(
                f"Could not publish checkpoint at step {metrics['step']:,}. "
                f"The complete snapshot is available at {path}."
            ) from error
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
            try:
                with snapshot:
                    # The model and optimizer tensors still reference live weights.
                    torch.save(checkpoint, snapshot)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            pending.append(worker.submit(upload, path, metrics))
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
