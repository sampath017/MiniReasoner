from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch

from .checkpoint_store import CheckpointStore


def publish_checkpoint(store: CheckpointStore, path: Path, metrics: dict, run) -> None:
    step = metrics["step"]
    store.save(
        path, step=step, val_loss=metrics["val_loss"],
        train_loss_interval_mean=metrics["train_loss_interval_mean"],
    )
    run.set_tag("checkpoint_step", str(step))
    print(f"Saved Drive checkpoint at step {step:,}.", flush=True)


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
            print(f"Queued Drive checkpoint at step {metrics['step']:,}.", flush=True)

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
