from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

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
    """Stage one immutable snapshot and upload it while the next batches train."""
    with (
        TemporaryDirectory(prefix="minireasoner-checkpoint-") as directory,
        ThreadPoolExecutor(max_workers=1) as worker,
    ):
        path = Path(directory) / "checkpoint.pt"
        pending = None

        def save(checkpoint: dict) -> None:
            nonlocal pending
            if pending is not None:
                pending.result()  # Finish reading the previous snapshot before reusing its path.
            torch.save(checkpoint, path)  # state_dict tensors still reference the live model.
            metrics = checkpoint["metrics"][-1].copy()
            pending = worker.submit(publish_checkpoint, store, path, metrics, run)

        try:
            yield save
        finally:
            if pending is not None:
                pending.result()


@contextmanager
def download_checkpoint(store: CheckpointStore, alias: str = "latest"):
    with TemporaryDirectory(prefix="minireasoner-checkpoint-") as directory:
        path = Path(directory) / "checkpoint.pt"
        store.download(alias, path)
        yield path
