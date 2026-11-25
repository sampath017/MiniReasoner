"""Save and restore complete training checkpoints from a run folder in Azure Blob Storage."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlunsplit

from azure.core.exceptions import ResourceNotFoundError

from .azure_blob import AzureBlobFiles, container_client
from .settings import AZURE_CONTAINER, CHECKPOINT_LIMIT, MLFLOW_EXPERIMENT
from .tokenizer import sha256


_LOSS = r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"
_FILENAME = re.compile(rf"step-(\d+)_val-({_LOSS})_train-({_LOSS})_([a-f0-9]{{64}})\.pt")
_RUN_ID = re.compile(r"[A-Za-z0-9_-]+")


def experiment_checkpoint_prefix(experiment: str = MLFLOW_EXPERIMENT) -> str:
    """Return the normalized Azure blob prefix holding per-run checkpoint folders."""
    folder = experiment.strip("/")
    if not folder or any(part in ("", ".", "..") for part in folder.split("/")):
        raise ValueError("A checkpoint store needs a valid experiment path.")
    return f"checkpoints/{folder}/"


def list_azure_run_blobs(
    container: str = AZURE_CONTAINER,
    *,
    experiment: str = MLFLOW_EXPERIMENT,
) -> dict[str, list[dict]]:
    """Group blobs in Azure under `checkpoints/<experiment>/<run_id>/` by run ID."""
    prefix = experiment_checkpoint_prefix(experiment)
    run_pattern = re.compile(rf"^{re.escape(prefix)}([A-Za-z0-9_-]+)/(.+)$")
    client = container_client(container)
    runs: dict[str, list[dict]] = {}
    for blob in client.list_blobs(name_starts_with=prefix):
        match = run_pattern.match(blob.name)
        if not match:
            continue
        run_id, filename = match.groups()
        if not _RUN_ID.fullmatch(run_id):
            continue
        runs.setdefault(run_id, []).append({
            "name": blob.name,
            "filename": filename,
            "size": int(blob.size or 0),
            "last_modified": blob.last_modified,
        })
    return runs


def delete_azure_blobs(container: str, blob_names: list[str]) -> int:
    """Delete the named blobs from `container`, ignoring already-missing blobs."""
    client = container_client(container)
    deleted = 0
    for name in blob_names:
        try:
            client.delete_blob(name)
            deleted += 1
        except ResourceNotFoundError:
            pass
    return deleted


@dataclass(frozen=True)
class _Snapshot:
    name: str
    step: int
    val_loss: float
    train_loss_interval_mean: float
    sha256: str

    def rank(self) -> tuple:
        return self.val_loss, self.train_loss_interval_mean, -self.step, self.name


class CheckpointStore:
    def __init__(self, container: str, run_id: str, *, experiment: str = MLFLOW_EXPERIMENT,
                 keep: int | None = CHECKPOINT_LIMIT):
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("A checkpoint store needs a valid run ID.")
        if keep is not None and (type(keep) is not int or keep < 1):
            raise ValueError("Set the checkpoint limit to None or a positive integer.")
        prefix = f"{experiment_checkpoint_prefix(experiment)}{run_id}"
        self.keep = keep
        self.files = AzureBlobFiles(container, prefix)
        # The SDK endpoint can include a SAS query string. Only publish the resource path.
        endpoint = urlsplit(self.files.container.url)
        self.location = urlunsplit((endpoint.scheme, endpoint.netloc, f"{endpoint.path.rstrip('/')}/{prefix}", "", ""))

    def prepare(self) -> None:
        self.files.prepare()

    def _versions(self) -> list[_Snapshot]:
        versions = []
        for name in self.files.names():
            match = _FILENAME.fullmatch(name)
            if match:
                step, val, train, digest = match.groups()
                versions.append(_Snapshot(name, int(step), float(val), float(train), digest))
        return sorted(versions, key=_Snapshot.rank)

    def save(self, path: Path, *, step: int, val_loss: float, train_loss_interval_mean: float) -> str:
        val, train = float(val_loss), float(train_loss_interval_mean)
        if type(step) is not int or step < 0 or not all(math.isfinite(x) and x >= 0 for x in (val, train)):
            raise ValueError("Checkpoint step and losses must be finite and nonnegative.")
        path = Path(path)
        digest = sha256(path)
        name = f"step-{step}_val-{abs(val)!r}_train-{abs(train)!r}_{digest}.pt"
        self.files.put(path, name, digest=digest)
        self.prune()
        return name

    def save_log(self, path: Path, *, name: str | None = None) -> str:
        """Upload or update a training log file under `<run_id>/logs/<filename>` in Azure."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Training log file not found: {path}")
        filename = Path(name or path.name).name
        if not filename or filename in (".", ".."):
            raise ValueError("A run log needs a valid filename.")
        blob_name = f"logs/{filename}"
        self.files.put(path, blob_name, overwrite=True)
        return blob_name

    def prune(self) -> None:
        if self.keep is None:
            return
        versions = self._versions()
        if not versions:
            return
        latest = max(versions, key=lambda item: item.step)
        kept_steps = {latest.step}
        for version in versions:
            if version == latest:
                continue
            if version.step not in kept_steps and len(kept_steps) < self.keep:
                kept_steps.add(version.step)
                continue
            self.files.delete(version.name)

    def download(self, alias: str, destination: Path) -> dict:
        versions = self._versions()
        if alias == "latest":
            selected = max(versions, key=lambda item: item.step, default=None)
        elif alias == "best":
            selected = next(iter(versions), None)
        elif match := re.fullmatch(r"step-(\d+)", alias):
            selected = next((item for item in versions if item.step == int(match[1])), None)
        else:
            raise ValueError("Choose 'latest', 'best', or 'step-N' for the checkpoint alias.")
        if selected is None:
            raise FileNotFoundError(f"No saved checkpoint for {alias!r} in {self.location}.")

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as tmp:
            staged = Path(tmp) / "checkpoint.pt"
            self.files.get(selected.name, staged)
            if sha256(staged) != selected.sha256:
                raise ValueError(f"Checksum mismatch for Azure checkpoint at step {selected.step}.")
            staged.replace(destination)
        return asdict(selected)
