"""Keep complete training checkpoints on mounted Drive or an authenticated rclone remote."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig
from tempfile import TemporaryDirectory

from .settings import CHECKPOINT_LIMIT
from .tokenizer import sha256


_LOSS = r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"
_FILENAME = re.compile(rf"step-(\d+)_val-({_LOSS})_train-({_LOSS})_([a-f0-9]{{64}})\.pt")


@dataclass(frozen=True)
class _Snapshot:
    name: str
    step: int
    val_loss: float
    train_loss_interval_mean: float
    sha256: str

    def rank(self) -> tuple:
        return self.val_loss, self.train_loss_interval_mean, -self.step, self.name


def _rclone(*args: str, allow_missing: bool = False) -> str:
    # A Jupyter kernel can use the project Python without its scripts directory on PATH.
    executable = shutil.which("rclone", path=sysconfig.get_path("scripts")) or shutil.which("rclone")
    if executable is None:
        raise FileNotFoundError(
            "The notebook kernel cannot find rclone. On a Linux GPU VM, run "
            "'uv sync --locked' from the project directory and select its .venv kernel. "
            "On other platforms, install the rclone CLI or use a writable Drive mount."
        )
    result = subprocess.run([executable, *args], capture_output=True, text=True, encoding="utf-8")
    if allow_missing and result.returncode in (3, 4):  # Directory or file absent.
        return ""
    if result.returncode:
        raise RuntimeError(f"rclone {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def _remote_name(root: str | Path) -> str | None:
    base = str(root).strip()
    if not base or "://" in base:
        raise ValueError("Use a writable mounted Drive directory or an authenticated rclone remote path.")
    if isinstance(root, str) and not re.match(r"^[A-Za-z]:[\\/]", base):
        match = re.match(r"^([A-Za-z0-9][\w .-]*):", base)
        if match:
            return match[1]
    return None


def check_checkpoint_root(root: str | Path) -> None:
    """Check the VM's named remote before preparing data or opening an MLflow run."""
    remote = _remote_name(root)
    if remote and f"{remote}:" not in _rclone("listremotes").splitlines():
        raise ValueError(
            f"The rclone remote '{remote}' is missing. In the project terminal run "
            f"'uv run rclone config', create a writable Google Drive remote named '{remote}', "
            "then rerun this settings cell."
        )


class CheckpointStore:
    def __init__(self, root: str | Path, run_id: str, *, keep: int | None = CHECKPOINT_LIMIT):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("A checkpoint store needs a valid run ID.")
        if keep is not None and (type(keep) is not int or keep < 1):
            raise ValueError("Set the checkpoint limit to None or a positive integer.")
        base = str(root).strip()
        self.keep = keep
        self._remote = _remote_name(root) is not None
        if self._remote:
            base = base.rstrip("/")
            self.location = base + ("" if base.endswith(":") else "/") + run_id
        else:
            self.location = str(Path(base).expanduser().resolve() / run_id)

    def prepare(self) -> None:
        """Create the run folder before spending GPU time on a new training run."""
        if self._remote:
            _rclone("mkdir", self.location)
        else:
            Path(self.location).mkdir(parents=True, exist_ok=True)

    def _versions(self) -> list[_Snapshot]:
        if self._remote:
            names = _rclone("lsf", "--files-only", self.location, allow_missing=True).splitlines()
        else:
            directory = Path(self.location)
            if not directory.exists():
                return []
            names = [p.name for p in directory.iterdir() if p.is_file() and not p.is_symlink()]

        versions = []
        for name in names:
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
        if self._remote:
            # Drive commits a new binary object only after its upload completes.
            _rclone("copyto", "--immutable", "--checksum", str(path.resolve()), f"{self.location}/{name}")
        else:
            directory = Path(self.location)
            directory.mkdir(parents=True, exist_ok=True)
            final = directory / name
            if not final.is_file() or sha256(final) != digest:
                with TemporaryDirectory(prefix=".upload-", dir=directory) as tmp:
                    staged = Path(tmp) / "checkpoint.pt"
                    with path.open("rb") as source, staged.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    if sha256(staged) != digest:
                        raise OSError(f"Drive copy failed its checksum at step {step}.")
                    staged.replace(final)
        self.prune()
        return name

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
            if self._remote:
                _rclone("deletefile", f"{self.location}/{version.name}", allow_missing=True)
            else:
                (Path(self.location) / version.name).unlink(missing_ok=True)

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
            if self._remote:
                _rclone("copyto", f"{self.location}/{selected.name}", str(staged.resolve()))
            else:
                shutil.copyfile(Path(self.location) / selected.name, staged)
            if sha256(staged) != selected.sha256:
                raise ValueError(f"Checksum mismatch for Drive checkpoint at step {selected.step}.")
            staged.replace(destination)
        return asdict(selected)
