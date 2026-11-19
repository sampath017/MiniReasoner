"""Back up the live MLflow database to Drive and restore it on a new runtime."""

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from time import monotonic

from .checkpoint_store import _rclone, _remote_name


def _md5(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "md5").hexdigest()


def _snapshot(database: Path, destination: Path) -> None:
    with (
        closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(destination)) as target,
    ):
        source.backup(target, pages=256)
        target.execute("PRAGMA journal_mode=DELETE")


class DriveTracking:
    """Manage the Drive copy of a project's SQLite history with one active writer."""

    def __init__(self, database: Path, checkpoint_root: str | Path):
        self.database = database.resolve()
        self._remote = _remote_name(checkpoint_root) is not None
        if self._remote:
            remote, path = str(checkpoint_root).strip().split(":", 1)
            base = Path(path.rstrip("/"))
            folder = (base.parent if base.name == "checkpoints" else base) / "tracking"
            self._folder = f"{remote}:{folder.as_posix()}"
        else:
            base = Path(checkpoint_root).expanduser().resolve()
            self._folder = (base.parent if base.name == "checkpoints" else base) / "tracking"
        self.location = f"{self._folder}/mlflow.db"
        self._marker = self.database.with_suffix(".drive.json")
        self._revision: str | None = None
        self._prepared = False

    def _drive_revision(self) -> str | None:
        if self._remote:
            hashes = _rclone("lsf", "--files-only", "--format", "h", "--hash", "MD5",
                             "--include", "/mlflow.db", self._folder, allow_missing=True).splitlines()
            if not hashes:
                return None
            if len(hashes) != 1 or len(hashes[0]) != 32:
                raise RuntimeError(f"Could not identify a unique checksum for {self.location}.")
            return hashes[0].lower()
        path = Path(self.location)
        return _md5(path) if path.is_file() else None

    def _marker_data(self, revision: str) -> dict:
        identity = self.database.stat()
        return {"location": self.location, "md5": revision, "database_id": [identity.st_dev, identity.st_ino]}

    def _remember(self, revision: str) -> None:
        with TemporaryDirectory(dir=self.database.parent) as directory:
            staged = Path(directory) / "revision.json"
            staged.write_text(json.dumps(self._marker_data(revision)), encoding="utf-8")
            staged.replace(self._marker)
        self._revision = revision

    def prepare(self) -> None:
        """Restore before opening MLflow, preserving an existing independent local history."""
        self.database.parent.mkdir(parents=True, exist_ok=True)
        remote = self._drive_revision()
        if remote is not None:
            with TemporaryDirectory(dir=self.database.parent) as directory:
                image = Path(directory) / "mlflow.db"
                if self.database.exists() and self.database.stat().st_size:
                    marker = json.loads(self._marker.read_text(encoding="utf-8")) if self._marker.exists() else {}
                    if marker != self._marker_data(remote):
                        _snapshot(self.database, image)
                        if _md5(image) != remote:
                            raise FileExistsError(
                                f"The MLflow histories in {self.database} and {self.location} differ. "
                                "Preserve the local runs folder and use a fresh workspace to restore Drive."
                            )
                else:
                    if any(Path(str(self.database) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
                        raise FileExistsError(f"Found SQLite recovery files beside {self.database}.")
                    if self._remote:
                        _rclone("copyto", self.location, str(image))
                    else:
                        shutil.copyfile(self.location, image)
                    if _md5(image) != remote:
                        raise ValueError(f"The Drive tracking database checksum changed: {self.location}")
                    with closing(sqlite3.connect(image.as_uri() + "?mode=ro", uri=True)) as db:
                        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                            raise ValueError("The tracking database on Drive failed SQLite's integrity check.")
                    image.replace(self.database)
                    print(f"Restored MLflow history from {self.location}.", flush=True)
            self._remember(remote)
        self._revision = remote
        self._prepared = True

    def save(self) -> None:
        """Publish a consistent image of all committed runs, including SQLite WAL changes."""
        if not self._prepared:
            raise RuntimeError("Prepare the Drive tracking database before saving.")
        if not self.database.is_file():
            return
        with TemporaryDirectory(dir=self.database.parent) as directory:
            image = Path(directory) / "mlflow.db"
            _snapshot(self.database, image)
            digest = _md5(image)
            current = self._drive_revision()
            if current == digest:
                self._remember(digest)
                return
            if current is not None and current != self._revision:
                raise RuntimeError(f"Another writer changed {self.location}. Local MLflow history is at {self.database}.")
            if self._remote:
                # A new Drive file revision is published when the upload completes.
                _rclone("copyto", "--checksum", str(image), self.location)
            else:
                folder = Path(self._folder)
                folder.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=folder) as staging:
                    staged = Path(staging) / "mlflow.db"
                    shutil.copyfile(image, staged)
                    if _md5(staged) != digest:
                        raise OSError(f"The copied tracking database failed verification at {folder}.")
                    staged.replace(self.location)
            self._remember(digest)

    @contextmanager
    def background(self):
        """Coalesce periodic uploads, then publish terminal status after MLflow closes."""
        with ThreadPoolExecutor(max_workers=1) as worker:
            pending = []
            next_backup = 0.0

            def request(*, validation: bool = False) -> None:
                nonlocal next_backup
                while pending and pending[0].done():
                    try:
                        pending.pop(0).result()
                    except Exception as error:
                        print(f"MLflow Drive backup will be retried: {error}", flush=True)
                now = monotonic()
                # One in flight and one waiting snapshot can cover any number of new steps.
                if len(pending) < 2 and (validation or now >= next_backup):
                    pending.append(worker.submit(self.save))
                    next_backup = now + 60

            training_error = None
            try:
                yield request
            except BaseException as error:
                training_error = error
                raise
            finally:
                if pending:
                    # The final save below also retries any unsuccessful background copy.
                    wait(pending)
                try:
                    self.save()
                except Exception as error:
                    if training_error is None:
                        raise
                    training_error.add_note(f"Final MLflow Drive backup also failed: {error}")
