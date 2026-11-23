"""Save, restore, and verify files under an Azure Blob Storage prefix."""

import hashlib
import os
from pathlib import Path

from azure.core import MatchConditions
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import ContainerClient

from .tokenizer import sha256


def container_client(container: str) -> ContainerClient:
    connection = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not connection:
        raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING in the project .env.")
    try:
        return ContainerClient.from_connection_string(connection, container_name=container)
    except ValueError:
        raise ValueError("Check the Azure storage connection string and container name.") from None


class AzureBlobFiles:
    def __init__(self, container: str, prefix: str):
        self.container = container_client(container)
        prefix = prefix.strip("/")
        self.prefix = f"{prefix}/" if prefix else ""

    def prepare(self) -> None:
        try:
            # Listing works with a container-scoped SAS, including on an empty run folder.
            blobs = self.container.list_blob_names(name_starts_with=self.prefix, results_per_page=1)
            next(iter(blobs), None)
        except ResourceNotFoundError as error:
            raise FileNotFoundError(f"Azure container {self.container.container_name!r} does not exist.") from error

    def names(self) -> list[str]:
        blobs = self.container.list_blob_names(name_starts_with=self.prefix)
        names = (name.removeprefix(self.prefix) for name in blobs)
        return [name for name in names if name and "/" not in name]

    def put(self, path: Path, name: str, *, digest: str | None = None) -> None:
        digest = digest or sha256(path)
        blob = self.container.get_blob_client(self.prefix + name)
        try:
            with path.open("rb") as stream:
                blob.upload_blob(stream, overwrite=False, metadata={"sha256": digest})
        except ResourceExistsError:
            # Retrying an upload whose commit succeeded is safe for identical data.
            saved = blob.get_blob_properties()
            if saved.size != path.stat().st_size:
                raise FileExistsError(f"An Azure blob with a different size exists at {blob.blob_name}.")
            stored_digest = (saved.metadata or {}).get("sha256")
            if stored_digest is None:
                checksum = hashlib.sha256()
                for chunk in blob.download_blob(etag=saved.etag, match_condition=MatchConditions.IfNotModified).chunks():
                    checksum.update(chunk)
                stored_digest = checksum.hexdigest()
            if stored_digest != digest:
                raise FileExistsError(f"An Azure blob with different contents exists at {blob.blob_name}.")

    def get(self, name: str, path: Path) -> None:
        download = self.container.download_blob(self.prefix + name)
        with path.open("wb") as stream:
            download.readinto(stream)

    def delete(self, name: str) -> None:
        try:
            self.container.delete_blob(self.prefix + name)
        except ResourceNotFoundError:
            pass
