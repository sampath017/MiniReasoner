"""Publish and restore verified SimpleStories data on an authenticated Google Drive."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

from ..core.tokenizer import (
    OUTPUT_FILES, prepared_corpus_manifest, publish_prepared_corpus, sha256, verify_corpus_files,
)


def _rclone(*args: str, allow_missing: bool = False) -> str:
    # Jupyter can select the project interpreter without activating its bin directory.
    executable = shutil.which("rclone", path=sysconfig.get_path("scripts")) or shutil.which("rclone")
    if executable is None:
        raise FileNotFoundError("Install rclone on the VM with 'uv sync --locked' and select the project kernel.")
    command = [executable]
    config_hint = Path(sys.prefix) / "minireasoner-rclone-path"
    if not os.environ.get("RCLONE_CONFIG"):
        if config_hint.is_file():
            command.extend(["--config", config_hint.read_text(encoding="utf-8").rstrip("\n")])
        elif os.environ.get("RCLONE_CONFIG_BASE64"):
            import base64
            target_conf = Path.home() / ".config" / "rclone" / "rclone.conf"
            if not target_conf.is_file():
                target_conf.parent.mkdir(parents=True, exist_ok=True)
                target_conf.write_bytes(base64.b64decode(os.environ["RCLONE_CONFIG_BASE64"]))
            command.extend(["--config", str(target_conf)])
    result = subprocess.run([*command, *args], capture_output=True, text=True, encoding="utf-8")
    if result.returncode and "didn't find section in config file" in result.stderr:
        raise RuntimeError(
            "The notebook cannot find its Google Drive connection. "
            "Run 'bash scripts/jarvis_labs.sh' in the VM project terminal, then rerun this cell."
        )
    if allow_missing and result.returncode in (3, 4):
        return ""
    if result.returncode:
        raise RuntimeError(f"Google Drive request failed (rclone {args[0]}): {result.stderr.strip()}")
    return result.stdout


def _remote_state(remote: str) -> tuple[dict | None, dict]:
    listing = _rclone("lsjson", "--files-only", "--hash-type", "sha256", remote, allow_missing=True)
    files = {}
    for item in json.loads(listing or "[]"):
        name = item["Name"]
        if name in files:
            raise ValueError(f"Google Drive has duplicate files named {name!r} in {remote}.")
        files[name] = item
    if "manifest.json" not in files:
        return None, files
    return json.loads(_rclone("cat", f"{remote}/manifest.json")), files


def _publish(remote: str, directory: Path, manifest: dict, digests: dict[str, str]) -> None:
    saved, files = _remote_state(remote)
    if saved is not None and saved != manifest:
        raise FileExistsError(f"A different corpus already exists in {remote}.")
    for name, digest in digests.items():
        existing = files.get(name)
        if existing is not None:
            hashes = {key.lower().replace("-", ""): value for key, value in existing.get("Hashes", {}).items()}
            remote_digest = hashes.get("sha256")
            if existing["Size"] != (directory / name).stat().st_size or (remote_digest and remote_digest != digest):
                raise FileExistsError(f"A different file already exists at {remote}/{name}.")
            if remote_digest:
                continue
        # Immutable checksum copies can finish an interrupted upload and also check older Drive files.
        _rclone("copyto", "--immutable", "--checksum", str(directory / name), f"{remote}/{name}")
    if saved is None:
        # Publish the manifest only once all payloads have reached Drive.
        _rclone("copyto", "--immutable", "--checksum", str(directory / "manifest.json"), f"{remote}/manifest.json")


def publish_tokenized(drive_data_root: str, directory: Path, manifest: dict) -> None:
    if json.loads((directory / "manifest.json").read_bytes()) != manifest:
        raise ValueError(f"The on-disk manifest changed in {directory}.")
    verify_corpus_files(directory, manifest)
    _publish(f"{drive_data_root.rstrip('/')}/tokenized/{directory.name}", directory, manifest, {
        "tokenizer.json": manifest["tokenizer_sha256"],
        "train.bin": manifest["train"]["sha256"],
        "val.bin": manifest["val"]["sha256"],
    })


def reuse_tokenized(drive_data_root: str, directory: Path, settings: dict) -> dict | None:
    manifest = prepared_corpus_manifest(directory, settings)
    if manifest is not None:
        publish_tokenized(drive_data_root, directory, manifest)
        return manifest

    remote = f"{drive_data_root.rstrip('/')}/tokenized/{directory.name}"
    manifest, files = _remote_state(remote)
    if manifest is None:
        return None
    if any(manifest.get(key) != value for key, value in settings.items()):
        raise ValueError(f"The prepared corpus in {remote} uses different settings.")
    missing = set(OUTPUT_FILES).difference(files)
    if missing:
        raise FileNotFoundError(f"Incomplete Google Drive corpus in {remote}. Missing {', '.join(sorted(missing))}.")
    print(f"Restoring prepared data from {remote}...", flush=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".restore-", dir=directory.parent) as temporary:
        stage = Path(temporary) / directory.name
        stage.mkdir()
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in OUTPUT_FILES:
            if name != "manifest.json":
                _rclone("copyto", f"{remote}/{name}", str(stage / name))
        prepared_corpus_manifest(stage, settings)
        verify_corpus_files(stage, manifest)
        try:
            publish_prepared_corpus(stage, directory)
        except OSError:
            # A second notebook may have restored the same corpus during our download.
            if prepared_corpus_manifest(directory, settings) != manifest:
                raise
            verify_corpus_files(directory, manifest)
    return manifest


def load_raw_training_stories(
    drive_data_root: str, cache_dir: Path, dataset_name: str, train_files: str, token: str | None,
    *,
    sync_drive: bool = True,
):
    from datasets import Dataset, load_dataset

    directory = cache_dir / dataset_name
    remote = f"{drive_data_root.rstrip('/')}/raw/{dataset_name}"
    source = directory / "train.parquet"
    marker = directory / "manifest.json"
    expected = {"dataset": dataset_name, "split": "train", "source_files": train_files}
    saved = _remote_state(remote)[0] if sync_drive else None
    manifest = saved
    if manifest is None and marker.is_file() and source.is_file():
        manifest = json.loads(marker.read_bytes())
    if manifest is not None:
        digest = manifest.get("sha256", "")
        if any(manifest.get(k) != v for k, v in expected.items()) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError(f"Raw training manifest does not match {dataset_name} and {train_files}.")
        if not source.is_file() or sha256(source) != digest:
            if saved is None:
                raise ValueError(f"The cached training stories do not match {marker}.")
            directory.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=directory) as temporary:
                staged = Path(temporary) / source.name
                print(f"Restoring training stories from {remote}...", flush=True)
                _rclone("copyto", f"{remote}/{source.name}", str(staged))
                if sha256(staged) != digest:
                    raise ValueError("The raw training Parquet file disagrees with its Google Drive manifest.")
                staged.replace(source)
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        if not marker.is_file() or marker.read_bytes() != manifest_bytes:
            with TemporaryDirectory(dir=directory) as temporary:
                staged = Path(temporary) / marker.name
                staged.write_bytes(manifest_bytes)
                staged.replace(marker)
        if sync_drive:
            _publish(remote, directory, manifest, {source.name: digest})
        return Dataset.from_parquet(str(source), cache_dir=str(cache_dir))

    print("Loading the published SimpleStories training shards...", flush=True)
    dataset = load_dataset(
        dataset_name, data_files={"train": train_files}, split="train", cache_dir=str(cache_dir),
        verification_mode="no_checks", token=token,
    ).select_columns(["story"])
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as temporary:
        stage = Path(temporary)
        parquet = stage / source.name
        dataset.to_parquet(str(parquet))
        digest = sha256(parquet)
        manifest = {**expected, "sha256": digest}
        (stage / marker.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if sync_drive:
            _publish(remote, stage, manifest, {source.name: digest})
        parquet.replace(source)
        (stage / marker.name).replace(marker)
    return dataset


SFT_FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json")


def _sft_digests(manifest: dict) -> dict[str, str]:
    files_info = manifest.get("files", {})
    return {name: files_info[name]["sha256"] for name in ("train.jsonl", "val.jsonl", "test.jsonl")}


def publish_sft_dataset(drive_data_root: str, directory: Path, manifest: dict) -> str:
    """Publish a prepared SFT dataset directory cleanly to Google Drive with SHA-256 verification."""
    saved_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if saved_manifest != manifest:
        raise ValueError(f"SFT manifest on disk does not match {directory}.")
    digests = _sft_digests(manifest)
    for name, expected in digests.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"SFT file {directory / name} fails SHA-256 check.")
    remote = f"{drive_data_root.rstrip('/')}/sft/{directory.name}"
    _publish(remote, directory, manifest, digests)
    return remote


def reuse_sft_dataset(drive_data_root: str, directory: Path, expected_identity: dict) -> dict | None:
    """Restore a prepared SFT dataset from Google Drive into `directory` if present and matching."""
    remote = f"{drive_data_root.rstrip('/')}/sft/{directory.name}"
    manifest, files = _remote_state(remote)
    if manifest is None:
        return None
    if any(manifest.get(k) != v for k, v in expected_identity.items()):
        raise ValueError(f"The SFT dataset in {remote} uses different settings.")
    missing = set(SFT_FILES).difference(files)
    if missing:
        raise FileNotFoundError(f"Incomplete Google Drive SFT dataset in {remote}. Missing {', '.join(sorted(missing))}.")
    print(f"Restoring prepared SFT dataset from {remote}...", flush=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".restore-sft-", dir=directory.parent) as temporary:
        stage = Path(temporary) / directory.name
        stage.mkdir()
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        digests = _sft_digests(manifest)
        for name, expected_sha in digests.items():
            _rclone("copyto", f"{remote}/{name}", str(stage / name))
            if sha256(stage / name) != expected_sha:
                raise ValueError(f"Downloaded SFT split {name} disagrees with Drive manifest checksum.")
        if directory.exists():
            shutil.rmtree(directory)
        stage.replace(directory)
    return manifest
