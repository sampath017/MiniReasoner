"""Prepare, cache, and verify packed SimpleStories training and validation corpora."""

from hashlib import blake2b
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer

from .drive_corpus import load_raw_training_stories, publish_tokenized, reuse_tokenized
from ..core.tokenizer import (
    EOS,
    TOKEN_DTYPE,
    pack_documents,
    publish_prepared_corpus,
    sha256,
    story_texts,
    train_tokenizer,
    verify_corpus_files,
    verify_packed_eos,
)
from ..settings import (
    DATASET_CACHE_DIR,
    DRIVE_DATA_ROOT,
    SIMPLESTORIES_TRAIN_FILES,
    TOKENIZED_DATA_DIR,
    VALIDATION_FRACTION,
    VOCAB_SIZE,
)


def ensure_tokenized_corpus(
    *,
    dataset_name: str = "SimpleStories/SimpleStories",
    source_splits: dict[str, str] | None = None,
    seed: int = 42,
    validation_fraction: float | None = None,
    target_vocab_size: int | None = None,
    cache_dir: Path | None = None,
    tokenized_path: Path | None = None,
    drive_data_root: str | None = None,
    tokenizer_device: str = "auto",
) -> dict:
    """Restore or prepare the packed `train.bin`/`val.bin` SimpleStories corpus."""
    if source_splits is None:
        source_splits = {"train": "train", "val": "train"}
    if validation_fraction is None:
        validation_fraction = VALIDATION_FRACTION
    if target_vocab_size is None:
        target_vocab_size = VOCAB_SIZE
    if cache_dir is None:
        cache_dir = DATASET_CACHE_DIR
    if tokenized_path is None:
        tokenized_path = TOKENIZED_DATA_DIR
    if drive_data_root is None:
        drive_data_root = DRIVE_DATA_ROOT

    corpus_settings = {
        "dataset": dataset_name,
        "source_splits": source_splits,
        "seed": seed,
        "split_method": "story_hash_v1",
        "validation_fraction": validation_fraction,
        "target_vocab_size": target_vocab_size,
    }

    manifest_path = tokenized_path / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_corpus_files(tokenized_path, manifest)
            print(f"Reusing verified packed data on local disk at {tokenized_path}.")
            return manifest
        except Exception as exc:
            print(f"Local tokenized corpus verification failed ({exc}); checking Google Drive...")

    manifest = reuse_tokenized(drive_data_root, tokenized_path, corpus_settings)
    if manifest is not None:
        print(f"Restored and verified packed data from Google Drive to {tokenized_path}.")
        return manifest

    print("No prepared corpus found. Preparing SimpleStories dataset from scratch...")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    dataset = load_raw_training_stories(
        drive_data_root, cache_dir, dataset_name, SIMPLESTORIES_TRAIN_FILES,
        token=os.getenv("HF_TOKEN", "").strip() or None,
    )

    split_indices = {"train": [], "val": []}
    validation_threshold = int(validation_fraction * (1 << 64))
    hash_key = str(seed).encode("ascii")
    row_index = 0
    for batch in dataset.iter(batch_size=2048):
        for text in batch["story"]:
            story = (text or "").strip()
            if story and EOS not in story:
                digest = blake2b(story.encode("utf-8"), digest_size=8, key=hash_key).digest()
                split_name = "val" if int.from_bytes(digest, "big") < validation_threshold else "train"
                split_indices[split_name].append(row_index)
            row_index += 1

    splits = {}
    for name, indices in split_indices.items():
        if not indices:
            raise ValueError(f"No usable {name} stories. Adjust VALIDATION_FRACTION.")
        splits[name] = dataset.select(indices)

    print(f"Training tokenizer with target vocab size {target_vocab_size:,}...")
    tokenizer = train_tokenizer(story_texts(splits["train"]), target_vocab_size, device=tokenizer_device)

    tokenized_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=tokenized_path.parent) as temporary:
        staging = Path(temporary) / tokenized_path.name
        staging.mkdir()
        tokenizer_path = staging / "tokenizer.json"
        tokenizer_path.write_text(tokenizer.to_str(pretty=True), encoding="utf-8")
        saved_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        eos_id = saved_tokenizer.token_to_id(EOS)

        manifest = {
            "version": 1,
            "dtype": TOKEN_DTYPE.str,
            **corpus_settings,
            "source_rows": len(dataset),
            "vocab_size": saved_tokenizer.get_vocab_size(),
            "eos_id": eos_id,
            "tokenizer_sha256": sha256(tokenizer_path),
        }
        for name, rows in splits.items():
            print(f"Packing {len(rows):,} {name} stories...", flush=True)
            split_info, story_ends = pack_documents(
                staging / f"{name}.bin", saved_tokenizer, story_texts(rows)
            )
            verify_packed_eos(staging / f"{name}.bin", split_info, story_ends, eos_id)
            manifest[name] = split_info

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        publish_prepared_corpus(staging, tokenized_path)

    publish_tokenized(drive_data_root, tokenized_path, corpus_settings)
    return manifest
