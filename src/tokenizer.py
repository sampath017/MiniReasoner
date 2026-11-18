from collections import Counter
from collections.abc import Iterable, Iterator
from hashlib import file_digest
from itertools import chain, islice
from pathlib import Path
import json

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


EOS = "<|eos|>"
TOKEN_DTYPE = np.dtype("<u2")
OUTPUT_FILES = ("train.bin", "val.bin", "tokenizer.json", "manifest.json")


def _word_chunks(words, vocab, storage_device):
    # A zero weight marks the end of a pre-token: a merge cannot cross it.
    token_ids, weights = [], []
    for word, count in words.items():
        token_ids.extend(vocab[char] for char in word)
        weights.extend([count] * (len(word) - 1))
        weights.append(0)
        if len(token_ids) >= 262_144:
            yield (torch.tensor(token_ids, dtype=torch.int64, device=storage_device),
                   torch.tensor(weights, dtype=torch.int64, device=storage_device))
            token_ids, weights = [], []
    if token_ids:
        yield (torch.tensor(token_ids, dtype=torch.int64, device=storage_device),
               torch.tensor(weights, dtype=torch.int64, device=storage_device))


def _sum_pair_counts(pairs, weights):
    keys, indices = torch.unique(pairs, sorted=True, return_inverse=True)
    counts = torch.zeros_like(keys)
    counts.scatter_add_(0, indices, weights)
    return keys, counts


def _best_bpe_pair(chunks, vocab_size, device):
    # At MiniReasoner's default size this histogram uses at most 8 MiB.
    # Larger vocabularies count just the pairs that actually occur.
    dense_counts = (torch.zeros(vocab_size ** 2, dtype=torch.int64, device=device)
                    if vocab_size <= 1024 else None)
    keys_by_chunk, counts_by_chunk = [], []
    for tokens, weights in chunks:
        tokens, weights = tokens.to(device), weights.to(device)
        pairs = tokens[:-1] * vocab_size + tokens[1:]
        if dense_counts is not None:
            dense_counts.scatter_add_(0, pairs, weights[:-1])
        else:
            valid = weights[:-1] > 0
            keys, counts = _sum_pair_counts(pairs[valid], weights[:-1][valid])
            keys_by_chunk.append(keys)
            counts_by_chunk.append(counts)
    if dense_counts is not None:
        counts = dense_counts
    elif keys_by_chunk:
        keys, counts = _sum_pair_counts(torch.cat(keys_by_chunk), torch.cat(counts_by_chunk))
    else:
        return None
    if not counts.numel():
        return None
    best = counts.argmax()  # Sorted keys break frequency ties by token ID.
    if counts[best].item() < 2:
        return None
    winner = best if dense_counts is not None else keys[best]
    return divmod(winner.item(), vocab_size)


def _merge_bpe_pair(tokens, weights, left, right, replacement):
    matches = (tokens[:-1] == left) & (tokens[1:] == right) & (weights[:-1] > 0)
    if left == right:
        # For a run like "aaaaa", merge positions 0 and 2 from left to right.
        positions = torch.arange(matches.numel(), device=tokens.device)
        last_break = torch.where(matches, -1, positions).cummax(0).values
        matches &= (positions - last_break) % 2 == 1
    keep = torch.ones_like(tokens, dtype=torch.bool)
    keep[1:] = ~matches
    tokens[:-1].masked_fill_(matches, replacement)
    weights[:-1] = torch.where(matches, weights[1:], weights[:-1])
    return tokens[keep], weights[keep]


def _train_cuda_bpe(tokenizer, documents, vocab_size, alphabet):
    print("Counting byte-level pre-tokens on CPU...", flush=True)
    words = Counter()
    pre_tokenize = tokenizer.pre_tokenizer.pre_tokenize_str
    for document in documents:
        pieces = pre_tokenize(document)
        words.update(piece for piece, _ in pieces if len(piece) > 1)

    print(f"Learning BPE merges on CUDA from {len(words):,} distinct multi-symbol pre-tokens...", flush=True)
    vocab = {EOS: 0}
    vocab.update((char, index) for index, char in enumerate(sorted(alphabet), start=1))
    id_to_piece = list(vocab)
    device = torch.device("cuda")
    # Each distinct spelling needs two int64 arrays. Leave room for counting
    # and compaction; larger datasets stage one chunk on the GPU at a time.
    bytes_needed = 16 * sum(len(word) for word in words)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    storage_device = device if bytes_needed <= free_bytes // 3 else torch.device("cpu")
    chunks = list(_word_chunks(words, vocab, storage_device))
    del words

    merges = []
    while len(vocab) < vocab_size:
        pair = _best_bpe_pair(chunks, len(vocab), device)
        if pair is None:
            break
        left, right = pair
        left_piece, right_piece = id_to_piece[left], id_to_piece[right]
        merged = left_piece + right_piece
        if merged not in vocab:
            vocab[merged] = len(id_to_piece)
            id_to_piece.append(merged)
        merges.append((left_piece, right_piece))
        for index, (tokens, weights) in enumerate(chunks):
            tokens, weights = _merge_bpe_pair(tokens.to(device), weights.to(device),
                                              left, right, vocab[merged])
            chunks[index] = tokens.to(storage_device), weights.to(storage_device)

    tokenizer.model = models.BPE(vocab=vocab, merges=merges)
    tokenizer.add_special_tokens([EOS])


def train_tokenizer(documents: Iterable[str], vocab_size: int, device: str = "auto") -> Tokenizer:
    if not 257 <= vocab_size <= 65_536:
        raise ValueError("Vocabulary size must be between 257 and 65,536")
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("Tokenizer training device must be auto, cuda, or cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for tokenizer training. Choose auto or cpu.")
    print(f"Tokenizer training device: {device.upper()}", flush=True)
    tokenizer = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Digits(individual_digits=True), byte_level]
    )
    tokenizer.decoder = decoders.ByteLevel()
    if device == "cuda":
        _train_cuda_bpe(tokenizer, documents, vocab_size, byte_level.alphabet())
        return tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=byte_level.alphabet(),
        special_tokens=[EOS],
        show_progress=False,
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    return tokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def prepared_corpus_manifest(directory: Path, corpus_settings: dict) -> dict | None:
    present = [name for name in OUTPUT_FILES if (directory / name).exists()]
    if not present:
        return None
    if len(present) != len(OUTPUT_FILES) or any(
        not (directory / name).is_file() for name in OUTPUT_FILES
    ):
        raise FileExistsError(
            f"Packed data is incomplete in {directory}. "
            "Move the incomplete folder aside or restore its missing files, then rerun this cell."
        )
    saved_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if saved_manifest.get("version") != 1 or saved_manifest.get("dtype") != TOKEN_DTYPE.str:
        raise ValueError(f"Unsupported packed dataset format in {directory}.")
    if any(saved_manifest.get(key) != value for key, value in corpus_settings.items()):
        raise ValueError(
            f"Packed data in {directory} uses different settings. "
            "Restore the original settings to reuse it or choose a new workspace_dir."
        )
    return saved_manifest


def publish_prepared_corpus(staging: Path, destination: Path) -> None:
    # Promote only a complete staged directory. Interrupted preparation stays unpublished.
    if any(not (staging / name).is_file() for name in OUTPUT_FILES):
        raise ValueError(f"The staged corpus is incomplete in {staging}.")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"The corpus destination contains existing files: {destination}")
        destination.rmdir()  # Remove only an empty destination directory.
    staging.rename(destination)


def verify_corpus_files(directory: Path, manifest: dict) -> None:
    if sha256(directory / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError(f"The tokenizer in {directory} does not match its manifest.")
    for split in ("train", "val"):
        path = directory / f"{split}.bin"
        info = manifest[split]
        if (path.stat().st_size != info["tokens"] * TOKEN_DTYPE.itemsize
                or sha256(path) != info["sha256"]):
            raise ValueError(f"{path} does not match the dataset manifest")


def story_texts(rows) -> Iterator[str]:
    for batch in rows.iter(batch_size=2048):
        yield from (text.strip() for text in batch["story"])


def pack_documents(path: Path, tokenizer: Tokenizer, documents: Iterator[str]) -> tuple[dict, np.ndarray]:
    eos_id = tokenizer.token_to_id(EOS)
    if eos_id is None:
        raise ValueError("The tokenizer needs an EOS token")
    tokens = examples = 0
    story_ends = []
    with path.open("wb") as stream:
        while batch := list(islice(documents, 2048)):
            encoded = [item.ids for item in tokenizer.encode_batch(batch, add_special_tokens=False)]
            lengths = [len(ids) + 1 for ids in encoded]
            story_ends.append(tokens + np.cumsum(lengths, dtype=np.int64) - 1)
            token_ids = chain.from_iterable(chain(ids, (eos_id,)) for ids in encoded)
            packed = np.fromiter(token_ids, dtype=TOKEN_DTYPE)
            stream.write(packed.tobytes())
            tokens += len(packed)
            examples += len(batch)
    if not story_ends:
        raise ValueError(f"No stories to pack in {path}")
    return {"documents": examples, "tokens": tokens, "sha256": sha256(path)}, np.concatenate(story_ends)


def verify_packed_eos(path: Path, info: dict, story_ends: np.ndarray, eos_id: int) -> None:
    if path.stat().st_size != info["tokens"] * TOKEN_DTYPE.itemsize:
        raise ValueError(f"{path}: file size does not match the saved token count")
    tokens = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
    if (len(story_ends) != info["documents"] or story_ends[-1] != len(tokens) - 1
            or np.any(tokens[story_ends] != eos_id)):
        raise ValueError(f"{path}: missing EOS at a story boundary")
    chunk_size = 1_048_576
    eos_count = sum(np.count_nonzero(tokens[i:i + chunk_size] == eos_id)
                    for i in range(0, len(tokens), chunk_size))
    if eos_count != len(story_ends):
        raise ValueError(f"{path}: expected exactly one EOS per story")
