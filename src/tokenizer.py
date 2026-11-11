from collections.abc import Iterable, Iterator
from hashlib import file_digest
from itertools import chain, islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


EOS = "<|eos|>"
TOKEN_DTYPE = np.dtype("<u2")


def train_tokenizer(documents: Iterable[str], vocab_size: int) -> Tokenizer:
    if not 257 <= vocab_size <= 65_536:
        raise ValueError("Vocabulary size must be between 257 and 65,536")
    tokenizer = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Digits(individual_digits=True), byte_level]
    )
    tokenizer.decoder = decoders.ByteLevel()
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


def pack_documents(path: Path, tokenizer: Tokenizer, documents: Iterator[str]) -> dict:
    eos_id = tokenizer.token_to_id(EOS)
    tokens = examples = 0
    with path.open("wb") as stream:
        while batch := list(islice(documents, 2048)):
            encoded = tokenizer.encode_batch(batch, add_special_tokens=False)
            token_ids = chain.from_iterable(chain(item.ids, (eos_id,)) for item in encoded)
            packed = np.fromiter(token_ids, dtype=TOKEN_DTYPE)
            stream.write(packed.tobytes())
            tokens += len(packed)
            examples += len(batch)
    return {"documents": examples, "tokens": tokens, "sha256": sha256(path)}
