"""BPE tokenizer training and encoding utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]


def train_tokenizer(
    texts: list[str],
    vocab_size: int = 12000,
    save_path: str | Path | None = None,
) -> Tokenizer:
    """Train a BPE tokenizer on the provided texts.

    Args:
        texts: List of raw text strings to train on.
        vocab_size: Target vocabulary size.
        save_path: If provided, save the tokenizer JSON to this path.

    Returns:
        Trained tokenizer instance.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))

    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer from a saved JSON file.

    Args:
        path: Path to the tokenizer JSON file.

    Returns:
        Loaded tokenizer instance.
    """
    return Tokenizer.from_file(str(path))


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode a text string into token IDs.

    Args:
        tokenizer: Trained tokenizer.
        text: Raw text to encode.

    Returns:
        List of integer token IDs.
    """
    return tokenizer.encode(text).ids


def encode_file(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Tokenize a text file and save as a numpy int32 array.

    Args:
        tokenizer: Trained tokenizer.
        input_path: Path to the input text file.
        output_path: Path to save the .npy token array.

    Returns:
        Number of tokens produced.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    ids = encode_text(tokenizer, text)
    arr = np.array(ids, dtype=np.int32)
    np.save(str(output_path), arr)
    return len(ids)
