"""Prepare all data: download NLTK corpora, save raw texts, train tokenizer, tokenize."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus import load_all_clusters, save_raw_corpora, CORPUS_CLUSTERS
from src.tokenizer import train_tokenizer, encode_file, load_tokenizer
from value_lexicons.mft_words import ALL_VALUE_WORDS


RAW_DIR = PROJECT_ROOT / "data" / "raw"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.json"


def main() -> None:
    print("=" * 60)
    print("Step 1: Downloading NLTK data and saving raw corpora")
    print("=" * 60)
    paths = save_raw_corpora(RAW_DIR)
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name}: {size_kb:.0f} KB")

    print()
    print("=" * 60)
    print("Step 2: Training BPE tokenizer on combined corpus")
    print("=" * 60)
    all_texts = []
    for path in paths.values():
        all_texts.append(path.read_text(encoding="utf-8"))

    tokenizer = train_tokenizer(all_texts, vocab_size=12000, save_path=TOKENIZER_PATH)
    print(f"  Tokenizer saved to {TOKENIZER_PATH}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    print()
    print("=" * 60)
    print("Step 3: Tokenizing each corpus cluster")
    print("=" * 60)
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    for name, raw_path in paths.items():
        out_path = TOKENIZED_DIR / f"{name}.npy"
        n_tokens = encode_file(tokenizer, raw_path, out_path)
        print(f"  {name}: {n_tokens:,} tokens -> {out_path}")

    print()
    print("=" * 60)
    print("Step 4: MFT word vocabulary coverage")
    print("=" * 60)
    vocab = tokenizer.get_vocab()
    found = [w for w in ALL_VALUE_WORDS if w in vocab]
    subword = [w for w in ALL_VALUE_WORDS if w not in vocab]
    print(f"  Total MFT words: {len(ALL_VALUE_WORDS)}")
    print(f"  Direct vocab matches: {len(found)}")
    print(f"  Requiring subword tokenization: {len(subword)}")
    if subword:
        print(f"  Examples: {subword[:10]}")

    print()
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
