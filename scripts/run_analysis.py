"""Run full analysis pipeline: extract embeddings, compare models, save figures."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import extract_and_save_all, load_model, extract_token_embeddings
from src.analyze import load_embeddings, compare_models
from value_lexicons.mft_words import ALL_VALUE_WORDS, WORD_TO_FOUNDATION


MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "analysis" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "analysis"


def main() -> None:
    print("=" * 60)
    print("Step 1: Extracting embeddings from all models + GPT-2")
    print("=" * 60)
    extract_and_save_all(
        models_dir=MODELS_DIR,
        tokenizer_path=TOKENIZER_PATH,
        words=ALL_VALUE_WORDS,
        output_dir=EMBEDDINGS_DIR,
    )

    print()
    print("=" * 60)
    print("Step 2: Loading embeddings and running comparative analysis")
    print("=" * 60)
    all_embs = load_embeddings(EMBEDDINGS_DIR)
    print(f"  Loaded embeddings for {len(all_embs)} models")

    metrics = compare_models(all_embs, WORD_TO_FOUNDATION, OUTPUT_DIR)

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    for model_name, m in metrics.items():
        probe = m["probe"]
        print(
            f"  {model_name}: "
            f"probe accuracy = {probe['accuracy']:.3f} +/- {probe['std']:.3f}, "
            f"n_words = {probe['n_words']}, "
            f"mean_sim = {m['mean_similarity']:.3f}"
        )

    print(f"\nFigures saved to {OUTPUT_DIR / 'figures'}")
    print(f"Metrics saved to {OUTPUT_DIR / 'results' / 'metrics.json'}")


if __name__ == "__main__":
    main()
