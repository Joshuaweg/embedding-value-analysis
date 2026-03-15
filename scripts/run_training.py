"""Train all value-aligned transformer models sequentially."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus import CORPUS_CLUSTERS
from src.model import ModelConfig
from src.train import TrainConfig, train


TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

ALL_CLUSTERS = list(CORPUS_CLUSTERS.keys()) + ["baseline"]


def main() -> None:
    model_config = ModelConfig()
    train_config = TrainConfig(checkpoint_dir=str(CHECKPOINT_DIR))

    timings: dict[str, float] = {}

    for cluster in ALL_CLUSTERS:
        data_path = TOKENIZED_DIR / f"{cluster}.npy"
        if not data_path.exists():
            print(f"Skipping {cluster}: {data_path} not found. Run prepare_data.py first.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Training model for: {cluster}")
        print(f"{'=' * 60}")

        start = time.time()
        train(cluster, data_path, model_config, train_config)
        elapsed = time.time() - start

        timings[cluster] = elapsed
        print(f"  {cluster} completed in {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    for name, t in timings.items():
        print(f"  {name}: {t:.1f}s ({t / 60:.1f} min)")
    total = sum(timings.values())
    print(f"  Total: {total:.1f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
