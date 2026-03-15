"""Compute pairwise Sliced Wasserstein Distances between all trained models.

Loads pre-extracted embeddings from analysis/embeddings/ and produces:
  - analysis/results/wasserstein_pairwise.json   — full matrix
  - analysis/results/wasserstein_by_foundation.json — per-foundation matrices
  - analysis/figures/wasserstein_pairwise.png    — heatmap
  - analysis/figures/wasserstein_<foundation>.png for each MFT foundation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze import (
    load_embeddings,
    pairwise_wasserstein_matrix,
    plot_wasserstein_heatmap,
    wasserstein_by_foundation,
)
from value_lexicons.mft_words import WORD_TO_FOUNDATION

EMBEDDINGS_DIR = PROJECT_ROOT / "analysis" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"

N_PROJECTIONS = 1000  # higher = more accurate estimate; ~1 s per pair


def main() -> None:
    print("Loading embeddings...")
    all_embs = load_embeddings(EMBEDDINGS_DIR)
    print(f"  {len(all_embs)} models: {sorted(all_embs.keys())}")

    # ------------------------------------------------------------------
    # 1. Full pairwise SWD (all value words pooled)
    # ------------------------------------------------------------------
    print(f"\nComputing pairwise SWD ({N_PROJECTIONS} projections per pair)...")
    dist_df = pairwise_wasserstein_matrix(all_embs, n_projections=N_PROJECTIONS)

    print("\nPairwise Sliced Wasserstein Distance matrix:")
    print(dist_df.round(4).to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dist_path = RESULTS_DIR / "wasserstein_pairwise.json"
    dist_path.write_text(json.dumps(dist_df.to_dict(), indent=2))
    print(f"\nSaved matrix -> {dist_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "wasserstein_pairwise.png"
    plot_wasserstein_heatmap(
        dist_df,
        title="Sliced Wasserstein Distance — All Value Words",
        output_path=fig_path,
    )
    print(f"Saved heatmap -> {fig_path}")

    # ------------------------------------------------------------------
    # 2. Per-foundation SWD
    # ------------------------------------------------------------------
    print("\nComputing per-foundation SWD...")
    foundation_results = wasserstein_by_foundation(
        all_embs, WORD_TO_FOUNDATION, n_projections=N_PROJECTIONS
    )

    foundation_json: dict[str, dict] = {}
    for foundation, df in foundation_results.items():
        foundation_json[foundation] = df.to_dict()
        fig_path = FIGURES_DIR / f"wasserstein_{foundation}.png"
        plot_wasserstein_heatmap(
            df,
            title=f"Sliced Wasserstein Distance — {foundation.capitalize()}",
            output_path=fig_path,
        )
        print(f"  {foundation}: saved heatmap -> {fig_path}")

    found_path = RESULTS_DIR / "wasserstein_by_foundation.json"
    found_path.write_text(json.dumps(foundation_json, indent=2))
    print(f"\nSaved per-foundation matrices -> {found_path}")

    # ------------------------------------------------------------------
    # 3. Summary: rank model pairs by distance
    # ------------------------------------------------------------------
    print("\nModel pairs ranked by overall SWD (most different -> most similar):")
    models = list(dist_df.index)
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pairs.append((models[i], models[j], dist_df.iloc[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, d in pairs:
        print(f"  {a:20s} vs {b:20s}: {d:.4f}")


if __name__ == "__main__":
    main()
