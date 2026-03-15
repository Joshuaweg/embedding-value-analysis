"""Analysis utilities for comparing value embeddings across models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance as wasserstein_1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_embeddings(embeddings_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load all saved embedding files from a directory.

    Args:
        embeddings_dir: Directory containing .npz files (one per model).

    Returns:
        Nested dict: model_name -> word -> embedding vector.
    """
    embeddings_dir = Path(embeddings_dir)
    all_embs: dict[str, dict[str, np.ndarray]] = {}

    for npz_path in sorted(embeddings_dir.glob("*.npz")):
        model_name = npz_path.stem
        data = np.load(str(npz_path))
        all_embs[model_name] = {k: data[k] for k in data.files}

    return all_embs


def cosine_similarity_matrix(embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute pairwise cosine similarity for a set of word embeddings.

    Args:
        embeddings: Dict mapping words to embedding vectors.

    Returns:
        DataFrame with words as both index and columns.
    """
    words = sorted(embeddings.keys())
    vecs = np.stack([embeddings[w] for w in words])

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = vecs / norms
    sim = normed @ normed.T

    return pd.DataFrame(sim, index=words, columns=words)


def plot_foundation_similarity_heatmap(
    embeddings: dict[str, np.ndarray],
    word_to_foundation: dict[str, str],
    model_name: str,
    output_path: str | Path,
) -> None:
    """Save a 6x6 heatmap of mean cosine similarity between MFT foundations.

    Args:
        embeddings: Word -> embedding vector.
        word_to_foundation: Word -> foundation name.
        model_name: Name for the plot title.
        output_path: Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    words = [w for w in embeddings if w in word_to_foundation]
    foundations = sorted(set(word_to_foundation[w] for w in words))

    vecs = np.stack([embeddings[w] for w in words])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.maximum(norms, 1e-8)

    n = len(foundations)
    matrix = np.zeros((n, n))
    for i, f_i in enumerate(foundations):
        idx_i = [j for j, w in enumerate(words) if word_to_foundation[w] == f_i]
        for j, f_j in enumerate(foundations):
            idx_j = [k for k, w in enumerate(words) if word_to_foundation[w] == f_j]
            sims = normed[idx_i] @ normed[idx_j].T
            matrix[i, j] = sims.mean()

    df = pd.DataFrame(matrix, index=foundations, columns=foundations)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        df,
        cmap="RdBu_r",
        center=0,
        vmin=-0.2,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
    )
    ax.set_title(f"Foundation Similarity — {model_name}")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_pca_projection(
    embeddings_dict: dict[str, np.ndarray],
    word_to_foundation: dict[str, str],
    model_name: str,
    output_path: str | Path,
) -> None:
    """PCA 2D scatter plot of word embeddings colored by MFT foundation.

    Args:
        embeddings_dict: Word -> embedding vector.
        word_to_foundation: Word -> foundation name mapping.
        model_name: Name for the plot title.
        output_path: Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    words = [w for w in embeddings_dict if w in word_to_foundation]
    if len(words) < 3:
        return

    vecs = np.stack([embeddings_dict[w] for w in words])
    labels = [word_to_foundation[w] for w in words]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(12, 10))
    foundations = sorted(set(labels))
    palette = sns.color_palette("husl", len(foundations))

    for i, f in enumerate(foundations):
        mask = [l == f for l in labels]
        pts = coords[mask]
        ws = [w for w, m in zip(words, mask) if m]
        ax.scatter(pts[:, 0], pts[:, 1], c=[palette[i]], label=f, s=60, alpha=0.7)
        for j, w in enumerate(ws):
            ax.annotate(w, (pts[j, 0], pts[j, 1]), fontsize=6, alpha=0.8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"PCA of Value Words — {model_name}")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_umap_projection(
    embeddings_dict: dict[str, np.ndarray],
    word_to_foundation: dict[str, str],
    model_name: str,
    output_path: str | Path,
) -> None:
    """UMAP 2D scatter plot of word embeddings colored by MFT foundation.

    Args:
        embeddings_dict: Word -> embedding vector.
        word_to_foundation: Word -> foundation name mapping.
        model_name: Name for the plot title.
        output_path: Path to save the figure.
    """
    import umap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    words = [w for w in embeddings_dict if w in word_to_foundation]
    if len(words) < 5:
        return

    vecs = np.stack([embeddings_dict[w] for w in words])
    labels = [word_to_foundation[w] for w in words]

    reducer = umap.UMAP(n_neighbors=min(15, len(words) - 1), random_state=42)
    coords = reducer.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(12, 10))
    foundations = sorted(set(labels))
    palette = sns.color_palette("husl", len(foundations))

    for i, f in enumerate(foundations):
        mask = [l == f for l in labels]
        pts = coords[mask]
        ws = [w for w, m in zip(words, mask) if m]
        ax.scatter(pts[:, 0], pts[:, 1], c=[palette[i]], label=f, s=60, alpha=0.7)
        for j, w in enumerate(ws):
            ax.annotate(w, (pts[j, 0], pts[j, 1]), fontsize=6, alpha=0.8)

    ax.set_title(f"UMAP of Value Words — {model_name}")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def train_mft_probe(
    embeddings_dict: dict[str, np.ndarray],
    word_to_foundation: dict[str, str],
) -> dict:
    """Train a logistic regression probe to classify MFT foundation from embeddings.

    Args:
        embeddings_dict: Word -> embedding vector.
        word_to_foundation: Word -> foundation name.

    Returns:
        Dict with 'accuracy' (cross-val mean), 'std', and 'n_words'.
    """
    words = [w for w in embeddings_dict if w in word_to_foundation]
    if len(words) < 10:
        return {"accuracy": 0.0, "std": 0.0, "n_words": len(words)}

    X = np.stack([embeddings_dict[w] for w in words])
    y_labels = [word_to_foundation[w] for w in words]

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    n_folds = min(5, min(np.bincount(y)))
    n_folds = max(2, n_folds)

    scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")

    return {
        "accuracy": float(scores.mean()),
        "std": float(scores.std()),
        "n_words": len(words),
        "classes": list(le.classes_),
    }


def _align_dims(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA-project X and Y to a shared dimensionality when they differ.

    If both arrays already have the same number of columns, they are returned
    unchanged.  Otherwise each is projected independently to
    ``min(d_X, d_Y, n_samples - 1)`` principal components so that sliced
    Wasserstein can operate in a common space.

    Args:
        X: Array of shape (n1, d1).
        Y: Array of shape (n2, d2).

    Returns:
        Tuple (X_aligned, Y_aligned) both with shape (ni, k).
    """
    d_X, d_Y = X.shape[1], Y.shape[1]
    if d_X == d_Y:
        return X, Y

    # Target: smallest safe PCA rank for both arrays
    k = min(d_X, d_Y, X.shape[0] - 1, Y.shape[0] - 1)
    if k < 1:
        raise ValueError(
            f"Cannot align embeddings: too few samples "
            f"(X={X.shape}, Y={Y.shape})"
        )

    pca_X = PCA(n_components=k).fit(X)
    pca_Y = PCA(n_components=k).fit(Y)
    return pca_X.transform(X), pca_Y.transform(Y)


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 500,
    seed: int = 42,
) -> float:
    """Compute the Sliced Wasserstein Distance between two sets of embedding vectors.

    Projects both distributions onto random unit vectors, computes the 1-D
    Wasserstein distance for each projection, and averages the results.
    Works correctly with unequal sample sizes and automatically aligns
    embedding dimensions via PCA when X and Y have different widths.

    Args:
        X: Array of shape (n1, d1).
        Y: Array of shape (n2, d2).
        n_projections: Number of random 1-D projections to average over.
        seed: Random seed for reproducibility.

    Returns:
        Scalar SWD estimate (non-negative).
    """
    X, Y = _align_dims(X, Y)

    rng = np.random.default_rng(seed)
    d = X.shape[1]

    # Sample random unit directions on the d-sphere
    directions = rng.standard_normal((n_projections, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, 1e-10)

    # Project: (n_projections, n_samples)
    proj_X = directions @ X.T  # (n_projections, n1)
    proj_Y = directions @ Y.T  # (n_projections, n2)

    total = 0.0
    for i in range(n_projections):
        total += wasserstein_1d(proj_X[i], proj_Y[i])

    return total / n_projections


def pairwise_wasserstein_matrix(
    all_model_embeddings: dict[str, dict[str, np.ndarray]],
    n_projections: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute pairwise Sliced Wasserstein Distances between all models.

    Args:
        all_model_embeddings: model_name -> word -> embedding vector.
        n_projections: Projections per pair (passed to sliced_wasserstein_distance).
        seed: Random seed.

    Returns:
        Symmetric DataFrame (models x models) of SWD values.
    """
    model_names = sorted(all_model_embeddings.keys())

    # Stack embeddings for each model into a matrix (n_words, d)
    stacked: dict[str, np.ndarray] = {}
    for name in model_names:
        words = sorted(all_model_embeddings[name].keys())
        stacked[name] = np.stack([all_model_embeddings[name][w] for w in words])

    n = len(model_names)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = sliced_wasserstein_distance(
                stacked[model_names[i]],
                stacked[model_names[j]],
                n_projections=n_projections,
                seed=seed,
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return pd.DataFrame(dist_matrix, index=model_names, columns=model_names)


def wasserstein_by_foundation(
    all_model_embeddings: dict[str, dict[str, np.ndarray]],
    word_to_foundation: dict[str, str],
    n_projections: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Compute pairwise SWD between models for each MFT foundation separately.

    Args:
        all_model_embeddings: model_name -> word -> embedding vector.
        word_to_foundation: word -> foundation name.
        n_projections: Projections per pair.
        seed: Random seed.

    Returns:
        Dict mapping foundation name -> symmetric DataFrame of SWD values.
    """
    foundations = sorted(set(word_to_foundation.values()))
    model_names = sorted(all_model_embeddings.keys())
    results: dict[str, pd.DataFrame] = {}

    for foundation in foundations:
        foundation_words = [w for w, f in word_to_foundation.items() if f == foundation]

        stacked: dict[str, np.ndarray] = {}
        for name in model_names:
            embs = all_model_embeddings[name]
            vecs = [embs[w] for w in foundation_words if w in embs]
            if vecs:
                stacked[name] = np.stack(vecs)

        available = [n for n in model_names if n in stacked]
        n = len(available)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = sliced_wasserstein_distance(
                    stacked[available[i]],
                    stacked[available[j]],
                    n_projections=n_projections,
                    seed=seed,
                )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        results[foundation] = pd.DataFrame(dist_matrix, index=available, columns=available)

    return results


def plot_wasserstein_heatmap(
    dist_matrix: pd.DataFrame,
    title: str,
    output_path: str | Path,
) -> None:
    """Save a heatmap of pairwise Sliced Wasserstein Distances.

    Args:
        dist_matrix: Symmetric DataFrame of SWD values.
        title: Plot title.
        output_path: Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        dist_matrix,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def compare_models(
    all_model_embeddings: dict[str, dict[str, np.ndarray]],
    word_to_foundation: dict[str, str],
    output_dir: str | Path,
) -> dict:
    """Run full analysis across all models and save results.

    Args:
        all_model_embeddings: model_name -> word -> embedding.
        word_to_foundation: Word -> foundation mapping.
        output_dir: Directory to save figures and metrics.

    Returns:
        Dict of all metrics.
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}

    for model_name, embs in all_model_embeddings.items():
        print(f"Analyzing {model_name}...")

        sim = cosine_similarity_matrix(embs)
        plot_foundation_similarity_heatmap(embs, word_to_foundation, model_name, fig_dir / f"{model_name}_similarity.png")
        plot_pca_projection(embs, word_to_foundation, model_name, fig_dir / f"{model_name}_pca.png")
        plot_umap_projection(embs, word_to_foundation, model_name, fig_dir / f"{model_name}_umap.png")

        probe_result = train_mft_probe(embs, word_to_foundation)
        metrics[model_name] = {
            "probe": probe_result,
            "n_words_with_embeddings": len(embs),
            "mean_similarity": float(sim.values[np.triu_indices_from(sim.values, k=1)].mean()),
        }

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "metrics.json"
    results_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {results_path}")

    return metrics
