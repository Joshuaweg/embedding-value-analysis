"""Corpus loading and preparation for value embedding research."""

from __future__ import annotations

import os
from pathlib import Path

import nltk


def _ensure_nltk_data() -> None:
    """Download required NLTK datasets if not already present."""
    for resource in ("gutenberg", "brown"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


CORPUS_CLUSTERS: dict[str, list[str]] = {
    "religious": [
        "bible-kjv.txt",
        "milton-paradise.txt",
    ],
    "transcendentalist": [
        "whitman-leaves.txt",
        "blake-poems.txt",
    ],
    "victorian": [
        "austen-emma.txt",
        "austen-persuasion.txt",
        "austen-sense.txt",
    ],
    "philosophical": [
        "chesterton-ball.txt",
        "chesterton-brown.txt",
        "chesterton-thursday.txt",
        "shakespeare-hamlet.txt",
        "shakespeare-macbeth.txt",
        "shakespeare-caesar.txt",
    ],
}


def load_cluster(cluster_name: str) -> str:
    """Load and concatenate all texts for a given corpus cluster.

    Args:
        cluster_name: One of the keys in CORPUS_CLUSTERS, or 'baseline'
            for the combined corpus plus Brown.

    Returns:
        Concatenated raw text for the cluster.
    """
    _ensure_nltk_data()
    from nltk.corpus import gutenberg, brown

    if cluster_name == "baseline":
        parts = [gutenberg.raw(fid) for fid in gutenberg.fileids()]
        parts.append(" ".join(brown.words()))
        return "\n\n".join(parts)

    if cluster_name not in CORPUS_CLUSTERS:
        raise ValueError(
            f"Unknown cluster '{cluster_name}'. "
            f"Choose from {list(CORPUS_CLUSTERS.keys())} or 'baseline'."
        )

    file_ids = CORPUS_CLUSTERS[cluster_name]
    return "\n\n".join(gutenberg.raw(fid) for fid in file_ids)


def load_all_clusters() -> dict[str, str]:
    """Load all corpus clusters including baseline.

    Returns:
        Dict mapping cluster name to concatenated text.
    """
    clusters = {name: load_cluster(name) for name in CORPUS_CLUSTERS}
    clusters["baseline"] = load_cluster("baseline")
    return clusters


def save_raw_corpora(output_dir: str | Path) -> dict[str, Path]:
    """Save each cluster as a plain text file.

    Args:
        output_dir: Directory to write cluster text files into.

    Returns:
        Dict mapping cluster name to the written file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = load_all_clusters()
    paths: dict[str, Path] = {}

    for name, text in clusters.items():
        path = output_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        paths[name] = path

    return paths
