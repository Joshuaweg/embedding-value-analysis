"""Embedding extraction from trained models and GPT-2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from src.model import ModelConfig, ValueTransformer


def load_model(checkpoint_path: str | Path, config: ModelConfig | None = None) -> ValueTransformer:
    """Load a trained ValueTransformer from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        config: Model config. If None, loaded from the checkpoint.

    Returns:
        Model in eval mode on CPU.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if config is None:
        config = ModelConfig(**ckpt["config"])

    model = ValueTransformer(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def extract_token_embeddings(
    model: ValueTransformer,
    tokenizer: Tokenizer,
    words: list[str],
) -> dict[str, np.ndarray]:
    """Extract static token embeddings for a list of words.

    For words that tokenize into multiple subword tokens, the embedding
    is the average of the constituent token embeddings.

    Args:
        model: Trained ValueTransformer.
        tokenizer: The BPE tokenizer used during training.
        words: Words to extract embeddings for.

    Returns:
        Dict mapping each word to its embedding vector.
    """
    embeddings: dict[str, np.ndarray] = {}

    with torch.no_grad():
        emb_weight = model.get_token_embeddings().weight

        for word in words:
            token_ids = tokenizer.encode(word).ids
            if not token_ids:
                continue
            vecs = emb_weight[token_ids]
            embeddings[word] = vecs.mean(dim=0).numpy()

    return embeddings


def extract_contextual_embeddings(
    model: ValueTransformer,
    tokenizer: Tokenizer,
    word: str,
    contexts: list[str],
) -> np.ndarray:
    """Extract contextual embeddings for a word across multiple contexts.

    Runs each context through the model, extracts the last-layer hidden
    state at the position of the target word, and averages across contexts.

    Args:
        model: Trained ValueTransformer.
        tokenizer: The BPE tokenizer.
        word: Target word.
        contexts: List of sentences containing the word.

    Returns:
        Averaged contextual embedding vector.
    """
    all_vecs: list[torch.Tensor] = []

    word_tokens = tokenizer.encode(word).ids
    if not word_tokens:
        raise ValueError(f"Word '{word}' produces no tokens")

    with torch.no_grad():
        for ctx in contexts:
            ids = tokenizer.encode(ctx).ids
            if len(ids) > model.config.context_len:
                ids = ids[: model.config.context_len]

            word_pos = _find_subsequence(ids, word_tokens)
            if word_pos == -1:
                continue

            idx = torch.tensor([ids], dtype=torch.long)

            # Run through all blocks to get last-layer hidden states
            x = model.drop(model.token_emb(idx) + model.pos_emb(torch.arange(len(ids))))
            for block in model.blocks:
                x = block(x)
            x = model.ln_f(x)

            word_vecs = x[0, word_pos : word_pos + len(word_tokens)]
            all_vecs.append(word_vecs.mean(dim=0))

    if not all_vecs:
        raise ValueError(f"Word '{word}' not found in any context")

    return torch.stack(all_vecs).mean(dim=0).numpy()


def _find_subsequence(seq: list[int], subseq: list[int]) -> int:
    """Find the start index of subseq in seq, or -1 if not found."""
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i : i + len(subseq)] == subseq:
            return i
    return -1


def extract_gpt2_embeddings(words: list[str]) -> dict[str, np.ndarray]:
    """Extract token embeddings from pretrained GPT-2.

    Args:
        words: Words to extract embeddings for.

    Returns:
        Dict mapping each word to its GPT-2 embedding vector.
    """
    from transformers import GPT2Model, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2Model.from_pretrained("gpt2")
    gpt2.eval()

    emb_weight = gpt2.wte.weight.detach()
    embeddings: dict[str, np.ndarray] = {}

    for word in words:
        token_ids = tok.encode(word)
        if not token_ids:
            continue
        ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        vecs = emb_weight[ids_tensor]
        embeddings[word] = vecs.mean(dim=0).numpy()

    return embeddings


def extract_and_save_all(
    models_dir: str | Path,
    tokenizer_path: str | Path,
    words: list[str],
    output_dir: str | Path,
) -> None:
    """Extract embeddings from all checkpoints and GPT-2, then save.

    Looks for `<corpus>/best.pt` checkpoints in models_dir.

    Args:
        models_dir: Directory containing per-corpus checkpoint folders.
        tokenizer_path: Path to the BPE tokenizer JSON.
        words: List of words to extract embeddings for.
        output_dir: Directory to save .npz embedding files.
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    for ckpt_dir in sorted(models_dir.iterdir()):
        best = ckpt_dir / "best.pt"
        if not best.exists():
            continue

        corpus_name = ckpt_dir.name
        print(f"Extracting embeddings for {corpus_name}...")
        model = load_model(best)
        embs = extract_token_embeddings(model, tokenizer, words)

        out_path = output_dir / f"{corpus_name}.npz"
        np.savez(str(out_path), **embs)

    print("Extracting GPT-2 embeddings...")
    gpt2_embs = extract_gpt2_embeddings(words)
    np.savez(str(output_dir / "gpt2.npz"), **gpt2_embs)

    # Save word list for reference
    (output_dir / "words.json").write_text(json.dumps(words, indent=2))
    print(f"All embeddings saved to {output_dir}")
