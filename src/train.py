"""Training loop for value-aligned transformer models."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model import ModelConfig, ValueTransformer


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    warmup_iters: int = 200
    min_lr: float = 3e-5
    checkpoint_dir: str = "models/checkpoints"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    """Loads tokenized .npy data and yields random batches."""

    def __init__(
        self, data_path: str | Path, batch_size: int, context_len: int, device: str
    ) -> None:
        self.data = np.load(str(data_path)).astype(np.int64)
        self.batch_size = batch_size
        self.context_len = context_len
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a random batch of (input, target) pairs."""
        max_start = len(self.data) - self.context_len - 1
        starts = np.random.randint(0, max_start, size=self.batch_size)
        x = np.stack([self.data[s : s + self.context_len] for s in starts])
        y = np.stack([self.data[s + 1 : s + self.context_len + 1] for s in starts])
        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device),
        )


def _get_lr(it: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / config.warmup_iters
    if it >= config.max_iters:
        return config.min_lr
    ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def _estimate_loss(
    model: ValueTransformer, loader: DataLoader, eval_iters: int
) -> float:
    """Estimate average loss over eval_iters random batches."""
    model.eval()
    total = 0.0
    for _ in range(eval_iters):
        x, y = loader.get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item()
    model.train()
    return total / eval_iters


def train(
    corpus_name: str,
    data_path: str | Path,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
) -> ValueTransformer:
    """Train a ValueTransformer on tokenized data.

    Args:
        corpus_name: Name of the corpus cluster (used for checkpoint naming).
        data_path: Path to the tokenized .npy file.
        model_config: Model architecture config. Uses defaults if None.
        train_config: Training hyperparameters. Uses defaults if None.

    Returns:
        Trained model.
    """
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainConfig()

    loader = DataLoader(
        data_path, train_config.batch_size, model_config.context_len, train_config.device
    )
    print(f"[{corpus_name}] Loaded {len(loader)} tokens from {data_path}")

    model = ValueTransformer(model_config).to(train_config.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{corpus_name}] Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    ckpt_dir = Path(train_config.checkpoint_dir) / corpus_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_loss = float("inf")

    pbar = tqdm(range(train_config.max_iters), desc=corpus_name)
    for it in pbar:
        lr = _get_lr(it, train_config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = loader.get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (it + 1) % train_config.eval_interval == 0 or it == train_config.max_iters - 1:
            eval_loss = _estimate_loss(model, loader, train_config.eval_iters)
            pbar.set_postfix(train=f"{loss.item():.3f}", val=f"{eval_loss:.3f}", lr=f"{lr:.1e}")
            history.append({"iter": it + 1, "train_loss": loss.item(), "eval_loss": eval_loss, "lr": lr})

            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(
                    {"model": model.state_dict(), "config": model_config.__dict__, "iter": it + 1},
                    ckpt_dir / "best.pt",
                )

    torch.save(
        {"model": model.state_dict(), "config": model_config.__dict__, "iter": train_config.max_iters},
        ckpt_dir / "final.pt",
    )

    history_path = ckpt_dir / "training_curve.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[{corpus_name}] Training complete. Best eval loss: {best_loss:.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a value-aligned transformer")
    parser.add_argument("--corpus", required=True, help="Corpus cluster name")
    parser.add_argument("--data", required=True, help="Path to tokenized .npy file")
    parser.add_argument("--iters", type=int, default=5000, help="Max training iterations")
    parser.add_argument("--checkpoint_dir", default="models/checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    tc = TrainConfig(max_iters=args.iters, checkpoint_dir=args.checkpoint_dir)
    train(args.corpus, args.data, train_config=tc)
