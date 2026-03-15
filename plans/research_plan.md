# Value Embedding Research — Implementation Plan

## Context
Existing AI alignment research measures values (e.g. via Moral Foundations Theory) but doesn't clearly show whether those measurements reflect how values are *encoded* in a model's representational geometry. The hypothesis here is that a model trained predominantly on texts of a particular value system will develop a measurably different embedding space — and that this difference is a more direct signal of value expression than post-hoc lexical scoring.

The approach: train multiple small transformers from scratch on value-distinct corpora, extract their embedding spaces, and compare geometric structure across models and against a GPT-2 baseline.

## Corpora

### NLTK Gutenberg (starting point)
| Value Cluster | Texts |
|---|---|
| Religious / Theological | `bible-kjv.txt`, `milton-paradise.txt` |
| Transcendentalist / Individual | `whitman-leaves.txt`, `blake-poems.txt` |
| Victorian Social / Moral | `austen-emma.txt`, `austen-persuasion.txt`, `austen-sense.txt` |
| Philosophical / Moral Struggle | `chesterton-ball.txt`, `chesterton-brown.txt`, `chesterton-thursday.txt`, `shakespeare-hamlet.txt`, `shakespeare-macbeth.txt`, `shakespeare-caesar.txt` |
| Baseline | All NLTK Gutenberg combined + Brown corpus |

### Supplementary via `gutenbergpy` (phase 2 expansion)
- Utilitarian: Mill's Utilitarianism (ID: 11224), On Liberty (34901)
- Eastern: Tao Te Ching (917), Dhammapada (2017)
- Political: Communist Manifesto (61), Locke's Two Treatises (7370)

## Value Lexicon (Moral Foundations Theory)
| Foundation | Poles |
|---|---|
| Care | care, compassion, harm, hurt, protect |
| Fairness | fair, equal, just, cheat, betray |
| Loyalty | loyal, faithful, traitor, duty |
| Authority | authority, obey, submit, rebel |
| Sanctity | pure, holy, sacred, corrupt, sin |
| Liberty | free, liberty, oppress, tyranny |

## Model Architecture
- 6 layers, 256 embed_dim, 8 heads, 1024 ffn_dim, 256 context window (~6M params)
- Shared BPE tokenizer (vocab 8k-16k) trained on combined corpus
- One model per value cluster + one baseline

## Analysis Methods
1. Geometric clustering: PCA/UMAP of value-word embeddings per model
2. Cosine similarity matrices: value word pair distances across models
3. Linear probes: classify MFT foundation from embeddings, compare accuracy across models
