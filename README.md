# Value Embedding Research

> **Status: Proof of Concept.** The current experiments establish the pipeline and demonstrate feasibility at small scale. More rigorous experimentation — controlled corpus design, larger models, statistical significance testing, and expanded analysis methods — is forthcoming.

Do models trained on texts with distinct value systems develop measurably different embedding spaces? This project tests that hypothesis using small transformers trained from scratch on value-distinct literary corpora, comparing their embedding geometry against each other and against GPT-2.

The core claim: embedding geometry is a more direct signal of value expression than post-hoc lexical scoring — because it reflects what the model learned to associate, not what we observe after the fact.

---

## Definition of Values (Operational)

> A value is a **cross-contextually stable evaluative orientation** toward a class of acts, concepts, or states — expressible in natural language and detectable as systematic geometric structure in a model's embedding space.

This definition is intentionally narrow in scope. It makes no claim about whether values are moral, aesthetic, epistemic, or practical in nature, nor does it assert a boundary between values and law — that boundary is itself an open empirical question this research may be able to test (see [Next Steps](#next-steps)). What it does assert is that values have three properties that make them tractable to study here:

1. **Expressible in language** — values can be anchored by seed words and probed through text corpora
2. **Cross-contextually stable** — a value orients consistently across situations rather than being situationally determined; this is what distinguishes a value from a preference or heuristic
3. **Geometrically detectable** — if a corpus encodes a value, that value should manifest as structure in the model's embedding space, measurable via clustering, linear separability, and distributional distance

An important constraint: this research tests what a model *learned to associate*, not how a model *behaves*. Embedding geometry is a necessary but not sufficient condition for value expression. A model whose embedding space cleanly separates care from harm has encoded something — but whether that encoding shapes generation or downstream behavior is a separate question, and one for future work.

---

## Approach

Five corpora were assembled around distinct value orientations drawn from [Moral Foundations Theory](https://moralfoundations.org/) (MFT). A small transformer (~6M parameters) was trained from scratch on each corpus, plus a baseline trained on mixed text. Embeddings for 204 MFT seed words were extracted from every model and compared geometrically.

| Corpus | Texts | Value Orientation |
|---|---|---|
| **religious** | Bible (KJV), Milton's *Paradise Lost* | Sanctity, Authority, Care |
| **transcendentalist** | Whitman's *Leaves of Grass*, Blake's *Poems* | Liberty, individual moral vision |
| **victorian** | Austen's *Emma*, *Persuasion*, *Sense & Sensibility* | Fairness, social propriety, Loyalty |
| **philosophical** | Chesterton (3 works), Shakespeare (*Hamlet*, *Macbeth*, *Julius Caesar*) | Moral ambiguity, Authority, conflict |
| **baseline** | Full NLTK Gutenberg + Brown corpus | Mixed / no value signal |

GPT-2 (pretrained on WebText) is included as an external reference point.

---

## Model Architecture

`ValueTransformer` — a NanoGPT-style decoder-only transformer:

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| Embedding dimension | 256 |
| Attention heads | 8 |
| FFN dimension | 1024 |
| Context window | 256 tokens |
| Parameters | ~6M |
| Tokenizer | BPE, vocab 12k, trained on combined corpus |

Weight tying is used between the token embedding and the LM head.

---

## Repository Structure

```
value_embedding/
├── src/
│   ├── model.py          # ValueTransformer architecture
│   ├── train.py          # Training loop (cosine LR schedule)
│   ├── tokenizer.py      # BPE tokenizer training & encoding
│   ├── corpus.py         # NLTK corpus loading and clustering
│   ├── embeddings.py     # Embedding extraction (custom models + GPT-2)
│   └── analyze.py        # Similarity, PCA, UMAP, linear probes, Wasserstein
├── scripts/
│   ├── prepare_data.py   # Download corpora, train tokenizer, tokenize
│   ├── run_training.py   # Train all 5 models sequentially
│   ├── run_analysis.py   # Extract embeddings, run full comparison pipeline
│   └── run_wasserstein.py# Pairwise Sliced Wasserstein Distance analysis
├── value_lexicons/
│   └── mft_words.py      # 204 MFT seed words across 6 foundations
├── plans/
│   └── research_plan.md  # Design notes and corpus rationale
└── requirements.txt
```

Generated artifacts (`data/`, `models/`, `analysis/`) are excluded from version control and can be fully recreated by running the scripts below.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ required. A CUDA-capable GPU is recommended for training but not required.

---

## Reproducing the Experiment

Run the three scripts in order:

```bash
# 1. Download corpora, train BPE tokenizer, tokenize each corpus
python scripts/prepare_data.py

# 2. Train one ValueTransformer per corpus (5 models + baseline)
python scripts/run_training.py

# 3. Extract embeddings, run geometric analysis, generate figures
python scripts/run_analysis.py

# 4. Compute pairwise Sliced Wasserstein Distances
python scripts/run_wasserstein.py
```

Outputs are written to:
- `models/checkpoints/<corpus>/` — `best.pt`, `final.pt`, `training_curve.json`
- `analysis/embeddings/` — `.npz` files, one per model (204 MFT words × embed_dim)
- `analysis/figures/` — PCA, UMAP, similarity heatmaps, Wasserstein heatmaps
- `analysis/results/` — `metrics.json`, `wasserstein_pairwise.json`, `wasserstein_by_foundation.json`

---

## Analysis Methods

### Cosine Similarity
Pairwise cosine similarity between MFT word embeddings within each model. Aggregated into 6×6 foundation-level heatmaps to show which moral foundations a model treats as semantically related.

### PCA / UMAP Projections
2D projections of all 204 value word embeddings, colored by MFT foundation. Cluster separation indicates how distinctly a model encodes foundation boundaries.

### Linear Probe
Logistic regression trained on word embeddings to predict MFT foundation (6-class, 5-fold cross-validation). Accuracy indicates how much foundation structure is encoded in the embedding geometry.

### Sliced Wasserstein Distance
For each pair of models, Sliced Wasserstein Distance (SWD) is estimated using 1,000 random 1D projections. SWD treats each model's embedding set as a distribution in R^d and measures distributional divergence without assuming a parametric form. When models have different embedding dimensions (e.g. GPT-2 at 768 vs custom models at 256), both are independently PCA-projected to the lower dimensionality before comparison.

Per-foundation SWD is also computed, isolating how much each moral foundation contributes to model divergence.

---

## Results Summary

### MFT Linear Probe Accuracy (6-class, chance = 16.7%)

| Model | Probe Accuracy | Mean Cosine Sim |
|---|---|---|
| GPT-2 | **52.9%** | 0.397 |
| Baseline | 37.3% | 0.336 |
| Religious | 33.8% | 0.371 |
| Victorian | 25.0% | 0.313 |
| Philosophical | 25.5% | 0.226 |
| Transcendentalist | 24.1% | 0.170 |

All models exceed chance, suggesting MFT structure is encoded across value-distinct corpora. GPT-2's higher accuracy likely reflects its larger embedding dimension and training data scale rather than targeted value learning.

### Pairwise Sliced Wasserstein Distance

|  | baseline | gpt2 | philosophical | religious | transcendentalist | victorian |
|---|---|---|---|---|---|---|
| **baseline** | 0.000 | 0.117 | 0.033 | 0.036 | 0.036 | 0.041 |
| **gpt2** | 0.117 | 0.000 | 0.104 | 0.106 | 0.096 | 0.096 |
| **philosophical** | 0.033 | 0.104 | 0.000 | 0.034 | 0.035 | 0.041 |
| **religious** | 0.036 | 0.106 | 0.034 | 0.000 | 0.041 | 0.045 |
| **transcendentalist** | 0.036 | 0.096 | 0.035 | 0.041 | 0.000 | 0.042 |
| **victorian** | 0.041 | 0.096 | 0.041 | 0.045 | 0.042 | 0.000 |

GPT-2 is a clear distributional outlier (SWD ~0.10–0.12 vs all custom models). Among the custom models, distances are tighter (0.033–0.045), consistent with shared architecture and tokenizer. Victorian and religious embeddings show the largest divergence among the value-aligned models.

---

## MFT Lexicon

204 seed words across 6 foundations, each with virtue and vice poles:

| Foundation | Example Virtues | Example Vices |
|---|---|---|
| Care | compassion, nurture, mercy | harm, cruel, torment |
| Fairness | just, equal, honest | cheat, deceive, corrupt |
| Loyalty | loyal, devoted, fellowship | traitor, forsake, treacherous |
| Authority | obey, tradition, discipline | rebel, anarchy, defy |
| Sanctity | pure, sacred, righteous | sin, defile, profane |
| Liberty | free, autonomous, emancipate | oppress, tyranny, enslave |

---

## Next Steps

### Corpus Expansion
- **Utilitarian ethics:** Mill's *Utilitarianism* and *On Liberty* (Project Gutenberg IDs 11224, 34901) — expected to cluster strongly around Fairness and Liberty
- **Eastern philosophy:** Tao Te Ching (917), Dhammapada (2017) — likely to produce distinctive sanctity geometry not captured by Western texts
- **Political philosophy:** Locke's *Two Treatises* (7370), *Communist Manifesto* (61) — opposing Liberty/Authority orientations make these a natural contrastive pair
- **Legal corpora:** U.S. constitutional documents and case law — Authority-heavy domain with minimal narrative framing

### Richer Embedding Analysis
- **Contextual embeddings over static:** replace token-embedding extraction with averaged last-layer hidden states from in-domain contexts, capturing how value words behave in sentence position rather than as isolated tokens
- **Analogy probes:** test whether value analogies (e.g. `loyal - loyalty + betrayal ≈ traitor`) are preserved or distorted across models
- **Anisotropy correction:** apply mean-centering or whitening before computing similarities — raw transformer embeddings are known to cluster in a narrow cone, which can inflate cosine similarity uniformly

### Distributional Analysis
- **Full 2-Wasserstein (Bures metric):** complement SWD with the closed-form Gaussian W2 using regularized covariance estimates (e.g. Ledoit-Wolf shrinkage), providing a lower bound and allowing decomposition into mean-shift vs. covariance-shift components
- **Per-word displacement:** for each MFT word, measure how far its embedding moves across models in a shared projected space — identify which specific words are most sensitive to value training signal
- **Foundation-level distributional shift:** extend per-foundation SWD into a structured transport plan to see which foundation pairs "move toward" each other under each value corpus

### Probing & Interpretability
- **Directional value probes:** learn a linear direction per foundation (virtue pole - vice pole centroid) and measure how each model's embedding space aligns that axis with the MFT structure
- **Causal intervention:** ablate attention heads and measure degradation in probe accuracy to locate where value structure is stored in the network
- **Cross-model token alignment:** use Procrustes alignment or CKA (Centered Kernel Alignment) to separate architecture-induced geometry from corpus-induced geometry

### Scaling
- **Model size sweep:** train at multiple scales (1M, 6M, 25M params) on the same corpora to test whether value encoding sharpens, dilutes, or shifts with capacity
- **Training data size control:** hold model size constant and vary corpus size to disentangle data quantity effects from value-content effects
- **Fine-tuning from GPT-2:** compare geometry of fine-tuned GPT-2 vs. trained-from-scratch models to test whether pretrained representations resist value-specific reorganization
