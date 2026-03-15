# Related Research

Papers relevant to value embedding analysis, embedding geometry, Moral Foundations Theory in NLP, and mechanistic interpretability. Each file contains a summary and notes on relevance to this project.

---

## Value Alignment & Moral Foundations in LLMs

| File | Paper | Relevance |
|---|---|---|
| [jin2025_conva.md](jin2025_conva.md) | Jin et al. (ACL 2025) — *ConVA: Controlled Value Vector Activation* | Validates that values are geometrically encoded in LLM activations; activation steering as complement to embedding analysis |
| [abdulhai2023_moral_foundations_llms.md](abdulhai2023_moral_foundations_llms.md) | Abdulhai et al. (EMNLP 2024) — *Moral Foundations of LLMs* | Closest behavioral analog: shows corpus composition drives MFT biases in model outputs |
| [yu2026_tracing_moral_foundations.md](yu2026_tracing_moral_foundations.md) | Yu et al. (2026) — *Tracing Moral Foundations in LLMs* | Layer-wise + SAE + causal steering analysis of moral encoding; methodological blueprint for extensions |
| [simmons2023_moral_mimicry.md](simmons2023_moral_mimicry.md) | Simmons (ACL 2023) — *Moral Mimicry* | Defines the null hypothesis: models may reproduce value patterns via surface mimicry; geometric analysis tests against this |
| [ji2024_moralbench.md](ji2024_moralbench.md) | Ji et al. (2024) — *MoralBench* | Behavioral MFT benchmark; potential validation target for geometric predictions |
| [survey2024_mft_plms.md](survey2024_mft_plms.md) | Survey (AI & Society 2024) — *MFT and Pre-Trained Language Models* | Maps the field; confirms the lexicon-to-geometry gap this project addresses |

## Embedding Geometry

| File | Paper | Relevance |
|---|---|---|
| [marks2023_geometry_of_truth.md](marks2023_geometry_of_truth.md) | Marks & Tegmark (2023) — *The Geometry of Truth* | Establishes linear encoding of high-level evaluative properties (truth); foundational precedent for value geometry |
| [razzhigaev2023_shape_of_learning.md](razzhigaev2023_shape_of_learning.md) | Razzhigaev et al. (EACL 2024) — *The Shape of Learning* | Anisotropy bell curve in decoders; intrinsic dimensionality dynamics; motivates layer-wise analysis |
| [machina2024_anisotropy_not_inherent.md](machina2024_anisotropy_not_inherent.md) | Machina & Mercer (NAACL 2024) — *Anisotropy is Not Inherent* | Isotropy is trainable; anisotropy correction may reveal cleaner value structure |
| [mikolov2013_word2vec.md](mikolov2013_word2vec.md) | Mikolov et al. (ICLR 2013) — *word2vec* | Foundational: semantic content decomposes into linear vector geometry via corpus statistics |

## Political & Evaluative Stance in Representations

| File | Paper | Relevance |
|---|---|---|
| [kim2025_linear_political.md](kim2025_linear_political.md) | Kim et al. (2025) — *Linear Representations of Political Perspective* | Strongest methodological parallel: political ideology linearly encoded in activations, transferable, causally steerable |

## Distributional Distance & Optimal Transport

| File | Paper | Relevance |
|---|---|---|
| [emnlp2023_wasserstein_embeddings.md](emnlp2023_wasserstein_embeddings.md) | EMNLP 2023 — *Linear Time Wasserstein with Embedding Selection* | Feature selection for Wasserstein; motivates value-aware SWD using probe-identified directions |

## Mechanistic Interpretability

| File | Paper | Relevance |
|---|---|---|
| [mechanistic2025_sae_social.md](mechanistic2025_sae_social.md) | 2025 — *SAEs for Socially-Relevant Concepts* | Sparse autoencoders as complement to linear probes; fine-grained feature identification for moral foundations |

## Representational Similarity & Data Effects

| File | Paper | Relevance |
|---|---|---|
| [klabunde2024_cka_bias.md](klabunde2024_cka_bias.md) | ICLR 2024 — *Correcting Biased CKA* | CKA bias in high-dim/low-sample regime (this project's exact setting); validates choice of SWD and Procrustes |
| [iclr2025_data_selection.md](iclr2025_data_selection.md) | ICLR 2025 — *Data Selection via Optimal Control* | Corpus composition predictably and fundamentally shapes embedding geometry |
