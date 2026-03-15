# The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models

**Authors:** Anton Razzhigaev, Matvey Mikhalchuk, Elizaveta Goncharova, Ivan Oseledets, Denis Dimitrov, Andrey Kuznetsov
**Year:** 2023
**Venue:** EACL 2024
**arXiv:** https://arxiv.org/abs/2311.05928

---

## Summary

Investigates how embedding geometry evolves during transformer training by tracking anisotropy and intrinsic dimensionality across layers and training steps. Key structural findings:

- **Decoder architectures** exhibit a bell-shaped anisotropy curve — highest concentration in middle layers — diverging from encoders, which distribute anisotropy more uniformly
- **Intrinsic dimensionality** follows a two-phase progression: expansion during early training (exploration of feature space) then compression toward completion (consolidation of learned structure)
- These patterns are consistent across model sizes and suggest fundamental architectural differences in how information is organized at different network depths

## Relevance

Directly informs the interpretation of this project's geometric analyses. This project uses static token embeddings (layer 0) from a 6-layer decoder transformer. If value structure follows the bell-shaped layer pattern, the most value-expressive representations may reside in middle layers (layers 3–4) rather than the embedding layer. The intrinsic dimensionality finding also suggests that training on focused value-distinct corpora may produce more compressed embedding geometry than baseline mixed-corpus training — measurable via dimensionality analysis as a complement to the current cosine and Wasserstein metrics.

## Key Takeaways

- Decoder middle layers likely carry the richest representational structure
- Intrinsic dimensionality rises then falls during training — compression signals convergence
- Anisotropy is architecturally shaped, not purely a function of training data
- Layer 0 (static token embeddings) may underrepresent value structure present in higher layers
