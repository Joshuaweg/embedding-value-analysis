# Correcting Biased Centered Kernel Alignment Measures

**Authors:** Multiple (ICLR 2024 Workshop)
**Year:** 2024
**Venue:** ICLR 2024 Workshop on Representational Alignment (Re-Align)
**arXiv:** https://arxiv.org/abs/2405.01012

---

## Summary

Addresses a systematic bias in Centered Kernel Alignment (CKA) when feature dimensions exceed sample count — a common scenario when comparing high-dimensional embedding spaces with limited vocabulary probes. CKA is widely used as a representational similarity metric between neural networks, but the authors show:

- CKA estimates are positively biased when n_samples < n_dimensions
- The bias inflates similarity scores and can produce misleading comparisons
- Corrections exist but require careful application; Procrustes alignment is more reliable in the high-dim/low-sample regime
- Cross-validation and bias-corrected estimators should be used when n < d

## Relevance

Directly applicable to this project's analysis regime: 204 MFT seed words (n_samples) vs. 256-dimensional embeddings (n_features). This is precisely the regime where naive CKA would be biased. Any future extension of this project that uses CKA to compare representational similarity across models should apply the corrections outlined here. More immediately, the paper reinforces the choice of Sliced Wasserstein Distance (which operates on projections and is not subject to the same bias) and Procrustes alignment as the more reliable metrics in this setting.

## Key Takeaways

- CKA is biased when n_features >= n_samples — this project operates in that regime (256 dims, 204 words)
- Procrustes alignment is recommended as a bias-robust alternative
- Sliced Wasserstein Distance (used in this project) avoids CKA's sample-size sensitivity
- Any future CKA analysis should use bias-corrected estimators
