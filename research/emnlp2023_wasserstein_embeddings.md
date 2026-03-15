# A Linear Time Approximation of Wasserstein Distance with Word Embedding Selection

**Authors:** Multiple (EMNLP 2023)
**Year:** 2023
**Venue:** EMNLP 2023
**URL:** https://aclanthology.org/2023.emnlp-main.935/

---

## Summary

Addresses the computational bottleneck of Wasserstein distance (in the Word Mover's Distance formulation) for document similarity, which classically requires O(n³) computation. Proposes a hybrid approach combining:

- Automatic feature selection — identifying which word embeddings are most informative for the transport problem
- Tree-based approximation of the transport plan — reducing complexity to linear time

Achieves linear time complexity while maintaining competitive accuracy on document classification and retrieval tasks. The feature selection component is particularly notable: it reveals that not all embedding dimensions contribute equally to distributional divergence; focusing on the most transport-relevant dimensions improves both speed and interpretability.

## Relevance

Directly informs the Wasserstein analysis in this project. The feature selection finding suggests that the current SWD analysis — which projects uniformly onto random directions — could be refined by projecting preferentially onto directions most relevant to value content (e.g., directions identified via linear probes for each MFT foundation). This would produce a *value-aware* Wasserstein distance that measures divergence specifically in the subspace where moral foundations are encoded, rather than aggregating over all 256 dimensions equally.

## Key Takeaways

- Not all embedding dimensions contribute equally to Wasserstein transport — feature selection matters
- Linear-time Wasserstein approximation is feasible without significant accuracy loss
- Combining probe-identified value directions with SWD could produce a value-aware distributional distance metric
- Selective projection is a natural bridge between linear probe analysis and distributional comparison
