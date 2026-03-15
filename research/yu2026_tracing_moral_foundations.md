# Tracing Moral Foundations in Large Language Models

**Authors:** Chenxiao Yu, Bowen Yi, Farzan Karimi-Malekabadi, Suhaib Abdurahman, Jinyi Ye, Shrikanth Narayanan, Yue Zhao, Morteza Dehghani
**Year:** 2026
**Venue:** arXiv preprint
**arXiv:** https://arxiv.org/abs/2601.05437

---

## Summary

Addresses whether LLMs genuinely understand morality or merely simulate it. Uses MFT to probe two instruction-tuned LLMs via three mechanistic methods:

1. **Layer-wise analysis** — how moral concepts are represented across network depth
2. **Sparse autoencoders (SAEs)** — identifying sparse features that support moral reasoning
3. **Causal steering** — intervening on value vectors and sparse features to confirm the encoding is causally active, not merely correlated

Key findings: both models represent and distinguish moral foundations in structured, layer-dependent ways. Moral mechanisms are distributed across layers, deepen with network depth, and are partially disentangled — suggesting moral structure emerges from training data statistics rather than hard-coded mechanisms.

## Relevance

This paper provides a multi-scale mechanistic blueprint that directly extends this project's approach. This project currently uses static token embeddings and aggregate geometric measures (cosine similarity, linear probes, Wasserstein distance). Yu et al. demonstrate that layer-wise analysis and causal steering offer complementary evidence at finer granularity. The layer-dependence finding is particularly relevant: moral encoding may concentrate in middle layers of the decoder, suggesting static token embeddings (layer 0) may underestimate the full value structure present in contextual representations.

## Key Takeaways

- Moral foundations are encoded in transformer activations in layer-dependent patterns
- SAEs reveal sparse, interpretable features that support moral reasoning
- Causal steering confirms encoding is active, not merely correlational
- Moral structure emerges from data statistics, not architectural induction
- Layer-wise analysis is essential — aggregate geometric measures miss depth-dependent organization
