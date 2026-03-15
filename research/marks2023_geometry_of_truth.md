# The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations

**Authors:** Samuel Marks, Max Tegmark
**Year:** 2023
**Venue:** arXiv preprint
**arXiv:** https://arxiv.org/abs/2310.06824

---

## Summary

Demonstrates that LLMs linearly encode truth — whether a statement is true or false manifests as a consistent linear direction in embedding space that generalizes across datasets, domains, and tasks without retraining. Three converging forms of evidence:

1. **Visualizations** — clear linear separation between true and false statement representations
2. **Transfer experiments** — probes trained on one dataset generalize to entirely different datasets, confirming the direction is universal rather than dataset-specific
3. **Causal interventions** — surgically rotating the truth direction in the forward pass causes the model to treat false statements as true and vice versa, confirming causality rather than mere correlation

A key finding: simple difference-in-means probes match the performance of complex nonlinear classifiers, suggesting truth is genuinely linearly encoded rather than entangled in ways that require nonlinear extraction.

## Relevance

This is foundational methodological precedent for this project. If a high-level evaluative property like truthfulness is linearly encoded in embedding space, the same principle should extend to moral value orientation — which is similarly a high-level evaluative property. The linear probe analysis in this project directly parallels Marks & Tegmark's approach applied to MFT foundations rather than truth. The causal intervention finding also motivates a next step: do analytic directions identified in this project's embedding space causally control value-relevant generation, or are they merely diagnostic?

## Key Takeaways

- High-level evaluative properties (truth) are linearly encoded in transformer representations
- Simple probes generalize across domains — the linear structure is universal, not task-specific
- Causal interventions confirm encoding is mechanistically active, not correlational
- This establishes the theoretical basis for expecting moral values to be linearly detectable in embedding geometry
