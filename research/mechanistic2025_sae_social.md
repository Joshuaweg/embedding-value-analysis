# Mechanistic Interpretability with SAEs: Probing Socially-Relevant Concepts

**Authors:** Multiple
**Year:** 2025
**Venue:** arXiv preprint
**arXiv:** https://arxiv.org/abs/2509.17665

---

## Summary

Uses Sparse Autoencoders (SAEs) to identify interpretable features encoding socially-relevant concepts (religion, violence, geography, political stance) in transformer activations. SAEs decompose dense activation vectors into sparse, human-interpretable "features" — each feature activates on a narrow class of inputs that often align with recognizable concepts.

Key findings:

- Many sparse features align with specific semantic or evaluative categories detectable by humans
- Multiple concepts can coexist in the same activation space without strong interference (partial disentanglement)
- Causal intervention on individual features allows surgical control over concept-specific outputs
- The approach scales to socially sensitive categories, including moral and value-laden content

## Relevance

SAEs provide a complementary analysis layer that this project's current approach does not include. Where linear probes identify aggregate foundation structure (is sanctity separable from care?), SAEs could reveal the individual features that constitute each foundation's representation — the precise neurons or directions most responsible for encoding "holy," "sacred," "pure" vs. "corrupt," "defile," "sin." This finer granularity would answer *how* value foundations are encoded, not just *whether* they are. Combined with the current probe and Wasserstein analysis, SAEs would provide a multi-scale picture of moral encoding across the ValueTransformer architecture.

## Key Takeaways

- SAEs decompose dense moral representations into sparse, interpretable features
- Value-relevant concepts can be localized to specific sparse features with causal effect
- SAEs complement linear probes — probes identify aggregate structure, SAEs identify constituent features
- Social concept analysis via SAEs is feasible and extends naturally to moral foundations
