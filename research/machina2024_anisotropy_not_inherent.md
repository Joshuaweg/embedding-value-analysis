# Anisotropy is Not Inherent to Transformers

**Authors:** Anemily Machina, Robert Mercer
**Year:** 2024
**Venue:** NAACL 2024
**URL:** https://aclanthology.org/2024.naacl-long.274/

---

## Summary

Challenges the widely-held assumption that transformer embedding spaces are inherently anisotropic — that vectors cluster in a narrow cone of the representational space regardless of training. By analyzing large Pythia models, the authors identify transformers that achieve isotropic embeddings, specifically where final Layer Normalization optimization produces uniform distributional properties across directions.

The central claim: isotropy/anisotropy is not an architectural inevitability but an optimization outcome. Previous theoretical justifications for unavoidable anisotropy are incomplete. Isotropy develops during training and can be shaped by objective and regularization choices.

## Relevance

This paper reframes a potential confound in this project's analysis. If anisotropy were universal and architecture-determined, differences in cosine similarity structure across models would be harder to attribute to corpus content. Machina & Mercer's finding that isotropy is trainable means that the geometric differences observed between value-distinct models could partly reflect anisotropy variation — another measurable signal beyond mean cosine similarity and Wasserstein distance. It also motivates testing whether anisotropy correction (mean-centering or whitening) changes the pattern of inter-model differences observed in this project.

## Key Takeaways

- Transformer embedding anisotropy is not architecturally fixed — it is trainable
- Final Layer Normalization choices strongly influence distributional geometry
- Isotropic models exist and their geometry is qualitatively different from anisotropic ones
- Anisotropy correction before computing similarity metrics may reveal cleaner value structure
