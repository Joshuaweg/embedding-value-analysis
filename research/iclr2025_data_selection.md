# Data Selection via Optimal Control for Language Models

**Authors:** Multiple (ICLR 2025)
**Year:** 2025
**Venue:** ICLR 2025
**arXiv:** https://arxiv.org/abs/2410.07064

---

## Summary

Proposes data selection methods for language model training that account for long-range training dynamics via an optimal control perspective — moving beyond static corpus metrics (deduplication, perplexity filtering) to dynamic optimization that considers how data composition shapes representation trajectories over training. Key contributions:

- Frames data curation as a control problem over the representation learning trajectory
- Shows that training data heterogeneity manifests as distributional disparities in learned features
- Demonstrates that corpus composition actively and predictably shapes the geometry of final representations
- Documents that these effects are not incidental but fundamental to how models learn

## Relevance

Provides theoretical grounding for this project's central design decision: that value-distinct corpora will produce measurably distinct embedding geometries. Rather than assuming this relationship, the data selection literature establishes it as a predictable consequence of how representation learning works. The optimal control framing also motivates a future research direction: rather than training on complete corpora and measuring the resulting geometry, could one select or weight training examples to produce a *target* value geometry?

## Key Takeaways

- Training data composition predictably and systematically shapes embedding geometry
- The corpus-geometry relationship is not incidental — it follows from representation learning dynamics
- Data heterogeneity produces distributional disparities in learned features (directly measurable via Wasserstein distance)
- Data selection as optimal control suggests corpus design is a lever for intentional value geometry engineering
