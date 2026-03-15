# Internal Value Alignment in Large Language Models through Controlled Value Vector Activation (ConVA)

**Authors:** Haoran Jin, Meng Li, Xiting Wang, Zhihao Xu, Minlie Huang, Yantao Jia, Defu Lian
**Year:** 2025
**Venue:** ACL 2025
**arXiv:** https://arxiv.org/abs/2507.11316

---

## Summary

ConVA (Controlled Value Vector Activation) proposes aligning LLMs with human values by directly identifying and modifying how values are encoded in latent representations — rather than through post-hoc behavioral alignment via fine-tuning or RLHF. The method interprets internal mechanisms using gated activation surgery to steer the model toward a target value orientation.

Experiments across ten fundamental moral values show ConVA achieves high control success while preserving generation fluency. Critically, the alignment persists under adversarial prompts designed to elicit opposite value expressions, suggesting the method captures genuine internal encoding rather than surface behavioral patterns.

## Relevance

This is among the closest prior work to this project's central hypothesis. ConVA implicitly validates that values like care, fairness, loyalty, authority, sanctity, and liberty are geometrically encoded in model activations — and that this encoding is surgically addressable. Where ConVA intervenes in activations, this project measures the geometry of static token embeddings as shaped by training corpus composition. The two approaches are complementary: ConVA asks *can we change value geometry?* — this project asks *does training data determine value geometry in the first place?*

## Key Takeaways

- Values are geometrically encoded in latent activations, not just surface outputs
- Gated activation steering can reliably shift value orientation without retraining
- Alignment robustness under adversarial prompts supports genuine encoding vs. mimicry
- Provides activation-level precedent for the embedding-level analysis in this project
