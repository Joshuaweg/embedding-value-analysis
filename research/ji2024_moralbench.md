# MoralBench: Moral Evaluation of LLMs

**Authors:** Jianchao Ji, Yutong Chen, Mingyu Jin, Wujiang Xu, Wenyue Hua, Yongfeng Zhang
**Year:** 2024
**Venue:** arXiv preprint
**arXiv:** https://arxiv.org/abs/2406.04428

---

## Summary

Introduces MoralBench, a benchmark for evaluating moral reasoning in LLMs using MFT as the organizing framework. The benchmark accounts for contextual nuance and alignment with human ethical standards, developed in consultation with ethics scholars. Key contributions:

- Structured evaluation across all six MFT foundations with scenario-based prompts
- Statistical methodology for measuring inter-model and intra-model variation in moral judgments
- Testing across multiple leading models reveals significant variance in moral reasoning profiles

Results show that models differ meaningfully in their MFT foundation profiles — some score higher on care/fairness, others on authority/loyalty — and these differences are consistent enough to treat as stable model characteristics.

## Relevance

MoralBench provides behavioral ground truth that could validate or challenge this project's geometric findings. If a model scores high on sanctity in MoralBench outputs, its embedding space should show tighter clustering of sanctity-related words and larger Wasserstein distance from models that score low on sanctity. Cross-validating embedding geometry against behavioral benchmarks like MoralBench would strengthen the claim that geometry is a meaningful signal of value expression rather than an artifact of corpus statistics.

## Key Takeaways

- LLMs have measurably distinct and stable moral foundation profiles at the output level
- MFT provides a consistent framework for cross-model moral comparison
- Behavioral benchmarks exist that could serve as validation targets for geometric analysis
- Embedding geometry predictions should correlate with MoralBench profiles if the geometric approach is valid
