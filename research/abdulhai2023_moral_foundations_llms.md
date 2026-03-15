# Moral Foundations of Large Language Models

**Authors:** Marwa Abdulhai, Gregory Serapio-García, Clément Crepy, Daria Valter, John Canny, Natasha Jaques
**Year:** 2023
**Venue:** EMNLP 2024
**arXiv:** https://arxiv.org/abs/2310.15337

---

## Summary

Applies Moral Foundations Theory to analyze whether LLMs exhibit systematic biases toward particular moral value systems. The core finding is that models trained on internet text internalize moral orientations that correlate with political ideology — a consequence of the demographic and ideological composition of the training corpus. These biases are consistent across different prompting strategies, and adversarial prompt engineering can shift which moral foundations a model emphasizes, with measurable downstream effects on task performance.

The key contribution is demonstrating that models implicitly encode moral stances without explicit alignment training — training corpus composition directly shapes which values a model prioritizes.

## Relevance

This is the closest prior behavioral analog to this project's approach. Where Abdulhai et al. measure value expression through model outputs, this project measures it through embedding geometry. Both treat training corpus as the causal variable and MFT as the analytic framework. The gap this project addresses: output-level measurement cannot distinguish whether values are represented internally or merely mimicked at the surface. Embedding geometry offers a more direct window into internal representation.

## Key Takeaways

- LLMs encode MFT-structured moral biases as a byproduct of corpus composition
- These biases are consistent across prompting variations, suggesting stable internal encoding
- Corpus content is the primary driver — not architecture or objective alone
- Behavioral measurement cannot confirm whether encoding is deep (geometric) or shallow (lexical)
