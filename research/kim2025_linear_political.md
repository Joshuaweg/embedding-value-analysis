# Linear Representations of Political Perspective Emerge in Large Language Models

**Authors:** Junsol Kim, James Evans, Aaron Schein
**Year:** 2025
**Venue:** arXiv preprint
**arXiv:** https://arxiv.org/abs/2503.02080

---

## Summary

Demonstrates that LLMs encode political ideology as linear directions in attention head activations. Researchers prompted models to generate text from the perspectives of U.S. lawmakers and identified which attention heads predicted DW-NOMINATE ideology scores (a standard political science measure). Key findings:

1. LLMs possess linear representations of political perspective — ideologically similar viewpoints cluster geometrically
2. Predictive attention heads concentrate in middle layers, consistent with existing anisotropy research showing middle layers carry the richest conceptual structure
3. Probes trained on lawmakers' ideology transfer successfully to predicting news outlets' political slant — the encoding is universal, not identity-specific
4. Linear activation interventions can steer model output toward liberal or conservative positions

The work treats political ideology — a high-level evaluative stance — as a geometric property of representations, directly paralleling this project's treatment of moral value orientation.

## Relevance

This is the strongest direct methodological parallel to this project. Political ideology and moral value orientation are both high-level evaluative stances encoded in training text. Kim et al. demonstrate that political ideology is linearly recoverable from activations, transfers across contexts, and is causally steerable. This project makes the analogous claim for MFT foundations. The layer concentration finding (middle layers most predictive) strongly suggests this project should extend analysis beyond token embeddings to contextual representations at layers 3–4 of the ValueTransformer.

## Key Takeaways

- Political ideology (analogous to moral value orientation) is linearly encoded in activations
- Middle layers are most predictive of evaluative stance — token embeddings alone may miss the richest signal
- Linear probes transfer across context types, confirming universality of the encoding
- Causal intervention via activation rotation is feasible and confirms causality
