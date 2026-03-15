# Moral Mimicry: Large Language Models Produce Moral Rationalizations Tailored to Political Identity

**Authors:** Gabriel Simmons
**Year:** 2023
**Venue:** ACL 2023 Student Research Workshop
**URL:** https://aclanthology.org/2023.acl-srw.40/

---

## Summary

Demonstrates that GPT-3/3.5 and OPT exhibit "moral mimicry" — the ability to reproduce moral reasoning patterns associated with political identities (liberal, conservative, libertarian, etc.) when prompted with identity anchors. Using MFT as the analytic lens, the paper shows that:

- Models generate text reflecting different moral foundation emphases depending on the identity anchor provided
- This behavior is robust across prompting strategies and models
- The term "moral mimicry" captures the key open question: does the model genuinely encode the value system, or is it retrieving and recombining surface patterns without internal representation?

## Relevance

Moral mimicry names the central interpretive problem that motivates this project. If a model can produce text that sounds care-aligned simply by imitating care-aligned authors, then output-level analysis cannot distinguish genuine encoding from retrieval. This project's hypothesis is that embedding geometry provides a test: a model that merely mimics care should not show systematically different geometric structure in its care-related embeddings compared to a model that mimics authority. If geometry *does* differ across value-distinct training corpora, that is evidence against pure mimicry. Simmons' framework thus provides a key falsifiability criterion for this research.

## Key Takeaways

- LLMs can reproduce culturally-specific moral reasoning via surface pattern matching
- Output-level analysis cannot distinguish mimicry from genuine encoding
- "Moral mimicry" defines the null hypothesis this project's geometric approach is designed to test
- Embedding geometry that varies with training corpus supports genuine encoding over mimicry
