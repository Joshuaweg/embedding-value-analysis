# A Survey on Moral Foundation Theory and Pre-Trained Language Models

**Authors:** Multiple (comprehensive survey)
**Year:** 2024
**Venue:** AI & Society (Springer Nature)
**arXiv:** https://arxiv.org/abs/2409.13521

---

## Summary

An authoritative survey integrating Moral Foundations Theory with recent advances in pre-trained language models. Covers three generations of approach:

1. **Dictionary-based methods** — lexicon scoring using MFT word lists (the approach this project uses as a starting point)
2. **Neural classification** — BERT/RoBERTa fine-tuned for foundation detection on labeled corpora
3. **Emerging methods** — zero-shot and few-shot approaches using GPT-family models, and early work on mechanistic analysis

Identifies a recurring challenge: traditional NLP approaches measure output-level signals and cannot distinguish genuine moral understanding from behavioral pattern-matching. Highlights the gap between lexical scoring (what words appear) and representational encoding (how concepts are organized internally).

## Relevance

This survey is useful both as a literature map and as a positioning document. It confirms that the lexical/dictionary approach used in this project (MFT seed words as probes) is well-established, and explicitly names the gap this project aims to address: moving from output-level scoring to representational geometry. The survey's coverage of neural classification methods also points toward a direct comparison opportunity — comparing linear probe performance from this project against fine-tuned BERT classifiers on the same foundation taxonomy.

## Key Takeaways

- Lexicon-based MFT analysis is mature but limited to surface measurement
- Neural classifiers improve coverage but still operate at output level
- The field has not systematically studied how MFT structure is encoded in embedding geometry
- This project's approach addresses an explicitly named gap in the survey's research agenda
