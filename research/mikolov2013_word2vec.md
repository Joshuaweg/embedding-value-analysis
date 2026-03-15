# Efficient Estimation of Word Representations in Vector Space (word2vec)

**Authors:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
**Year:** 2013
**Venue:** ICLR 2013
**arXiv:** https://arxiv.org/abs/1301.3781

---

## Summary

The foundational paper establishing that semantic content decomposes into vector geometry. Introduces Skip-gram and CBOW models trained via simple unsupervised prediction objectives, demonstrating that learned word embeddings exhibit linear structure preserving semantic relationships:

- `king - man + woman ≈ queen`
- `Paris - France + Germany ≈ Berlin`

Despite the simplicity of the learning objective — predict neighboring words — rich semantic and syntactic structure emerges in the geometry of the learned space. This is the earliest evidence that meaning organizes itself geometrically in learned representations, and that the geometry is regular enough to support arithmetic operations.

## Relevance

This paper is the theoretical and historical foundation for this entire project. The word2vec findings establish the precedent: if semantic content (lexical meaning) decomposes into vector geometry via corpus statistics, then value content — which is high-level evaluative semantic structure embedded in the same corpora — should also decompose into geometry. This project extends the word2vec intuition from lexical semantics to moral value orientation: just as `king` and `queen` are geometrically related in a word-prediction model, `care` and `compassion` should be geometrically proximate in a model trained on care-dominant text, and that proximity should shift predictably when the training corpus changes.

## Key Takeaways

- Semantic structure decomposes into linear vector geometry via corpus statistics alone
- Simple unsupervised objectives are sufficient to produce rich representational structure
- Geometric analogies are preserved — suggesting arithmetic structure in semantic space
- Provides the foundational precedent for expecting value structure to be geometrically encoded in transformer embeddings
