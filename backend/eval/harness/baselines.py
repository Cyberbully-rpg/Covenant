"""
Covenant Phase 3, Step 2 — Trivial baseline retrievers.

There is no real retriever until Phase 6, so the scorer needs something to
exercise it. These two exist to bracket the problem before embeddings are
ever built:

  1. RandomRetriever — picks segments at random. Establishes the FLOOR.
     Any real retriever that can't clear this decisively is broken.

  2. LexicalRetriever — TF-IDF over the contract's own segments, scoring
     them against the category probe question. Establishes whether Phase
     6's embedding retrieval is worth its cost: if dense embeddings can't
     beat plain lexical matching by a meaningful margin, the added
     infrastructure (Chroma, embedding model, ingestion pipeline) isn't
     buying anything and that's worth knowing BEFORE building it.

Both return segment spans as (start_char, end_char) offsets into the
original contract text, which is exactly what scorer.score_query() takes —
no adapter layer.

Note on offsets: segments come from the Phase 2 segmenter, whose
start_char/end_char BOUND a segment's extent (they can include stripped
boilerplate — see Segment's docstring). That inflation is small
(p90 8.86%) and, importantly, it inflates both baselines identically, so
it can't skew a comparison between them.
"""

from __future__ import annotations

import random

from sklearn.feature_extraction.text import TfidfVectorizer


class RandomRetriever:
    """Picks k segments uniformly at random. The floor."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def retrieve(self, question: str, segments: list, k: int = 5) -> list[tuple[int, int]]:
        if not segments:
            return []
        chosen = self.rng.sample(segments, min(k, len(segments)))
        return [(s.start_char, s.end_char) for s in chosen]


class LexicalRetriever:
    """
    TF-IDF cosine similarity between the question and each segment.

    Fitted per contract (the corpus is that contract's own segments), which
    is the honest analogue of what Phase 6 will do per-contract, and avoids
    leaking global corpus statistics a real single-document retriever
    wouldn't have.
    """

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 1)):
        self.ngram_range = ngram_range

    def retrieve(self, question: str, segments: list, k: int = 5) -> list[tuple[int, int]]:
        if not segments:
            return []
        texts = [s.embedding_text for s in segments]
        try:
            vec = TfidfVectorizer(
                stop_words="english",
                ngram_range=self.ngram_range,
                min_df=1,
            )
            matrix = vec.fit_transform(texts)
            q = vec.transform([question])
        except ValueError:
            # empty vocabulary (e.g. every segment is stopwords/punctuation)
            return [(s.start_char, s.end_char) for s in segments[:k]]

        sims = (matrix @ q.T).toarray().ravel()
        top = sims.argsort()[::-1][:k]
        return [(segments[i].start_char, segments[i].end_char) for i in top]
