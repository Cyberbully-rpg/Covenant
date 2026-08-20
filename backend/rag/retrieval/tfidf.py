"""
Covenant Phase 6 — lexical (TF-IDF) retriever.

The stronger half of the default retrieval path, but no longer the whole
of it. ARCHITECTURE.md §6D.2 measured dense MiniLM losing to per-contract
TF-IDF by −0.1116 hit_rate@5 and concluded TF-IDF should ship alone;
§6D.3 then found most of that gap was a silent truncation confound, and
that RRF fusion of the two beats either alone (0.7195 vs this retriever's
0.6934). Default ranking into Phase 7 is therefore `hybrid.py`, with this
class as its lexical input and Chroma as its dense one.

This retriever remains directly useful on its own: it is the control that
every variant is measured against, and it is the fallback path if the
Chroma collection is unavailable.

Interface matches Phase 3's LexicalRetriever: per-contract fit on that
contract's own segments, returning (start_char, end_char) spans. A second
method, retrieve_with_text, is what Phase 7 generation will call.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 1)):
        self.ngram_range = ngram_range

    def _rank(self, question: str, segments: list, k: int) -> list[tuple[int, float]]:
        """Return (segment_index, score) pairs, highest first, length <= k."""
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
            return [(i, 0.0) for i in range(min(k, len(segments)))]

        sims = (matrix @ q.T).toarray().ravel()
        top = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in top]

    def retrieve(self, question: str, segments: list, k: int = 5) -> list[tuple[int, int]]:
        ranked = self._rank(question, segments, k)
        return [(segments[i].start_char, segments[i].end_char) for i, _ in ranked]

    def rank_spans(self, question: str, segments: list, depth: int) -> list[tuple[int, int]]:
        """Deeper ranked span list, for fusion (see hybrid.py)."""
        return self.retrieve(question, segments, k=depth)

    def retrieve_with_text(self, question: str, segments: list, k: int = 5) -> list[dict]:
        ranked = self._rank(question, segments, k)
        out = []
        for i, score in ranked:
            seg = segments[i]
            out.append({
                "start_char": seg.start_char,
                "end_char": seg.end_char,
                "text": seg.embedding_text,
                "header": seg.header,
                "score": score,
            })
        return out
