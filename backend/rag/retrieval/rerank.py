"""
Covenant Phase 6 — cross-encoder reranking.

WHY
---
The rank diagnostic (ARCHITECTURE.md §6D.4) found the ranker is close, not
lost: the correct segment is inside the top 20 for **92.3%** of questions
while only 72% reach the top 5. So a second pass that reorders 20
candidates has a large, measured headroom to convert, and nothing else on
the ladder is aimed at that.

Bi-encoders (everything in `retrieve.py`) must compress a segment into one
vector *before* seeing the question, so the vector has to be a general
summary. A cross-encoder reads the question and the segment **together**
and scores the pair directly, which lets it use the interaction between
them — which clause of a long segment answers *this* question — at the
cost of one forward pass per candidate instead of one per corpus item.
That cost is only affordable over a short candidate list, which is exactly
what the first-stage retriever now provides.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — the standard small reranker
trained on MS MARCO relevance pairs. Baseline-first, same discipline as
TRD §7.1's embedding ladder: this is the smallest credible reranker, not
the strongest available, and no CUDA is available (TRD §6.1).

WHAT THIS CANNOT FIX
--------------------
Reranking only reorders what the first stage returned. Its ceiling is the
first stage's recall at `candidate_depth` — 0.9230 at depth 20 — and the
7.7% of questions whose answer never enters the candidate list are
untouched no matter how good the reranker is. It also cannot help the
categories where the query's words are absent from the gold text
(Volume Restriction 98.8% absent, §6D.4) unless it genuinely understands
the clause rather than matching wording; whether it does is precisely
what the measurement decides.
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_DEPTH = 20
# Cross-encoders truncate like any transformer; ms-marco-MiniLM takes 512
# word-pieces for the PAIR. Segments are trimmed rather than silently cut
# mid-encode, so the truncation is visible here instead of being the kind
# of hidden confound §6D.3 already caught once.
MAX_SEGMENT_CHARS = 1400


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """Reorders a first-stage retriever's candidates by pairwise relevance."""

    name = "rerank"

    def __init__(self, base, candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
                 model_name: str = DEFAULT_MODEL, name: str | None = None):
        self.base = base
        self.candidate_depth = candidate_depth
        self.model_name = model_name
        if name:
            self.name = name

    def _text_for(self, span, by_span) -> str:
        seg = by_span.get(span)
        if seg is None:
            return ""
        text = seg.embedding_text or seg.text or ""
        return text[:MAX_SEGMENT_CHARS]

    def retrieve_batch(self, questions, segments, contract_id, k: int = 5):
        by_span = {(s.start_char, s.end_char): s for s in segments}
        candidates = self.base.retrieve_batch(questions, segments, contract_id,
                                              self.candidate_depth)

        # One encode call for the whole contract's pairs rather than one per
        # question — same batching rationale as retrieve.py `_query_many`.
        pairs, owner = [], []
        for i, (q, spans) in enumerate(zip(questions, candidates)):
            for span in spans:
                pairs.append((q, self._text_for(span, by_span)))
                owner.append(i)

        if not pairs:
            return [[] for _ in questions]

        scores = get_cross_encoder(self.model_name).predict(pairs, show_progress_bar=False)

        per_question: list[list[tuple[float, tuple[int, int]]]] = [[] for _ in questions]
        cursor = 0
        for i, spans in enumerate(candidates):
            for span in spans:
                per_question[i].append((float(scores[cursor]), span))
                cursor += 1

        out = []
        for i, scored in enumerate(per_question):
            # Stable sort on score alone, so equal scores keep first-stage
            # order rather than being shuffled arbitrarily.
            order = sorted(range(len(scored)), key=lambda j: -scored[j][0])
            out.append([scored[j][1] for j in order[:k]])
        return out

    def retrieve(self, question: str, segments: list, contract_id: str, k: int = 5):
        return self.retrieve_batch([question], segments, contract_id, k)[0]
