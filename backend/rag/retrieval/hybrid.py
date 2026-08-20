"""
Covenant Phase 6 — hybrid (lexical + dense) retrieval via Reciprocal Rank
Fusion.

WHY THIS EXISTS
---------------
§6D.2 measured dense MiniLM at 0.5818 hit_rate@5 against TF-IDF's 0.6934
and concluded dense loses. That conclusion is correct and it is also
incomplete: "which single retriever wins" is a different question from
"what is the best retrieval this project can ship." The two retrievers
fail on *different* queries — TF-IDF misses when the question's wording
and the clause's wording diverge, dense misses when the discriminating
signal is one rare literal term that mean-pooling washes out. Where two
rankers have uncorrelated errors, fusing them beats either alone, and
that is the standard result in retrieval literature (BM25 + dense hybrid).

Fusion here is Reciprocal Rank Fusion, not score averaging, deliberately:
TF-IDF cosine scores and Chroma cosine distances are not on a common
scale and normalizing them per-query would introduce a tuning knob with
no principled setting. RRF uses only *rank position*, so it needs no
calibration between the two systems — which keeps this an evidence-gated
addition rather than a hyperparameter surface.

    RRF(span) = Σ_over_rankers  weight / (RRF_C + rank_in_that_ranker)

RRF_C = 60 is the value from the original Cormack et al. formulation and
is left at its published default rather than tuned against this eval set;
tuning it on the same 6,702 rows the result is reported on would be
fitting to the test set. Same reasoning for the equal default weights.

Fusion is over *spans*, matching what scorer.py scores, so a span found by
both rankers accumulates both contributions — that agreement bonus is the
mechanism by which fusion outperforms its inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tfidf import TfidfRetriever  # noqa: E402
from retrieve import ChromaRetriever  # noqa: E402

RRF_C = 60
DEFAULT_DEPTH = 20


class HybridRetriever:
    """Fuses TfidfRetriever and ChromaRetriever rankings by RRF.

    Needs both `segments` (TF-IDF fits per contract, in memory) and
    `contract_id` (Chroma is a persisted store scoped by metadata), so its
    signature is the union of the two it wraps rather than matching either
    exactly.
    """

    name = "hybrid_rrf"

    def __init__(
        self,
        collection_name: str | None = None,
        depth: int = DEFAULT_DEPTH,
        w_lexical: float = 1.0,
        w_dense: float = 1.0,
        lexical_ngram_range: tuple[int, int] = (1, 1),
        name: str | None = None,
    ):
        self.depth = depth
        self.w_lexical = w_lexical
        self.w_dense = w_dense
        self.lexical = TfidfRetriever(ngram_range=lexical_ngram_range)
        self.dense = ChromaRetriever(collection_name=collection_name)
        if name:
            self.name = name

    def _fuse(self, lex_spans: list, dense_spans: list, k: int) -> list[tuple[int, int]]:
        scores: dict[tuple[int, int], float] = {}
        for weight, ranking in ((self.w_lexical, lex_spans), (self.w_dense, dense_spans)):
            if weight == 0:
                continue
            for rank, span in enumerate(ranking, start=1):
                scores[span] = scores.get(span, 0.0) + weight / (RRF_C + rank)

        # Ties broken by lexical rank, so the fused ranking degrades to the
        # stronger single ranker rather than to arbitrary dict order when
        # the two disagree completely.
        lex_pos = {span: i for i, span in enumerate(lex_spans)}
        ordered = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], lex_pos.get(kv[0], len(lex_spans))),
        )
        return [span for span, _ in ordered[:k]]

    def _fused(self, question: str, segments: list, contract_id: str, k: int) -> list[tuple[int, int]]:
        return self._fuse(
            self.lexical.rank_spans(question, segments, self.depth),
            self.dense.rank_spans(question, contract_id, self.depth),
            k,
        )

    def retrieve(self, question: str, segments: list, contract_id: str, k: int = 5) -> list[tuple[int, int]]:
        return self._fused(question, segments, contract_id, k)

    def retrieve_batch(self, questions: list[str], segments: list, contract_id: str,
                       k: int = 5) -> list[list[tuple[int, int]]]:
        """Batched fusion for one contract — the dense half is the expensive
        one and batches per contract (retrieve.py `_query_many`); the lexical
        half stays per-question, which is cheap since it fits on that one
        contract's segments."""
        dense_batch = self.dense.rank_spans_batch(questions, contract_id, self.depth)
        return [
            self._fuse(self.lexical.rank_spans(q, segments, self.depth), dense_batch[i], k)
            for i, q in enumerate(questions)
        ]

    def retrieve_with_text(self, question: str, segments: list, contract_id: str, k: int = 5) -> list[dict]:
        """Fused spans re-attached to their segment text, for Phase 7."""
        by_span = {(s.start_char, s.end_char): s for s in segments}
        out = []
        for span in self._fused(question, segments, contract_id, k):
            seg = by_span.get(span)
            out.append({
                "start_char": span[0],
                "end_char": span[1],
                "text": seg.embedding_text if seg else "",
                "header": (seg.header if seg else "") or "",
                "contract_id": contract_id,
            })
        return out
