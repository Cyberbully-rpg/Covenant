"""
Covenant Phase 6 — dense retrieval over the Chroma collection built by
ingestion/ingest.py.

Default retrieval mode is full-document search (TRD §5.1): a query is
always scoped to one contract via `where={"contract_id": ...}`, never
searched across the whole 510-contract corpus. No classifier metadata
filter here — that toggle is Phase 8, off by construction in this module
(RAG must run correctly with zero classifier involvement, TRD §5.5).

`ChromaRetriever.retrieve()` intentionally matches the SAME interface as
Phase 3's `RandomRetriever`/`LexicalRetriever` (baselines.py) — a
`.retrieve(question, ..., k)` method returning `(start_char, end_char)`
spans — with the addition of a required `contract_id`, since unlike the
in-memory baselines this one queries a persisted store scoped per
contract rather than re-deriving everything from `segments` each call.
That shared shape is what lets it run through the exact same eval harness
(Phase 3's scorer.py) for a fair, apples-to-apples comparison against the
TF-IDF baseline already measured there.

WINDOW DEDUPLICATION
--------------------
Under windowed ingestion a single segment is stored as several records
that all carry the same parent `(start_char, end_char)`. Ranking must be
over *segments*, not records, or a long segment would occupy several of
the k slots with different slices of itself and crowd out other clauses.

So the query oversamples (`k * OVERSAMPLE` records) and collapses to
distinct parent spans, keeping each span's best-ranked window. Because
Chroma returns records in ascending distance, first-seen == best-scoring,
which makes the collapse a max-pool over a segment's windows: a segment
is ranked by its single best-matching part. Retrieved spans stay
parent-sized, so `mean_retrieved_chars` remains directly comparable to
the unwindowed and TF-IDF runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding import embed  # noqa: E402
from ingestion.ingest import get_collection  # noqa: E402

# How many records to pull per k slots before collapsing to parent spans.
# 5 covers the observed worst case: the longest segments in the corpus
# (~5,000 chars) split into ~7 windows, and even if one segment's windows
# swept every slot, 5x still leaves room to fill k distinct spans.
OVERSAMPLE = 5


class ChromaRetriever:
    name = "chroma_dense"

    def __init__(self, collection_name: str | None = None, name: str | None = None):
        self._collection = None
        self._collection_name = collection_name
        if name:
            self.name = name

    @property
    def collection(self):
        if self._collection is None:
            self._collection = get_collection(reset=False, name=self._collection_name)
        return self._collection

    @staticmethod
    def _dedupe(metadatas, documents, distances, k: int) -> list[dict]:
        """Collapse window records to distinct parent spans, best window first."""
        seen: set[tuple[int, int]] = set()
        out: list[dict] = []
        for meta, doc, dist in zip(metadatas, documents, distances):
            span = (meta["start_char"], meta["end_char"])
            if span in seen:
                continue  # a lower-ranked window of an already-selected segment
            seen.add(span)
            out.append({**meta, "text": doc, "distance": dist})
            if len(out) == k:
                break
        return out

    def _query_many(self, questions: list[str], contract_id: str, k: int) -> list[list[dict]]:
        """Ranked, window-deduped hits for several questions against ONE contract.

        Batched deliberately. Chroma's metadata-filtered search costs
        roughly the same per call regardless of how many query vectors ride
        along, and every question in the eval set shares the same
        `where={"contract_id": ...}` clause within a contract — so issuing
        one call per contract instead of one per question turns a
        6,702-call sweep into a 510-call one. That is the difference
        between a multi-hour harness run and a few minutes, with identical
        results.
        """
        if not questions:
            return []
        # is_query=True: models with an asymmetric retrieval instruction (BGE)
        # need it on this side only. Ingestion never sets it (embedding.py).
        vecs = embed(questions, is_query=True)
        result = self.collection.query(
            query_embeddings=[v.tolist() for v in vecs],
            n_results=max(k, k * OVERSAMPLE),
            where={"contract_id": contract_id},
        )
        if not result["metadatas"]:
            return [[] for _ in questions]
        return [
            self._dedupe(result["metadatas"][i], result["documents"][i],
                         result["distances"][i], k)
            for i in range(len(questions))
        ]

    def _query(self, question: str, contract_id: str, k: int) -> list[dict]:
        """Ranked, window-deduped hits for one contract. Best first."""
        batch = self._query_many([question], contract_id, k)
        return batch[0] if batch else []

    def retrieve(self, question: str, contract_id: str, k: int = 5) -> list[tuple[int, int]]:
        return [(h["start_char"], h["end_char"]) for h in self._query(question, contract_id, k)]

    def retrieve_with_text(self, question: str, contract_id: str, k: int = 5) -> list[dict]:
        """Same as retrieve(), but returns full metadata + document text —
        what Phase 7's generation call site will actually need.

        Note the `text` of a windowed hit is the matching window, not the
        whole parent segment; Phase 7 re-slices the parent span from the
        contract when it needs full clause text for a prompt.
        """
        return self._query(question, contract_id, k)

    def rank_spans(self, question: str, contract_id: str, depth: int) -> list[tuple[int, int]]:
        """Deeper ranked span list, for fusion (see hybrid.py)."""
        return [(h["start_char"], h["end_char"]) for h in self._query(question, contract_id, depth)]

    def rank_spans_batch(self, questions: list[str], contract_id: str,
                         depth: int) -> list[list[tuple[int, int]]]:
        """Batched rank_spans for one contract — see `_query_many`."""
        return [
            [(h["start_char"], h["end_char"]) for h in hits]
            for hits in self._query_many(questions, contract_id, depth)
        ]

    def retrieve_batch(self, questions: list[str], contract_id: str,
                       k: int = 5) -> list[list[tuple[int, int]]]:
        return self.rank_spans_batch(questions, contract_id, k)
