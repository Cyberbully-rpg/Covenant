"""
Covenant Phase 3, Step 2 — Retrieval-correctness metric.

Mechanical character-span overlap between what a retriever returned and
CUAD's gold answer span(s). No LLM involved, no judgment calls — pure
arithmetic on offsets.

WHAT THIS METRIC IS GAMEABLE BY (state this wherever results are reported):
chunk size. A segmenter that returns huge chunks will score higher on raw
overlap for free — in the limit, "retrieve the entire contract" scores a
perfect hit rate while being useless. Raw hit rate alone is therefore
never a defensible number to report.

To make that gaming visible instead of hidden, score_retrieval() always
returns hit rate PAIRED with volume-normalized companions:
  - `gold_density`: gold-span chars ÷ total retrieved chars. How much of
    what you handed back was actually the answer. Falls as chunks bloat,
    so chunk-size gaming shows up here as a collapsing number even while
    hit rate climbs.
  - `mean_retrieved_chars`: raw size of what was returned, per query.
    Makes the bloat itself directly visible.
A hit-rate improvement that comes with a gold_density drop is chunk-size
gaming, not a retrieval improvement. Report all three together, always.

SCOPE (inherited from Step 1, restated because it travels with the
numbers): "validated against CUAD's contract distribution using CUAD's 41
templated category probes" — not diverse user questions, not legal
documents in general.

EMPTY-CATEGORY ROWS: rows where CUAD marks the category absent
(has_gold_span=False) are EXCLUDED from these scores — span overlap is
undefined with no gold span to overlap. They are counted and reported as
`n_skipped_no_gold` so the exclusion is visible, never silent. Scoring
correct abstention on those rows is deferred (it needs a confidence
threshold, which needs a real retriever/classifier to calibrate against —
neither exists before Phase 6). The rows remain in the eval set for that
later use.
"""

from __future__ import annotations

from dataclasses import dataclass


Span = tuple[int, int]  # (start, end), end exclusive


def overlap_chars(a: Span, b: Span) -> int:
    """Number of characters shared by two spans. 0 if they don't touch."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def spans_overlap(a: Span, b: Span) -> bool:
    """True if two spans share at least one character."""
    return overlap_chars(a, b) > 0


@dataclass
class QueryScore:
    """Per-query result. Aggregate these with aggregate_scores()."""
    hit: bool                    # did ANY retrieved chunk touch ANY gold span
    gold_chars_hit: int          # gold chars covered by retrieved chunks (deduped)
    gold_chars_total: int        # total gold chars available for this query
    retrieved_chars: int         # total chars handed back (deduped)


def _merge(spans: list[Span]) -> list[Span]:
    """Merge overlapping/adjacent spans so char counts never double-count."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def score_query(retrieved: list[Span], gold_spans: list[Span]) -> QueryScore:
    """
    Score one query's retrieved chunks against its gold span(s).

    A hit means overlap with ANY gold span — CUAD often labels the same
    category in several places in one contract, and finding any one of
    them is a correct retrieval.

    Pure function: takes offsets, returns numbers. No retriever, no eval
    set, no I/O.
    """
    retrieved_merged = _merge(retrieved)
    gold_merged = _merge(gold_spans)

    retrieved_chars = sum(e - s for s, e in retrieved_merged)
    gold_chars_total = sum(e - s for s, e in gold_merged)

    gold_chars_hit = 0
    for g in gold_merged:
        covered = _merge([
            (max(g[0], r[0]), min(g[1], r[1]))
            for r in retrieved_merged
            if spans_overlap(g, r)
        ])
        gold_chars_hit += sum(e - s for s, e in covered)

    hit = any(
        spans_overlap(r, g)
        for r in retrieved_merged
        for g in gold_merged
    )

    return QueryScore(
        hit=hit,
        gold_chars_hit=gold_chars_hit,
        gold_chars_total=gold_chars_total,
        retrieved_chars=retrieved_chars,
    )


def aggregate_scores(scores: list[QueryScore], n_skipped_no_gold: int = 0) -> dict:
    """
    Roll per-query scores into the reportable metric set.

    Always returns hit_rate together with its volume-normalized companions,
    so chunk-size gaming is visible in the reported numbers (see module
    docstring).
    """
    n = len(scores)
    if n == 0:
        return {
            "n_scored": 0,
            "n_skipped_no_gold": n_skipped_no_gold,
            "hit_rate": 0.0,
            "gold_recall": 0.0,
            "gold_density": 0.0,
            "mean_retrieved_chars": 0.0,
        }

    total_retrieved = sum(s.retrieved_chars for s in scores)
    total_gold_hit = sum(s.gold_chars_hit for s in scores)
    total_gold = sum(s.gold_chars_total for s in scores)

    return {
        # headline: fraction of queries where we touched a gold span at all
        "n_scored": n,
        "n_skipped_no_gold": n_skipped_no_gold,
        "hit_rate": sum(1 for s in scores if s.hit) / n,
        # how much of the gold answer text we actually covered
        "gold_recall": (total_gold_hit / total_gold) if total_gold else 0.0,
        # volume-normalized: how much of what we returned was the answer.
        # collapses when chunks bloat — this is the anti-gaming companion.
        "gold_density": (total_gold_hit / total_retrieved) if total_retrieved else 0.0,
        "mean_retrieved_chars": total_retrieved / n,
    }
