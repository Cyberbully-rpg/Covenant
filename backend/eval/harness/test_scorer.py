"""
Covenant Phase 3, Step 2 — Unit tests for the retrieval-correctness scorer.

Synthetic offsets only: the scorer is a pure function over spans, so these
prove its arithmetic without needing a retriever, an eval set, or any I/O.

Required cases (per Phase 3 spec): exact-match span, partial overlap, no
overlap, multi-span gold set, empty-category row. Plus the anti-gaming
property the metric exists to expose.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scorer import (
    overlap_chars,
    spans_overlap,
    score_query,
    aggregate_scores,
)


# ---------------------------------------------------------------------------
# Span arithmetic primitives
# ---------------------------------------------------------------------------

def test_overlap_chars_counts_shared_characters():
    assert overlap_chars((0, 10), (5, 20)) == 5
    assert overlap_chars((0, 10), (10, 20)) == 0   # touching, not overlapping
    assert overlap_chars((0, 10), (20, 30)) == 0   # disjoint
    assert overlap_chars((5, 15), (0, 100)) == 10  # fully contained


def test_spans_overlap_requires_at_least_one_shared_char():
    assert spans_overlap((0, 10), (9, 20)) is True
    assert spans_overlap((0, 10), (10, 20)) is False


# ---------------------------------------------------------------------------
# Case 1: exact-match span
# ---------------------------------------------------------------------------

def test_exact_match_span_is_a_perfect_hit():
    score = score_query(retrieved=[(100, 200)], gold_spans=[(100, 200)])
    assert score.hit is True
    assert score.gold_chars_hit == 100
    assert score.gold_chars_total == 100
    assert score.retrieved_chars == 100

    agg = aggregate_scores([score])
    assert agg["hit_rate"] == 1.0
    assert agg["gold_recall"] == 1.0
    assert agg["gold_density"] == 1.0  # everything returned was the answer


# ---------------------------------------------------------------------------
# Case 2: partial overlap
# ---------------------------------------------------------------------------

def test_partial_overlap_counts_as_hit_but_scores_below_one():
    # retrieved covers only the back half of the gold span
    score = score_query(retrieved=[(150, 250)], gold_spans=[(100, 200)])
    assert score.hit is True
    assert score.gold_chars_hit == 50     # 150..200
    assert score.gold_chars_total == 100
    assert score.retrieved_chars == 100

    agg = aggregate_scores([score])
    assert agg["hit_rate"] == 1.0
    assert agg["gold_recall"] == 0.5
    assert agg["gold_density"] == 0.5


# ---------------------------------------------------------------------------
# Case 3: no overlap
# ---------------------------------------------------------------------------

def test_no_overlap_is_a_miss():
    score = score_query(retrieved=[(0, 50)], gold_spans=[(100, 200)])
    assert score.hit is False
    assert score.gold_chars_hit == 0

    agg = aggregate_scores([score])
    assert agg["hit_rate"] == 0.0
    assert agg["gold_recall"] == 0.0
    assert agg["gold_density"] == 0.0


# ---------------------------------------------------------------------------
# Case 4: multi-span gold set — hitting ANY gold span counts
# ---------------------------------------------------------------------------

def test_hitting_any_one_of_several_gold_spans_counts_as_hit():
    gold = [(100, 200), (5000, 5100), (9000, 9050)]
    # retrieved touches only the middle gold span
    score = score_query(retrieved=[(5050, 5150)], gold_spans=gold)
    assert score.hit is True
    assert score.gold_chars_hit == 50          # 5050..5100
    assert score.gold_chars_total == 250       # 100 + 100 + 50


def test_multi_span_gold_accumulates_coverage_across_spans():
    gold = [(100, 200), (5000, 5100)]
    score = score_query(retrieved=[(100, 200), (5000, 5100)], gold_spans=gold)
    assert score.hit is True
    assert score.gold_chars_hit == 200
    assert score.gold_chars_total == 200
    assert aggregate_scores([score])["gold_recall"] == 1.0


def test_overlapping_retrieved_chunks_are_not_double_counted():
    # two retrieved chunks overlapping each other and the gold span
    score = score_query(retrieved=[(100, 180), (150, 200)], gold_spans=[(100, 200)])
    assert score.retrieved_chars == 100   # merged, not 80 + 50
    assert score.gold_chars_hit == 100    # not counted twice


# ---------------------------------------------------------------------------
# Case 5: empty-category row (no gold span)
# ---------------------------------------------------------------------------

def test_empty_gold_set_produces_no_hit_and_no_gold_chars():
    """
    Rows where CUAD marks the category absent have no gold span, so span
    overlap is undefined. score_query stays well-defined (no crash, no
    fake hit) but such rows are excluded from aggregates by the runner —
    see test_skipped_rows_are_reported_not_hidden.
    """
    score = score_query(retrieved=[(0, 100)], gold_spans=[])
    assert score.hit is False
    assert score.gold_chars_hit == 0
    assert score.gold_chars_total == 0
    assert score.retrieved_chars == 100


def test_skipped_rows_are_reported_not_hidden():
    scores = [score_query([(100, 200)], [(100, 200)])]
    agg = aggregate_scores(scores, n_skipped_no_gold=14208)
    assert agg["n_scored"] == 1
    assert agg["n_skipped_no_gold"] == 14208


def test_aggregate_of_nothing_is_zeroed_not_a_crash():
    agg = aggregate_scores([], n_skipped_no_gold=5)
    assert agg["n_scored"] == 0
    assert agg["hit_rate"] == 0.0
    assert agg["n_skipped_no_gold"] == 5


# ---------------------------------------------------------------------------
# The anti-gaming property this metric exists to expose
# ---------------------------------------------------------------------------

def test_chunk_size_gaming_shows_up_as_collapsing_gold_density():
    """
    Returning the whole document guarantees a hit — that's exactly why hit
    rate alone is not reportable. gold_density must fall off a cliff to
    make the gaming visible.
    """
    honest = score_query(retrieved=[(100, 200)], gold_spans=[(100, 200)])
    gamed = score_query(retrieved=[(0, 100_000)], gold_spans=[(100, 200)])

    honest_agg = aggregate_scores([honest])
    gamed_agg = aggregate_scores([gamed])

    # both "hit" — hit rate cannot tell them apart
    assert honest_agg["hit_rate"] == gamed_agg["hit_rate"] == 1.0
    # gold_density can, decisively
    assert honest_agg["gold_density"] == 1.0
    assert gamed_agg["gold_density"] < 0.01
    assert gamed_agg["mean_retrieved_chars"] > honest_agg["mean_retrieved_chars"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
