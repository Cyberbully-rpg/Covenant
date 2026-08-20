"""
Tests for HybridRetriever (hybrid.py).

The fusion math is tested directly, with both wrapped rankers stubbed out.
That's deliberate: RRF's correctness claim is about rank arithmetic, not
about embeddings, and testing it through a real Chroma index would make a
pure ranking bug look like a retrieval-quality wobble. The properties that
matter:

  1. a span both rankers agree on outranks a span only one ranker likes,
     even when the latter sits at rank 1 in its own list -- the agreement
     bonus IS the mechanism by which fusion beats its inputs;
  2. zeroing a weight degrades fusion exactly to the other ranker, which
     is what makes hybrid a strict generalization of the tfidf baseline
     rather than a different thing that happens to score similarly;
  3. output stays capped at k and free of duplicates.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid import HybridRetriever, RRF_C  # noqa: E402


class _StubRanker:
    def __init__(self, spans):
        self.spans = spans

    def rank_spans(self, question, second, depth):
        return self.spans[:depth]


def _hybrid(lex_spans, dense_spans, **kwargs):
    h = HybridRetriever.__new__(HybridRetriever)
    h.depth = kwargs.get("depth", 20)
    h.w_lexical = kwargs.get("w_lexical", 1.0)
    h.w_dense = kwargs.get("w_dense", 1.0)
    h.w_char = kwargs.get("w_char", 0.0)
    h.lexical = _StubRanker(lex_spans)
    h.dense = _StubRanker(dense_spans)
    h.char = None
    return h


A, B, C, D = (0, 10), (10, 20), (20, 30), (30, 40)


def test_span_both_rankers_agree_on_beats_either_rank_one_pick():
    # B is 2nd for lexical and 2nd for dense; A is 1st for lexical only,
    # C is 1st for dense only. Agreement should lift B above both.
    h = _hybrid([A, B], [C, B])
    assert h.retrieve("q", [], "cid", k=3)[0] == B


def test_zero_dense_weight_reproduces_the_lexical_ranking_exactly():
    lex = [A, B, C, D]
    h = _hybrid(lex, [D, C, B, A], w_dense=0.0)
    assert h.retrieve("q", [], "cid", k=4) == lex


def test_zero_lexical_weight_reproduces_the_dense_ranking_exactly():
    dense = [D, C, B, A]
    h = _hybrid([A, B, C, D], dense, w_lexical=0.0)
    assert h.retrieve("q", [], "cid", k=4) == dense


def test_result_is_capped_at_k_and_has_no_duplicate_spans():
    h = _hybrid([A, B, C, D], [B, C, D, A])
    out = h.retrieve("q", [], "cid", k=2)
    assert len(out) == 2
    assert len(set(out)) == 2


def test_fusion_score_matches_the_published_rrf_formula():
    # A: lexical rank 1 only -> 1/(C+1). B: dense rank 1 only -> 1/(C+1).
    # Tie, broken toward the lexical ordering (see hybrid.py).
    h = _hybrid([A], [B])
    assert h.retrieve("q", [], "cid", k=2) == [A, B]
    # With dense weighted above lexical, B must overtake A.
    h2 = _hybrid([A], [B], w_dense=2.0)
    assert h2.retrieve("q", [], "cid", k=2) == [B, A]
    assert RRF_C == 60  # published default, deliberately untuned


def test_depth_limits_how_deep_each_ranker_contributes():
    h = _hybrid([A, B, C, D], [D, C, B, A], depth=1)
    out = h.retrieve("q", [], "cid", k=4)
    assert set(out) == {A, D}  # only rank-1 from each list participates


def test_empty_rankings_yield_no_spans_rather_than_raising():
    h = _hybrid([], [])
    assert h.retrieve("q", [], "cid", k=5) == []


def test_char_ranker_contributes_when_weighted_and_is_absent_when_not():
    """The char n-gram ranker is a third RRF input, off by default so adding
    the capability changes no previously-measured result."""
    h = _hybrid([A], [A], w_char=1.0)
    h.char = _StubRanker([D])
    out = h.retrieve("q", [], "cid", k=2)
    assert D in out, "a weighted char ranking must reach the fused result"

    off = _hybrid([A], [A])
    off.char = _StubRanker([D])
    assert D not in off.retrieve("q", [], "cid", k=2)
