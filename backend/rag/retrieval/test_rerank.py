"""
Tests for cross-encoder reranking (rerank.py).

The cross-encoder itself is stubbed. Loading a real model here would make
these slow and would test HuggingFace rather than this module; what needs
asserting is the plumbing, which is where reranking usually goes wrong:

  1. candidates come from the base retriever at `candidate_depth`, not at
     k -- reranking a 5-item list can only shuffle it, and the whole point
     is to reach the ~20 deep where the answer usually sits;
  2. scores are attributed to the right (question, span) pair when several
     questions are batched into one predict() call. An off-by-one in that
     bookkeeping silently scrambles relevance across questions and still
     "works";
  3. output honours k and the pairing survives contract-level batching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import rerank  # noqa: E402
from rerank import CrossEncoderReranker, MAX_SEGMENT_CHARS  # noqa: E402


class FakeSeg:
    def __init__(self, start, end, text):
        self.start_char, self.end_char = start, end
        self.embedding_text, self.text = text, text


SEGMENTS = [FakeSeg(0, 10, "alpha"), FakeSeg(10, 20, "bravo"),
            FakeSeg(20, 30, "charlie"), FakeSeg(30, 40, "delta")]
SPANS = [(s.start_char, s.end_char) for s in SEGMENTS]


class _StubBase:
    def __init__(self):
        self.asked_k = None

    def retrieve_batch(self, questions, segments, contract_id, k):
        self.asked_k = k
        return [list(SPANS) for _ in questions]


class _StubEncoder:
    """Scores by keyword: a pair scores 1.0 when the question names the segment."""
    def __init__(self):
        self.calls = 0

    def predict(self, pairs, show_progress_bar=False):
        self.calls += 1
        return [1.0 if seg_text and seg_text in q else 0.0 for q, seg_text in pairs]


def _reranker(monkeypatch, depth=20):
    encoder = _StubEncoder()
    monkeypatch.setattr(rerank, "get_cross_encoder", lambda *a, **kw: encoder)
    base = _StubBase()
    return CrossEncoderReranker(base, candidate_depth=depth), base, encoder


def test_candidates_are_pulled_at_candidate_depth_not_at_k(monkeypatch):
    r, base, _ = _reranker(monkeypatch, depth=20)
    r.retrieve_batch(["find alpha"], SEGMENTS, "c1", k=3)
    assert base.asked_k == 20


def test_the_segment_the_question_names_is_promoted_to_first(monkeypatch):
    r, _, _ = _reranker(monkeypatch)
    out = r.retrieve_batch(["find charlie"], SEGMENTS, "c1", k=4)
    assert out[0][0] == (20, 30)


def test_scores_are_attributed_to_the_right_question_when_batched(monkeypatch):
    """One predict() call covers every (question, span) pair for the whole
    contract; mixing up the offsets scrambles relevance while still
    returning plausible-looking spans."""
    r, _, encoder = _reranker(monkeypatch)
    out = r.retrieve_batch(["find delta", "find alpha"], SEGMENTS, "c1", k=4)
    assert out[0][0] == (30, 40)
    assert out[1][0] == (0, 10)
    assert encoder.calls == 1  # batched, not one call per question


def test_output_respects_k(monkeypatch):
    r, _, _ = _reranker(monkeypatch)
    out = r.retrieve_batch(["find bravo"], SEGMENTS, "c1", k=2)
    assert len(out[0]) == 2


def test_no_candidates_yields_empty_results_rather_than_raising(monkeypatch):
    encoder = _StubEncoder()
    monkeypatch.setattr(rerank, "get_cross_encoder", lambda *a, **kw: encoder)

    class _Empty:
        def retrieve_batch(self, questions, segments, contract_id, k):
            return [[] for _ in questions]

    assert CrossEncoderReranker(_Empty()).retrieve_batch(["q"], SEGMENTS, "c1", k=5) == [[]]


def test_long_segments_are_trimmed_before_encoding(monkeypatch):
    """Cross-encoders truncate silently past their pair limit; trimming here
    keeps that visible rather than hidden (the confound in 6D.3)."""
    r, _, _ = _reranker(monkeypatch)
    long_seg = FakeSeg(0, 10, "x" * (MAX_SEGMENT_CHARS * 3))
    assert len(r._text_for((0, 10), {(0, 10): long_seg})) == MAX_SEGMENT_CHARS
