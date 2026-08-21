"""
Tests for the faithfulness judge (judge.py) and the harness log schema.

No network — the judge's cloud backend is stubbed. What is asserted is the
set of rules TRD locks, each of which is the kind that fails silently and
flatters the number if it breaks:

  1. cloud-only, and never the same model that generated the answer
     (§6.2). A model grading its own output is not a weak signal, it is
     no signal, so this raises rather than warns.
  2. unreadable judge output is UNPARSEABLE, never a pass. Defaulting it
     to SUPPORTED would inflate the exact number this module reports.
  3. abstentions are counted but excluded from the faithfulness
     denominator, so a model that refuses everything scores zero rows
     rather than 100%.
  4. the log keeps retrieval correctness and faithfulness in separate
     fields (§3.3) and carries backend identity (§3.5).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "generation"))

from backends import Backend, Generation, OllamaBackend  # noqa: E402
from judge import (ABSTAINED, PARTIAL, SUPPORTED, UNPARSEABLE,  # noqa: E402
                   UNSUPPORTED, FaithfulnessJudge, Verdict, parse_verdict, summarize)
from log_schema import GenerationLogRow, GenerationLogWriter, read_log  # noqa: E402

CHUNKS = [{"header": "12. GOVERNING LAW", "start_char": 1, "end_char": 9,
           "text": "Delaware law governs."}]


class _StubCloud(Backend):
    name = "groq"
    default_model = "stub-judge"

    def __init__(self, reply="SUPPORTED\nit matches", **kw):
        super().__init__(**kw)
        self.reply = reply
        self.calls = 0

    def _call(self, prompt, system):
        self.calls += 1
        return self.reply


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("text,label", [
    ("SUPPORTED\nreason", SUPPORTED),
    ("UNSUPPORTED\nreason", UNSUPPORTED),
    ("PARTIALLY_SUPPORTED\nreason", PARTIAL),
    ("supported\nlowercase still counts", SUPPORTED),
])
def test_labels_parse_from_the_first_line(text, label):
    assert parse_verdict(text)[0] == label


def test_partially_supported_is_not_misread_as_supported():
    """PARTIALLY_SUPPORTED contains the substring SUPPORTED; a naive check
    grades every hedged answer as fully grounded."""
    assert parse_verdict("PARTIALLY_SUPPORTED\nhalf of it")[0] == PARTIAL


def test_unreadable_output_is_unparseable_not_a_pass():
    for junk in ("", "   ", "I think it's probably fine?"):
        assert parse_verdict(junk)[0] == UNPARSEABLE


def test_rationale_is_captured_from_the_second_line():
    assert parse_verdict("SUPPORTED\nthe excerpt says Delaware")[1] == "the excerpt says Delaware"


# --- routing rules ---------------------------------------------------------

def test_judge_refuses_a_local_backend():
    with pytest.raises(ValueError, match="cloud"):
        FaithfulnessJudge(OllamaBackend())


def test_judge_refuses_to_grade_its_own_output():
    j = FaithfulnessJudge(_StubCloud())
    with pytest.raises(ValueError, match="same model"):
        j.judge("q", CHUNKS, "an answer", generator_identity=j.identity)


def test_judge_grades_a_different_model_normally():
    j = FaithfulnessJudge(_StubCloud())
    v = j.judge("q", CHUNKS, "Delaware law governs.", "ollama:llama3.2:3b")
    assert v.label == SUPPORTED
    assert v.judge_identity == "groq:stub-judge"


def test_a_non_transient_backend_error_becomes_unparseable_and_keeps_the_message():
    """A malformed request is a real judge failure, unlike a rate limit --
    see the NOT_JUDGED test below for that distinction."""
    class _Boom(Backend):
        name = "groq"
        default_model = "m"

        def _call(self, prompt, system):
            raise RuntimeError("BadRequestError: unsupported parameter")

    v = FaithfulnessJudge(_Boom()).judge("q", CHUNKS, "answer", "ollama:x")
    assert v.label == UNPARSEABLE
    assert "unsupported parameter" in (v.error or "")


def test_an_abstention_is_labelled_without_spending_a_judge_call():
    """An answer that asserts nothing cannot be unfaithful, and paying an
    API call to discover that would be waste on a large fraction of rows."""
    stub = _StubCloud()
    v = FaithfulnessJudge(stub).judge("q", CHUNKS, "NOT_IN_EXCERPTS", "ollama:x")
    assert v.label == ABSTAINED
    assert stub.calls == 0


# --- aggregation -----------------------------------------------------------

def _v(label):
    return Verdict(label, "", "groq:stub", 0)


def test_abstentions_are_excluded_from_the_faithfulness_denominator():
    s = summarize([_v(SUPPORTED), _v(UNSUPPORTED), _v(ABSTAINED), _v(ABSTAINED)])
    assert s["n_scored"] == 2
    assert s["faithful_rate"] == 0.5      # 1 of 2 claims, not 1 of 4 rows
    assert s["n_abstained"] == 2
    assert s["abstention_rate"] == 0.5


def test_a_model_that_abstains_on_everything_scores_no_rows_not_a_perfect_score():
    s = summarize([_v(ABSTAINED) for _ in range(5)])
    assert s["n_scored"] == 0
    assert s["faithful_rate"] is None


def test_unparseable_rows_are_reported_and_never_counted_as_supported():
    s = summarize([_v(SUPPORTED), _v(UNPARSEABLE)])
    assert s["n_unparseable"] == 1
    assert s["faithful_rate"] == 1.0  # denominator is claims actually graded
    assert s["n_scored"] == 1


def test_partial_is_tracked_separately_from_fully_supported():
    s = summarize([_v(SUPPORTED), _v(PARTIAL)])
    assert s["faithful_rate"] == 0.5
    assert s["supported_or_partial_rate"] == 1.0


# --- log schema (TRD §3.5) -------------------------------------------------

def test_log_row_carries_backend_identity_for_both_models(tmp_path):
    row = GenerationLogRow(
        run_id="r1", contract_id="c1", question="q", answer="a",
        generator_backend="ollama", generator_model="llama3.2:3b",
        judge_backend="groq", judge_model="gpt-oss-120b",
    )
    assert row.generator_identity == "ollama:llama3.2:3b"
    path = tmp_path / "run.jsonl"
    with GenerationLogWriter(path) as w:
        w.write(row)
    back = read_log(path)[0]
    assert back["generator_model"] == "llama3.2:3b"
    assert back["judge_model"] == "gpt-oss-120b"
    assert back["timestamp"]


def test_log_keeps_retrieval_and_faithfulness_in_separate_fields(tmp_path):
    """TRD §3.3 -- blending them destroys the attribution the whole harness
    exists to provide."""
    row = GenerationLogRow(run_id="r1", retrieval_hit=True, judge_label=UNSUPPORTED)
    path = tmp_path / "run.jsonl"
    with GenerationLogWriter(path) as w:
        w.write(row)
    back = read_log(path)[0]
    # Retrieval succeeded AND the model was unfaithful: a single blended
    # score could not express this, and it is the most diagnostic case there is.
    assert back["retrieval_hit"] is True
    assert back["judge_label"] == UNSUPPORTED
    assert "score" not in back


def test_rows_are_flushed_as_they_are_written(tmp_path):
    """A paid, rate-limitable run must keep what it already bought."""
    path = tmp_path / "run.jsonl"
    w = GenerationLogWriter(path)
    w.write(GenerationLogRow(run_id="r1"))
    assert len(read_log(path)) == 1  # readable before close()
    w.close()


def test_reasoning_model_think_blocks_do_not_break_parsing():
    """qwen3.6-27b (a Groq option) wraps output in <think>...</think>, which
    would push the label off line 1 and silently zero the faithfulness
    denominator for every row."""
    text = "<think>\nLet me consider the excerpt carefully.\n</think>\nSUPPORTED\nit matches"
    assert parse_verdict(text) == (SUPPORTED, "it matches")


def test_a_think_block_with_no_verdict_after_it_is_unparseable():
    assert parse_verdict("<think>thinking forever</think>")[0] == UNPARSEABLE


def test_default_judge_model_is_the_small_one():
    """A 120B judge is disproportionate for a three-way label task; the
    small model was verified correct on grounded/fabricated/partial cases."""
    from backends import GROQ_JUDGE_MODEL
    assert GROQ_JUDGE_MODEL == "openai/gpt-oss-20b"


def test_a_rate_limited_row_is_not_judged_rather_than_unparseable():
    """UNPARSEABLE means the judge replied unreadably -- a grading problem.
    A 429 means no verdict was ever obtained. Conflating them once produced
    a run where 57 rate-limited rows looked like judge failures."""
    from judge import NOT_JUDGED

    class _Limited(Backend):
        name = "groq"
        default_model = "m"

        def _call(self, prompt, system):
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")

    v = FaithfulnessJudge(_Limited(max_retries=0)).judge("q", CHUNKS, "answer", "ollama:x")
    assert v.label == NOT_JUDGED


def test_never_judged_rows_are_excluded_from_every_rate_and_surfaced(monkeypatch):
    from judge import NOT_JUDGED
    s = summarize([_v(SUPPORTED), _v(UNSUPPORTED)] + [_v(NOT_JUDGED)] * 8)
    assert s["n_scored"] == 2
    assert s["faithful_rate"] == 0.5          # not 1/10
    assert s["n_not_judged"] == 8
    assert s["judged_coverage"] == 0.2        # 2 of 10 -- the run was partial
