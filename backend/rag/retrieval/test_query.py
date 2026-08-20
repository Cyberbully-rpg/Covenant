"""
Tests for query boilerplate stripping (query.py).

Two properties matter. First, stripping must keep every informative token
-- the category and the details clause -- because they are what the
rankers actually match on; a "cleaner" that dropped the category would
silently destroy retrieval rather than improve it. Second, it must be a
no-op on anything that isn't a CUAD probe, since a free-text /ask
question has no template to remove and must reach the retriever intact.

The wrapper's placement is also asserted: CleanQueryRetriever hands the
CLEANED text down to the base retriever, while callers above it (the lead
prior, which parses the category out of the original probe) still see the
original. Getting that order wrong would disable the prior entirely.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from query import strip_boilerplate, CleanQueryRetriever  # noqa: E402
from lead_prior import LeadPrior, PriorRetriever, category_from_question  # noqa: E402

PROBE = ('Highlight the parts (if any) of this contract related to "Governing Law" '
         'that should be reviewed by a lawyer. Details: Which state\'s law governs '
         'the interpretation of the contract?')


def test_stripping_keeps_the_category_and_the_details():
    out = strip_boilerplate(PROBE)
    assert out.startswith("Governing Law")
    assert "law governs" in out


def test_stripping_removes_the_constant_instruction_wrapper():
    out = strip_boilerplate(PROBE)
    for noise in ("Highlight", "reviewed by a lawyer", "parts (if any)"):
        assert noise not in out


def test_a_probe_without_a_details_clause_reduces_to_the_category():
    probe = ('Highlight the parts (if any) of this contract related to "Parties" '
             'that should be reviewed by a lawyer.')
    assert strip_boilerplate(probe) == "Parties"


def test_free_text_questions_pass_through_untouched():
    for q in ("who signed this?", "", "Details: something else entirely"):
        assert strip_boilerplate(q) == q


class _RecordingBase:
    def __init__(self):
        self.seen = None

    def retrieve_batch(self, questions, segments, contract_id, k):
        self.seen = list(questions)
        return [[(0, 10)] * k for _ in questions]


def test_wrapper_hands_the_cleaned_text_to_the_base_retriever():
    base = _RecordingBase()
    CleanQueryRetriever(base).retrieve_batch([PROBE], [], "c1", k=1)
    assert base.seen == ["Governing Law. Which state's law governs the interpretation "
                         "of the contract?"]


class FakeSeg:
    def __init__(self, start, end):
        self.start_char, self.end_char = start, end


def test_prior_still_sees_the_original_question_through_the_wrapper():
    """The prior parses `related to "X"` -- if cleaning happened above it
    instead of below, the category would be gone and the prior dead."""
    segments = [FakeSeg(0, 100), FakeSeg(100, 200)]
    parties_probe = ('Highlight the parts (if any) of this contract related to "Parties" '
                     'that should be reviewed by a lawyer. Details: who signed')
    assert category_from_question(strip_boilerplate(parties_probe)) is None  # cleaning kills it

    base = _RecordingBase()
    stacked = PriorRetriever(CleanQueryRetriever(base), LeadPrior(categories={"Parties"}))
    out = stacked.retrieve_batch([parties_probe], segments, "c1", k=2)
    assert out[0][0] == (0, 100)          # prior fired -> it saw the original
    assert "Highlight" not in base.seen[0]  # base got the cleaned text
