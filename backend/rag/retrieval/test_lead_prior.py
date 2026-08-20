"""
Tests for the leading-segment positional prior (lead_prior.py).

The properties worth asserting are the ones that keep this from becoming
a cheat rather than a retrieval component:

  1. it fires ONLY for categories it was fitted to, and is the identity
     function otherwise -- a prior that quietly reorders every query would
     be trading the 37 other categories for the 4 it helps;
  2. what it fits comes from the rows it is GIVEN, so the harness's
     train-only fit is real rather than decorative;
  3. a category below threshold, or with too few rows to judge, is not
     adopted -- otherwise noise on a rare category becomes a rule;
  4. it never returns duplicates and never returns more than k.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from lead_prior import LeadPrior, PriorRetriever, category_from_question  # noqa: E402


class FakeSeg:
    def __init__(self, start, end):
        self.start_char = start
        self.end_char = end


SEGMENTS = [FakeSeg(0, 100), FakeSeg(100, 200), FakeSeg(200, 300), FakeSeg(300, 400)]
LEAD = (0, 100)
Q_PARTIES = 'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.'
Q_OTHER = 'Highlight the parts (if any) of this contract related to "Insurance" that should be reviewed by a lawyer.'


def _row(cid, category, gold_start, gold_end, question=None):
    return {
        "contract_id": cid,
        "category": category,
        "question": question or f'related to "{category}" that',
        "has_gold_span": True,
        "gold_spans": [{"start": gold_start, "end": gold_end}],
    }


def test_category_is_parsed_out_of_a_cuad_probe():
    assert category_from_question(Q_PARTIES) == "Parties"


def test_free_text_question_yields_no_category_so_the_prior_cannot_fire():
    assert category_from_question("who signed this thing?") is None
    prior = LeadPrior(categories={"Parties"})
    assert not prior.applies_to("who signed this thing?")


def test_fit_adopts_a_category_whose_answers_sit_in_the_first_segment():
    segs = {f"c{i}": SEGMENTS for i in range(30)}
    rows = [_row(f"c{i}", "Parties", 10, 50) for i in range(30)]
    prior = LeadPrior.fit(rows, segs)
    assert prior.categories == {"Parties"}


def test_fit_rejects_a_category_whose_answers_sit_deep_in_the_document():
    segs = {f"c{i}": SEGMENTS for i in range(30)}
    rows = [_row(f"c{i}", "Insurance", 310, 350) for i in range(30)]
    assert LeadPrior.fit(rows, segs).categories == set()


def test_fit_rejects_a_category_with_too_few_rows_to_judge():
    """Below min_rows, a perfect-looking rate is noise, not a rule."""
    segs = {f"c{i}": SEGMENTS for i in range(5)}
    rows = [_row(f"c{i}", "Rare Category", 10, 50) for i in range(5)]
    assert LeadPrior.fit(rows, segs).categories == set()


def test_fit_rejects_a_category_that_only_sometimes_leads():
    segs = {f"c{i}": SEGMENTS for i in range(40)}
    rows = ([_row(f"c{i}", "Mixed", 10, 50) for i in range(16)] +      # 40% lead
            [_row(f"c{i}", "Mixed", 310, 350) for i in range(16, 40)])
    assert LeadPrior.fit(rows, segs).categories == set()


def test_fit_only_sees_the_rows_it_is_given():
    """The harness fits on train contracts only; that has to be real."""
    segs = {f"c{i}": SEGMENTS for i in range(60)}
    train = [_row(f"c{i}", "Parties", 10, 50) for i in range(30)]
    held_out = [_row(f"c{i}", "Insurance", 10, 50) for i in range(30, 60)]
    prior = LeadPrior.fit(train, segs)
    assert "Insurance" not in prior.categories
    assert LeadPrior.fit(held_out, segs).categories == {"Insurance"}


def test_apply_promotes_the_lead_segment_for_a_fitted_category():
    prior = LeadPrior(categories={"Parties"})
    ranked = [(200, 300), (300, 400), (100, 200)]
    out = prior.apply(ranked, SEGMENTS, Q_PARTIES, k=3)
    assert out[0] == LEAD


def test_apply_is_the_identity_for_an_unfitted_category():
    prior = LeadPrior(categories={"Parties"})
    ranked = [(200, 300), (300, 400), (100, 200)]
    assert prior.apply(ranked, SEGMENTS, Q_OTHER, k=3) == ranked


def test_apply_never_duplicates_a_lead_segment_already_ranked():
    prior = LeadPrior(categories={"Parties"})
    ranked = [(200, 300), LEAD, (300, 400)]
    out = prior.apply(ranked, SEGMENTS, Q_PARTIES, k=3)
    assert out.count(LEAD) == 1
    assert len(out) == len(set(out))


def test_apply_respects_k():
    prior = LeadPrior(categories={"Parties"})
    ranked = [(200, 300), (300, 400), (100, 200)]
    assert len(prior.apply(ranked, SEGMENTS, Q_PARTIES, k=2)) == 2


def test_round_trip_through_dict_preserves_the_fitted_rule():
    prior = LeadPrior(categories={"Parties", "Document Name"}, n_lead=2)
    restored = LeadPrior.from_dict(prior.to_dict())
    assert restored.categories == prior.categories
    assert restored.n_lead == prior.n_lead


class _StubBase:
    def retrieve_batch(self, questions, segments, contract_id, k):
        return [[(200, 300), (300, 400), (100, 200)][:k] for _ in questions]


def test_prior_retriever_promotes_only_the_matching_question_in_a_batch():
    r = PriorRetriever(_StubBase(), LeadPrior(categories={"Parties"}))
    out = r.retrieve_batch([Q_PARTIES, Q_OTHER], SEGMENTS, "c1", k=3)
    assert out[0][0] == LEAD
    assert out[1][0] == (200, 300)
    assert all(len(spans) == 3 for spans in out)
