"""
Tests for the Phase 4 contract-level split.

The property that actually matters is contract-disjointness: no contract
may have segments on both sides of the boundary. A segment-level split
would leak shared vocabulary, defined terms and boilerplate across the
boundary and inflate every downstream metric, and it would do so silently
— nothing about the reported numbers would look wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from split import split_contract_ids, split_examples, split_examples_three_way  # noqa: E402


def make_examples(n_contracts=50, per_contract=10):
    return [
        {"contract_id": f"c{i:03d}", "segment_id": f"c{i:03d}_s{j}",
         "text": f"clause text {i} {j}", "labels": []}
        for i in range(n_contracts)
        for j in range(per_contract)
    ]


def test_no_contract_appears_on_both_sides():
    train, test = split_examples(make_examples())
    train_ids = {e["contract_id"] for e in train}
    test_ids = {e["contract_id"] for e in test}
    assert train_ids & test_ids == set(), (
        f"contract leakage across split boundary: {train_ids & test_ids}"
    )


def test_split_covers_every_example_exactly_once():
    examples = make_examples()
    train, test = split_examples(examples)
    assert len(train) + len(test) == len(examples)
    seen = {e["segment_id"] for e in train} | {e["segment_id"] for e in test}
    assert len(seen) == len(examples)


def test_split_is_deterministic_across_calls():
    examples = make_examples()
    a_train, a_test = split_examples(examples)
    b_train, b_test = split_examples(examples)
    assert [e["segment_id"] for e in a_train] == [e["segment_id"] for e in b_train]
    assert [e["segment_id"] for e in a_test] == [e["segment_id"] for e in b_test]


def test_split_is_independent_of_input_order():
    """A contract must land on the same side regardless of row ordering."""
    examples = make_examples()
    _, test_a = split_examples(examples)
    _, test_b = split_examples(list(reversed(examples)))
    assert {e["contract_id"] for e in test_a} == {e["contract_id"] for e in test_b}


def test_test_fraction_is_approximately_respected():
    train_ids, test_ids = split_contract_ids([f"c{i}" for i in range(100)],
                                             test_frac=0.2)
    assert len(test_ids) == 20
    assert len(train_ids) == 80


def test_different_seeds_produce_different_partitions():
    _, test_a = split_contract_ids([f"c{i}" for i in range(100)], seed=1)
    _, test_b = split_contract_ids([f"c{i}" for i in range(100)], seed=2)
    assert set(test_a) != set(test_b)


def test_tiny_corpus_still_yields_a_nonempty_test_set():
    train_ids, test_ids = split_contract_ids(["a", "b", "c"], test_frac=0.2)
    assert len(test_ids) >= 1
    assert len(train_ids) >= 1


def test_three_way_split_all_sides_are_mutually_disjoint():
    examples = make_examples(n_contracts=100)
    train, val, test = split_examples_three_way(examples)
    train_ids = {e["contract_id"] for e in train}
    val_ids = {e["contract_id"] for e in val}
    test_ids = {e["contract_id"] for e in test}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_three_way_split_covers_every_example_exactly_once():
    examples = make_examples(n_contracts=100)
    train, val, test = split_examples_three_way(examples)
    assert len(train) + len(val) + len(test) == len(examples)


def test_three_way_split_test_side_matches_two_way_split():
    """
    The test set carved out by the three-way split must be IDENTICAL to
    the one the two-way split (used by Phase 4's baseline) produces at the
    same seed — otherwise Phase 5's reported numbers aren't comparable to
    Phase 4's on the test side, which is the whole point of using the same
    seed.
    """
    examples = make_examples(n_contracts=100)
    _, two_way_test = split_examples(examples)
    _, _, three_way_test = split_examples_three_way(examples)
    assert ({e["contract_id"] for e in two_way_test} ==
            {e["contract_id"] for e in three_way_test})


def test_three_way_split_val_is_nonempty():
    examples = make_examples(n_contracts=100)
    _, val, _ = split_examples_three_way(examples)
    assert len(val) > 0
