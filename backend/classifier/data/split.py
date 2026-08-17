"""
Covenant Phase 4 — contract-level train/test split.

The split is BY CONTRACT, never by segment. Segments from one contract
share vocabulary, defined terms, party names, drafting firm formatting and
often near-verbatim boilerplate; splitting by segment puts near-duplicates
on both sides of the boundary and inflates every score reported afterwards.
This is the single easiest way to produce a fraudulent-looking number on
this dataset, so it lives in one function with a test asserting
contract-disjointness rather than being re-hand-rolled per experiment.

The split is deterministic given (contract_ids, seed) — it depends only on
the sorted contract id list, not on segment order or corpus size, so a
contract stays on the same side of the boundary across reruns as long as
the seed and the corpus are unchanged.
"""

from __future__ import annotations

import random

DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.2
DEFAULT_VAL_FRAC = 0.16   # of the whole corpus, i.e. 20% of the training pool


def split_contract_ids(
    contract_ids, test_frac: float = DEFAULT_TEST_FRAC, seed: int = DEFAULT_SEED
) -> tuple[list[str], list[str]]:
    """Deterministically partition contract ids into (train_ids, test_ids)."""
    ids = sorted(set(contract_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, round(len(ids) * test_frac))
    return sorted(ids[n_test:]), sorted(ids[:n_test])


def split_examples(
    examples, test_frac: float = DEFAULT_TEST_FRAC, seed: int = DEFAULT_SEED
) -> tuple[list[dict], list[dict]]:
    """Partition segment-level examples by their contract_id."""
    train_ids, test_ids = split_contract_ids(
        (e["contract_id"] for e in examples), test_frac, seed
    )
    test_set = set(test_ids)
    train = [e for e in examples if e["contract_id"] not in test_set]
    test = [e for e in examples if e["contract_id"] in test_set]
    return train, test


def split_examples_three_way(
    examples,
    test_frac: float = DEFAULT_TEST_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Partition into (train, val, test), still by contract.

    The test set is carved out FIRST with exactly the same call the
    two-way split uses, so it contains exactly the same contracts as the
    Phase 4 baseline's test set at the same seed. Phase 5 numbers are
    therefore directly comparable to Phase 4's on the test side; only the
    training pool shrinks, which is why the ladder is measured against a
    same-data control run rather than against Phase 4's printed figure.

    Validation exists because per-category decision thresholds (ladder
    step 3) have to be chosen on data the model did not train on AND that
    is not the test set. Tuning thresholds on test would be selecting the
    reported metric directly — the number would be meaningless.
    """
    train_pool, test = split_examples(examples, test_frac, seed)
    # val_frac is expressed as a fraction of the WHOLE corpus, so rescale
    # it to a fraction of the remaining training pool.
    pool_frac = val_frac / (1.0 - test_frac)
    train, val = split_examples(train_pool, pool_frac, seed)
    return train, val, test
