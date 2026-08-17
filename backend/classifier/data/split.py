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
