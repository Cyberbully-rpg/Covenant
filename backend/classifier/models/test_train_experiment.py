"""
Tests for the Phase 5 ladder runner.

The property that matters most here is evidence-gating: each lever is
applied on top of the CURRENT CHAMPION, and is kept only if it improves
macro-F1 over that champion. A bug that instead chains every lever onto
the previous step regardless of whether it won would silently let a
rejected lever keep influencing every later result — "evidence-gated"
would become decorative rather than real. That is exercised directly
below by forcing a lever to lose and checking the next lever's input.

Also covered: threshold tuning falls back to 0.5 (not some arbitrary
grid value) when a category has zero validation positives, since nothing
was actually measured for that category.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import train_experiment as te  # noqa: E402


def make_rows(n=60):
    """
    Two learnable categories with distinct, easy vocabulary plus a
    "Sparse" category with only 2 positives -- realistic enough that a
    lever can plausibly win or lose on it.
    """
    rows = []
    for i in range(n):
        cid = f"c{i:03d}"
        rows.append({
            "contract_id": cid, "segment_id": f"{cid}_a",
            "text": "this agreement shall be governed by the laws of delaware " * 4,
            "labels": ["Governing Law"], "position": 0,
            "relative_position": 0.0, "n_segments_in_contract": 3,
            "char_len": 200, "header": "[preamble]", "scheme": "bare_nn",
            "is_oversized_split": False, "is_undersized": False, "has_parent": False,
        })
        rows.append({
            "contract_id": cid, "segment_id": f"{cid}_b",
            "text": "buyer may audit the books and records of seller annually " * 4,
            "labels": ["Audit Rights"], "position": 1,
            "relative_position": 0.5, "n_segments_in_contract": 3,
            "char_len": 220, "header": "2.1 Audit", "scheme": "bare_nn",
            "is_oversized_split": False, "is_undersized": False, "has_parent": False,
        })
        rows.append({
            "contract_id": cid, "segment_id": f"{cid}_c",
            "text": "the parties acknowledge receipt of the foregoing schedules " * 4,
            "labels": (["Sparse"] if i < 2 else []), "position": 2,
            "relative_position": 1.0, "n_segments_in_contract": 3,
            "char_len": 210, "header": "2.2 Misc", "scheme": "bare_nn",
            "is_oversized_split": False, "is_undersized": False, "has_parent": False,
        })
    return rows


@pytest.fixture
def split():
    rows = make_rows()
    categories = sorted({c for r in rows for c in r["labels"]})
    train, val, test = te.split_examples_three_way(rows, seed=42)
    return train, val, test, categories


def test_run_step_returns_metrics_and_artifacts(split):
    train, val, test, categories = split
    metrics, artifacts = te.run_step(te.CONTROL, train, val, test, categories, seed=42)
    assert "macro_f1" in metrics and "micro_f1" in metrics
    assert set(artifacts) >= {"vectorizer", "models", "thresholds", "categories"}


def test_learnable_categories_are_actually_learned(split):
    train, val, test, categories = split
    metrics, _ = te.run_step(te.CONTROL, train, val, test, categories, seed=42)
    by_cat = {r["category"]: r for r in metrics["per_category"]}
    for cat in ("Governing Law", "Audit Rights"):
        assert by_cat[cat]["f1"] > 0.5, f"{cat} should be trivially separable: {by_cat[cat]}"


def test_threshold_tuning_falls_back_to_half_with_no_val_positives():
    y_val = np.zeros(20, dtype=int)  # no positives at all for this category
    prob_val = np.random.RandomState(0).rand(20)
    assert te.tune_threshold(y_val, prob_val) == 0.5


def test_threshold_tuning_picks_from_the_grid_when_positives_exist():
    rng = np.random.RandomState(0)
    y_val = np.array([0] * 15 + [1] * 5)
    prob_val = np.concatenate([rng.rand(15) * 0.4, 0.6 + rng.rand(5) * 0.4])
    t = te.tune_threshold(y_val, prob_val)
    assert 0.05 <= t <= 0.95


def test_candidate_from_only_adds_the_new_lever_not_previous_rejects():
    """
    Ladder-gating contract: a candidate built from a champion carries
    exactly that champion's config plus the one new change -- it must not
    accumulate settings from steps that were never adopted.
    """
    champion_step = te.CONTROL  # ngram_range=(1,1), structural=False
    candidate = te.candidate_from(
        champion_step, "stepX", "label", {"structural": True}, "why"
    )
    assert candidate.ngram_range == (1, 1)   # inherited from champion, unchanged
    assert candidate.structural is True       # the new lever
    assert candidate.tune_thresholds is False  # never set anywhere -> stays off


def test_rejected_lever_does_not_leak_into_the_next_lever(monkeypatch, split):
    """
    If step1 loses to the control, step2's candidate must be built on top
    of the CONTROL's config, not step1's -- otherwise a rejected lever
    keeps silently influencing every later result.
    """
    train, val, test, categories = split

    real_run_step = te.run_step
    seen_configs = []

    def spying_run_step(step, *a, **kw):
        seen_configs.append(step)
        metrics, artifacts = real_run_step(step, *a, **kw)
        if step.name == "step1_ngrams":
            # force this lever to lose regardless of its real result
            metrics["macro_f1"] = -1.0
        return metrics, artifacts

    monkeypatch.setattr(te, "run_step", spying_run_step)

    champion, champion_artifacts = te.run_step(te.CONTROL, train, val, test, categories, 42)
    champion_step = te.CONTROL
    for name, label, changes, rationale in te.LEVERS[:2]:  # ngrams, then structural
        step = te.candidate_from(champion_step, name, label, changes, rationale)
        res, artifacts = te.run_step(step, train, val, test, categories, 42)
        if res["macro_f1"] > champion["macro_f1"]:
            champion, champion_step, champion_artifacts = res, step, artifacts

    # step1_ngrams was forced to lose, so step2_structural's actual config
    # must show ngram_range still (1,1) -- inherited from CONTROL, not step1.
    step2_config = seen_configs[-1]
    assert step2_config.ngram_range == (1, 1), (
        f"rejected step1_ngrams leaked into step2's config: {step2_config}"
    )
    assert step2_config.structural is True
