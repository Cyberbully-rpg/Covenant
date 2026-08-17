"""
Tests for the Phase 4 baseline trainer.

These are contract tests on the training/reporting path, not accuracy
tests — model quality is measured by the run itself, not asserted here.
What is asserted is the stuff that would silently corrupt a reported
number: that the split used for training is contract-disjoint, that
macro AND micro AND per-category figures are all present (TRD §4.1), and
that a category with no training positives is surfaced as untrainable
rather than quietly scoring 0.0 and dragging the macro average down
without explanation.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import train_baseline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from split import split_contract_ids  # noqa: E402


def make_rows():
    """Two learnable categories with distinct vocabulary, one with zero positives."""
    rows = []
    for i in range(40):
        cid = f"c{i:03d}"
        rows.append({
            "contract_id": cid, "segment_id": f"{cid}_a",
            "text": "this agreement shall be governed by the laws of delaware " * 3,
            "labels": ["Governing Law"],
        })
        rows.append({
            "contract_id": cid, "segment_id": f"{cid}_b",
            "text": "buyer may audit the books and records of seller annually " * 3,
            "labels": ["Audit Rights"],
        })
        rows.append({
            "contract_id": cid, "segment_id": f"{cid}_c",
            "text": "the parties acknowledge receipt of the foregoing schedules " * 3,
            "labels": [],
        })
    return rows


@pytest.fixture
def trained(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "training_segments.jsonl", "w", encoding="utf-8") as f:
        for row in make_rows():
            f.write(json.dumps(row) + "\n")
    # "Ghost Category" exists in the label space but has no positive anywhere.
    monkeypatch.setattr(train_baseline, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_baseline, "OUT_DIR", tmp_path / "artifacts")
    return train_baseline.train()


def test_split_used_for_training_is_contract_disjoint(trained):
    assert trained["n_contracts_train"] + trained["n_contracts_test"] == 40
    assert trained["n_contracts_test"] > 0
    assert trained["n_contracts_train"] > 0


def test_reports_macro_and_micro_and_per_category(trained):
    for key in ("macro_precision", "macro_recall", "macro_f1",
                "micro_precision", "micro_recall", "micro_f1"):
        assert key in trained, f"{key} missing — TRD §4.1 requires macro AND micro"
    assert len(trained["per_category"]) == trained["n_categories"]
    for row in trained["per_category"]:
        assert {"precision", "recall", "f1", "support_test", "support_train"} <= set(row)


def test_never_reports_a_blended_accuracy(trained):
    """TRD §4.1: a single blended accuracy number is never acceptable here."""
    assert not any("accuracy" in k for k in trained)


def test_learnable_categories_are_actually_learned(trained):
    by_cat = {r["category"]: r for r in trained["per_category"]}
    for cat in ("Governing Law", "Audit Rights"):
        assert by_cat[cat]["f1"] > 0.5, f"{cat} should be trivially separable, got {by_cat[cat]}"


def test_category_with_no_train_positives_is_flagged_untrainable(tmp_path, monkeypatch):
    """
    A category whose only positives land on the test side must be named in
    untrainable_categories — otherwise it scores 0.0, drags the macro
    average down, and looks like a model failure instead of a data gap.
    """
    rows = make_rows()
    _, test_ids = split_contract_ids({r["contract_id"] for r in rows})
    held_out = test_ids[0]
    rows = [
        dict(r, labels=r["labels"] + ["Ghost Category"])
        if r["contract_id"] == held_out and r["segment_id"].endswith("_c") else r
        for r in rows
    ]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "training_segments.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    monkeypatch.setattr(train_baseline, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_baseline, "OUT_DIR", tmp_path / "artifacts")

    results = train_baseline.train()
    assert "Ghost Category" in results["untrainable_categories"]
    ghost = next(r for r in results["per_category"] if r["category"] == "Ghost Category")
    assert ghost["trainable"] is False
    assert ghost["support_train"] == 0
    assert ghost["support_test"] > 0


def test_artifacts_written(trained, tmp_path):
    out = tmp_path / "artifacts"
    assert (out / "baseline_metrics.json").exists()
    assert (out / "tfidf_logreg_baseline.joblib").exists()
