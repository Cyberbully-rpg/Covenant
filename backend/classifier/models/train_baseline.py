"""
Covenant Phase 4 — classical multi-label classifier baseline.

TF-IDF + per-category weighted logistic regression over all 41 CUAD
categories (TRD §4.2, baseline-first and locked). One binary
one-vs-rest model per category, `class_weight="balanced"` so the rare
categories are not simply predicted-away by the majority class.

Reporting rules (TRD §4.1, non-negotiable):
  * macro-F1 AND micro-F1 AND full per-category precision/recall/F1.
  * A single blended accuracy number is never acceptable for this task.
    Accuracy is not computed here at all — with ~2% positive rate per
    category, predicting all-negative scores >97% "accurate" and is
    worthless.

Why unigrams here: TF-IDF n-grams are step 1 of the locked escalation
ladder (TRD §4.3). This run is the floor that step has to beat, so it
deliberately uses the weakest reasonable representation. Don't fold ladder
steps into this file — each one needs its own MLflow run to be evidence.

The train/test split is BY CONTRACT (see data/split.py). Segments are not
independent within a contract.

MLflow tracking starts here (Phase 4 in ARCHITECTURE.md §5): params, the
headline metrics, and the per-category table are logged to a local
file-backed store under `mlruns/`.

Usage:
    python backend/classifier/models/train_baseline.py [--limit N] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from split import split_examples  # noqa: E402

DATA_DIR = Path("data/processed/classifier")
OUT_DIR = Path("backend/classifier/models/artifacts")
EXPERIMENT = "covenant-classifier"


def load_examples(limit: int | None = None) -> list[dict]:
    path = DATA_DIR / "training_segments.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run "
            "`python backend/classifier/data/build_training_data.py` first."
        )
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit:
        keep = sorted({r["contract_id"] for r in rows})[:limit]
        keep = set(keep)
        rows = [r for r in rows if r["contract_id"] in keep]
    return rows


def train(limit: int | None = None, seed: int = 42) -> dict:
    examples = load_examples(limit)
    train_rows, test_rows = split_examples(examples, seed=seed)

    categories = sorted({c for r in examples for c in r["labels"]})
    mlb = MultiLabelBinarizer(classes=categories)
    y_train = mlb.fit_transform([r["labels"] for r in train_rows])
    y_test = mlb.transform([r["labels"] for r in test_rows])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform([r["text"] for r in train_rows])
    x_test = vectorizer.transform([r["text"] for r in test_rows])

    t0 = time.time()
    models: dict[str, LogisticRegression] = {}
    y_pred = np.zeros_like(y_test)
    untrainable: list[str] = []

    for i, cat in enumerate(categories):
        col = y_train[:, i]
        if col.sum() == 0:
            # No positives on the train side — a model cannot be fit at all.
            # Recorded explicitly rather than silently scoring as 0.0.
            untrainable.append(cat)
            continue
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(x_train, col)
        models[cat] = clf
        y_pred[:, i] = clf.predict(x_test)

    train_seconds = time.time() - t0

    p, r, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0, labels=range(len(categories))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0
    )

    per_category = [
        {
            "category": cat,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support_test": int(support[i]),
            "support_train": int(y_train[:, i].sum()),
            "trainable": cat in models,
        }
        for i, cat in enumerate(categories)
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "models": models, "categories": categories},
        OUT_DIR / "tfidf_logreg_baseline.joblib",
    )

    results = {
        "model": "tfidf-unigram + logreg(class_weight=balanced), one-vs-rest",
        "seed": seed,
        "n_contracts_train": len({r["contract_id"] for r in train_rows}),
        "n_contracts_test": len({r["contract_id"] for r in test_rows}),
        "n_segments_train": len(train_rows),
        "n_segments_test": len(test_rows),
        "n_features": int(x_train.shape[1]),
        "n_categories": len(categories),
        "untrainable_categories": untrainable,
        "train_seconds": train_seconds,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "per_category": per_category,
    }
    with open(OUT_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def log_to_mlflow(results: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="phase4-tfidf-logreg-baseline"):
        mlflow.log_params({
            "representation": "tfidf-unigram",
            "model": "logreg-ovr",
            "class_weight": "balanced",
            "min_df": 3,
            "max_df": 0.9,
            "split": "by-contract",
            "seed": results["seed"],
            "n_features": results["n_features"],
            "n_contracts_train": results["n_contracts_train"],
            "n_contracts_test": results["n_contracts_test"],
        })
        mlflow.log_metrics({
            k: results[k] for k in (
                "macro_precision", "macro_recall", "macro_f1",
                "micro_precision", "micro_recall", "micro_f1",
                "train_seconds",
            )
        })
        for row in results["per_category"]:
            key = row["category"].replace("/", "-").replace(" ", "_")[:80]
            mlflow.log_metrics({
                f"f1__{key}": row["f1"],
                f"precision__{key}": row["precision"],
                f"recall__{key}": row["recall"],
            })
        mlflow.log_artifact(str(OUT_DIR / "baseline_metrics.json"))


def report(results: dict) -> None:
    print(f"model     : {results['model']}")
    print(f"split     : by contract, seed {results['seed']}  "
          f"({results['n_contracts_train']} train / "
          f"{results['n_contracts_test']} test contracts)")
    print(f"segments  : {results['n_segments_train']} train / "
          f"{results['n_segments_test']} test")
    print(f"features  : {results['n_features']}   "
          f"fit in {results['train_seconds']:.1f}s")
    print()
    print(f"macro  P {results['macro_precision']:.3f}  "
          f"R {results['macro_recall']:.3f}  F1 {results['macro_f1']:.3f}")
    print(f"micro  P {results['micro_precision']:.3f}  "
          f"R {results['micro_recall']:.3f}  F1 {results['micro_f1']:.3f}")
    print("(no single blended accuracy is reported — see TRD §4.1)")
    print()
    print(f"{'category':<45} {'P':>6} {'R':>6} {'F1':>6} {'n_test':>7} {'n_train':>8}")
    print("-" * 82)
    for row in sorted(results["per_category"], key=lambda r: r["f1"]):
        print(f"{row['category']:<45} {row['precision']:>6.3f} {row['recall']:>6.3f} "
              f"{row['f1']:>6.3f} {row['support_test']:>7} {row['support_train']:>8}")

    if results["untrainable_categories"]:
        print(f"\nuntrainable (zero train positives): "
              f"{results['untrainable_categories']}")
    zero = [r["category"] for r in results["per_category"] if r["f1"] == 0.0]
    print(f"\ncategories at F1 = 0.000: {len(zero)}/{results['n_categories']}")
    print(f"wrote {OUT_DIR / 'baseline_metrics.json'} and "
          f"{OUT_DIR / 'tfidf_logreg_baseline.joblib'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="contracts to use (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    results = train(args.limit or None, args.seed)
    report(results)
    if not args.no_mlflow:
        log_to_mlflow(results)
        print(f"logged to MLflow experiment '{EXPERIMENT}' (file:./mlruns)")


if __name__ == "__main__":
    main()
