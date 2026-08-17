"""
Covenant Phase 5 follow-up — richer text representation, tried BEFORE
SMOTE/ensembling (TRD §4.3 steps 4-5), per the zero-shot diagnostic
finding (§6C.5): a zero-shot LLM with ZERO training examples cleared
~80-90% on the 3 categories the classical champion scores F1 = 0.000 on.
If the bottleneck were data quantity, giving a model LESS data (zero
examples) should have made it worse, not better. That rules out "not
enough examples" and points at "not reading the relationship the category
actually asks about" — see features/category_similarity.py's docstring
for the concrete per-category argument.

Same evidence-gating discipline as train_experiment.py: each candidate is
applied ON TOP OF the Phase 5 champion (bigrams + structural + tuned
thresholds) and kept only if it beats that champion's macro-F1. Neither
candidate here is on TRD's locked ladder — both are a deliberate,
documented deviation prompted by the diagnostic, not a silent add-on.

Two candidates:
  A) trigrams   — extend the champion's ngram_range from (1,2) to (1,3).
                  Cheap sanity check: does a wider bag-of-words window
                  alone help, before reaching for anything more targeted?
  B) similarity — for each category's own binary classifier, add ONE
                  extra column: cosine similarity between the clause's
                  TF-IDF vector and CUAD's own category-definition text
                  (also TF-IDF'd in the same vector space). This is a
                  per-category feature, not a shared matrix column, which
                  is why it needs its own training loop rather than
                  reusing train_experiment.run_step directly.

Usage:
    python backend/classifier/models/train_richer_features.py [--limit N] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))

from split import split_examples_three_way  # noqa: E402
from structural import StructuralFeatures, hstack_with_text  # noqa: E402
from category_similarity import (  # noqa: E402
    CategorySimilarity, load_category_questions, hstack_similarity,
)
import train_experiment as te  # noqa: E402

DATA_DIR = Path("data/processed/classifier")
OUT_DIR = Path("backend/classifier/models/artifacts")
EXPERIMENT = "covenant-classifier"

TARGET_CATEGORIES = [   # the 3 categories this follow-up is aimed at (F1=0.000 champion)
    "Competitive Restriction Exception",
    "Price Restrictions",
    "Third Party Beneficiary",
]

CHAMPION_CONFIG = dict(ngram_range=(1, 2), structural=True, tune_thresholds=True)


def run_similarity_step(train, val, test, categories, seed=42) -> dict:
    """Champion's text+structural features, plus a per-category
    similarity-to-CUAD-definition column added only for that category's
    own classifier."""
    mlb = MultiLabelBinarizer(classes=categories)
    y_train = mlb.fit_transform([r["labels"] for r in train])
    y_val = mlb.transform([r["labels"] for r in val])
    y_test = mlb.transform([r["labels"] for r in test])

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=CHAMPION_CONFIG["ngram_range"],
        min_df=3, max_df=0.9, sublinear_tf=True, strip_accents="unicode",
    )
    x_train_text = vectorizer.fit_transform([r["text"] for r in train])
    x_val_text = vectorizer.transform([r["text"] for r in val])
    x_test_text = vectorizer.transform([r["text"] for r in test])

    struct = StructuralFeatures()
    x_train_base = hstack_with_text(x_train_text, struct.fit_transform(train))
    x_val_base = hstack_with_text(x_val_text, struct.transform(val))
    x_test_base = hstack_with_text(x_test_text, struct.transform(test))

    questions = load_category_questions()
    sim = CategorySimilarity(vectorizer, questions)

    t0 = time.time()
    y_pred = np.zeros_like(y_test)
    thresholds, untrainable = {}, []

    for i, cat in enumerate(categories):
        col = y_train[:, i]
        if col.sum() == 0:
            untrainable.append(cat)
            continue
        sim_train = sim.column_for(cat, x_train_text)
        sim_val = sim.column_for(cat, x_val_text)
        sim_test = sim.column_for(cat, x_test_text)
        x_train = hstack_similarity(x_train_base, sim_train)
        x_val = hstack_similarity(x_val_base, sim_val)
        x_test = hstack_similarity(x_test_base, sim_test)

        clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                  solver="liblinear", random_state=seed)
        clf.fit(x_train, col)
        t = te.tune_threshold(y_val[:, i], clf.predict_proba(x_val)[:, 1])
        thresholds[cat] = t
        y_pred[:, i] = (clf.predict_proba(x_test)[:, 1] >= t).astype(int)

    train_seconds = time.time() - t0
    p, r, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0, labels=range(len(categories)))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0)

    return {
        "step": "similarity_to_definition", "label": "+ category-definition similarity",
        "rationale": "diagnostic showed relational gap, not data-quantity gap",
        "n_features": int(x_train_base.shape[1]) + 1,
        "train_seconds": train_seconds, "untrainable_categories": untrainable,
        "thresholds": thresholds,
        "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "per_category": [
            {"category": cat, "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f1[i]), "support_test": int(support[i]),
             "support_train": int(y_train[:, i].sum())}
            for i, cat in enumerate(categories)
        ],
    }


def log_to_mlflow(result: dict, split_info: dict, adopted: bool) -> None:
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"phase5-followup-{result['step']}"):
        mlflow.log_params({
            "phase": "5-followup", "step": result["step"],
            "on_locked_ladder": False, "n_features": result["n_features"], **split_info,
        })
        mlflow.log_metrics({k: result[k] for k in (
            "macro_precision", "macro_recall", "macro_f1",
            "micro_precision", "micro_recall", "micro_f1", "train_seconds")})
        mlflow.set_tag("adopted", str(adopted))
        mlflow.set_tag("rationale", result["rationale"])
        for row in result["per_category"]:
            key = row["category"].replace("/", "-").replace(" ", "_")[:80]
            mlflow.log_metrics({f"f1__{key}": row["f1"], f"precision__{key}": row["precision"],
                                 f"recall__{key}": row["recall"]})


def print_row(res: dict, delta: float, adopted: bool) -> None:
    print(f"  macro-F1 {res['macro_f1']:.4f}  micro-F1 {res['micro_f1']:.4f}  "
          f"delta {delta:+.4f}  -> {'ADOPTED' if adopted else 'REJECTED'}")
    by_cat = {r["category"]: r for r in res["per_category"]}
    for cat in TARGET_CATEGORIES:
        r = by_cat.get(cat)
        if r is None:
            print(f"    {cat:<38} (no examples in this run)")
            continue
        print(f"    {cat:<38} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    examples = te.load_examples(args.limit or None)
    train, val, test = split_examples_three_way(examples, seed=args.seed)
    categories = sorted({c for r in examples for c in r["labels"]})
    split_info = {
        "n_contracts_train": len({r["contract_id"] for r in train}),
        "n_contracts_val": len({r["contract_id"] for r in val}),
        "n_contracts_test": len({r["contract_id"] for r in test}),
    }

    print("=== rebuilding Phase 5 champion (bigrams + structural + tuned thresholds) ===")
    champion_step = te.LadderStep("champion", "Phase 5 champion", **CHAMPION_CONFIG)
    champion, _ = te.run_step(champion_step, train, val, test, categories, args.seed)
    print(f"  champion macro-F1 {champion['macro_f1']:.4f}\n")
    champ_by_cat = {x["category"]: x for x in champion["per_category"]}
    for cat in TARGET_CATEGORIES:
        r = champ_by_cat.get(cat)
        if r is None:
            print(f"    {cat:<38} (no examples in this run)")
            continue
        print(f"    {cat:<38} F1={r['f1']:.3f} (champion baseline)")

    print("\n=== candidate A: trigrams on top of champion ===")
    trigram_step = te.candidate_from(
        champion_step, "trigrams", "+ trigrams", {"ngram_range": (1, 3)},
        "cheap sanity check before a targeted feature")
    trigram_res, _ = te.run_step(trigram_step, train, val, test, categories, args.seed)
    trigram_delta = trigram_res["macro_f1"] - champion["macro_f1"]
    trigram_adopted = trigram_delta > 0
    print_row(trigram_res, trigram_delta, trigram_adopted)
    if not args.no_mlflow:
        log_to_mlflow(trigram_res, split_info, trigram_adopted)

    print("\n=== candidate B: category-definition similarity on top of champion ===")
    sim_res = run_similarity_step(train, val, test, categories, args.seed)
    sim_delta = sim_res["macro_f1"] - champion["macro_f1"]
    sim_adopted = sim_delta > 0
    print_row(sim_res, sim_delta, sim_adopted)
    if not args.no_mlflow:
        log_to_mlflow(sim_res, split_info, sim_adopted)

    print("\n" + "=" * 80)
    print(f"champion (bigrams+structural+thresholds)  macro-F1 {champion['macro_f1']:.4f}")
    print(f"+ trigrams                                 macro-F1 {trigram_res['macro_f1']:.4f}  "
          f"({trigram_delta:+.4f})  {'ADOPTED' if trigram_adopted else 'rejected'}")
    print(f"+ category-definition similarity           macro-F1 {sim_res['macro_f1']:.4f}  "
          f"({sim_delta:+.4f})  {'ADOPTED' if sim_adopted else 'rejected'}")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"phase5_followup_results_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed, "split": split_info,
            "champion": {"macro_f1": champion["macro_f1"], "per_category": champion["per_category"]},
            "trigrams": {"result": trigram_res, "delta": trigram_delta, "adopted": trigram_adopted},
            "similarity": {"result": sim_res, "delta": sim_delta, "adopted": sim_adopted},
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
