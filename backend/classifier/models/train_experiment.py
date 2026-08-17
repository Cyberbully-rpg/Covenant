"""
Covenant Phase 5 — classifier feature improvements (escalation ladder).

Runs TRD §4.3's locked priority order as separate, individually-measured
experiments rather than one bundled "improved model":

  step 0  control      unigram TF-IDF + weighted logreg (Phase 4's recipe)
  step 1  n-grams      bigrams in TF-IDF
  step 2  structural   + segmenter structural features alongside TF-IDF
  step 3  thresholds   + per-category decision thresholds tuned on val

Steps 4 (SMOTE) and 5 (logreg+XGBoost ensembling) are deliberately NOT run
here. They are the expensive end of the ladder and TRD §4.2's baseline-first
rule means they need evidence from steps 1-3 first — specifically, evidence
that the remaining gap is an imbalance/model-capacity problem rather than a
representation or threshold problem.

EVIDENCE GATING (the point of this file): each step is adopted into the
running configuration only if it improves macro-F1 over the current
champion. A step that doesn't win is reported as a negative result and
discarded, not silently kept because it was on the roadmap. Negative
results are logged to MLflow exactly like positive ones.

WHY A CONTROL RUN EXISTS: Phase 5 needs a validation split for threshold
tuning, which is carved out of Phase 4's training pool. Every ladder step
therefore trains on less data than Phase 4 did. Comparing step 1 against
Phase 4's printed 0.430 would confound "bigrams helped" with "trained on
fewer contracts". Step 0 re-runs Phase 4's exact recipe on Phase 5's data
so every delta below is attributable to the lever alone. The test set is
identical to Phase 4's at the same seed.

Usage:
    python backend/classifier/models/train_experiment.py [--limit N] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))

from split import split_examples_three_way  # noqa: E402
from structural import StructuralFeatures, hstack_with_text, FEATURE_NAMES  # noqa: E402

DATA_DIR = Path("data/processed/classifier")
OUT_DIR = Path("backend/classifier/models/artifacts")
EXPERIMENT = "covenant-classifier"

THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


@dataclass
class LadderStep:
    name: str
    label: str
    ngram_range: tuple = (1, 1)
    structural: bool = False
    tune_thresholds: bool = False
    rationale: str = ""


CONTROL = LadderStep(
    "step0_control", "control (Phase 4 recipe, Phase 5 data)",
    rationale="baseline every delta is measured against",
)

# Each lever is applied ON TOP OF THE CURRENT CHAMPION, not on top of the
# previous step. If bigrams lose, the structural run must not silently
# inherit bigrams — otherwise a rejected lever keeps influencing every
# later result and "evidence-gated" becomes decorative.
LEVERS = [
    ("step1_ngrams", "+ bigrams", {"ngram_range": (1, 2)},
     "TRD 4.3 step 1 — attacks similar-words/different-meaning confusion"),
    ("step2_structural", "+ structural features", {"structural": True},
     "TRD 4.3 step 2 — position/length/header signal TF-IDF cannot see"),
    ("step3_thresholds", "+ tuned thresholds", {"tune_thresholds": True},
     "TRD 4.3 step 3 — Phase 4 showed recall ~2x precision"),
]


def candidate_from(champion_step: LadderStep, name, label, changes, rationale) -> LadderStep:
    """Champion config + one new lever."""
    base = {k: v for k, v in asdict(champion_step).items()
            if k not in ("name", "label", "rationale")}
    base.update(changes)
    return LadderStep(name=name, label=label, rationale=rationale, **base)


def load_examples(limit: int | None = None) -> list[dict]:
    path = DATA_DIR / "training_segments.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run "
            "`python backend/classifier/data/build_training_data.py` first."
        )
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if "position" not in rows[0]:
        raise SystemExit(
            "training_segments.jsonl predates Phase 5 and has no structural "
            "fields — rebuild it with build_training_data.py."
        )
    if limit:
        keep = set(sorted({r["contract_id"] for r in rows})[:limit])
        rows = [r for r in rows if r["contract_id"] in keep]
    return rows


def tune_threshold(y_true_val: np.ndarray, prob_val: np.ndarray) -> float:
    """
    Pick the decision threshold maximizing F1 for ONE category on the
    validation split. Falls back to 0.5 when val has no positives for the
    category — with nothing to measure, a tuned threshold would be noise
    dressed up as a decision.
    """
    if y_true_val.sum() == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        pred = (prob_val >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_val, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def run_step(step: LadderStep, train, val, test, categories, seed: int = 42) -> tuple[dict, dict]:
    """Returns (metrics, artifacts). artifacts holds the fitted objects
    needed to reuse this exact step's model later — kept separate from the
    JSON-serializable metrics dict since sklearn objects aren't JSON-safe."""
    mlb = MultiLabelBinarizer(classes=categories)
    y_train = mlb.fit_transform([r["labels"] for r in train])
    y_val = mlb.transform([r["labels"] for r in val])
    y_test = mlb.transform([r["labels"] for r in test])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=step.ngram_range,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform([r["text"] for r in train])
    x_val = vectorizer.transform([r["text"] for r in val])
    x_test = vectorizer.transform([r["text"] for r in test])

    struct = None
    if step.structural:
        struct = StructuralFeatures()
        x_train = hstack_with_text(x_train, struct.fit_transform(train))
        x_val = hstack_with_text(x_val, struct.transform(val))
        x_test = hstack_with_text(x_test, struct.transform(test))

    t0 = time.time()
    y_pred = np.zeros_like(y_test)
    thresholds: dict[str, float] = {}
    untrainable: list[str] = []
    models: dict[str, LogisticRegression] = {}

    for i, cat in enumerate(categories):
        col = y_train[:, i]
        if col.sum() == 0:
            untrainable.append(cat)
            continue
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000,
            solver="liblinear", random_state=seed,
        )
        clf.fit(x_train, col)
        models[cat] = clf
        t = (tune_threshold(y_val[:, i], clf.predict_proba(x_val)[:, 1])
             if step.tune_thresholds else 0.5)
        thresholds[cat] = t
        y_pred[:, i] = (clf.predict_proba(x_test)[:, 1] >= t).astype(int)

    train_seconds = time.time() - t0

    p, r, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0, labels=range(len(categories))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0)

    metrics = {
        "step": step.name,
        "label": step.label,
        "rationale": step.rationale,
        "config": asdict(step),
        "n_features": int(x_train.shape[1]),
        "train_seconds": train_seconds,
        "untrainable_categories": untrainable,
        "thresholds": thresholds,
        "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "per_category": [
            {
                "category": cat, "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support_test": int(support[i]),
                "support_train": int(y_train[:, i].sum()),
                "threshold": thresholds.get(cat, 0.5),
            }
            for i, cat in enumerate(categories)
        ],
    }
    artifacts = {
        "vectorizer": vectorizer,
        "structural": struct,
        "models": models,
        "thresholds": thresholds,
        "categories": categories,
        "config": asdict(step),
    }
    return metrics, artifacts


def log_to_mlflow(result: dict, split_info: dict, adopted: bool) -> None:
    import mlflow

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"phase5-{result['step']}"):
        cfg = result["config"]
        mlflow.log_params({
            "phase": 5,
            "ladder_step": result["step"],
            "ngram_range": str(cfg["ngram_range"]),
            "structural_features": cfg["structural"],
            "tuned_thresholds": cfg["tune_thresholds"],
            "model": "logreg-ovr",
            "class_weight": "balanced",
            "split": "by-contract train/val/test",
            "n_features": result["n_features"],
            **split_info,
        })
        mlflow.log_metrics({
            k: result[k] for k in (
                "macro_precision", "macro_recall", "macro_f1",
                "micro_precision", "micro_recall", "micro_f1", "train_seconds")
        })
        # A step that lost is still evidence; record the verdict, don't hide it.
        mlflow.set_tag("adopted", str(adopted))
        mlflow.set_tag("rationale", result["rationale"])
        for row in result["per_category"]:
            key = row["category"].replace("/", "-").replace(" ", "_")[:80]
            mlflow.log_metrics({
                f"f1__{key}": row["f1"],
                f"precision__{key}": row["precision"],
                f"recall__{key}": row["recall"],
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    examples = load_examples(args.limit or None)
    train, val, test = split_examples_three_way(examples, seed=args.seed)
    categories = sorted({c for r in examples for c in r["labels"]})

    split_info = {
        "n_contracts_train": len({r["contract_id"] for r in train}),
        "n_contracts_val": len({r["contract_id"] for r in val}),
        "n_contracts_test": len({r["contract_id"] for r in test}),
    }
    print(f"split (by contract): {split_info['n_contracts_train']} train / "
          f"{split_info['n_contracts_val']} val / {split_info['n_contracts_test']} test")
    print(f"segments: {len(train)} / {len(val)} / {len(test)}")
    print(f"structural feature block: {len(FEATURE_NAMES)} columns\n")

    print(f"running {CONTROL.name} ({CONTROL.label}) ...", flush=True)
    champion, champion_artifacts = run_step(CONTROL, train, val, test, categories, args.seed)
    champion["macro_f1_delta_vs_champion"] = 0.0
    champion["adopted"] = True
    champion_step = CONTROL
    results, adopted_steps = [champion], [CONTROL.name]
    print(f"  macro-F1 {champion['macro_f1']:.4f}  "
          f"micro-F1 {champion['micro_f1']:.4f}\n")
    if not args.no_mlflow:
        log_to_mlflow(champion, split_info, True)

    for name, label, changes, rationale in LEVERS:
        step = candidate_from(champion_step, name, label, changes, rationale)
        print(f"running {name} ({label} on top of {champion_step.name}) ...", flush=True)
        res, artifacts = run_step(step, train, val, test, categories, args.seed)
        delta = res["macro_f1"] - champion["macro_f1"]
        adopted = delta > 0
        res["macro_f1_delta_vs_champion"] = delta
        res["applied_on_top_of"] = champion_step.name
        res["adopted"] = adopted
        if adopted:
            champion, champion_step, champion_artifacts = res, step, artifacts
            adopted_steps.append(name)
        results.append(res)
        print(f"  macro-F1 {res['macro_f1']:.4f}  micro-F1 {res['micro_f1']:.4f}  "
              f"delta {delta:+.4f}  -> {'ADOPTED' if adopted else 'REJECTED'}\n")
        if not args.no_mlflow:
            log_to_mlflow(res, split_info, adopted)

    print("=" * 92)
    print(f"{'step':<20} {'macro-P':>8} {'macro-R':>8} {'macro-F1':>9} "
          f"{'micro-F1':>9} {'delta':>8}  verdict")
    print("-" * 92)
    for res in results:
        print(f"{res['step']:<20} {res['macro_precision']:>8.4f} "
              f"{res['macro_recall']:>8.4f} {res['macro_f1']:>9.4f} "
              f"{res['micro_f1']:>9.4f} {res['macro_f1_delta_vs_champion']:>+8.4f}"
              f"  {'ADOPTED' if res['adopted'] else 'rejected'}")
    print("=" * 92)

    control = results[0]
    print(f"\nchampion: {champion['step']} ({champion['label']})")
    print(f"  macro-F1 {control['macro_f1']:.4f} -> {champion['macro_f1']:.4f} "
          f"({champion['macro_f1'] - control['macro_f1']:+.4f} vs control)")
    print(f"  micro-F1 {control['micro_f1']:.4f} -> {champion['micro_f1']:.4f} "
          f"({champion['micro_f1'] - control['micro_f1']:+.4f} vs control)")
    print(f"  macro-P  {control['macro_precision']:.4f} -> {champion['macro_precision']:.4f}"
          f"   macro-R {control['macro_recall']:.4f} -> {champion['macro_recall']:.4f}")
    print(f"  adopted steps: {adopted_steps}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "split": split_info, "categories": categories,
               "champion": champion["step"], "adopted_steps": adopted_steps,
               "results": results}
    # Seed-scoped filename: seed-stability runs must not clobber each other,
    # and a marginal lever's verdict is only trustworthy across seeds.
    out_path = OUT_DIR / f"phase5_ladder_results_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")

    model_path = OUT_DIR / f"phase5_champion_seed{args.seed}.joblib"
    joblib.dump(champion_artifacts, model_path)
    print(f"wrote {model_path}  (champion: {champion['step']})")


if __name__ == "__main__":
    main()
