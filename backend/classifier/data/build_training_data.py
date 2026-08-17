"""
Covenant Phase 4 — Training data construction + pre-training data check.

Turns segments (Phase 2) + CUAD gold spans (Phase 3 eval set) into labeled
multi-label training examples:

    a segment carries label C if the segment's character span overlaps ANY
    gold answer span for category C in that contract.

This reuses Phase 3's span-overlap primitive rather than reimplementing it,
so "what counts as a hit" is defined in exactly one place across eval and
training.

Two things this script deliberately surfaces BEFORE any model is trained,
because both change how Phase 4 must be built:

  1. Per-category positive counts. CUAD is severely imbalanced (TRD §4.1);
     categories with very few positives will not produce meaningful F1 no
     matter what model is used, and that needs reporting as a finding
     rather than being buried in a macro average.

  2. Gold-span capture rate. A gold span that no segment overlaps is a
     label the classifier can never learn and retrieval can never hit —
     it's a segmentation ceiling, not a model failure. Measuring it here
     keeps Phase 2's quality visible in Phase 4's numbers.

Split discipline: any train/test split MUST be BY CONTRACT, never by
segment. Segments from one contract share vocabulary, formatting, and
defined terms; splitting by segment leaks that across the boundary and
inflates scores.

Usage:
    python backend/classifier/data/build_training_data.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmenter"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "harness"))

from segmenter import segment_contract   # noqa: E402
from scorer import spans_overlap         # noqa: E402

EVAL_DIR = Path("data/processed/eval")
OUT_DIR = Path("data/processed/classifier")


def load_jsonl(path: Path, limit: int | None = None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build(limit: int | None = None) -> dict:
    contracts = load_jsonl(EVAL_DIR / "contracts.jsonl", limit)
    contract_ids = {c["contract_id"] for c in contracts}

    # gold spans grouped by (contract, category)
    gold: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    categories: set[str] = set()
    for row in load_jsonl(EVAL_DIR / "eval_set.jsonl"):
        categories.add(row["category"])
        if row["contract_id"] in contract_ids and row["has_gold_span"]:
            gold[(row["contract_id"], row["category"])] = [
                (g["start"], g["end"]) for g in row["gold_spans"]
            ]

    all_categories = sorted(categories)

    examples = []
    pos_segments = Counter()          # category -> n segments labeled positive
    pos_contracts = defaultdict(set)  # category -> contracts with >=1 positive segment
    spans_total = 0
    spans_captured = 0

    for c in contracts:
        cid = c["contract_id"]
        segments = segment_contract(c["context"], doc_id=cid)
        seg_spans = [(s.start_char, s.end_char) for s in segments]

        for cat in all_categories:
            g_spans = gold.get((cid, cat), [])
            if not g_spans:
                continue
            for g in g_spans:
                spans_total += 1
                if any(spans_overlap(g, s) for s in seg_spans):
                    spans_captured += 1

        for seg, span in zip(segments, seg_spans):
            labels = [
                cat for cat in all_categories
                if any(spans_overlap(span, g) for g in gold.get((cid, cat), []))
            ]
            for cat in labels:
                pos_segments[cat] += 1
                pos_contracts[cat].add(cid)
            examples.append({
                "contract_id": cid,
                "segment_id": seg.segment_id,
                "text": seg.text,
                "labels": labels,
                "n_labels": len(labels),
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "training_segments.jsonl", "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_seg = len(examples)
    labeled = sum(1 for e in examples if e["labels"])
    return {
        "n_contracts": len(contracts),
        "n_segments": n_seg,
        "n_segments_with_any_label": labeled,
        "n_segments_unlabeled": n_seg - labeled,
        "gold_spans_total": spans_total,
        "gold_spans_captured": spans_captured,
        "gold_span_capture_rate": spans_captured / spans_total if spans_total else 0.0,
        "categories": all_categories,
        "pos_segments": dict(pos_segments),
        "pos_contracts": {k: len(v) for k, v in pos_contracts.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="contracts to use (0 = all 510)")
    args = ap.parse_args()

    stats = build(args.limit or None)

    print(f"contracts              : {stats['n_contracts']}")
    print(f"segments               : {stats['n_segments']}")
    print(f"  with >=1 label       : {stats['n_segments_with_any_label']} "
          f"({stats['n_segments_with_any_label']/stats['n_segments']*100:.1f}%)")
    print(f"  unlabeled (negatives): {stats['n_segments_unlabeled']} "
          f"({stats['n_segments_unlabeled']/stats['n_segments']*100:.1f}%)")
    print()
    print(f"gold spans             : {stats['gold_spans_total']}")
    print(f"  captured by a segment: {stats['gold_spans_captured']} "
          f"({stats['gold_span_capture_rate']*100:.2f}%)  <- segmentation ceiling")
    print()

    rows = sorted(stats["categories"], key=lambda c: stats["pos_segments"].get(c, 0))
    print(f"{'category':<45} {'pos_segs':>9} {'contracts':>10} {'seg_rate':>9}")
    print("-" * 76)
    for cat in rows:
        ps = stats["pos_segments"].get(cat, 0)
        pc = stats["pos_contracts"].get(cat, 0)
        print(f"{cat:<45} {ps:>9} {pc:>10} {ps/stats['n_segments']*100:>8.2f}%")

    counts = [stats["pos_segments"].get(c, 0) for c in stats["categories"]]
    counts_sorted = sorted(counts)
    print()
    print(f"positives per category: min {counts_sorted[0]}, "
          f"median {counts_sorted[len(counts_sorted)//2]}, max {counts_sorted[-1]}")
    print(f"categories with <50 positive segments : "
          f"{sum(1 for c in counts if c < 50)}/{len(counts)}")
    print(f"categories with <100 positive segments: "
          f"{sum(1 for c in counts if c < 100)}/{len(counts)}")

    with open(OUT_DIR / "label_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'training_segments.jsonl'} and label_stats.json")


if __name__ == "__main__":
    main()
