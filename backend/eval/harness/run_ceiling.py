"""
Covenant Phase 6 — segmentation ceiling.

Answers one question that has to be settled before any retrieval number
can be interpreted: **is the segmenter throwing answers away?**

A retriever can only ever return segments. If CUAD's gold span for a
question overlaps no segment at all, no ranker — however good — can hit
it. That fraction is a hard ceiling on hit_rate, and without it, a score
like 0.72 is uninterpretable: it could mean a mediocre ranker against an
easy ceiling, or an excellent ranker against a broken segmenter. Those
call for opposite work.

This is not a retriever and not a baseline. No embeddings, no ranking, no
model — just character-offset arithmetic against `scorer.spans_overlap`,
the same overlap definition every reported number uses. It writes no
results file.

Measured over the full corpus, the ceiling is 0.9985: 10 of 6,702 scored
rows have an unreachable gold span. So segmentation is effectively maxed
out and every remaining miss is a ranking failure. Note this cuts both
ways — it also means "improve the segmenter" is NOT a lever for retrieval
accuracy, whatever intuition suggests (ARCHITECTURE.md §6D.4).

Usage:
    python backend/eval/harness/run_ceiling.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmenter"))

from segmenter import segment_contract  # noqa: E402
from scorer import spans_overlap  # noqa: E402

EVAL_DIR = Path("data/processed/eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="contracts (0 = all 510)")
    ap.add_argument("--show-worst", type=int, default=8,
                    help="how many worst-covered categories to list")
    args = ap.parse_args()
    limit = args.limit if args.limit > 0 else None

    contracts = {}
    with open(EVAL_DIR / "contracts.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            contracts[row["contract_id"]] = row["context"]
            if limit and len(contracts) >= limit:
                break

    print(f"segmenting {len(contracts)} contracts...")
    segs = {cid: segment_contract(t, doc_id=cid) for cid, t in contracts.items()}

    n = reachable = 0
    per_cat = defaultdict(lambda: [0, 0])
    with open(EVAL_DIR / "eval_set.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["contract_id"] not in contracts or not row["has_gold_span"]:
                continue
            n += 1
            gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
            spans = [(s.start_char, s.end_char) for s in segs[row["contract_id"]]]
            ok = any(spans_overlap(sp, g) for sp in spans for g in gold)
            reachable += ok
            stat = per_cat[row["category"]]
            stat[0] += 1
            stat[1] += ok

    print(f"\nscored rows        : {n}")
    print(f"reachable          : {reachable}")
    print(f"unreachable        : {n - reachable} ({(n - reachable) / n:.2%})")
    print(f"CEILING on hit_rate: {reachable / n:.4f}")

    worst = sorted(((hits / tot, tot, cat) for cat, (tot, hits) in per_cat.items()),
                   key=lambda x: x[0])[:args.show_worst]
    print(f"\nworst-covered categories")
    print(f"  {'ceiling':>8}  {'n':>5}  category")
    for rate, tot, cat in worst:
        print(f"  {rate:>8.4f}  {tot:>5}  {cat}")


if __name__ == "__main__":
    main()
