"""
Covenant Phase 3, Step 2 — Run the baseline retrievers over the eval set.

Wires Step 1's eval set + the Phase 2 segmenter + the baselines + the
scorer into one end-to-end pass, producing the first real numbers this
project has for retrieval quality.

Scoring follows the locked Step 2 decision: only rows where CUAD marks the
category PRESENT (has_gold_span=True) are scored, since span overlap is
undefined without a gold span. Absent-category rows are counted and
reported as n_skipped_no_gold, never silently dropped.

Every reported hit_rate is printed alongside gold_density and
mean_retrieved_chars, because hit rate alone is gameable by chunk size
(see scorer.py's module docstring).

Usage:
    python backend/eval/harness/run_baselines.py [--limit N] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmenter"))

from segmenter import segment_contract           # noqa: E402
from scorer import score_query, aggregate_scores  # noqa: E402
from baselines import RandomRetriever, LexicalRetriever  # noqa: E402

EVAL_DIR = Path("data/processed/eval")
SCOPE_CLAIM = (
    "validated against CUAD's contract distribution using CUAD's 41 "
    "templated category probes"
)


def load_contracts(limit: int | None) -> dict[str, str]:
    contracts = {}
    with open(EVAL_DIR / "contracts.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            contracts[row["contract_id"]] = row["context"]
            if limit and len(contracts) >= limit:
                break
    return contracts


def load_eval_rows(contract_ids: set[str]) -> list[dict]:
    rows = []
    with open(EVAL_DIR / "eval_set.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["contract_id"] in contract_ids:
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50,
                    help="number of contracts to evaluate (default 50; use 0 for all 510)")
    ap.add_argument("--k", type=int, default=5, help="chunks retrieved per query")
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
    contracts = load_contracts(limit)
    rows = load_eval_rows(set(contracts))

    print(f"scope: {SCOPE_CLAIM}")
    print(f"contracts: {len(contracts)}   eval rows: {len(rows)}   k={args.k}\n")

    # segment each contract once, reused across every retriever and query
    print("segmenting contracts...")
    segments_by_contract = {
        cid: segment_contract(text, doc_id=cid) for cid, text in contracts.items()
    }
    seg_counts = [len(v) for v in segments_by_contract.values()]
    print(f"  {sum(seg_counts)} segments total, "
          f"mean {sum(seg_counts)/len(seg_counts):.1f} per contract\n")

    retrievers = [RandomRetriever(seed=0), LexicalRetriever()]
    results = {}

    for r in retrievers:
        scores = []
        skipped = 0
        for row in rows:
            if not row["has_gold_span"]:
                skipped += 1
                continue
            segments = segments_by_contract[row["contract_id"]]
            retrieved = r.retrieve(row["question"], segments, k=args.k)
            gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
            scores.append(score_query(retrieved, gold))

        agg = aggregate_scores(scores, n_skipped_no_gold=skipped)
        results[r.name] = agg

        print(f"[{r.name}]")
        print(f"  scored rows        : {agg['n_scored']}")
        print(f"  skipped (no gold)  : {agg['n_skipped_no_gold']}")
        print(f"  hit_rate@{args.k}         : {agg['hit_rate']:.4f}")
        print(f"  gold_recall        : {agg['gold_recall']:.4f}")
        print(f"  gold_density       : {agg['gold_density']:.4f}")
        print(f"  mean_retrieved_chars: {agg['mean_retrieved_chars']:.1f}")
        print()

    lift = results["tfidf"]["hit_rate"] - results["random"]["hit_rate"]
    print(f"tfidf lift over random (hit_rate@{args.k}): {lift:+.4f}")

    out = EVAL_DIR / "baseline_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "scope_claim": SCOPE_CLAIM,
            "k": args.k,
            "n_contracts": len(contracts),
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
