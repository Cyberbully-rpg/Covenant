"""
Covenant Phase 6 — dense (Chroma + sentence-transformers) retrieval vs.
the Phase 3 baselines.

SUPERSEDED by run_retrieval_variants.py, which measures this same dense
path alongside every other variant in one pass and re-runs the baselines
as controls. Kept, and pinned to the original unwindowed collection, so
the historical §6D.2 result stays reproducible exactly as it was first
measured — including the truncation confound §6D.3 later found in it.
Use the variant sweep for anything new.

Runs the exact same eval set, the exact same scorer.score_query()/
aggregate_scores(), and the exact same k as run_baselines.py — the only
variable that changes is the retriever. This is the harness validation
the roadmap requires before Phase 7 (generation) gets built on top of
retrieval: Phase 2's ARCHITECTURE.md note framed the question precisely —
"Dense embedding retrieval must beat [TF-IDF's] 0.693 hit_rate@5 by a
worthwhile margin to justify Chroma + an embedding model + an ingestion
pipeline" — and this script is what actually measures that, rather than
assuming it.

REQUIRES the Chroma collection already built by
backend/rag/ingestion/ingest.py over the SAME set of contracts being
evaluated here (or a superset) — this script does not ingest, only
queries. Run ingest.py first.

Usage:
    python backend/eval/harness/run_dense_baseline.py [--limit N] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "retrieval"))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag"))

from scorer import score_query, aggregate_scores  # noqa: E402
from retrieve import ChromaRetriever  # noqa: E402
from ingestion.ingest import LEGACY_COLLECTION_NAME  # noqa: E402

EVAL_DIR = Path("data/processed/eval")


def load_contract_ids(limit: int | None) -> set[str]:
    ids = set()
    with open(EVAL_DIR / "contracts.jsonl", encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["contract_id"])
            if limit and len(ids) >= limit:
                break
    return ids


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
                     help="number of contracts to evaluate (default 50; use 0 for all 510 -- "
                          "must match what was ingested)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
    contract_ids = load_contract_ids(limit)
    rows = load_eval_rows(contract_ids)

    print(f"contracts: {len(contract_ids)}   eval rows: {len(rows)}   k={args.k}")

    # Pinned to the original unwindowed collection ON PURPOSE. Collections
    # are now namespaced per model and windowing mode, so an unpinned
    # ChromaRetriever() here would silently measure the *windowed* index and
    # write the result under the "chroma_dense" key — relabelling a
    # different experiment as the historical one. This script exists to
    # reproduce §6D.2, and nothing else.
    retriever = ChromaRetriever(collection_name=LEGACY_COLLECTION_NAME)
    n_in_collection = retriever.collection.count()
    print(f"chroma collection size: {n_in_collection} segments\n")

    scores = []
    skipped = 0
    for i, row in enumerate(rows):
        if not row["has_gold_span"]:
            skipped += 1
            continue
        retrieved = retriever.retrieve(row["question"], row["contract_id"], k=args.k)
        gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
        scores.append(score_query(retrieved, gold))
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(rows)} rows scored", flush=True)

    agg = aggregate_scores(scores, n_skipped_no_gold=skipped)

    print(f"\n[chroma_dense]")
    print(f"  scored rows        : {agg['n_scored']}")
    print(f"  skipped (no gold)  : {agg['n_skipped_no_gold']}")
    print(f"  hit_rate@{args.k}         : {agg['hit_rate']:.4f}")
    print(f"  gold_recall        : {agg['gold_recall']:.4f}")
    print(f"  gold_density       : {agg['gold_density']:.4f}")
    print(f"  mean_retrieved_chars: {agg['mean_retrieved_chars']:.1f}")

    # merge into the existing baseline_results.json rather than overwriting it,
    # so tfidf/random (Phase 3) and chroma_dense (Phase 6) live side by side.
    out_path = EVAL_DIR / "baseline_results.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    # A partial run must never land in the results file. Every number there is
    # quoted as "510 contracts, 6,702 scored rows"; merging a --limit 3 run
    # under the same key silently replaces a full-corpus result with a 37-row
    # one that still reads as authoritative.
    if limit is not None:
        print(f"\nNOT writing results: partial run ({len(contract_ids)} contracts). "
              f"Re-run with --limit 0 to record full-corpus numbers.")
        return

    existing.setdefault("results", {})["chroma_dense"] = agg
    existing["k"] = args.k
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    if "tfidf" in existing.get("results", {}):
        lift = agg["hit_rate"] - existing["results"]["tfidf"]["hit_rate"]
        print(f"\nchroma_dense vs tfidf (hit_rate@{args.k}): {lift:+.4f}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
