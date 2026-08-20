"""
Covenant Phase 6 — rank diagnostic.

ARCHITECTURE.md §6D.3 leaves one question unanswered. The default
retriever hits 0.7195 at k=5, and the segmenter ceiling is 0.9985, so
~28% of questions have a findable answer that the ranker fails to
surface. That number says the misses exist. It does not say where they
are, and the right next move depends entirely on that:

  - if the gold segment usually sits just below the cutoff (rank 6-10),
    the ranker is nearly right and small refinements — a better lexical
    scorer, tuned weights, a reranker over the top 20 — will convert a
    lot of misses cheaply;
  - if it sits deep (rank 30+) or isn't found at all, the ranker is not
    seeing the relevant signal, and no amount of reordering the top of
    the list helps. That calls for a different representation (stronger
    embedding model, query rewriting), not tuning.

So this script reports the RANK of the first gold-overlapping segment,
not just whether it landed in the top 5. Three outputs:

  1. hit_rate@k curve (k = 1,3,5,10,20,50) — how much is bought by simply
     showing more chunks, which is the cheapest available lever and needs
     to be quantified before anything expensive is attempted;
  2. the rank distribution of the first gold hit, plus MRR;
  3. per-category hit_rate@5, since CUAD's 41 categories are wildly
     uneven and an aggregate number hides which ones are actually broken.

Diagnostic only — writes no results file and changes no reported number.
Fusion depth is raised to 50 here so ranks past the production depth of
20 are observable; the top of the ranking is unaffected for the k values
that matter, but this is why the @5 figure here can differ in the last
digit from the §6D.3 table.

Usage:
    python backend/eval/harness/run_rank_diagnostic.py --limit 0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmenter"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "retrieval"))

from segmenter import segment_contract  # noqa: E402
from scorer import spans_overlap  # noqa: E402
from tfidf import TfidfRetriever  # noqa: E402
from retrieve import ChromaRetriever  # noqa: E402
from hybrid import HybridRetriever  # noqa: E402
from ingestion.ingest import collection_name, split_windows, window_chars_for  # noqa: E402

EVAL_DIR = Path("data/processed/eval")
DIAG_DEPTH = 50
K_CURVE = [1, 3, 5, 10, 20, 50]


def load_contracts(limit):
    contracts = {}
    with open(EVAL_DIR / "contracts.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            contracts[row["contract_id"]] = row["context"]
            if limit and len(contracts) >= limit:
                break
    return contracts


def load_eval_rows(contract_ids):
    rows = []
    with open(EVAL_DIR / "eval_set.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["contract_id"] in contract_ids:
                rows.append(row)
    return rows


def first_gold_rank(ranked_spans, gold):
    """1-indexed rank of the first retrieved span touching any gold span.
    None if no ranked span touches gold within the depth searched."""
    for i, span in enumerate(ranked_spans, start=1):
        if any(spans_overlap(span, g) for g in gold):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", default="hybrid_bigram",
                    choices=["hybrid_bigram", "tfidf_bigram", "dense_win"])
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
    contracts = load_contracts(limit)
    rows = load_eval_rows(set(contracts))
    print(f"contracts: {len(contracts)}   eval rows: {len(rows)}   depth={DIAG_DEPTH}")
    print(f"variant: {args.variant}\n")

    print("segmenting...")
    segs = {cid: segment_contract(t, doc_id=cid) for cid, t in contracts.items()}

    # Chroma rejects n_results larger than the matching record count, so
    # derive each contract's window count locally -- same arithmetic
    # ingestion used, no extra query.
    win_chars = window_chars_for()
    n_windows = {
        cid: sum(len(split_windows((s.embedding_text.strip() or s.text.strip() or s.header),
                                   win_chars))
                 for s in slist if (s.embedding_text.strip() or s.text.strip() or s.header))
        for cid, slist in segs.items()
    }

    lexical = TfidfRetriever(ngram_range=(1, 2))
    dense = ChromaRetriever(collection_name=collection_name(windowed=True))
    hybrid = HybridRetriever(collection_name=collection_name(windowed=True),
                             depth=DIAG_DEPTH, lexical_ngram_range=(1, 2))

    by_contract = defaultdict(list)
    skipped = 0
    for row in rows:
        if not row["has_gold_span"]:
            skipped += 1
            continue
        by_contract[row["contract_id"]].append(row)

    ranks = []            # None means "not found within DIAG_DEPTH"
    per_category = defaultdict(list)
    for n, (cid, crows) in enumerate(by_contract.items(), start=1):
        segments = segs[cid]
        questions = [r["question"] for r in crows]
        depth = min(DIAG_DEPTH, len(segments))
        dense_depth = min(depth, max(1, n_windows[cid]))

        if args.variant == "tfidf_bigram":
            batch = [lexical.rank_spans(q, segments, depth) for q in questions]
        elif args.variant == "dense_win":
            batch = dense.rank_spans_batch(questions, cid, dense_depth)
        else:
            hybrid.depth = dense_depth
            batch = hybrid.retrieve_batch(questions, segments, cid, depth)

        for row, ranked in zip(crows, batch):
            gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
            r = first_gold_rank(ranked, gold)
            ranks.append(r)
            per_category[row["category"]].append(r)

        if n % 100 == 0:
            print(f"  ...{n}/{len(by_contract)} contracts", flush=True)

    total = len(ranks)
    found = [r for r in ranks if r is not None]

    print(f"\nscored rows: {total}   (skipped, no gold: {skipped})\n")

    print("hit_rate@k curve")
    print(f"  {'k':>4}  {'hit_rate':>9}  {'gain vs k=5':>12}")
    at5 = sum(1 for r in found if r <= 5) / total
    for k in K_CURVE:
        hr = sum(1 for r in found if r <= k) / total
        marker = "  <- current" if k == 5 else ""
        print(f"  {k:>4}  {hr:>9.4f}  {hr - at5:>+12.4f}{marker}")

    print("\nwhere the first gold segment actually ranks")
    buckets = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50)]
    for lo, hi in buckets:
        c = sum(1 for r in found if lo <= r <= hi)
        label = f"rank {lo}" if lo == hi else f"rank {lo}-{hi}"
        print(f"  {label:<14} {c:>6}  {c / total:>7.2%}")
    missing = total - len(found)
    print(f"  {'not in top ' + str(DIAG_DEPTH):<14} {missing:>6}  {missing / total:>7.2%}")

    mrr = sum(1 / r for r in found) / total
    print(f"\n  MRR                 : {mrr:.4f}")
    print(f"  median rank (found) : {statistics.median(found):.0f}")
    print(f"  mean rank (found)   : {statistics.mean(found):.1f}")

    print("\nworst categories by hit_rate@5 (n >= 20)")
    stats = []
    for cat, rs in per_category.items():
        if len(rs) < 20:
            continue
        f = [r for r in rs if r is not None]
        stats.append((sum(1 for r in f if r <= 5) / len(rs), len(rs),
                      statistics.median(f) if f else float("nan"), cat))
    stats.sort()
    print(f"  {'hit@5':>7}  {'n':>5}  {'med rank':>8}  category")
    for hr, n_rows, med, cat in stats[:12]:
        print(f"  {hr:>7.3f}  {n_rows:>5}  {med:>8.0f}  {cat}")
    print("\n  best:")
    for hr, n_rows, med, cat in stats[-5:]:
        print(f"  {hr:>7.3f}  {n_rows:>5}  {med:>8.0f}  {cat}")


if __name__ == "__main__":
    main()
