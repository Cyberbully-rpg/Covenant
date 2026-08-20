"""
Covenant Phase 6 — retrieval variant sweep.

One runner, several retrievers, ONE eval set. Every variant here is scored
by the same scorer.score_query()/aggregate_scores() over the same rows at
the same k as Phase 3's run_baselines.py and Phase 6's
run_dense_baseline.py, so every number in baseline_results.json stays
mutually comparable. The only variable that changes between variants is
the retriever.

Variants
--------
  tfidf          Phase 3's lexical baseline, recomputed here as a control —
                 if this doesn't reproduce 0.6934 the harness has drifted.
  tfidf_bigram   Same, with ngram_range=(1,2). Tests whether multi-word
                 legal terms of art ("change of control") beat unigrams.
  dense          Unwindowed MiniLM, the original §6D.2 collection. Present
                 so the truncation fix is measured against its own
                 predecessor rather than against a remembered number.
  dense_win      Windowed MiniLM — same model, same corpus, truncation
                 removed (ingest.py's WINDOWING note).
  hybrid         RRF fusion of tfidf + dense_win (hybrid.py).

Each variant's result is merged into baseline_results.json under its own
key; nothing is overwritten except the variant being re-run.

Every reported hit_rate is printed alongside gold_density and
mean_retrieved_chars, because hit rate alone is gameable by chunk size
(scorer.py's module docstring). Windowed dense retrieval returns
parent-sized spans specifically so that pair stays honest.

Usage:
    python backend/eval/harness/run_retrieval_variants.py --variants tfidf,dense_win,hybrid
    python backend/eval/harness/run_retrieval_variants.py --variants all --limit 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmenter"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "retrieval"))

from segmenter import segment_contract  # noqa: E402
from scorer import score_query, aggregate_scores  # noqa: E402
from tfidf import TfidfRetriever  # noqa: E402
from retrieve import ChromaRetriever  # noqa: E402
from hybrid import HybridRetriever  # noqa: E402
from rerank import CrossEncoderReranker  # noqa: E402
from query import CleanQueryRetriever  # noqa: E402
from lead_prior import LeadPrior, PriorRetriever  # noqa: E402
from ingestion.ingest import collection_name, LEGACY_COLLECTION_NAME  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "classifier" / "data"))
from split import split_contract_ids  # noqa: E402

EVAL_DIR = Path("data/processed/eval")
ALL_VARIANTS = ["tfidf", "tfidf_bigram", "dense", "dense_win", "hybrid", "hybrid_bigram",
                "tfidf_bigram_prior", "hybrid_bigram_prior", "hybrid_char_prior",
                "hybrid_bigram_rerank", "hybrid_bigram_prior_rerank",
                "hybrid_bigram_prior_cleanq", "hybrid_bigram_prior_cleanq_rerank"]


def build_variant(key: str, prior: LeadPrior | None = None):
    """Returns call(questions, segments, contract_id, k) -> list of span lists.

    Everything is batched per contract, not per question. The dense half of
    any variant pays a fixed Chroma metadata-filter cost per call, and all
    of a contract's questions share the same filter, so per-contract
    batching is a ~13x reduction in calls with identical results (see
    retrieve.py `_query_many`).
    """
    if key == "tfidf":
        r = TfidfRetriever()
        return lambda qs, s, c, k: [r.retrieve(q, s, k) for q in qs]
    if key == "tfidf_bigram":
        r = TfidfRetriever(ngram_range=(1, 2))
        return lambda qs, s, c, k: [r.retrieve(q, s, k) for q in qs]
    if key == "dense":
        r = ChromaRetriever(collection_name=LEGACY_COLLECTION_NAME)
        return lambda qs, s, c, k: r.retrieve_batch(qs, c, k)
    if key == "dense_win":
        r = ChromaRetriever(collection_name=collection_name(windowed=True))
        return lambda qs, s, c, k: r.retrieve_batch(qs, c, k)
    if key == "hybrid":
        r = HybridRetriever(collection_name=collection_name(windowed=True))
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    if key == "hybrid_bigram":
        r = HybridRetriever(collection_name=collection_name(windowed=True),
                            lexical_ngram_range=(1, 2))
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    if key == "tfidf_bigram_prior":
        if prior is None:
            raise SystemExit("tfidf_bigram_prior needs a fitted LeadPrior")
        base = TfidfRetriever(ngram_range=(1, 2))
        # Ask whether the dense half still earns its keep once the prior
        # handles the metadata categories: if this matches the hybrid
        # version, Chroma is carrying no weight and the simpler system wins
        # (CLAUDE.md — don't add complexity ahead of demonstrated need).
        wrapped = PriorRetriever(
            type("_LexBatch", (), {
                "retrieve_batch": staticmethod(
                    lambda qs, s, c, k: [base.retrieve(q, s, k) for q in qs])
            })(),
            prior, name=key)
        return lambda qs, s, c, k: wrapped.retrieve_batch(qs, s, c, k)
    if key == "hybrid_bigram_prior":
        if prior is None:
            raise SystemExit("hybrid_bigram_prior needs a fitted LeadPrior")
        base = HybridRetriever(collection_name=collection_name(windowed=True),
                               lexical_ngram_range=(1, 2))
        r = PriorRetriever(base, prior, name=key)
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    if key == "hybrid_char_prior":
        # Adds character n-grams as a third RRF ranker — the "fuzzy search"
        # lever. Equal weight, untuned, same discipline as the other two.
        if prior is None:
            raise SystemExit("hybrid_char_prior needs a fitted LeadPrior")
        base = HybridRetriever(collection_name=collection_name(windowed=True),
                               lexical_ngram_range=(1, 2), w_char=1.0)
        r = PriorRetriever(base, prior, name=key)
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    if key == "hybrid_bigram_rerank":
        base = HybridRetriever(collection_name=collection_name(windowed=True),
                               lexical_ngram_range=(1, 2))
        r = CrossEncoderReranker(base, name=key)
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    if key == "hybrid_bigram_prior_rerank":
        if prior is None:
            raise SystemExit("hybrid_bigram_prior_rerank needs a fitted LeadPrior")
        base = HybridRetriever(collection_name=collection_name(windowed=True),
                               lexical_ngram_range=(1, 2))
        # Rerank FIRST, then apply the prior. The other order would let the
        # cross-encoder demote the lead segment straight back out of the top
        # k, undoing the one thing it cannot score for itself: the metadata
        # categories whose answer shares no vocabulary with the question.
        r = PriorRetriever(CrossEncoderReranker(base), prior, name=key)
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    if key == "hybrid_bigram_prior_cleanq":
        if prior is None:
            raise SystemExit("hybrid_bigram_prior_cleanq needs a fitted LeadPrior")
        base = HybridRetriever(collection_name=collection_name(windowed=True),
                               lexical_ngram_range=(1, 2))
        # CleanQuery sits BELOW the prior: the prior reads the category out of
        # the original probe text, which stripping would remove.
        r = PriorRetriever(CleanQueryRetriever(base), prior, name=key)
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    if key == "hybrid_bigram_prior_cleanq_rerank":
        if prior is None:
            raise SystemExit("hybrid_bigram_prior_cleanq_rerank needs a fitted LeadPrior")
        base = HybridRetriever(collection_name=collection_name(windowed=True),
                               lexical_ngram_range=(1, 2))
        # The reranker was measured HURTING when fed the raw instruction
        # template (0.85 -> 0.65 on a 20-contract probe). This variant asks
        # whether that was the cross-encoder failing or the input being wrong.
        r = PriorRetriever(CleanQueryRetriever(CrossEncoderReranker(base)), prior, name=key)
        return lambda qs, s, c, k: r.retrieve_batch(qs, s, c, k)
    raise SystemExit(f"unknown variant: {key} (choose from {ALL_VARIANTS})")


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
    ap.add_argument("--variants", default="tfidf,dense_win,hybrid",
                    help=f"comma-separated, or 'all'. Options: {','.join(ALL_VARIANTS)}")
    ap.add_argument("--limit", type=int, default=50,
                    help="contracts to evaluate (0 = all 510; must match what was ingested)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-write", action="store_true", help="print only, don't touch results file")
    ap.add_argument("--split", default="all", choices=["all", "train", "test"],
                    help="which contracts to SCORE on. Any fitted component (the lead "
                         "prior) is always fitted on the train side only, so --split test "
                         "is the only honest setting for a variant that learns anything.")
    args = ap.parse_args()

    keys = ALL_VARIANTS if args.variants == "all" else [v.strip() for v in args.variants.split(",")]

    limit = args.limit if args.limit > 0 else None
    contracts = load_contracts(limit)

    # Contract-level split, reusing Phase 4's function and seed so a contract
    # sits on the same side of the boundary here as it does for the
    # classifier. Splitting by segment instead would put near-duplicate text
    # on both sides and inflate everything downstream (split.py's docstring).
    train_ids, test_ids = split_contract_ids(contracts.keys())
    scored_ids = {"all": set(contracts), "train": set(train_ids), "test": set(test_ids)}[args.split]
    rows = load_eval_rows(scored_ids)
    print(f"contracts: {len(contracts)}   scoring on: {args.split} ({len(scored_ids)} contracts)")
    print(f"eval rows: {len(rows)}   k={args.k}")

    print("segmenting contracts...")
    segments_by_contract = {
        cid: segment_contract(text, doc_id=cid) for cid, text in contracts.items()
    }
    print(f"  {sum(len(v) for v in segments_by_contract.values())} segments total")

    # Fitted on train contracts only, always — never on whatever is being scored.
    prior = LeadPrior.fit(load_eval_rows(set(train_ids)), segments_by_contract)
    print(f"lead prior fitted on {len(train_ids)} train contracts -> "
          f"{len(prior.categories)} categories: {sorted(prior.categories)}")
    if args.split != "test" and any(k.endswith("_prior") for k in
                                    (ALL_VARIANTS if args.variants == "all"
                                     else args.variants.split(","))):
        print("  ⚠ scoring a fitted variant on contracts it was fitted on — "
              "use --split test for a number worth reporting")
    print()

    out_path = EVAL_DIR / "baseline_results.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    results = existing.setdefault("results", {})

    # Group scored rows by contract so every variant can batch its queries.
    # Rows without a gold span are counted, never silently dropped (Phase 3's
    # locked scoring decision: span overlap is undefined without a gold span).
    by_contract: dict[str, list[dict]] = {}
    skipped_total = 0
    for row in rows:
        if not row["has_gold_span"]:
            skipped_total += 1
            continue
        by_contract.setdefault(row["contract_id"], []).append(row)

    for key in keys:
        call = build_variant(key, prior=prior)
        t0 = time.time()
        scores = []
        for n, (cid, crows) in enumerate(by_contract.items(), start=1):
            questions = [r["question"] for r in crows]
            batch = call(questions, segments_by_contract[cid], cid, args.k)
            for row, spans in zip(crows, batch):
                gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
                scores.append(score_query(spans, gold))
            if n % 100 == 0:
                print(f"  [{key}] ...{n}/{len(by_contract)} contracts", flush=True)

        agg = aggregate_scores(scores, n_skipped_no_gold=skipped_total)
        agg["elapsed_sec"] = round(time.time() - t0, 1)
        results[key] = agg

        print(f"[{key}]")
        print(f"  scored rows        : {agg['n_scored']}")
        print(f"  hit_rate@{args.k}         : {agg['hit_rate']:.4f}")
        print(f"  gold_recall        : {agg['gold_recall']:.4f}")
        print(f"  gold_density       : {agg['gold_density']:.4f}")
        print(f"  mean_retrieved_chars: {agg['mean_retrieved_chars']:.1f}")
        print(f"  elapsed            : {agg['elapsed_sec']}s\n")

    baseline = results.get("tfidf", {}).get("hit_rate")
    if baseline:
        print(f"{'variant':<16}{'hit_rate@' + str(args.k):>12}{'vs tfidf':>12}")
        for key in keys:
            hr = results[key]["hit_rate"]
            print(f"{key:<16}{hr:>12.4f}{hr - baseline:>+12.4f}")

    # A partial run must never land in the results file. Every number in
    # baseline_results.json is quoted as "510 contracts, 6,702 scored rows";
    # merging a --limit 5 smoke run under the same key silently replaces a
    # full-corpus result with a 50-row one that still *looks* authoritative.
    partial = limit is not None or args.split != "all"
    if partial and not args.no_write:
        why = (f"partial run ({len(contracts)} contracts)" if limit is not None
               else f"--split {args.split}, not the full corpus")
        print(f"\nNOT writing results: {why}. Every entry in baseline_results.json is "
              f"quoted as 510 contracts / 6,702 rows; a split run under the same key would "
              f"silently replace one. Re-run with --limit 0 --split all to record.")
    if not args.no_write and not partial:
        existing["k"] = args.k
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
