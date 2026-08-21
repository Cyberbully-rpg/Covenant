"""
Covenant Phase 7 — end-to-end generation eval.

retrieve -> generate -> judge -> log, over a sampled subset of the
held-out contracts, reporting the two Phase 3 signals **separately**:

  1. retrieval correctness — mechanical span overlap against CUAD's own
     labelled answers. No LLM involved, non-gameable, has ground truth.
  2. faithfulness — LLM judge, cloud, offline, different model from the
     generator. A trend indicator with NO ground truth (TRD §3.4).

They are never averaged into one number (TRD §3.3). A bad answer is either
"retrieval didn't find it" or "retrieval found it and the model didn't use
it", and one blended score destroys that attribution — which is the entire
reason the harness was built before the classifier or RAG existed.

RETRIEVAL IS REUSED, NOT REIMPLEMENTED
---------------------------------------
Candidates come from `run_retrieval_variants.build_variant`, the same
function that produced Phase 6's published numbers. Re-deriving retrieval
here would let this runner drift from the measured configuration and
quietly report generation quality on top of a retrieval path nobody
validated.

SAMPLING
--------
Generation eval is one LLM call per row plus one judge call, so the full
6,702 scored rows would be ~13,400 API calls. This samples deterministically
(seeded) and records the sample size in the run summary — an unrecorded
sample size makes two runs incomparable.

Two strata, kept apart:
  - **gold rows** (CUAD marks the category present): both signals defined.
  - **absent rows** (CUAD marks it absent): retrieval correctness is
    undefined — there is no span to overlap — so these are NOT scored for
    it. They are sampled to observe whether the model correctly abstains.
    Reported as a **diagnostic, not a metric**: scorer.py records that
    scoring correct abstention needs a calibrated confidence threshold,
    which does not exist yet, so no precision/recall claim is made here.

Usage:
    python backend/eval/harness/run_generation_eval.py --gold-rows 60 --absent-rows 20
    python backend/eval/harness/run_generation_eval.py --gen-backend groq --k 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmenter"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "generation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "classifier" / "data"))

from segmenter import segment_contract  # noqa: E402
from scorer import score_query  # noqa: E402
from split import split_contract_ids  # noqa: E402
from lead_prior import LeadPrior  # noqa: E402
from backends import OllamaBackend, build_backend  # noqa: E402
from generate import generate_for_eval  # noqa: E402
from prompt import build_prompt, is_no_answer  # noqa: E402
from judge import (FaithfulnessJudge, summarize, ABSTAINED,  # noqa: E402
                   NOT_JUDGED, default_judge_backend)
from log_schema import GenerationLogRow, GenerationLogWriter  # noqa: E402
from run_retrieval_variants import build_variant, load_contracts, load_eval_rows  # noqa: E402

EVAL_DIR = Path("data/processed/eval")
LOG_DIR = EVAL_DIR / "generation_runs"
DEFAULT_RETRIEVER = "hybrid_bigram_prior_cleanq"
# Consecutive judge failures that mean the daily budget is gone.
BUDGET_STOP_AFTER = 5


def chunks_for(spans, by_span) -> list[dict]:
    """Attach segment text to retrieved spans, in rank order."""
    out = []
    for start, end in spans:
        seg = by_span.get((start, end))
        out.append({
            "start_char": start,
            "end_char": end,
            "header": (getattr(seg, "header", "") or "") if seg else "",
            "text": (getattr(seg, "embedding_text", "") or getattr(seg, "text", "")) if seg else "",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default=DEFAULT_RETRIEVER)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gold-rows", type=int, default=60)
    ap.add_argument("--absent-rows", type=int, default=20,
                    help="rows where CUAD marks the category absent; abstention diagnostic only")
    ap.add_argument("--gen-backend", default="ollama", choices=["ollama", "groq", "gemini"],
                    help="TRD §6.2 permits a per-RUN choice here, never per-request")
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-backend", default="groq", choices=["groq", "gemini"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    contracts = load_contracts(None)
    train_ids, test_ids = split_contract_ids(contracts.keys())
    test_ids = set(test_ids)
    rows = load_eval_rows(test_ids)

    print(f"run {run_id}")
    print(f"scoring on held-out test split: {len(test_ids)} contracts, {len(rows)} rows")
    print("segmenting...")
    segs = {cid: segment_contract(t, doc_id=cid) for cid, t in contracts.items()}

    # Fitted on TRAIN only, exactly as the retrieval harness does.
    prior = LeadPrior.fit(load_eval_rows(set(train_ids)), segs)
    call = build_variant(args.retriever, prior=prior)

    gold_pool = [r for r in rows if r["has_gold_span"]]
    absent_pool = [r for r in rows if not r["has_gold_span"]]
    rng = random.Random(args.seed)
    sample = (rng.sample(gold_pool, min(args.gold_rows, len(gold_pool)))
              + rng.sample(absent_pool, min(args.absent_rows, len(absent_pool))))
    rng.shuffle(sample)

    gen_backend = (OllamaBackend(model=args.gen_model) if args.gen_backend == "ollama"
                   else build_backend(args.gen_backend, model=args.gen_model))
    # Use the configured judge backend unless explicitly overridden -- its
    # token budget and reasoning effort were both set by measured failures,
    # and build_backend() alone would silently drop them.
    if args.judge_backend == "groq" and not args.judge_model:
        judge = FaithfulnessJudge(default_judge_backend())
    else:
        judge = FaithfulnessJudge(build_backend(args.judge_backend, model=args.judge_model))
    gen_identity = f"{gen_backend.name}:{gen_backend.model}"

    print(f"retriever  : {args.retriever} (k={args.k})")
    print(f"generator  : {gen_identity}")
    print(f"judge      : {judge.identity}")
    print(f"sample     : {len(sample)} rows "
          f"({min(args.gold_rows, len(gold_pool))} gold + "
          f"{min(args.absent_rows, len(absent_pool))} absent), seed={args.seed}\n")

    by_contract = defaultdict(list)
    for r in sample:
        by_contract[r["contract_id"]].append(r)

    log_path = LOG_DIR / f"{run_id}.jsonl"
    verdicts_gold, verdicts_absent = [], []
    retrieval_scores = []
    n_done, t0 = 0, time.time()
    stopped_early = False
    # Budget guard. Free-tier Groq allows ~200k tokens/day for a model,
    # which is roughly 90 judge calls at this prompt size. Once the budget
    # is gone every remaining call fails, and a run that keeps going
    # produces a long tail of NOT_JUDGED rows that costs time and teaches
    # nothing. Stopping early and saying so is the honest failure.
    consecutive_budget_failures = 0


    with GenerationLogWriter(log_path) as log:
        for cid, crows in by_contract.items():
            segments = segs[cid]
            by_span = {(s.start_char, s.end_char): s for s in segments}
            questions = [r["question"] for r in crows]
            if stopped_early:
                break
            batch = call(questions, segments, cid, args.k)

            for row, spans in zip(crows, batch):
                chunks = chunks_for(spans, by_span)
                gen = generate_for_eval(row["question"], chunks, gen_backend)
                answer = gen.text.strip()

                entry = GenerationLogRow(
                    run_id=run_id,
                    contract_id=cid,
                    category=row["category"],
                    question=row["question"],
                    has_gold_span=row["has_gold_span"],
                    retriever=args.retriever,
                    k=args.k,
                    retrieved_spans=[list(s) for s in spans],
                    retrieved_chunks=chunks,
                    prompt=build_prompt(row["question"], chunks),
                    answer=answer,
                    abstained=is_no_answer(answer),
                    generator_backend=gen.backend,
                    generator_model=gen.model,
                    generator_temperature=gen.temperature,
                    generation_latency_ms=gen.latency_ms,
                    generation_error=gen.error,
                )

                # Signal 1 — mechanical, only where a gold span exists.
                if row["has_gold_span"]:
                    gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
                    qs = score_query(spans, gold)
                    retrieval_scores.append(qs)
                    entry.retrieval_hit = qs.hit
                    entry.gold_chars_hit = qs.gold_chars_hit
                    entry.gold_chars_total = qs.gold_chars_total
                    entry.retrieved_chars = qs.retrieved_chars

                # Signal 2 — LLM judge, separate call, different model.
                v = judge.judge(row["question"], chunks, answer, gen_identity)
                entry.judge_label = v.label
                entry.judge_rationale = v.rationale
                entry.judge_backend = judge.backend.name
                entry.judge_model = judge.backend.model
                entry.judge_latency_ms = v.latency_ms
                entry.judge_error = v.error
                (verdicts_gold if row["has_gold_span"] else verdicts_absent).append(v)

                log.write(entry)
                n_done += 1

                if v.label == NOT_JUDGED:
                    consecutive_budget_failures += 1
                    if consecutive_budget_failures >= BUDGET_STOP_AFTER:
                        print(f"\n  STOPPING EARLY: {BUDGET_STOP_AFTER} consecutive "
                              f"judge failures (daily token budget exhausted).")
                        print(f"  {n_done} of {len(sample)} rows completed; the rest "
                              f"are unjudged and excluded, not counted as failures.")
                        stopped_early = True
                        break
                else:
                    consecutive_budget_failures = 0

                if n_done % 10 == 0:
                    rate = n_done / max(1e-9, time.time() - t0)
                    print(f"  ...{n_done}/{len(sample)} rows ({rate:.2f}/s)", flush=True)

    # ---- report: two signals, side by side, never combined --------------
    n_hits = sum(1 for s in retrieval_scores if s.hit)
    n_ret = len(retrieval_scores)
    print(f"\n{'='*62}\nRUN {run_id}\n{'='*62}")
    print(f"retriever {args.retriever}  k={args.k}")
    print(f"generator {gen_identity}   judge {judge.identity}\n")

    print("[1] retrieval correctness — mechanical, ground-truthed, non-gameable")
    print(f"  rows scored        : {n_ret}")
    print(f"  hit_rate@{args.k}         : {n_hits / n_ret:.4f}" if n_ret else "  (none)")

    fs = summarize(verdicts_gold)
    print("\n[2] faithfulness — LLM judge, TREND INDICATOR ONLY (TRD §3.4)")
    print("    no ground truth; known to reward fluency and length;")
    print("    never to be reported without [1] beside it")
    if stopped_early:
        print(f"  !! PARTIAL RUN: stopped at {n_done}/{len(sample)} rows "
              f"(daily token budget). Coverage below is not 100%.")
    print(f"  judged coverage           : {fs['judged_coverage']:.1%}"
          if fs.get("judged_coverage") is not None else "")
    print(f"  never judged (budget)     : {fs['n_not_judged']}")
    print(f"  answers that made a claim : {fs['n_scored']}")
    print(f"  SUPPORTED                 : {fs['counts']['SUPPORTED']}")
    print(f"  PARTIALLY_SUPPORTED       : {fs['counts']['PARTIALLY_SUPPORTED']}")
    print(f"  UNSUPPORTED               : {fs['counts']['UNSUPPORTED']}")
    print(f"  abstained (not scored)    : {fs['n_abstained']}")
    print(f"  unparseable (not scored)  : {fs['n_unparseable']}")
    if fs["faithful_rate"] is not None:
        print(f"  faithful_rate             : {fs['faithful_rate']:.4f}")
        print(f"  supported_or_partial      : {fs['supported_or_partial_rate']:.4f}")

    if verdicts_absent:
        n_abs = sum(1 for v in verdicts_absent if v.label == ABSTAINED)
        print(f"\n[diagnostic] absent-category rows (CUAD says the clause isn't there)")
        print(f"  rows                      : {len(verdicts_absent)}")
        print(f"  model correctly abstained : {n_abs} ({n_abs/len(verdicts_absent):.1%})")
        print("  NOT a metric — scoring correct abstention needs a calibrated")
        print("  confidence threshold, which does not exist yet (scorer.py).")

    summary_path = LOG_DIR / f"{run_id}.summary.json"
    summary_path.write_text(json.dumps({
        "run_id": run_id,
        "retriever": args.retriever,
        "k": args.k,
        "generator": gen_identity,
        "judge": judge.identity,
        "seed": args.seed,
        "split": "test",
        "n_sampled": len(sample),
        "n_gold_rows": n_ret,
        "n_absent_rows": len(verdicts_absent),
        "retrieval": {"hit_rate": (n_hits / n_ret) if n_ret else None, "n_scored": n_ret},
        "faithfulness": {k: v for k, v in fs.items() if k != "counts"} | {"counts": fs["counts"]},
        "abstention_on_absent_rows": (
            sum(1 for v in verdicts_absent if v.label == ABSTAINED) / len(verdicts_absent)
            if verdicts_absent else None),
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {log_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
