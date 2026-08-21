"""
Covenant Phase 7 — re-grade an existing generation run with a different judge.

Generation and judging cost wildly different amounts. On this hardware a
row costs ~45s to generate (llama3.2:3b on CPU, long contract excerpts)
and ~1.5s to judge. Re-running the whole eval to change the judge would
therefore throw away ~97% of the work for no reason — the answers are
already recorded verbatim in the log, and grading them again needs nothing
else.

That makes the judge genuinely swappable, which matters beyond
convenience. TRD §3.4 says the faithfulness score is a trend indicator
with no ground truth; the honest way to see how much of a number is the
judge's opinion rather than the system's behaviour is to run two different
judges over identical answers and compare. This makes that a two-minute
operation instead of an hour-long one.

Every rule from judge.py still applies here — cloud-only, and never the
model that produced the answer. The generator identity is read from each
log row rather than supplied by the caller, so re-judging cannot
accidentally pair a model with its own output.

Usage:
    python backend/eval/harness/rejudge.py <run.jsonl>
    python backend/eval/harness/rejudge.py <run.jsonl> --judge-model groq/compound-mini
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "generation"))

from backends import build_backend  # noqa: E402
from judge import (ABSTAINED, NOT_JUDGED, FaithfulnessJudge, Verdict,  # noqa: E402
                   default_judge_backend, summarize)
from log_schema import GenerationLogRow, GenerationLogWriter, read_log  # noqa: E402


BUDGET_STOP_AFTER = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to a run's .jsonl")
    ap.add_argument("--judge-backend", default="groq", choices=["groq", "gemini"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--suffix", default=None,
                    help="output tag; defaults to the judge model name")
    args = ap.parse_args()

    src = Path(args.log)
    rows = read_log(src)
    if not rows:
        raise SystemExit(f"no rows in {src}")

    # Use the configured judge backend (token budget + reasoning effort set
    # by measured failures) unless explicitly overridden.
    if args.judge_backend == "groq" and not args.judge_model:
        judge = FaithfulnessJudge(default_judge_backend())
    else:
        judge = FaithfulnessJudge(build_backend(args.judge_backend, model=args.judge_model))
    tag = args.suffix or judge.backend.model.replace("/", "_")
    out_path = src.with_name(f"{src.stem}.rejudged-{tag}.jsonl")
    if out_path.exists():
        out_path.unlink()

    print(f"source : {src} ({len(rows)} rows)")
    print(f"was    : {rows[0].get('judge_backend')}:{rows[0].get('judge_model')}")
    print(f"now    : {judge.identity}\n")

    verdicts_gold, verdicts_absent = [], []
    n_hits = n_ret = 0
    t0 = time.time()
    # Same budget guard as the main runner: once the daily token budget is
    # gone every remaining call fails, and grinding through the rest costs
    # minutes to learn nothing. An earlier pass wasted 50 calls this way.
    consecutive_budget_failures = 0
    stopped_early = False

    with GenerationLogWriter(out_path) as log:
        for i, r in enumerate(rows, start=1):
            gen_identity = f"{r.get('generator_backend')}:{r.get('generator_model')}"
            v = judge.judge(r.get("question", ""), r.get("retrieved_chunks", []),
                            r.get("answer", ""), gen_identity)

            row = GenerationLogRow(**{k: val for k, val in r.items()
                                      if k in GenerationLogRow.__dataclass_fields__})
            row.judge_label = v.label
            row.judge_rationale = v.rationale
            row.judge_backend = judge.backend.name
            row.judge_model = judge.backend.model
            row.judge_latency_ms = v.latency_ms
            row.judge_error = v.error
            log.write(row)

            if r.get("has_gold_span"):
                verdicts_gold.append(v)
                n_ret += 1
                n_hits += bool(r.get("retrieval_hit"))
            else:
                verdicts_absent.append(v)

            if v.label == NOT_JUDGED:
                consecutive_budget_failures += 1
                if consecutive_budget_failures >= BUDGET_STOP_AFTER:
                    print(f"\n  STOPPING EARLY at {i}/{len(rows)}: "
                          f"{BUDGET_STOP_AFTER} consecutive judge failures "
                          f"(daily token budget exhausted).")
                    stopped_early = True
                    break
            else:
                consecutive_budget_failures = 0

            if i % 20 == 0:
                print(f"  ...{i}/{len(rows)} ({i / max(1e-9, time.time() - t0):.1f}/s)",
                      flush=True)

    fs = summarize(verdicts_gold)
    print(f"\n{'='*62}\nRE-JUDGED {src.stem} with {judge.identity}\n{'='*62}")
    print("[1] retrieval correctness — unchanged, carried from the source run")
    print(f"  rows scored   : {n_ret}")
    if n_ret:
        print(f"  hit_rate      : {n_hits / n_ret:.4f}")
    print("\n[2] faithfulness — LLM judge, TREND INDICATOR ONLY (TRD §3.4)")
    if stopped_early:
        print("  !! PARTIAL: stopped on budget exhaustion; coverage below is not 100%")
    print(f"  coverage      : {fs['judged_coverage']:.1%}"
          if fs.get('judged_coverage') is not None else "")
    print(f"  never judged  : {fs['n_not_judged']}")
    print(f"  answers judged: {fs['n_scored']}")
    for k in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
        print(f"  {k:<22}: {fs['counts'][k]}")
    print(f"  abstained     : {fs['n_abstained']}")
    print(f"  unparseable   : {fs['n_unparseable']}")
    if fs["faithful_rate"] is not None:
        print(f"  faithful_rate : {fs['faithful_rate']:.4f}")
        print(f"  supported+partial: {fs['supported_or_partial_rate']:.4f}")

    if verdicts_absent:
        n_abs = sum(1 for v in verdicts_absent if v.label == ABSTAINED)
        print(f"\n[diagnostic] absent-category rows: {n_abs}/{len(verdicts_absent)} "
              f"correctly abstained ({n_abs / len(verdicts_absent):.1%}) — not a metric")

    summary = out_path.with_suffix(".summary.json")
    summary.write_text(json.dumps({
        "source_run": src.stem,
        "judge": judge.identity,
        "n_rows": len(rows),
        "retrieval": {"hit_rate": (n_hits / n_ret) if n_ret else None, "n_scored": n_ret},
        "faithfulness": fs,
        "abstention_on_absent_rows": (
            sum(1 for v in verdicts_absent if v.label == ABSTAINED) / len(verdicts_absent)
            if verdicts_absent else None),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}\nwrote {summary}")


if __name__ == "__main__":
    main()
