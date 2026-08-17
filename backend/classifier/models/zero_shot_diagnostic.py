"""
Covenant Phase 5, TRD §4.4 — zero-shot LLM diagnostic.

DIAGNOSTIC ROLE ONLY, per TRD §4.4 — this never gates a build decision.
Its one job: for the categories the classical classifier (Phase 5 champion,
§6C.2) still fails on, check whether a zero-shot LLM also fails on the same
examples. If it does, that's evidence of label ambiguity/noise in CUAD
itself, and engineering effort should NOT be spent chasing those categories
further with more classical levers (TRD §4.3 steps 4-5).

Also doubles as the sandbox comparison the project owner asked for between
three candidate inference backends (Ollama local, Gemini cloud, Groq cloud)
before committing to one for this call site. TRD §5.5's routing rule —
batch eval may use Ollama OR cloud, per-run config — means any of the three
is architecturally valid here; this script measures which is actually
usable.

Task: reuse CUAD's OWN per-category question (already in eval_set.jsonl,
Phase 3) rather than writing new prompts — same rationale as Phase 4 reusing
spans_overlap: what counts as "the question for this category" is defined
once, not reinvented per consumer. For each sampled segment, the LLM is
asked CUAD's question and must answer YES/NO, compared against whether that
segment carries the gold label.

This is NOT a rigorous eval (small sample, single prompt template, no
retries beyond one). It is a diagnostic pass, sized to answer "gap or
noise?" cheaply — matching TRD §4.4's stated role.

Usage:
    python backend/classifier/models/zero_shot_diagnostic.py --backend ollama
    python backend/classifier/models/zero_shot_diagnostic.py --backend gemini
    python backend/classifier/models/zero_shot_diagnostic.py --backend groq
    python backend/classifier/models/zero_shot_diagnostic.py --backend all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

DATA_DIR = Path("data/processed/classifier")
EVAL_DIR = Path("data/processed/eval")
OUT_DIR = Path("backend/classifier/models/artifacts")

# The three categories the Phase 5 champion scores F1 = 0.000 on (§6C.2).
TARGET_CATEGORIES = [
    "Competitive Restriction Exception",
    "Price Restrictions",
    "Third Party Beneficiary",
]

N_PER_CLASS = 8   # positives and negatives per category — cheap, not rigorous (see module docstring)
OLLAMA_MODEL = "llama3.2:3b"
GEMINI_MODEL = "gemini-3.6-flash"
GROQ_MODEL = "qwen/qwen3.6-27b"

PROMPT_TEMPLATE = """You are reviewing one clause from a commercial contract.

Clause:
\"\"\"
{text}
\"\"\"

Question: {question}

Answer with exactly one word: YES if this clause is relevant to the question, NO if it is not."""


def load_category_questions() -> dict[str, str]:
    questions = {}
    with open(EVAL_DIR / "eval_set.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["category"] not in questions:
                questions[row["category"]] = row["question"]
    return questions


def sample_examples(category: str, n_per_class: int) -> list[dict]:
    """n_per_class positives (label present) and n_per_class negatives, deterministic order."""
    pos, neg = [], []
    with open(DATA_DIR / "training_segments.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if len(row["text"].strip()) < 20:
                continue
            if category in row["labels"] and len(pos) < n_per_class:
                pos.append(row)
            elif category not in row["labels"] and len(neg) < n_per_class:
                neg.append(row)
            if len(pos) >= n_per_class and len(neg) >= n_per_class:
                break
    return [{**r, "gold": True} for r in pos] + [{**r, "gold": False} for r in neg]


def parse_yes_no(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    return None  # unparseable — recorded, not silently coerced to a guess


def call_ollama(prompt: str) -> tuple[str, float]:
    import ollama
    t0 = time.time()
    resp = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, options={"temperature": 0})
    return resp["response"], time.time() - t0


def call_gemini(client, prompt: str) -> tuple[str, float]:
    t0 = time.time()
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config={"temperature": 0, "automatic_function_calling": {"disable": True}},
    )
    return resp.text, time.time() - t0


def call_groq(client, prompt: str) -> tuple[str, float]:
    # qwen3.6-27b is a reasoning model that otherwise prepends a <think>...</think>
    # block before the actual answer, which parse_yes_no would never see.
    # reasoning_format="hidden" asks Groq to strip that block server-side.
    t0 = time.time()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        reasoning_format="hidden",
    )
    return resp.choices[0].message.content, time.time() - t0


def run_backend(backend: str, questions: dict[str, str]) -> dict:
    load_dotenv()
    client = None
    if backend == "gemini":
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY not set (checked .env and environment)")
        client = genai.Client(api_key=key)
    elif backend == "groq":
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("GROQ_API_KEY not set (checked .env and environment)")
        client = Groq(api_key=key)

    rows = []
    for category in TARGET_CATEGORIES:
        question = questions[category]
        examples = sample_examples(category, N_PER_CLASS)
        for ex in examples:
            prompt = PROMPT_TEMPLATE.format(text=ex["text"][:2000], question=question)
            try:
                if backend == "ollama":
                    raw, latency = call_ollama(prompt)
                elif backend == "gemini":
                    raw, latency = call_gemini(client, prompt)
                else:
                    raw, latency = call_groq(client, prompt)
                error = None
            except Exception as e:  # noqa: BLE001 — diagnostic script, record and continue
                raw, latency, error = "", None, str(e)
            pred = parse_yes_no(raw) if not error else None
            rows.append({
                "backend": backend, "category": category,
                "segment_id": ex["segment_id"], "gold": ex["gold"],
                "pred": pred, "correct": (pred == ex["gold"]) if pred is not None else None,
                "latency_s": latency, "raw_response": raw[:200], "error": error,
            })
            print(f"  [{backend}] {category[:30]:<30} gold={ex['gold']!s:<5} "
                  f"pred={pred!s:<5} {latency:.2f}s" if not error
                  else f"  [{backend}] {category[:30]:<30} ERROR: {error[:80]}")

    n_scored = sum(1 for r in rows if r["correct"] is not None)
    n_correct = sum(1 for r in rows if r["correct"])
    n_unparseable = sum(1 for r in rows if r["pred"] is None and r["error"] is None)
    n_errors = sum(1 for r in rows if r["error"] is not None)
    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]

    model_by_backend = {"ollama": OLLAMA_MODEL, "gemini": GEMINI_MODEL, "groq": GROQ_MODEL}
    return {
        "backend": backend,
        "model": model_by_backend[backend],
        "n_examples": len(rows),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": n_correct / n_scored if n_scored else None,
        "n_unparseable": n_unparseable,
        "n_errors": n_errors,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "total_latency_s": sum(latencies) if latencies else None,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "gemini", "groq", "all"], default="all")
    args = ap.parse_args()

    questions = load_category_questions()
    backends = ["ollama", "gemini", "groq"] if args.backend == "all" else [args.backend]

    results = {}
    for backend in backends:
        print(f"\n=== {backend} ===")
        results[backend] = run_backend(backend, questions)

    print("\n" + "=" * 70)
    print(f"{'backend':<10} {'model':<20} {'acc':>8} {'unparse':>8} {'errors':>7} {'mean_s':>8}")
    print("-" * 70)
    for backend, r in results.items():
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "n/a"
        mean_s = f"{r['mean_latency_s']:.2f}" if r["mean_latency_s"] is not None else "n/a"
        print(f"{backend:<10} {r['model']:<20} {acc:>8} {r['n_unparseable']:>8} "
              f"{r['n_errors']:>7} {mean_s:>8}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase5_zero_shot_diagnostic.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    existing.update(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
