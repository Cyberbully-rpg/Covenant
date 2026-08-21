"""
Covenant Phase 3 step 3 / Phase 7 — faithfulness judge.

Deferred from Phase 3 because it needs generated answers to score, and
nothing generated answers until Phase 7. It lives here, in the harness,
rather than beside `rag/generation/` on purpose: TRD §6.2 requires the
judge to be a separate call that is **never the same model that produced
the answer being judged**, and physical separation makes that a deliberate
act rather than a one-line mistake.

Locked constraints (TRD §6.2, §3.3, §3.4):
  - ALWAYS cloud. Never Ollama, never the live serving path.
  - ALWAYS a separate call, made offline after generation.
  - NEVER the same model that generated the answer. Enforced here by
    raising, not by documentation — a model grading its own output is not
    a weak signal, it is no signal.
  - Faithfulness is reported ALONGSIDE mechanical retrieval correctness
    and never blended into one number (§3.3). The two answer different
    questions: did we find the text, and did the model stick to it.

WHAT THIS SCORE IS AND IS NOT (§3.4 — must travel with the number)
------------------------------------------------------------------
It is a trend indicator. LLM judges are known to reward fluency and
length, and this one has no ground truth behind it — unlike retrieval
correctness, which is arithmetic on CUAD's own labelled spans. A rising
faithfulness score with flat retrieval correctness means the model got
more careful; it does not mean the system found more answers.

ABSTENTIONS ARE SCORED SEPARATELY, NOT AS FAILURES
---------------------------------------------------
Phase 6 measured that the retrieved set misses the answer for ~15% of
questions, and CUAD marks the category absent for 14,208 of 20,910 rows.
Saying "not in the excerpts" is therefore frequently the *correct*
behaviour. An abstention asserts nothing about the contract, so it cannot
be unfaithful — it is counted in its own bucket and excluded from the
faithfulness rate, whose denominator is answers that actually made a
claim. Folding abstentions in either direction would let a model that
refuses everything score perfectly.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag" / "generation"))

from backends import (Backend, GeminiBackend, GroqBackend,  # noqa: E402
                      GROQ_JUDGE_MODEL, JUDGE_MAX_TOKENS,
                      JUDGE_REASONING_EFFORT, is_cloud, is_transient)
from prompt import format_chunks, is_no_answer  # noqa: E402

SUPPORTED = "SUPPORTED"
PARTIAL = "PARTIALLY_SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
ABSTAINED = "ABSTAINED"
UNPARSEABLE = "UNPARSEABLE"
# Distinct from UNPARSEABLE on purpose. UNPARSEABLE means the judge replied
# and the reply could not be read -- a judge-quality problem. NOT_JUDGED
# means no verdict was ever obtained: the free-tier daily token budget ran
# out, or the connection failed. Conflating them once already produced a
# run where 57 rate-limited rows looked like judge failures and the
# faithfulness rate was computed from 5 surviving answers.
NOT_JUDGED = "NOT_JUDGED"

LABELS = (SUPPORTED, PARTIAL, UNSUPPORTED)

JUDGE_SYSTEM = (
    "You grade whether an answer is supported by contract excerpts. You are "
    "not grading whether the answer is correct, useful, or well written — "
    "only whether every claim it makes is present in the excerpts shown.\n"
    f"Reply with exactly one label on the first line: {SUPPORTED}, {PARTIAL}, "
    f"or {UNSUPPORTED}.\n"
    f"  {SUPPORTED} — every claim traces to the excerpts.\n"
    f"  {PARTIAL} — the central claim traces to the excerpts but some detail "
    "does not.\n"
    f"  {UNSUPPORTED} — the answer asserts something the excerpts do not say.\n"
    "On the second line give a one-sentence reason. Output nothing else."
)


@dataclass
class Verdict:
    label: str
    rationale: str
    judge_identity: str
    latency_ms: int
    error: str | None = None

    @property
    def is_scored(self) -> bool:
        """Whether this row belongs in the faithfulness denominator."""
        return self.label in LABELS


def build_judge_prompt(question: str, chunks: list[dict], answer: str) -> str:
    return (
        f"CONTRACT EXCERPTS\n-----------------\n{format_chunks(chunks)}\n\n"
        f"QUESTION\n--------\n{question.strip()}\n\n"
        f"ANSWER TO GRADE\n---------------\n{answer.strip()}\n\n"
        f"Label this answer."
    )


def parse_verdict(text: str) -> tuple[str, str]:
    """(label, rationale). Returns UNPARSEABLE rather than guessing.

    A judge whose output couldn't be read is missing data, not a neutral
    or passing grade — silently defaulting it to SUPPORTED would inflate
    the exact number this module exists to report honestly.
    """
    if not text or not text.strip():
        return UNPARSEABLE, ""
    # Reasoning models (qwen3.6-27b among the Groq options) prepend a
    # <think> block, which would push the label off line 1 and make every
    # verdict UNPARSEABLE. Strip it so the judge model can be swapped
    # without the parser silently zeroing the faithfulness denominator.
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return UNPARSEABLE, ""
    head = lines[0].upper()
    # PARTIALLY_SUPPORTED contains SUPPORTED, so test the longer label first.
    for label in (PARTIAL, UNSUPPORTED, SUPPORTED):
        if re.search(rf"\b{label}\b", head):
            return label, (lines[1] if len(lines) > 1 else "")
    return UNPARSEABLE, lines[0][:200]


def default_judge_backend() -> Backend:
    """Gemini 3.1 Flash Lite. Chosen on measured coverage, not preference.

    Over the same 80-row run, judged by three models:
      gemini-3.1-flash-lite   52 judged, 0 unparseable, 100% coverage
      groq gpt-oss-120b       44 judged, 8 unparseable
      groq gpt-oss-20b        40 judged, 12 unparseable
    The gpt-oss models are reasoning models that intermittently return
    empty content; a judge that silently declines to grade 15-20% of rows
    shrinks the faithfulness denominator without announcing it.

    Gemini also sits on a DIFFERENT provider quota from Groq, which matters
    on free tiers: batch-eval generation can use Groq without competing for
    the same daily token budget the judge needs.
    """
    return GeminiBackend(max_tokens=JUDGE_MAX_TOKENS)


def groq_judge_backend() -> Backend:
    """The Groq alternative, for the judge-variance comparison (TRD §3.4).
    Faithfulness moved 0.591 -> 0.700 across judges on identical answers,
    so running two is how that spread stays visible instead of assumed."""
    return GroqBackend(model=GROQ_JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS,
                       reasoning_effort=JUDGE_REASONING_EFFORT)


class FaithfulnessJudge:
    """Cloud-only grader for generated answers."""

    def __init__(self, backend: Backend | None = None):
        # Groq by default, not Gemini: both are cloud and both satisfy
        # TRD 6.2, but Gemini's usable models on this account return
        # 404/429/503 (see backends.py). Verified working beats preferred.
        backend = backend or default_judge_backend()
        if not is_cloud(backend.name):
            raise ValueError(
                f"judge backend must be cloud (TRD §6.2), got {backend.name!r}. "
                "The judge never runs on the local serving path."
            )
        self.backend = backend

    @property
    def identity(self) -> str:
        return f"{self.backend.name}:{self.backend.model}"

    def judge(self, question: str, chunks: list[dict], answer: str,
              generator_identity: str) -> Verdict:
        """Grade one answer.

        `generator_identity` is REQUIRED, not optional, so that the
        never-self-judge rule cannot be bypassed by a caller that simply
        forgot to pass it.
        """
        if generator_identity == self.identity:
            raise ValueError(
                f"judge and generator are the same model ({self.identity}) — "
                "TRD §6.2 forbids a model grading its own output."
            )

        if is_no_answer(answer):
            # Asserts nothing about the contract, so there is nothing to be
            # unfaithful about. Counted, not scored (see module docstring).
            return Verdict(ABSTAINED, "model declined for lack of grounding",
                           self.identity, 0)

        gen = self.backend.generate(
            build_judge_prompt(question, chunks, answer), system=JUDGE_SYSTEM
        )
        if gen.error:
            # Budget/transport failures are missing data, not bad grading.
            label = NOT_JUDGED if is_transient(gen.error) else UNPARSEABLE
            return Verdict(label, "", self.identity, gen.latency_ms, gen.error)
        label, rationale = parse_verdict(gen.text)
        return Verdict(label, rationale, self.identity, gen.latency_ms)


def summarize(verdicts: list[Verdict]) -> dict:
    """Aggregate. Abstentions and unparseable rows are reported, never folded in."""
    counts = {k: 0 for k in (SUPPORTED, PARTIAL, UNSUPPORTED, ABSTAINED,
                             UNPARSEABLE, NOT_JUDGED)}
    for v in verdicts:
        counts[v.label] = counts.get(v.label, 0) + 1
    scored = sum(counts[k] for k in LABELS)
    return {
        "n_total": len(verdicts),
        "n_scored": scored,
        "n_abstained": counts[ABSTAINED],
        "n_unparseable": counts[UNPARSEABLE],
        "n_not_judged": counts[NOT_JUDGED],
        "counts": counts,
        # Denominator is answers that actually made a claim. A model that
        # abstains on everything scores 0 rows, not 100%.
        "faithful_rate": (counts[SUPPORTED] / scored) if scored else None,
        "supported_or_partial_rate": (
            (counts[SUPPORTED] + counts[PARTIAL]) / scored) if scored else None,
        "abstention_rate": counts[ABSTAINED] / len(verdicts) if verdicts else None,
        # Coverage is reported so a partial run cannot be mistaken for a
        # complete one: a faithfulness rate computed over 5 of 60 rows is
        # not a smaller version of the same number, it is a different one.
        "judged_coverage": (scored + counts[ABSTAINED]) / len(verdicts) if verdicts else None,
    }
