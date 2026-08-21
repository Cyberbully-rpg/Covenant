"""
Covenant Phase 7 — prompt construction.

One place where retrieved chunks become an LLM prompt, so the exact text a
model saw is reproducible from a log row rather than reconstructed by
guesswork.

The prompt is built to make the Phase 3 §3.3 split measurable. Retrieval
correctness and faithfulness are scored separately, which only works if an
answer's failure can be attributed to one layer or the other — so the
instructions push the model to answer **strictly from the supplied
excerpts** and to say so plainly when they don't contain the answer.
Without that, a model that confidently answers from parametric knowledge
would paper over retrieval misses, and the retrieval metric and the
faithfulness metric would both be measuring something other than what they
claim.

`NO_ANSWER_MARKER` matters more than it looks. Measured in Phase 6, the
retrieved set misses the answer for ~15% of questions even at the adopted
configuration, and CUAD also marks a category outright absent in 14,208 of
20,910 eval rows. "I cannot find this" is therefore the *correct* answer a
large fraction of the time, and a model that never says it is wrong in a
way average answer quality would hide.
"""

from __future__ import annotations

NO_ANSWER_MARKER = "NOT_IN_EXCERPTS"

SYSTEM_PROMPT = (
    "You are a contract analysis assistant. You answer strictly from the "
    "contract excerpts provided in the user message, never from outside "
    "knowledge or assumption.\n"
    f"If the excerpts do not contain the answer, reply exactly: {NO_ANSWER_MARKER}\n"
    "Otherwise answer in at most three sentences, quoting the operative "
    "language from the excerpt where possible. Do not speculate, do not "
    "give legal advice, and do not describe what a contract of this type "
    "usually says."
)


def format_chunks(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered excerpts.

    Numbered so an answer can cite one and a human reading a log row can
    check the citation. Each carries its character span, which is what ties
    a generated answer back to the mechanical retrieval metric.
    """
    if not chunks:
        return "(no excerpts retrieved)"
    parts = []
    for i, ch in enumerate(chunks, start=1):
        header = (ch.get("header") or "").strip()
        label = f"[{i}]" + (f" {header}" if header else "")
        span = f"(chars {ch.get('start_char')}-{ch.get('end_char')})"
        text = (ch.get("text") or "").strip()
        parts.append(f"{label} {span}\n{text}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """The full user-side prompt. Pair with SYSTEM_PROMPT."""
    return (
        f"CONTRACT EXCERPTS\n"
        f"-----------------\n"
        f"{format_chunks(chunks)}\n\n"
        f"QUESTION\n"
        f"--------\n"
        f"{question.strip()}\n\n"
        f"Answer using only the excerpts above."
    )


def is_no_answer(text: str) -> bool:
    """True when the model declined for lack of grounding.

    Deliberately lenient about surrounding punctuation and casing: small
    local models rarely emit a bare marker even when instructed to, and
    counting a hedged refusal as a confident answer would misattribute a
    correct abstention as a faithfulness failure.
    """
    if not text:
        return False
    return NO_ANSWER_MARKER.lower() in text.strip().lower()
