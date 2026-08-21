"""
Covenant Phase 7 — inference backends.

Low-level clients only. This module knows how to *call* Ollama, Gemini and
Groq; it deliberately does NOT know which one should be used for what.
That decision belongs to `generate.py` and `judge.py`, because TRD §6.2
locks routing as **code-path-determined, never a user-facing toggle** —
and a module that accepts a backend name from its caller is one refactor
away from becoming exactly the runtime switch that rule forbids.

Every call returns a `Generation`, which carries the backend identity
alongside the text. TRD §3.5 requires backend identity on every harness
log row so a regression can be attributed to a model/version swap rather
than to a genuine retrieval or chunking change; carrying it in the return
value means a caller cannot log an answer without also having its
provenance.

Backends available in this environment (verified, see CLAUDE.md):
  ollama  — local, llama3.2:3b, no API cost
  gemini  — cloud, gemini-2.5-flash
  groq    — cloud, fast, free tier
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

OLLAMA_DEFAULT_MODEL = "llama3.2:3b"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
# Faithfulness judge model. Chosen deliberately small: a 120B judge is
# disproportionate for a three-way label task, and grading is the cheapest
# place in this pipeline to overspend without noticing.
#
# There is no ~70B model on this Groq account -- the full list is 13
# models, and between 20b and 120b there is only qwen3.6-27b. Every
# text-capable model below 120B was tested on the real judge task
# (grounded / fabricated / partially-supported):
#   openai/gpt-oss-20b   all three correct, 1.3-2.1s   <- chosen
#   groq/compound-mini   all three correct, 2.0-3.4s   (viable fallback)
#   qwen/qwen3.6-27b     unusable: emits <think> blocks, label never
#                        lands on line 1 (parse_verdict now tolerates
#                        this, but it still wastes tokens reasoning)
#   allam-2-7b           DISQUALIFIED: graded an answer containing a
#                        fabricated "30 days" detail as fully SUPPORTED.
#                        That is exactly the miss a faithfulness judge
#                        exists to catch, so size is beside the point.
GROQ_JUDGE_MODEL = "openai/gpt-oss-20b"
GROQ_JUDGE_FALLBACK_MODEL = "groq/compound-mini"

# Gemini model selection, entirely determined by testing rather than
# preference. Most of this family did NOT work on this account:
#   - `gemini-2.5-flash` appears in models.list() but returns 404 on
#     generateContent ("no longer available to new users"). Being listed
#     does not mean being callable.
#   - `gemini-3.6-flash` returned 429 RESOURCE_EXHAUSTED.
#   - `gemini-3.7-flash` returned 503 (high demand).
#   - `gemini-3.6-flash` / `gemini-flash-latest` return EMPTY text even
#     when they succeed -- reasoning consumes the output budget.
#   - `gemini-3.1-flash-lite` works, returns text, and graded all 80 rows
#     of a real run with ZERO unparseable verdicts.
# Pinned explicitly: `gemini-flash-latest` is a moving alias, and an eval
# number whose judge silently changed underneath it is not reproducible.
GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Deterministic by default. An eval run that cannot be reproduced is not a
# measurement, and this project's whole reporting discipline depends on a
# number meaning the same thing twice.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512

# Judges need more headroom than generators. Measured: 10 of 80 judge calls
# returned EMPTY content with no error, because gpt-oss models reason before
# answering and a 512-token budget was consumed before any visible output.
# Same failure shape as gemini-2.5-flash returning "". A silently unscored
# row is worse than a failed one -- it shrinks the faithfulness denominator
# without announcing it.
JUDGE_MAX_TOKENS = 2048
# Groq exposes reasoning_effort on gpt-oss. A three-way label task does not
# need deep reasoning, and low effort leaves the budget for the answer.
JUDGE_REASONING_EFFORT = "low"

DEFAULT_MAX_RETRIES = 4
RETRY_BASE_DELAY_S = 2.0
# Substrings identifying transient failures worth retrying. Free-tier Groq
# rate-limits during a long batch run, and losing rows to a 429 corrupts the
# denominator of every number computed from that run.
TRANSIENT_MARKERS = (
    "429", "rate limit", "ratelimit", "resource_exhausted",
    "500", "502", "503", "504", "unavailable", "overloaded",
    "timeout", "timed out", "connection",
)


def is_transient(message: str) -> bool:
    low = (message or "").lower()
    return any(m in low for m in TRANSIENT_MARKERS)


@dataclass
class Generation:
    """One model response plus the provenance TRD §3.5 requires."""
    text: str
    backend: str
    model: str
    latency_ms: int
    temperature: float = DEFAULT_TEMPERATURE
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def identity(self) -> str:
        """Single string identifying exactly what produced this text."""
        return f"{self.backend}:{self.model}"


def load_env() -> None:
    """Load .env by absolute path.

    Bare `load_dotenv()` walks the caller's stack to find the .env and
    raises AssertionError when the calling frame has no file (stdin-piped
    code, some REPLs). Passing the path explicitly avoids that entirely.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001 — a missing .env is not fatal here;
        pass          # the backend that needs a key will say so clearly.


class Backend:
    """Common shape. Subclasses implement `_call`."""

    name = "base"
    default_model = ""

    def __init__(self, model: str | None = None,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.model = model or self.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _call(self, prompt: str, system: str | None) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, system: str | None = None) -> Generation:
        """Never raises for a backend failure — returns a Generation whose
        `error` is set instead. A batch eval over thousands of rows must not
        die on one timeout, and a silently-dropped row would corrupt the
        denominator of every number computed from the run."""
        t0 = time.time()
        text, error, attempts = "", None, 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                text, error = self._call(prompt, system), None
                break
            except Exception as e:  # noqa: BLE001 — recorded, not swallowed
                error = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries and is_transient(error):
                    # Exponential backoff. Retrying a rate limit immediately
                    # just burns the next slot too.
                    time.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
                    continue
                text = ""
                break
        return Generation(
            text=text,
            backend=self.name,
            model=self.model,
            latency_ms=int((time.time() - t0) * 1000),
            temperature=self.temperature,
            error=error,
            extra={"attempts": attempts},
        )


class OllamaBackend(Backend):
    """Local. The only backend permitted for interactive generation (TRD §6.2)."""

    name = "ollama"
    default_model = OLLAMA_DEFAULT_MODEL

    def _call(self, prompt: str, system: str | None) -> str:
        import ollama
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        resp = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return resp["message"]["content"]


class GeminiBackend(Backend):
    name = "gemini"
    default_model = GEMINI_DEFAULT_MODEL

    def _call(self, prompt: str, system: str | None) -> str:
        from google import genai
        from google.genai import types
        load_env()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (expected in .env)")
        client = genai.Client(api_key=key)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system or None,
        )
        resp = client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        return resp.text or ""


class GroqBackend(Backend):
    name = "groq"
    default_model = GROQ_DEFAULT_MODEL

    def __init__(self, *args, reasoning_effort: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort

    def _call(self, prompt: str, system: str | None) -> str:
        from groq import Groq
        load_env()
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set (expected in .env)")
        client = Groq(api_key=key)
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        kwargs = {}
        # Only gpt-oss accepts reasoning_effort; sending it elsewhere 400s.
        if self.reasoning_effort and "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content or ""


CLOUD_BACKENDS = {"gemini": GeminiBackend, "groq": GroqBackend}
LOCAL_BACKENDS = {"ollama": OllamaBackend}
ALL_BACKENDS = {**LOCAL_BACKENDS, **CLOUD_BACKENDS}


def build_backend(name: str, model: str | None = None, **kwargs) -> Backend:
    """Construct by name. Intended for *config-driven* call sites only —
    the batch-eval runner, where TRD §6.2 permits a per-run choice. The
    interactive and judge paths must never reach for this."""
    if name not in ALL_BACKENDS:
        raise SystemExit(f"unknown backend {name!r} (have: {sorted(ALL_BACKENDS)})")
    return ALL_BACKENDS[name](model=model, **kwargs)


def is_cloud(name: str) -> bool:
    return name in CLOUD_BACKENDS
