"""
Tests for the Phase 7 generation layer (prompt.py, backends.py, generate.py).

No network. Backends are stubbed, because what needs asserting here is the
routing and the failure handling, not whether Ollama or Groq work — those
were verified live once and would make this suite slow and flaky.

The properties that matter are the ones TRD §6.2 and §3.5 lock:

  1. `generate_interactive` cannot be pointed at a cloud backend. Not
     "shouldn't be" -- there is no parameter to do it with, and that is
     asserted by inspecting the signature, so a future refactor that adds
     one fails a test rather than quietly becoming the user-facing toggle
     §6.2 forbids.
  2. A backend failure returns a Generation carrying the error, never
     raises. A batch eval over hundreds of paid API calls must not die on
     one timeout, and a silently dropped row would corrupt the denominator
     of every number computed from the run.
  3. Backend identity rides along with the text, so a caller cannot log an
     answer without its provenance (§3.5).
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import backends  # noqa: E402
import generate  # noqa: E402
from backends import (Backend, Generation, GeminiBackend, GroqBackend,  # noqa: E402
                      OllamaBackend, build_backend, is_cloud)
from prompt import (NO_ANSWER_MARKER, SYSTEM_PROMPT, build_prompt,  # noqa: E402
                    format_chunks, is_no_answer)

CHUNKS = [
    {"header": "12. GOVERNING LAW", "start_char": 100, "end_char": 260,
     "text": "Governed by the laws of Delaware."},
    {"header": "", "start_char": 260, "end_char": 400, "text": "Notices in writing."},
]


# --- prompt ----------------------------------------------------------------

def test_chunks_are_numbered_and_carry_their_character_spans():
    """Numbering lets an answer cite an excerpt; spans are what tie a
    generated answer back to the mechanical retrieval metric."""
    out = format_chunks(CHUNKS)
    assert "[1]" in out and "[2]" in out
    assert "chars 100-260" in out
    assert "GOVERNING LAW" in out


def test_empty_retrieval_is_stated_not_silently_blank():
    assert "no excerpts" in format_chunks([]).lower()


def test_prompt_contains_the_question_and_the_excerpt_text():
    p = build_prompt("What law governs?", CHUNKS)
    assert "What law governs?" in p
    assert "Delaware" in p


def test_system_prompt_demands_grounding_and_offers_an_abstention():
    """Without an abstention path a model answers from parametric knowledge
    and papers over retrieval misses, which would corrupt both signals."""
    assert NO_ANSWER_MARKER in SYSTEM_PROMPT
    assert "excerpt" in SYSTEM_PROMPT.lower()


@pytest.mark.parametrize("text,expected", [
    (NO_ANSWER_MARKER, True),
    (f"  {NO_ANSWER_MARKER}.  ", True),
    (f"{NO_ANSWER_MARKER.lower()}", True),
    ("The agreement is governed by Delaware law.", False),
    ("", False),
])
def test_abstention_detection_is_lenient_about_punctuation_and_case(text, expected):
    """Small local models rarely emit a bare marker even when told to;
    counting a hedged refusal as a confident answer would misreport a
    correct abstention as a faithfulness failure."""
    assert is_no_answer(text) is expected


# --- backends --------------------------------------------------------------

def test_generation_identity_names_backend_and_model():
    g = Generation(text="x", backend="ollama", model="llama3.2:3b", latency_ms=1)
    assert g.identity == "ollama:llama3.2:3b"


def test_backend_failure_is_returned_not_raised():
    class _Boom(Backend):
        name = "boom"
        default_model = "m"

        def _call(self, prompt, system):
            raise RuntimeError("rate limited")

    g = _Boom().generate("hi")
    assert g.error is not None and "rate limited" in g.error
    assert g.text == ""
    assert g.backend == "boom"  # provenance survives the failure


def test_successful_generation_records_provenance_and_latency():
    class _Ok(Backend):
        name = "ok"
        default_model = "m"

        def _call(self, prompt, system):
            return "answer"

    g = _Ok().generate("hi")
    assert (g.text, g.error) == ("answer", None)
    assert g.latency_ms >= 0
    assert g.temperature == backends.DEFAULT_TEMPERATURE


def test_cloud_and_local_backends_are_classified_correctly():
    """The judge's cloud-only rule (TRD §6.2) is enforced off this."""
    assert is_cloud("gemini") and is_cloud("groq")
    assert not is_cloud("ollama")


def test_build_backend_rejects_an_unknown_name():
    with pytest.raises(SystemExit):
        build_backend("gpt-9")


def test_default_temperature_is_zero_for_reproducibility():
    """An eval run that cannot be reproduced is not a measurement."""
    assert backends.DEFAULT_TEMPERATURE == 0.0
    for cls in (OllamaBackend, GeminiBackend, GroqBackend):
        assert cls().temperature == 0.0


# --- routing (TRD §6.2) ----------------------------------------------------

def test_interactive_generation_has_no_backend_parameter():
    """TRD §6.2: interactive generation is ALWAYS Ollama, and routing is
    never a user-facing toggle. Structural, not documentary -- adding a
    backend argument here fails this test."""
    params = set(inspect.signature(generate.generate_interactive).parameters)
    assert params == {"question", "chunks", "model"}
    assert "backend" not in params


def test_interactive_generation_constructs_the_local_backend(monkeypatch):
    seen = {}

    class _Fake:
        name, model, temperature = "ollama", "stub", 0.0

        def generate(self, prompt, system=None):
            seen["prompt"], seen["system"] = prompt, system
            return Generation("ok", self.name, self.model, 1)

    monkeypatch.setattr(generate, "OllamaBackend", lambda model=None: _Fake())
    g = generate.generate_interactive("q?", CHUNKS)
    assert g.backend == "ollama"
    assert seen["system"] == SYSTEM_PROMPT
    assert "q?" in seen["prompt"]


def test_eval_generation_uses_whichever_backend_it_is_handed():
    """TRD §6.2 permits a per-RUN choice on the batch-eval path; passing the
    backend in is what keeps that decision at run level, not per request."""
    class _Cloud(Backend):
        name = "groq"
        default_model = "m"

        def _call(self, prompt, system):
            return "cloud answer"

    g = generate.generate_for_eval("q?", CHUNKS, _Cloud())
    assert (g.backend, g.text) == ("groq", "cloud answer")
