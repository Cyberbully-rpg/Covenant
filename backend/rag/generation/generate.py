"""
Covenant Phase 7 — the generation call sites.

TRD §6.2 locks three `generate()` call sites with different routing, and
requires that they "must not be conflated in code or in logging":

  | call site           | backend                                        |
  |---------------------|------------------------------------------------|
  | interactive demo    | ALWAYS Ollama                                  |
  | batch eval          | Ollama or cloud, per-run config                |
  | faithfulness judge  | ALWAYS cloud, separate call, never the         |
  |                     | same model that produced the answer            |

Two of the three live here. The judge lives in
`backend/eval/harness/judge.py` — physically separate, because it is not a
generation path at all: it runs offline, after the fact, scoring text
someone else produced. Putting it beside these two would make "same model
judged its own answer" a one-line mistake rather than an impossible one.

The routing rule is enforced structurally, not by comment:
`generate_interactive()` takes **no backend argument**. There is no
parameter to pass, so no call site can make it use a cloud model, and no
future refactor can turn it into the user-facing toggle TRD §6.2 forbids.
`generate_for_eval()` does take a configured backend, because a per-run
choice is exactly what that rule permits there.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends import Backend, Generation, OllamaBackend  # noqa: E402
from prompt import SYSTEM_PROMPT, build_prompt  # noqa: E402


def generate_interactive(question: str, chunks: list[dict],
                         model: str | None = None) -> Generation:
    """Answer one question for the live demo. ALWAYS Ollama (TRD §6.2).

    `model` selects which *local* model; there is deliberately no way to
    select a different backend. Cloud inference is never on the live
    serving path — that is the accepted tradeoff in TRD §6.3, not an
    oversight to be optimized away later.
    """
    return OllamaBackend(model=model).generate(
        build_prompt(question, chunks), system=SYSTEM_PROMPT
    )


def generate_for_eval(question: str, chunks: list[dict],
                      backend: Backend) -> Generation:
    """Answer one question inside a batch eval run.

    Takes an already-constructed backend: the choice is made once per run
    by the runner (from config), never per request. Passing it in rather
    than naming it here is what keeps that a run-level decision.
    """
    return backend.generate(build_prompt(question, chunks), system=SYSTEM_PROMPT)
