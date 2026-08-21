"""
Covenant Phase 3 step 3 / Phase 7 — harness logging schema.

TRD §3.5 requires every harness log row to record **backend identity**
alongside the prompt, the retrieved chunks and the answer, with a stated
reason: without it, a regression cannot be attributed to a backend swap
(an Ollama model version bump, a cloud model deprecated underneath you)
versus a genuine retrieval or chunking change. That is not hypothetical
here — building Phase 7 turned up `gemini-2.5-flash` returning 404 despite
appearing in `models.list()`. A run logged without model identity would
have been unexplainable a week later.

One JSONL row per eval question, written as the run proceeds rather than
at the end, so an interrupted or rate-limited run keeps everything it
already paid for.

WHAT IS DELIBERATELY KEPT SEPARATE
-----------------------------------
Retrieval correctness and faithfulness are stored as distinct fields and
never combined into a single score (TRD §3.3). The whole point of the
decomposition is being able to say "the answer was wrong because
retrieval missed" versus "retrieval was fine and the model invented
something" — a blended number destroys exactly that attribution. Anything
reading these logs must preserve the split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class GenerationLogRow:
    # --- identity of the run -------------------------------------------
    run_id: str
    schema_version: int = SCHEMA_VERSION
    timestamp: str = ""

    # --- the question --------------------------------------------------
    contract_id: str = ""
    category: str = ""
    question: str = ""
    has_gold_span: bool = False

    # --- retrieval (what was fed to the model) -------------------------
    retriever: str = ""
    k: int = 0
    retrieved_spans: list = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)

    # --- retrieval correctness: mechanical, non-gameable (TRD §3.3) ----
    retrieval_hit: bool | None = None
    gold_chars_hit: int | None = None
    gold_chars_total: int | None = None
    retrieved_chars: int | None = None

    # --- generation ----------------------------------------------------
    prompt: str = ""
    answer: str = ""
    abstained: bool = False
    # TRD §3.5: backend identity, non-optional in practice.
    generator_backend: str = ""
    generator_model: str = ""
    generator_temperature: float | None = None
    generation_latency_ms: int | None = None
    generation_error: str | None = None

    # --- faithfulness: LLM judge, trend indicator only (TRD §3.4) ------
    judge_label: str | None = None
    judge_rationale: str = ""
    judge_backend: str = ""
    judge_model: str = ""
    judge_latency_ms: int | None = None
    judge_error: str | None = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def generator_identity(self) -> str:
        return f"{self.generator_backend}:{self.generator_model}"


class GenerationLogWriter:
    """Append-only JSONL writer. Flushes per row on purpose.

    A batch eval is long, costs API calls, and can die to a rate limit at
    any point. Buffering would trade real money for a marginally faster
    write.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.n_written = 0

    def write(self, row: GenerationLogRow) -> None:
        self._fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        self._fh.flush()
        self.n_written += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "GenerationLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
