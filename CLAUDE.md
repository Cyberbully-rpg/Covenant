# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Covenant is a solo ML/AI systems **portfolio** project (not a product), built on CUAD (Contract
Understanding Atticus Dataset — 510 expert-labeled contracts, 41 clause categories). It demonstrates
three roughly-equal-depth pillars: classical ML (multi-label clause classification), RAG/retrieval
engineering (contract Q&A), and MLOps (experiment tracking, data versioning, drift monitoring, eval
harnesses). FastAPI + Next.js are a thin exposure layer built last, not the point of the project.

Full context lives in three companion docs — read them before making any non-trivial design or scope
decision, since this project treats prior decisions as locked unless explicitly reopened:
- `PRD.md` — product framing, scope, what's explicitly out of scope, success criteria
- `TRD.md` — every locked technical decision with rationale and rejected alternatives
- `ARCHITECTURE.md` — system diagrams, data flow, the 13-phase build order, current phase status

Treat any decision marked "locked" in these docs as stable. Don't silently re-litigate it — if it looks
wrong, surface that explicitly and ask before deviating.

## Commands

```powershell
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
dvc init

# Run the whole test suite (41 tests)
pytest backend -q

# Run the full segmenter test suite
pytest backend/segmenter/test_segmenter.py -v

# Run a single test
pytest backend/segmenter/test_segmenter.py::test_article_scheme_detected -v

# Run segmenter manually against a sample contract
python backend/segmenter/segmenter.py data/interim/harpoon_sample.txt

# Classifier (Phase 4) — build labels, then train the baseline
python backend/classifier/data/build_training_data.py  # + gold-span capture rate
python backend/classifier/models/train_baseline.py     # + MLflow run
```

There is no lint/format/build tooling configured yet, and no `pytest.ini`/`pyproject.toml` — pytest
runs with defaults. Each `backend/<subsystem>/` package is a plain module with an `__init__.py`; test
files currently live next to the module they test (`backend/segmenter/test_segmenter.py`), not under
`backend/tests/`.

## Architecture

### The segmenter is the highest-leverage component

`backend/segmenter/segmenter.py` turns raw CUAD contract text into two different downstream artifacts
from a single pass: **classifier training segments** (raw `text`, unmodified) and **RAG chunks**
(`embedding_text`, which gets parent-section-title context injected for undersized fragments). A
segmentation bug propagates into both the classifier and RAG subsystems, which is why it's treated as
its own phase rather than folded into classifier work.

Pipeline, in order (see module docstring in `segmenter.py` for full detail):
1. **Boilerplate strip** — removes SEC EDGAR extraction footers/page markers, tracking an
   offset map so cleaned-text positions can be translated back to original CUAD character offsets
   (needed later to resolve CUAD's answer-span labels against segments).
2. **TOC detection/removal** — table-of-contents blocks reuse the same numbering patterns as real
   headers; disambiguated by checking for absence of prose between consecutive header matches.
3. **Top-level scheme selection** — exactly one numbering scheme wins for the whole document, tried
   in priority order `article` → `section_nn` → `bare_nn` → `fallback` (single segment). A scheme is
   trusted only with 3+ matches (`MIN_SCHEME_MATCHES`) after TOC removal.
4. **Oversized sub-splitting** — segments over `OVERSIZED_THRESHOLD_CHARS` (4500) get re-cut using
   whatever numbering is nested inside them; sub-segments carry `parent_id` back to the top-level
   segment and the parent itself is dropped (not duplicated).
5. **Undersized context injection** — segments under `UNDERSIZED_THRESHOLD_CHARS` (175) get the
   parent header prepended into `embedding_text` only; raw `text` is left untouched so classifier
   labels are never contaminated by injected context.

The critical correctness invariant, enforced by tests: `raw_text[segment.start_char:segment.end_char]
== segment.text` must hold exactly, even after boilerplate stripping and TOC removal shift character
positions internally. Anything touching offset math must preserve this round-trip.

Thresholds in `segmenter.py` (`MIN_SCHEME_MATCHES`, `OVERSIZED_THRESHOLD_CHARS`,
`UNDERSIZED_THRESHOLD_CHARS`, TOC window/prose constants) are empirically grounded in measurements
across all 510 CUAD contracts, not arbitrary — don't tune them without re-measuring against the corpus.

### Two-consumer data flow (governs the rest of the pipeline)

```
CUAD JSON → Segmenter → ⎡ raw text      → Classifier (41-label, TF-IDF+LogReg/XGBoost) → predicted
                         ⎣ embedding text → Chroma (RAG chunks)                            categories
                                                                                            written as
                                                                                            chunk metadata
                                                            ↓
                                          RAG retrieval (full-doc default; optional
                                          lawyer-toggled hard `where={"category":...}`
                                          pre-ranking filter) → generation → faithfulness judge
                                                            ↓
                                          Risk rating layer (presence/absence lookup on
                                          classifier output only, zero new inference calls)
```

- **Classifier→RAG dependency is one-directional and optional, never standing**: RAG must run
  correctly with the filter toggle off and zero classifier involvement.
- **Inference backend routing is code-path-determined, never a user-facing parameter.** Three
  `generate()` call sites exist and must stay architecturally distinct: interactive demo (always
  Ollama), batch eval (Ollama or cloud, per-run config), faithfulness judge (always cloud, offline,
  separate call, never the same model that generated the answer being judged).
- **Eval harness (Phase 3) precedes the classifier and RAG builds** — everything after it is measured
  against a real yardstick (CUAD's native question/answer-span triples) from day one. Retrieval
  correctness (mechanical span overlap, non-gameable) and faithfulness (LLM-judge) are two decomposed
  signals, never conflated into one number.
- **Risk rating has no ground truth** — it's a manually-built presence/absence lookup table on
  classifier output, never reported with precision/recall/F1, and must be documented as heuristic-only
  wherever it's presented.

### Current phase

Phase 5 (classifier feature improvements) — not started; next up. Phases 0–4 are complete: repo
scaffolding, CUAD acquisition/EDA, segmenter, eval harness (steps 1–2; step 3 deferred to Phase 7),
and the classical classifier baseline (`backend/classifier/`, macro-F1 0.430 / micro-F1 0.488).
See `ARCHITECTURE.md` §5 for the full 13-phase roadmap and status table before assuming what's built,
and §6B for the Phase 4 numbers Phase 5 has to beat.

## Working conventions specific to this project

- **Baseline-first, evidence-gated escalation.** The classifier is TF-IDF + logistic regression/XGBoost
  by default; transformer fine-tuning is shelved and only reopened with evidence the classical ceiling
  is insufficient. The same discipline applies to infra choices (SQLite over Postgres, Chroma over
  Qdrant) — don't add complexity ahead of demonstrated need.
- **No CUDA/discrete GPU** (AMD Ryzen 7 8840HS, integrated Radeon 780M) — vLLM is fully out of
  consideration; local inference is Ollama-only, cloud API for batch eval and the LLM-judge.
- **A phase isn't done until committed and pushed** to `main` the same day it finishes — local and
  remote should never diverge by more than one phase.
- **Never report a single blended accuracy number** for the classifier — always macro-F1 + micro-F1 +
  per-category precision/recall/F1, given severe class imbalance across the 41 categories.
- Repo lives at `C:\Dev\Covenant\` specifically (not under OneDrive) to avoid DVC/Chroma file-lock
  conflicts under cloud sync — don't relocate it.
