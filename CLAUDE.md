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

# Classifier (Phase 5) — escalation ladder (n-grams -> structural -> tuned thresholds)
python backend/classifier/models/train_experiment.py   # + MLflow run per step

# RAG (Phase 6) — build the vector store, then measure retrieval variants
python backend/rag/ingestion/ingest.py --limit 0       # 510 contracts -> 45,254 windowed records
python backend/eval/harness/run_retrieval_variants.py --variants all --limit 0
python backend/eval/harness/run_retrieval_variants.py --variants hybrid_bigram_prior --limit 0 --split test
python backend/eval/harness/run_ceiling.py            # segmentation ceiling (0.9985)
python backend/eval/harness/run_rank_diagnostic.py    # rank distribution + per-category

# RAG generation (Phase 7) — retrieve -> generate -> judge -> log
python backend/eval/harness/run_generation_eval.py --gold-rows 60 --absent-rows 20
python backend/eval/harness/rejudge.py <run.jsonl>                    # re-grade, no regeneration
python backend/eval/harness/rejudge.py <run.jsonl> --judge-backend groq  # second judge for §6.5 variance
```

The variant sweep is the only way retrieval numbers should be produced — it runs every variant through
the same `scorer.score_query()` on the same 6,702 rows at the same k, and re-runs `tfidf`/`dense` as
controls so harness drift shows up immediately. Ingesting under a different embedding model means
setting `COVENANT_EMBED_MODEL` for *both* the ingest and the sweep; collections are namespaced per
model so the two can never silently mix. Any variant that fits something (the lead prior) must be
reported on `--split test` — the harness fits on the train side only and refuses to write split or
partial runs into `baseline_results.json`.

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

Phase 6 (RAG retrieval path) is complete. The adopted default is `hybrid_bigram_prior_cleanq` — RRF fusion
of bigram TF-IDF and windowed dense retrieval, plus a leading-segment positional prior for the four
document-metadata categories and query-boilerplate stripping. **0.8460 hit_rate@5 on 102 held-out
contracts**, against 0.6934 for the Phase 3 TF-IDF baseline. Four further escalations were measured
and three rejected on evidence (char n-grams, cross-encoder reranking, a retrieval-tuned embedding
model) — see TRD §7.5 before re-proposing any of them. Three things must travel with that number: the segmentation ceiling is 0.9985
(so remaining misses are ranking, never chunking); the dense half does **not** separate from lexical
alone out of sample and is kept on an argued basis, not a measured one (TRD §7.2); and anything fitted
is reported on `--split test` only. See `ARCHITECTURE.md` §6D for the variant tables, the truncation
confound found in the first measurement, and the rank/coverage diagnostics.
**Phase 7 (generation + inference backend routing) is built and has taken its first measurement**,
which also discharges Phase 3's deferred step 3 (faithfulness judge + logging schema). First run,
80 rows on the held-out split: retrieval hit_rate@5 0.8500, faithfulness 0.6154 (gemini judge, 100%
coverage), and the model correctly abstained on only 35% of rows where CUAD says the clause is absent.

Three things must travel with any faithfulness number. **Name the judge** — swapping graders on
identical answers moved the score 0.591 -> 0.700 (TRD §6.5), so headline results are reported as a
range across two judges, never a single figure. **Check coverage** — `NOT_JUDGED` (budget exhausted)
is a distinct label from `UNPARSEABLE` (judge replied unreadably), and a run whose `judged_coverage`
is below 100% is partial, not a smaller version of the same number. **Respect the quota** — Groq free
tier is 200k tokens/day *per model*, roughly 90 judge calls, so an 80-row run costs most of one
model's daily budget. Cloud batch-eval generation does NOT scale here; beyond ~80 rows local Ollama is
the only option without paying (TRD §6.6).

Still open in Phase 7: the generator is `llama3.2:3b` (small — fabrication and weak abstention are
both characteristic of undersized models; a 7-8B pull is the untried next lever), and abstention
discipline needs work as a product concern in its own right.

Phase 5 (classifier feature improvements) is partially open: steps 1–3 are complete (n-grams,
structural features, tuned thresholds; macro-F1 0.430 → 0.503) and steps 4–5 (SMOTE/ensembling) were
explicitly deferred. Two items previously recorded as blocked are **no longer blocked** and can be
run whenever it's worth the time: the zero-shot LLM diagnostic (see "Inference backends" below), and
§6C's parked idea of reusing the Phase 6 sentence-transformer embeddings as a classifier feature —
those embeddings now exist.

### Inference backends (verified working, 2026-08-21)

All three are configured and were checked live, so **do not assume generation work is blocked**:

- **Ollama** — running on `localhost:11434` with `llama3.2:3b` pulled. Only that model is present;
  anything larger needs an explicit `ollama pull` first, and on this CPU-only box a 7–8B model will be
  noticeably slower per answer.
- **Gemini** — `GEMINI_API_KEY` in `.env`, validated (50 models visible, incl. `gemini-2.5-flash`).
- **Groq** — `GROQ_API_KEY` in `.env`, validated (13 models visible).

`.env` is gitignored and holds only those two key names. Load it explicitly with an absolute path
(`load_dotenv(r"C:\Dev\Covenant\.env")`) — bare `load_dotenv()` raises an AssertionError when the
calling frame has no file, e.g. code piped in via stdin.

Routing is locked in TRD §6.2 and is **code-path-determined, never a user-facing toggle**: interactive
generation always Ollama; batch eval Ollama or cloud per run; the faithfulness judge always cloud,
always a separate call, and **never the same model that produced the answer being judged** — so with
Ollama generating, the judge must be Gemini or Groq.

Cost note for Phase 7: generation eval is one LLM call per row plus one judge call, so the full 6,702
rows is ~13,400 API calls. Use a sampled subset and record the sample size the same way the retrieval
scope claim is recorded, or faithfulness numbers won't be comparable across runs.

Phases 0–4 are complete: repo scaffolding, CUAD acquisition/EDA, segmenter, eval harness (steps 1–2),
and the classical classifier baseline. See `ARCHITECTURE.md` §5 for the full 13-phase roadmap and
status table before assuming what's built.

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
