# Covenant — Architecture & Phase Roadmap

**Companion documents:** `PRD.md` (product framing, scope, audience), `TRD.md` (locked technical decisions and rationale).
**Purpose:** System-level view of how components connect, data flows end-to-end, and exactly where the project currently stands in its 13-phase build order.

---

## 1. System Overview

```
                          ┌─────────────────────┐
                          │   CUAD v1 Dataset    │
                          │  (510 contracts,     │
                          │   41 categories)     │
                          └──────────┬───────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  SEGMENTATION ENGINE (Ph 2) ✅  │
                    │  - Boilerplate strip (1st step) │
                    │  - Cascade: strict schemes       │
                    │    (ARTICLE → Section N.N →      │
                    │    bare N.N) → inline variants   │
                    │    → window-split fallback       │
                    │  - Oversized sub-split, recursive│
                    │    (parent-child metadata)       │
                    │  - Undersized: context injection │
                    │    at embed time only            │
                    └───────┬───────────────┬──────────┘
                            │               │
              (classifier training)   (RAG chunks)
                            │               │
                            ▼               ▼
        ┌───────────────────────┐   ┌──────────────────────┐
        │  CLASSIFIER (Ph 4-5)  │   │  CHROMA VECTOR DB     │
        │  TF-IDF + LogReg/     │   │  (Phase 6)            │
        │  XGBoost, 41 labels   │   │  chunk embeddings +   │
        │  MLflow tracked       │   │  classifier category  │
        │                       │   │  metadata (multi-label)│
        └───────────┬───────────┘   └───────────┬────────────┘
                    │                            │
                    │  predicted categories      │
                    │  written as chunk metadata  │
                    │  at ingestion (one-way)     │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────┐
                │   RAG RETRIEVAL + GENERATION     │
                │   (Phase 6-7)                    │
                │  - default: full-doc similarity  │
                │  - optional hard filter:         │
                │    where={"category": selected}  │
                │    (lawyer-toggled, pre-ranking)  │
                │  - generation via Ollama/cloud   │
                │    (routing = code-path, not     │
                │     user toggle)                  │
                └──────────────┬────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │   EVAL HARNESS (Phase 3, built    │
                │   before classifier/RAG exist)    │
                │  - eval set from CUAD QA triples ✅│
                │  - retrieval correctness ✅         │
                │    (span overlap, reported with    │
                │     gold_density anti-gaming pair) │
                │  - random + TF-IDF baselines ✅     │
                │  - faithfulness judge (cloud LLM,  │
                │    offline, separate call) → Ph 7  │
                │  - filter-on vs filter-off          │
                │    precision@k ablation (Ph 8)      │
                └──────────────┬────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │   RISK RATING LAYER (Phase 10)    │
                │  - lookup table on classifier      │
                │    present/absent output only      │
                │  - zero new inference calls        │
                │  - heuristic only, no P/R/F1 claim │
                └──────────────┬────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │   MLOPS LAYER (cross-cutting,      │
                │   formalized Phase 9)              │
                │  - MLflow (classifier experiments) │
                │  - DVC (data versioning)           │
                │  - drift monitoring (input dist +   │
                │    embedding drift)                 │
                │  - SQLite (logs, metadata, preds)  │
                └──────────────┬────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │   API LAYER — FastAPI (Phase 11)  │
                │  /classify   /ask   /eval          │
                └──────────────┬────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │   FRONTEND — Next.js (Phase 12)   │
                │  thin display layer only           │
                └─────────────────────────────────┘
```

---

## 2. Component Responsibilities

| Component | Status | Responsibility | Depends on | Consumed by |
|---|---|---|---|---|
| Segmentation Engine | ✅ `backend/segmenter/` | Turn raw contract text into classification/retrieval units | CUAD raw JSON | Classifier, Chroma ingestion |
| Eval Harness | 🔶 `backend/eval/harness/` (retrieval ✅, faithfulness → Ph 7) | Score retrieval correctness + faithfulness | CUAD QA triples, Segmentation, RAG output | Every phase's validation gate |
| Classifier | ⬜ Ph 4–5 | Predict 41-category multi-label presence per segment | Segmentation output | Chroma metadata, Risk Rating |
| Chroma Vector DB | ⬜ Ph 6 | Store chunk embeddings + classifier metadata | Segmentation output, Classifier output | RAG retrieval |
| RAG Retrieval/Generation | ⬜ Ph 6–7 | Answer contract questions, optionally filtered | Chroma, Inference backends | Eval harness, API layer |
| Risk Rating Layer | ⬜ Ph 10 | Flag presence/absence risk signals | Classifier output only (no new inference) | API layer |
| MLOps Layer | ⬜ Ph 9 (cross-cutting) | Track, version, monitor | All components above | Portfolio evidence, drift alerts |
| FastAPI Layer | ⬜ Ph 11 | Expose classify/ask/eval over HTTP | All components above | Frontend, direct API consumers |
| Next.js Frontend | ⬜ Ph 12 | Thin UI over backend | FastAPI layer | Portfolio demo viewers |

**Critical dependency rule (locked, see TRD §5.5):** RAG must be able to run with **zero classifier involvement** when the filter toggle is off. The classifier→RAG dependency is one-directional and optional, never standing.

---

## 3. Data Flow — Two Consumers, One Segmenter

The segmenter is the single highest-leverage component in the pipeline because it feeds two otherwise-independent subsystems:

1. **Classifier training path:** raw (context-uninjected) segment text → TF-IDF vectorization → per-category classification. Raw text is preserved specifically so label training isn't contaminated by injected parent-title context.
2. **RAG chunking path:** context-injected segment text (parent section title prepended for undersized fragments) → embedding → Chroma storage. Context injection here improves retrieval quality without affecting classifier label integrity, because the two paths diverge immediately after the shared cascade/sub-split logic.

This is why segmentation quality is described in the TRD as setting the **ceiling** on both downstream systems — a segmentation bug doesn't just hurt one subsystem, it propagates into both.

---

## 4. Inference Backend Routing (Architectural View)

```
generate() call sites (must remain architecturally distinct):

1. Interactive demo generation  → Ollama (always)
2. Batch eval generation        → Ollama or cloud (per-run config)
3. Faithfulness judge           → cloud (always, offline, separate call,
                                          never same model as generator)
4. [Shelved, Phase 10 if built] → cloud (offline, cached at ingestion,
   Risk severity scoring (B2)      never the generating model)
```

Routing is resolved by **which code path is executing**, never by a request-time or user-facing parameter. This is enforced architecturally, not just as a convention — there is no API parameter anywhere in the design that lets a caller select a backend.

---

## 5. Locked Phase Roadmap (13 Phases)

| Phase | Name | Status | Key deliverable |
|---|---|---|---|
| **0** | Repo & environment scaffolding | ✅ **Complete** | `git init`, `.gitignore`, monorepo skeleton, DVC init, initial commit + push |
| **1** | Data acquisition + exploration | ✅ **Complete** | CUAD pulled, label distribution / span length / contract length examined, structure variance spot-checked (EDA notebook) |
| **2** | Segmentation engine | ✅ **Complete** | Cascading pattern-detection segmenter, 15 tests, validated against real CUAD contracts. Generalization gap found and fixed during Phase 3 (see §6) |
| **3** | Eval harness skeleton | 🔶 **Steps 1–2 complete; step 3 deferred to Phase 7 (current phase)** | Eval set from CUAD's native QA triples + retrieval-correctness metric + baselines ✅. Faithfulness judge & logging schema → Phase 7 |
| **4** | Classical classifier baseline | ⬜ Not started (next) | TF-IDF + weighted logreg/XGBoost, all 41 categories, MLflow tracking begins |
| **5** | Classifier feature improvements | ⬜ Not started | Evidence-gated, locked priority order (n-grams → structural features → threshold tuning → imbalance handling → ensembling) |
| **6** | RAG retrieval path | ⬜ Not started | Chroma ingestion, retrieval logic, full-document search validated against harness before generation added |
| **7** | RAG generation + inference backend | ⬜ Not started | Ollama/cloud routing wired in, faithfulness judge + harness logging schema built and go live (deferred here from Phase 3, step 3) |
| **8** | Classifier-to-RAG hard filter integration | ⬜ Not started | Metadata filter wired in, filter-on/filter-off precision@k ablation run as portfolio artifact |
| **9** | MLOps wrap-up | ⬜ Not started | Drift monitoring (classifier + embeddings) live, query logging fully live, DVC formalized across all datasets |
| **10** | Risk rating heuristic | ⬜ Not started | Presence/absence lookup table, Method 1 + Method 3 validation |
| **11** | API layer (FastAPI) | ⬜ Not started | `/classify`, `/ask` (with filter params), `/eval` |
| **12** | Frontend (Next.js) | ⬜ Not started | Thin layer over already-working, already-evaluated backend |

**Rule governing every phase:** a phase is not done until it is committed **and pushed** to GitHub — working locally is not sufficient to mark a phase complete.

---

## 6. Phase 2 (Segmentation) — Complete, With One Post-Hoc Correction

Implemented in `backend/segmenter/segmenter.py`, 15 tests in `test_segmenter.py`. Locked design (TRD §2) implemented as specified: boilerplate strip → TOC removal → single top-level scheme selection (3+ match threshold) → oversized sub-split → undersized context injection.

### 6.1 Generalization gap found by Phase 3's harness (and fixed)

Phase 2 passed all its tests and was validated against a real contract — but that contract (Harpoon) was blank-line-delimited, and **only ~23% of CUAD contracts are** (measured across 120). Every header pattern required a preceding `\n\n`, so **78% of contracts silently fell back to a single whole-document segment.**

This surfaced only once Phase 3's harness ran a *random* retriever and it scored `hit_rate@5 = 0.744` — impossible unless "the chunk" was the entire contract. The paired `gold_density = 0.009` confirmed it. Fixes:

- **Inline header patterns** (`section_inline`, `article_inline`, `bare_nn_inline`, `int_dot_inline`) for headers running inline in flowing prose, ranked **below** all strict blank-line-anchored schemes so stricter evidence always wins first.
- **Window-split fallback** (`FALLBACK_WINDOW_CHARS = 3000`) so unstructured documents still yield usable retrieval units; documents under `FALLBACK_MIN_DOC_CHARS = 6000` stay whole.
- **Recursive sub-splitting** — one level deep still left 25KB sub-segments.

| | Before | After |
|---|---|---|
| Contracts hitting fallback | 78% | 22% |
| Max segment size | 125,364 chars | 5,987 chars |
| Segments/contract (median) | 1 | 18 |

**Lesson recorded deliberately:** this is the eval-first ordering (TRD §3.1) paying for itself. A segmentation defect that all unit tests missed was caught before any classifier or retriever was built on top of it.

### 6.2 Offset semantics (locked by this phase)

`start_char`/`end_char` **bound** a segment's original extent rather than exactly reproducing `segment.text`, because boilerplate removed mid-body can't be represented by one contiguous span. Empirically <10% inflation (p90 8.86%). Verified invariant: `clean_boilerplate(raw[start:end]) == segment.text` for **323/323** real segments. Exact sub-span exclusion deferred until a consumer needs it.

---

## 6A. Phase 3 (Eval Harness) — Steps 1–2 Complete

Built in `backend/eval/harness/`. Step 3 (faithfulness judge + logging schema) **deferred to Phase 7**, where the generator it judges actually exists.

### 6A.1 Step 1 — Eval set (`build_eval_set.py`)

Derived from CUAD's native (question, contract, answer-span) triples — no hand-written questions.

| Metric | Count |
|---|---|
| Rows | 20,910 (510 contracts × 41 categories) |
| Rows with ≥1 gold span | 6,702 |
| Absent-category rows | 14,208 |
| Multi-span cells | 2,605 |
| CUAD offset mismatches | 0 |

**All** gold spans preserved per cell (correct = overlap with **any**). Absent-category rows are kept and tagged `has_gold_span: false`, never dropped. Outputs to `data/processed/eval/` (git-ignored, DVC's domain — regenerate by running the script).

**Scope claim (locked, travels with every reported number):** *"validated against CUAD's contract distribution using CUAD's 41 templated category probes"* — not diverse user questions, not legal documents in general.

### 6A.2 Step 2 — Retrieval-correctness metric (`scorer.py`)

Pure span-overlap function; no retriever, no I/O, 12 unit tests.

**This metric is explicitly NOT claimed to be non-gameable** — it is gameable by chunk size (retrieve the whole document, score a perfect hit rate). Mitigation is transparency, not correction: every result reports `hit_rate` **paired with** `gold_density` (gold chars ÷ retrieved chars) and `mean_retrieved_chars`. A unit test pins the property — whole-document and exact retrieval both score `hit_rate = 1.0`, and only `gold_density` separates them (1.0 vs <0.01). A hit-rate gain accompanied by a density drop is chunk-size gaming, not retrieval improvement.

**Absent-category rows are excluded from scoring** (span overlap is undefined with no gold span) and surfaced as `n_skipped_no_gold` — never silently dropped. Scoring correct *abstention* is deferred: it needs a confidence threshold, which needs a real retriever/classifier to calibrate against. The rows remain in the eval set for that later use.

### 6A.3 Baselines (`baselines.py`, `run_baselines.py`)

Two trivial retrievers bracket the problem before Phase 6 exists. Full corpus, 510 contracts, 6,702 scored rows, k=5:

| | random | tfidf |
|---|---|---|
| hit_rate@5 | 0.2226 | **0.5768** |
| gold_recall | 0.1402 | 0.5398 |
| gold_density | 0.0102 | 0.0332 |
| mean_retrieved_chars | 7,401 | 8,741 |

**TF-IDF lift: +0.354** at comparable retrieval volume, so the gain is genuine retrieval quality rather than chunk-size inflation.

**Phase 6 decision input (the reason these baselines exist):** plain lexical TF-IDF already reaches 0.577 hit_rate@5. Dense embedding retrieval must beat that by a worthwhile margin to justify Chroma + an embedding model + an ingestion pipeline. That bar is now measured, not assumed.

---

## 7. Environment & Tooling Snapshot

- **Repo:** `C:\Dev\Covenant\` (monorepo; relocated from OneDrive Desktop to avoid DVC/Chroma file-lock conflicts)
- **Python:** 3.12.10, `venv`, explicit interpreter path at creation
- **IDE:** Cursor, agent mode disabled
- **Package pins:** `pathspec<1.0.0`; ChromaDB pinned to 1.x line
- **Version control:** git (single `main` branch), DVC hooked in from Phase 0
- **Experiment tracking:** MLflow (from Phase 4)
- **Local inference:** Ollama
- **Cloud inference:** used for batch eval + LLM-as-judge only, never live serving path
- **Structured data store:** SQLite
- **Multi-agent roles:** Claude = architect, Gemini = debugger, Grok = tester
- **Coordination artifact:** `PROJECT_LEDGER.md` — *not currently in the repo; `ledger.md` was removed in the Phase 2 commit. This doc set (PRD/TRD/ARCHITECTURE) plus `CLAUDE.md` is the de facto handoff artifact.*
- **Testing:** pytest, defaults (no `pytest.ini`/`pyproject.toml`). Test files sit beside the module they test. Current suite: 27 tests (15 segmenter + 12 eval scorer)

### 7.1 Generated artifacts (not in git)

`data/processed/eval/` (`contracts.jsonl`, `eval_set.jsonl`, `metadata.json`, `baseline_results.json`) is git-ignored — regenerate with:

```powershell
python backend/eval/harness/build_eval_set.py          # rebuild eval set
python backend/eval/harness/run_baselines.py --limit 0 # re-run baselines (all 510)
```

DVC-tracking these is Phase 9 work (TRD §9.1), not yet done.

---

## 8. How to Use This Document Set With Another Model

Paste `PRD.md`, `TRD.md`, and this file (`ARCHITECTURE.md`) together into a new session. That gives any model:
- **PRD** → what the project is, who it's for, what's explicitly out of scope, why it's built the way it's built.
- **TRD** → every locked technical decision with rationale and rejected alternatives — enough to answer "why not X instead?" without re-deriving from scratch.
- **ARCHITECTURE** → how components connect, current build status, and exactly what phase is active right now.

Any agent picking this up should treat every decision marked "locked" as stable unless the user explicitly says they want to reopen it — this mirrors the working style already established with Claude.