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
                    │   SEGMENTATION ENGINE (Phase 2) │
                    │  - Boilerplate strip (1st step) │
                    │  - Cascade: ARTICLE → Section   │
                    │    N.N → bare numbering →       │
                    │    fallback                     │
                    │  - Oversized sub-split (parent-  │
                    │    child metadata)               │
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
                │  - retrieval correctness           │
                │    (mechanical span overlap)       │
                │  - faithfulness judge (cloud LLM,  │
                │    offline, separate call)         │
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

| Component | Responsibility | Depends on | Consumed by |
|---|---|---|---|
| Segmentation Engine | Turn raw contract text into classification/retrieval units | CUAD raw JSON | Classifier, Chroma ingestion |
| Classifier | Predict 41-category multi-label presence per segment | Segmentation output | Chroma metadata, Risk Rating |
| Chroma Vector DB | Store chunk embeddings + classifier metadata | Segmentation output, Classifier output | RAG retrieval |
| RAG Retrieval/Generation | Answer contract questions, optionally filtered | Chroma, Inference backends | Eval harness, API layer |
| Eval Harness | Score retrieval correctness + faithfulness | CUAD QA triples, Segmentation, RAG output | Every phase's validation gate |
| Risk Rating Layer | Flag presence/absence risk signals | Classifier output only (no new inference) | API layer |
| MLOps Layer | Track, version, monitor — cross-cutting | All components above | Portfolio evidence, drift alerts |
| FastAPI Layer | Expose classify/ask/eval over HTTP | All components above | Frontend, direct API consumers |
| Next.js Frontend | Thin UI over backend | FastAPI layer | Portfolio demo viewers |

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
| **2** | Segmentation engine | 🔶 **Design locked, implementation pending (current phase)** | Cascading pattern-detection segmenter, validated manually against a real contract sample |
| **3** | Eval harness skeleton | ⬜ Not started | Retrieval-correctness metric, faithfulness judge scaffolding, logging schema — built before classifier/RAG exist |
| **4** | Classical classifier baseline | ⬜ Not started | TF-IDF + weighted logreg/XGBoost, all 41 categories, MLflow tracking begins |
| **5** | Classifier feature improvements | ⬜ Not started | Evidence-gated, locked priority order (n-grams → structural features → threshold tuning → imbalance handling → ensembling) |
| **6** | RAG retrieval path | ⬜ Not started | Chroma ingestion, retrieval logic, full-document search validated against harness before generation added |
| **7** | RAG generation + inference backend | ⬜ Not started | Ollama/cloud routing wired in, faithfulness judge goes live |
| **8** | Classifier-to-RAG hard filter integration | ⬜ Not started | Metadata filter wired in, filter-on/filter-off precision@k ablation run as portfolio artifact |
| **9** | MLOps wrap-up | ⬜ Not started | Drift monitoring (classifier + embeddings) live, query logging fully live, DVC formalized across all datasets |
| **10** | Risk rating heuristic | ⬜ Not started | Presence/absence lookup table, Method 1 + Method 3 validation |
| **11** | API layer (FastAPI) | ⬜ Not started | `/classify`, `/ask` (with filter params), `/eval` |
| **12** | Frontend (Next.js) | ⬜ Not started | Thin layer over already-working, already-evaluated backend |

**Rule governing every phase:** a phase is not done until it is committed **and pushed** to GitHub — working locally is not sufficient to mark a phase complete.

---

## 6. Phase 2 Detail — Current Active Phase

Design is fully locked (see TRD §2 for full technical detail). Summary of locked decisions for quick reference:

- **Cascade order:** boilerplate strip → ARTICLE → Section N.N → bare numbering → fallback.
- **Pattern confirmation threshold:** 3+ hits to confirm a real numbering scheme.
- **Oversized cutoff:** ~4000–5000 chars (p75–p80 of real CUAD measurements).
- **Undersized floor:** ~150–200 chars.
- **TOC disambiguation:** absence of prose between consecutive header matches = TOC, not real headers.
- **Output schema:** intentionally loose until Phase 3's harness defines required fields.

**Remaining work in this phase:** write the actual Python segmenter module implementing the above, then manually validate against a real contract sample (including at least one long contract with a TOC, to exercise the disambiguation rule) before moving to Phase 3.

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
- **Coordination artifact:** `PROJECT_LEDGER.md`

---

## 8. How to Use This Document Set With Another Model

Paste `PRD.md`, `TRD.md`, and this file (`ARCHITECTURE.md`) together into a new session. That gives any model:
- **PRD** → what the project is, who it's for, what's explicitly out of scope, why it's built the way it's built.
- **TRD** → every locked technical decision with rationale and rejected alternatives — enough to answer "why not X instead?" without re-deriving from scratch.
- **ARCHITECTURE** → how components connect, current build status, and exactly what phase is active right now.

Any agent picking this up should treat every decision marked "locked" as stable unless the user explicitly says they want to reopen it — this mirrors the working style already established with Claude.