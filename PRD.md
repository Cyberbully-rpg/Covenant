# Covenant — Product Requirements Document (PRD)

**Project name history:** Redline → ClauseLens → **Covenant** (current, locked)
**Document purpose:** Full context handoff. This file, together with `TRD.md` and `ARCHITECTURE.md`, is meant to be pasted into any other model/agent to fully reconstruct project context with no prior conversation history required.

---

## 1. What This Project Is

Covenant is a solo **ML/AI systems portfolio project**, not a commercial product and not a full-stack project. It is a contract-analysis pipeline built on CUAD (Contract Understanding Atticus Dataset — 510 real contracts, expert-labeled across 41 clause categories).

The point of the project is **skill demonstration across three roughly-equal-depth pillars**:
1. Classical ML (multi-label clause classification)
2. RAG / retrieval engineering (contract Q&A)
3. MLOps / production practices (experiment tracking, data versioning, drift monitoring, eval harnesses)

A FastAPI layer and a Next.js frontend exist only to **expose** the ML/RAG core for demonstration purposes. They are explicitly not the point of the project and are built last.

**This is NOT:**
- A product being built to sell or pitch to law firms
- A project aiming for state-of-the-art transformer fine-tuning by default
- A full-stack-first or frontend-first project

---

## 2. Target Audience / Use Case

- **Primary audience:** Portfolio reviewers (recruiters, hiring managers, ML engineers evaluating the builder's skill).
- **Secondary audience:** The builder's own learning — this is explicitly a skill-building exercise where complexity must be *earned by evidence*, not assumed.
- **Named collaborator/reviewer:** Kavish (friend the project has been explained to; may be referenced as an informal audience/sounding board).
- **No end users in the product-management sense.** There is no customer, no legal team, no deployment target beyond a portfolio demo.

---

## 3. Core Problem Statement

Legal contract review is manual, slow, and expert-bottlenecked. CUAD provides a labeled benchmark for this domain (span-extraction QA over 41 clause categories). Covenant reframes this as a **multi-label classification + retrieval-augmented Q&A** problem to demonstrate an end-to-end ML systems build: from raw legal text → structured segments → classified clauses → retrieval-grounded answers → heuristic risk flags — with MLOps instrumentation at every stage.

---

## 4. Product Scope (What Covenant Does)

### 4.1 Multi-label Clause Classifier
- Classifies contract segments against the **full 41 CUAD categories** (not a simplified subset).
- Severe class imbalance is expected and reported honestly (macro-F1 + micro-F1, per-category precision/recall/F1 — never a single blended accuracy number).

### 4.2 RAG-based Contract Q&A
- Full-document similarity search by default.
- Optional lawyer-toggled hard filter that constrains retrieval to a single classifier-predicted category before ranking.
- Answers are generated and then separately judged for faithfulness (grounding) against retrieved chunks.

### 4.3 Heuristic Risk-Rating Layer
- A **non-learned**, manually-built lookup table mapping specific CUAD categories to a risk direction (flag-if-absent for protective clauses, flag-if-present for one-sided clauses).
- Explicitly documented as a heuristic, never presented as validated with precision/recall (no ground truth for "risk" exists in CUAD or anywhere else for this project).

### 4.4 MLOps Instrumentation
- Experiment tracking (MLflow), data versioning (DVC), a purpose-built eval harness, drift monitoring (classifier input distribution + embedding drift on incoming documents), and structured query/prediction logging (SQLite).

### 4.5 Exposure Layer (secondary, built last)
- FastAPI endpoints: `/classify`, `/ask`, `/eval`.
- Next.js frontend: thin display layer over an already-working, already-evaluated backend.

---

## 5. Explicitly Out of Scope

- Transformer fine-tuning (shelved; classical-only unless classical ceiling is proven insufficient with evidence).
- Content-severity risk judgment (e.g., "how bad is this specific liability clause") — out of scope; only presence/absence heuristics are in scope.
- Cross-clause interaction risk analysis — out of scope.
- vLLM / discrete-GPU-dependent inference — hardware does not support it (see constraints).
- Postgres or any multi-writer relational DB — no evidence of need for a solo project; SQLite chosen deliberately.
- Qdrant or any vector DB beyond Chroma — deferred pending evidence of a real filtering bottleneck.
- Building a commercial-grade UI, auth system, multi-tenant infra, or billing — none of this is relevant to a portfolio ML project.
- An "other relevant categories detected" always-on field in RAG responses — explicitly rejected to avoid an implicit standing dependency of RAG on the classifier.

---

## 6. Success Criteria

Since there is no business metric (no users, no revenue), success is defined as:

1. **Evidentiary rigor** — every claim about model quality is backed by a real metric on a real (CUAD-derived) eval set, never vibes-based.
2. **Architectural defensibility** — every major decision (filter design, DB choice, backend routing, risk-rating scope) has a documented rationale and known tradeoffs, and can be explained/defended in an interview setting.
3. **Complete, working, portfolio-demoable pipeline** — from raw CUAD JSON to a working `/ask` endpoint with retrieval + generation + faithfulness scoring + optional classifier filter + risk flags.
4. **Honest reporting of limitations** — e.g., risk-rating is explicitly labeled heuristic-only; RAG eval scope is explicitly bounded to CUAD's contract distribution, not "legal documents in general."
5. **Clean, version-controlled build history** — every phase committed and pushed the same day it's completed; local and remote never diverge by more than one phase.

---

## 7. Constraints

### 7.1 Hardware
- AMD Ryzen 7 8840HS, integrated Radeon 780M, 32GB RAM.
- **No CUDA, no discrete GPU.** This rules out vLLM entirely and shapes the local-inference strategy toward Ollama (CPU/iGPU-capable local inference).

### 7.2 Team / Process
- Solo builder.
- Multi-agent AI-assisted workflow: **Claude = architect** (design decisions, this document set), **Gemini = debugger**, **Grok = tester**.
- `PROJECT_LEDGER.md` is a shared ground-truth handoff artifact used to keep multiple AI agents synchronized on project state.
- Cursor is the IDE; **agent mode (autonomous file generation) is deliberately kept disabled** — it has previously caused unwanted file creation.

### 7.3 Decision-making style (process constraint, not a technical one)
- Architectural decisions are discussed conversationally with tradeoffs laid out in plain language before being locked.
- Ambiguous requests are resolved by presenting interpretations/options explicitly — never guessed silently.
- Once a decision is locked, it is treated as stable and not silently re-litigated (it can be explicitly reopened with evidence).

---

## 8. Build Order Philosophy (Product-level Rationale)

Covenant follows an **eval-first, ML-core-first** philosophy:

- The eval harness is built in **Phase 3** — before the classifier or RAG system exist — so every subsequent component is measured against a real yardstick from day one, rather than bolting evaluation on retroactively.
- The classifier and RAG core are built and validated via API/CLI only; the frontend is the **last** phase (Phase 12), a thin layer over an already-validated backend.
- This ordering is a deliberate rejection of "build the demo UI first" — the product is the pipeline, not the interface.

See `ARCHITECTURE.md` §Phase Roadmap for the full 13-phase build order and current status.

---

## 9. Current Status (as of this document)

- **Phase 0 (repo scaffolding):** ✅ Complete, committed to `main`.
- **Phase 1 (CUAD data exploration/EDA):** ✅ Complete, committed to `main`.
- **Phase 2 (contract segmenter):** 🔶 Design fully locked; implementation not yet written. This is the current active phase.
- **Phases 3–12:** Not started.

See `TRD.md` for locked technical decisions and `ARCHITECTURE.md` for system design, data flow, and phase-by-phase detail.