# COVENANT PROJECT STATE LEDGER
*Multi-agent handoff doc — Claude (architect), Gemini (debugger), Grok (tester)*
*Last updated: 2026-08-14*

## 0. STATUS FLAG
**Phase: 0 — Repo & environment scaffolding. NOT STARTED (blocked).**
No application code, schema, or API exists yet. Everything below Section 1
is architecture that has been decided but not implemented. Do not debug or
test against anything in this document as if it were running code — it
isn't yet. This ledger exists so all three agents share the same accurate
picture of what's real vs. what's planned.

**Current blocker**: waiting on a verified clean `COVENANT.zip` (previous
attempt had a folder-nesting bug + premature DVC init; repo and local
folder were deleted and a restart from a verified zip was chosen instead
of patching in place).

---

## 1. PROJECT METADATA & ARCHITECTURE

### 1.1 What this project is (and isn't)
Covenant is a **solo ML/AI portfolio project**, not a product and not a
CLM system. It demonstrates classical ML, RAG/retrieval engineering, and
MLOps at roughly equal depth, built on the CUAD dataset (510 expert-labeled
contracts, 41 clause categories, native span-extraction task reformulated
as multi-label classification). It has clean API boundaries so it *could*
later slot into a larger CLM as an ingestion-time intelligence module —
but it does not implement contract lifecycle workflows, e-signature,
storage, or ingestion of arbitrary live contracts. There is no "document
upload → status pipeline" in scope.

### 1.2 Core Tech Stack (decided, mostly unimplemented)
* **Frontend**: Next.js — Phase 12 only, deferred, zero code written.
* **Backend**: Python, FastAPI — Phase 11 only. Two endpoints planned:
  `/classify`, `/ask` (plus `/eval` for harness runs). Zero code written.
* **Database**: SQLite for structured/relational data (logs, metadata,
  predictions). No ORM chosen yet, no schema written.
* **Vector DB**: Chroma (chosen over Qdrant — no filtering-bottleneck
  evidence yet to justify Qdrant). No collections created yet.
* **ML tooling**: scikit-learn (TF-IDF, logistic regression), XGBoost.
  Transformer fine-tuning explicitly out of scope unless classical ceiling
  is proven insufficient with evidence.
* **Inference backends**: Ollama (local, interactive demo + batch),
  cloud API (batch eval + LLM-as-judge only, always offline, never the
  generating model for judge calls). vLLM explicitly excluded — no ROCm
  path for integrated AMD Radeon 780M graphics.
* **MLOps**: MLflow (experiment tracking, starts Phase 4), DVC (data
  versioning, initialized Phase 0).
* **Third-party integration APIs**: **none.** No e-sign, OCR, or document
  storage APIs are part of this project's scope. Do not add these unless
  the scope is explicitly reopened.

### 1.3 System Architecture Overview
Covenant runs two parallel pipelines off one shared segmenter: a classical
multi-label clause classifier (TF-IDF/logreg/XGBoost across 41 CUAD
categories) and a RAG contract Q&A system backed by Chroma, with a
manually-defined presence/absence risk-heuristic layer reading the
classifier's output post-hoc. The classifier can optionally constrain RAG
retrieval via a lawyer-toggled hard metadata filter (`where={"category":
selected}` applied pre-cosine-similarity), never automatic. There is no
ingestion, parsing, or status-workflow system in the CLM sense — input is
CUAD's fixed 510-contract dataset, not a live document intake pipeline.

---

## 2. DATABASE SCHEMA & DATA OBJECTS

### 2.1 Current schema: NONE EXISTS
No SQL DDL, Prisma schema, or Mongoose models have been written. Do not
generate migrations or write tests against a schema — there isn't one.

### 2.2 Planned data shapes (design intent only, not implemented)
```
# Chroma chunk metadata (planned, not built)
{
  "chunk_id": str,
  "contract_id": str,
  "text": str,
  "category": list[str],       # multi-label, classifier-predicted at ingestion
  "parent_section_id": str | None,   # for sibling-pull on sub-split segments
  "position": int,
}

# SQLite tables (planned, not built — no column types finalized)
- predictions(contract_id, chunk_id, category, confidence, model_version, timestamp)
- query_log(query_id, question, contract_id, filter_applied, retrieved_chunk_ids,
            answer, faithfulness_score, backend_used, timestamp)
- risk_flags(contract_id, category, flag_direction, reason, timestamp)
```
These are subject to change once Phase 1 (data exploration) and Phase 2
(segmenter) are actually built — do not treat as final.

---

## 3. CORE SERVICE APIS & WEBHOOK ENVELOPES

### 3.1 Active API Endpoints: NONE
FastAPI layer is Phase 11. Zero routes exist. Planned contract only:
* `POST /classify` — planned, not implemented
* `POST /ask` — planned, response includes `filter_applied` and
  `classifier_confidence_on_filter`, both `null` when filter toggle is OFF
* `GET /eval` — planned, triggers harness run

No webhook envelopes exist or are in scope — there is no external system
sending Covenant webhooks.

---

## 4. THE CURRENT SPRINT TARGET

### 4.1 Feature in Progress
* **Target**: Phase 0 — repo initialization, `.gitignore`, monorepo
  skeleton, DVC init, first commit + push.
* **Blocker**: waiting on verified clean project zip (prior attempt hit a
  double-nesting unzip bug compounded by premature DVC init).
* **Current active code block**: none. No code has been written this
  session.

### 4.2 Notes for Gemini (debugger) and Grok (tester)
There is nothing to debug or test yet. The next real artifacts you'll see
are: repo skeleton + `.gitignore` (Phase 0), then CUAD exploration
scripts (Phase 1), then the segmenter (Phase 2) — that's the first
component with real logic worth testing (cascading pattern detection,
CUAD-span-containment validation). Flag it immediately if either of you
receives a task referencing anything in Section 2.2 or Section 3.1 as if
it already exists — it doesn't yet, and building against it will diverge
from what Claude actually ships.