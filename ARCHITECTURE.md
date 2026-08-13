# Architecture

## Overview

Covenant reformulates CUAD's native span-extraction QA task into two
parallel, evidence-gated pipelines:

1. **Classifier** — multi-label clause classification (41 categories)
2. **RAG** — retrieval-augmented contract Q&A

Both consume a shared **segmenter** output. A **risk-rating** layer sits
on top of the classifier's output (zero additional inference calls). A
thin **FastAPI** layer exposes both pipelines. A **Next.js frontend** is
the last layer, built only after the backend is validated.

## Design philosophy

**Baseline-first, evidence-gated escalation.** Every component starts at
the simplest viable version. More complexity (transformer fine-tuning,
soft-boost re-ranking, LLM severity scoring) is only added after the
simpler baseline is proven insufficient — not by default.

## Classifier

- **Scope**: full 41-category multi-label task, weighted loss, per-category
  precision/recall/F1, macro-F1 + micro-F1 (never blended accuracy)
- **Approach**: TF-IDF + weighted logistic regression / XGBoost per
  category. No transformer fine-tune unless the classical ceiling is
  proven insufficient with evidence.
- **Improvement levers, in priority order**:
  1. N-grams (bigrams/trigrams) in TF-IDF
  2. Structural features from the segmenter (position, header type,
     length, trigger phrases)
  3. Per-category threshold tuning
  4. Imbalance handling (e.g. SMOTE) for rare categories
  5. Logreg + XGBoost ensembling (last resort)
- **Key distinction**: segmenter quality sets the *ceiling*; feature
  representation determines how close the classifier gets to that
  ceiling. Sequential, not interchangeable.
- Zero-shot LLM baseline is a *diagnostic tool*, not a build gate — if it
  fails on the same examples the classical model fails on, that's CUAD
  label noise, not an architecture gap.

## Segmenter

Cascading pattern detection: ARTICLE headers → Section N.N numbering →
bare numbering → fallback. Numbering creates new segments; blank lines
alone do not. Oversized segments sub-split with parent-child metadata.
Undersized fragments get parent-title context injected at embed time;
raw text stays unchanged for classifier labels. Serves both the
classifier and RAG from one pipeline.

## RAG

- **Default**: full-document cosine similarity search, no filter.
- **Classifier-to-RAG filter (hard filter, locked)**: classifier runs at
  ingestion, writes predicted categories as Chroma metadata. Toggle ON =
  lawyer selects a category, `where={"category": selected}` applied
  *before* similarity search — pre-search constraint, not post-rank
  boost. 100% lawyer-driven, never automatic.
- **Merge semantics**: `/ask` response includes `filter_applied` and
  `classifier_confidence_on_filter`, populated only when the filter
  toggle was ON for that query; both null when OFF.
- Soft-boost re-ranking is a documented Phase 2 escalation, triggered
  only if the hard-filter ablation shows evidence of failure.

## Vector DB & inference

- **Chroma** (zero-setup). Qdrant deferred pending evidence of a real
  filtering bottleneck.
- **Ollama** (local) and **cloud API** only — vLLM ruled out (no CUDA).
  Interactive demo → Ollama. Batch eval → Ollama or cloud. LLM-as-judge →
  always cloud, offline, separate call, never the generating model.

## Eval harness

- Eval set derived from CUAD's native (question, contract, answer-span)
  triples.
- Two separately measured signals: **retrieval correctness** (mechanical
  span overlap, non-gameable) and **faithfulness** (LLM-as-judge, cloud,
  offline, trend indicator only).
- Every log row records backend identity alongside prompt/chunks/answer.
- Filter-on vs. filter-off comparison run as a portfolio artifact.
- Built in Phase 3 — before the classifier or RAG exist, after chunking
  is stable.

## MLOps

- Classifier: MLflow + DVC + drift monitoring.
- RAG: fixed eval set on every pipeline change, query log, embedding
  drift monitoring. Diagnostic, not curative.
- Structured data: SQLite (baseline-first applied to infra too).

## Risk rating

- **Not ML** — no ground truth exists for contract risk. No
  precision/recall/F1 ever claimed.
- Presence/absence lookup table, built manually. Most categories flag if
  ABSENT (protective clauses); some flag if PRESENT (one-sided clauses).
  Each entry carries a citable legal-consensus reason.
- Runs on the classifier's existing output — zero new inference calls.
- Validated via face validity (Method 1) + manual spot-check calibration
  (Method 3). Built in Phase 10.

## Phase list

0. Repo & environment scaffolding
1. Data acquisition + exploration
2. Segmentation engine
3. Eval harness skeleton
4. Classical classifier baseline
5. Classifier feature improvements
6. RAG retrieval path
7. RAG generation + inference backend
8. Classifier-to-RAG hard filter integration
9. MLOps wrap-up
10. Risk rating heuristic
11. API layer (FastAPI)
12. Frontend (Next.js)

Each phase ends with a commit + push the same day it finishes.