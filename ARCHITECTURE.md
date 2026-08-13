# Architecture

Locked design decisions for Covenant. This document is the source of
truth — decisions here are not re-litigated without explicitly reopening
them.

## Data & Classifier Scope

- **Dataset**: CUAD, 510 contracts, expert-labeled. Native task is
  span-extraction QA, reformulated here as multi-label classification.
- **Classifier scope**: full 41-category multi-label task, not a subset.
  Severe class imbalance expected — weighted loss, per-category
  precision/recall/F1, macro-F1 + micro-F1 reported. Never a single blended
  accuracy number.
- **Model approach**: baseline-first. TF-IDF + weighted logistic regression
  / XGBoost per category, before any transformer fine-tune.

### Classifier escalation policy

Fully classical for now. No transformer fine-tune unless the classical
ceiling is proven insufficient with evidence. Improvement levers, in
priority order:

1. N-grams (bigrams/trigrams) in TF-IDF instead of unigrams — attacks
   same-vocabulary/different-meaning category confusion.
2. Structural features from the segmenter (section position, header type,
   clause length, trigger phrases) fed alongside TF-IDF.
3. Per-category decision threshold tuning (not fixed 0.5) — no retraining
   needed.
4. Imbalance handling beyond weighted loss (e.g. SMOTE) — watch for
   synthetic-vector artifacts on TF-IDF.
5. Ensembling logreg + XGBoost per category — last classical lever.

Zero-shot LLM baseline is a diagnostic, not a build gate: if it fails on the
same examples the classical model fails on, that signals CUAD label
ambiguity, not an architecture gap.

**Key distinction**: chunking/segmentation quality sets the ceiling on
classifier performance; feature representation quality determines how close
the classifier gets to that ceiling. Sequential, not interchangeable.

## Segmentation

Cascading pattern-detection: ARTICLE headers → Section N.N → bare numbering
→ fallback. Numbering creates new segments; blank lines alone do not.
Oversized segments sub-split with parent-child metadata. Undersized
fragments get parent section title prepended at embed time only; raw text
preserved unchanged for classifier labels.

Serves two consumers from one pipeline: classifier training data and RAG
chunking.

## RAG Architecture

- **Default**: full-document similarity search, no filter.
- **Classifier-to-RAG filter (Interpretation A — hard filter)**: classifier
  runs at ingestion, writes predicted categories as metadata on each Chroma
  chunk (multi-label). Toggle OFF (default) = full-document search, no
  `where` clause. Toggle ON = lawyer selects one category, retrieval adds
  `where={"category": selected}` before cosine similarity runs. 100%
  lawyer-driven, never automatic.
  - Soft boost / re-rank is a documented Phase 2 escalation candidate,
    triggered only if hard-filter ablation shows evidence of failure.
  - Eval harness runs filter-on vs. filter-off comparisons to produce a
    precision@k delta as a portfolio artifact.
- **Merge semantics (thin Interpretation B)**: `/ask` response includes
  `filter_applied` and `classifier_confidence_on_filter` fields, populated
  only when the filter toggle was ON for that query; both null when OFF. No
  always-on "other relevant categories detected" field — RAG has no standing
  dependency on classifier output.

## Vector DB & Inference

- **Vector DB**: Chroma. Qdrant deferred pending evidence of a real
  filtering bottleneck.
- **Inference backends**: Ollama (local) and cloud API only. vLLM removed
  from consideration (CUDA-dependent, not viable on integrated graphics).
  - Interactive demo → always Ollama.
  - Batch eval → Ollama or cloud.
  - LLM-as-judge → always cloud, always offline, never the generating model.
  - Backend selection is code-path-determined, never user-facing.

## Eval Harness

Built in Phase 3, before the classifier or RAG exist.

- Eval set derived from CUAD's native (question, contract, answer-span)
  triples — not hand-written questions.
- Scope claim: "validated against CUAD's contract distribution," not
  "legal documents in general."
- Two signals measured separately:
  1. **Retrieval correctness** — mechanical character-span overlap between
     retrieved chunk and CUAD's labeled answer span. No judge needed.
  2. **Faithfulness** — LLM-as-judge (cloud, offline) scores whether the
     generated answer is entailed by retrieved chunks. Trend indicator, not
     ground truth — always paired with the mechanical metric.
- Every log row records backend identity alongside prompt, retrieved
  chunks, and answer — enables tracing regressions to backend swap vs.
  retrieval/chunking change.

## MLOps

- **Classifier side**: MLflow + DVC + drift monitoring (input distribution
  vs. training data).
- **RAG side**: fixed eval set run on every pipeline change, query logging
  (chunks + answer + faithfulness score), embedding drift monitoring.
  Diagnostic, not curative.
- **Structured data**: SQLite. Chosen over Postgres per baseline-first
  discipline — no evidence yet that concurrent writes or complex joins are
  needed at solo-portfolio scale.

## Risk Rating (Phase 10)

- No ground truth exists for contract risk — never presented with
  precision/recall/F1 like the classifier.
- **Scope**: clause presence/absence heuristics only. Content-severity
  judgment and cross-clause interaction risk are out of scope.
- **Mechanics**: predefined lookup table, built manually, maps each
  risk-relevant CUAD category to a flag direction:
  - "Flag if ABSENT" — protective clauses (Limitation of Liability,
    Indemnification, Insurance) where missing means unbounded exposure.
  - "Flag if PRESENT" — one-sided clauses (Termination for Convenience,
    Uncapped Liability, Most Favored Nation) where existing at all is the
    risk signal.
  - Each entry carries a citable legal-consensus reason.
  - At inference: classifier's existing per-category output is checked
    against the table — zero new inference calls.
- **Validation**: Method 1 (face validity via legal-consensus grounding) +
  Method 3 (human spot-check calibration). Method 2 (internal consistency)
  dropped.
- **Shelved**: LLM-assisted severity scoring (Interpretation B2) — would
  need a third distinct LLM call site, has no mechanical validation check,
  loses Method 1 grounding since severity varies by jurisdiction. Layers on
  top of presence/absence, never replaces it. Not built until there's
  evidence the presence/absence layer is insufficient.

## Repo & Version Control

- Monorepo (backend + frontend + infra together).
- Git from day one, initialized before any code exists.
- `.gitignore` configured in the first commit.
- Commit + push at the end of every phase, minimum.

## Build Order

ML-core-first. Classifier + RAG + eval harness validated via API/CLI before
the Next.js frontend is touched. Frontend is the last layer.

**Sequencing: eval-first.** The eval harness skeleton is built in Phase 3,
before the classifier or RAG exist, so every component is measured against
a real yardstick from the start.

## Phase List

| Phase | Contents |
|---|---|
| 0 | Repo & environment scaffolding |
| 1 | Data acquisition + exploration |
| 2 | Segmentation engine |
| 3 | Eval harness skeleton |
| 4 | Classical classifier baseline |
| 5 | Classifier feature improvements (evidence-gated) |
| 6 | RAG retrieval path |
| 7 | RAG generation + inference backend |
| 8 | Classifier-to-RAG hard filter integration |
| 9 | MLOps wrap-up |
| 10 | Risk rating heuristic |
| 11 | API layer (FastAPI) |
| 12 | Frontend (Next.js) |

Each phase ends with a commit + push.
