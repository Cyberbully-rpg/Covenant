# Covenant — Technical Requirements Document (TRD)

**Companion documents:** `PRD.md` (product framing, scope, audience), `ARCHITECTURE.md` (system diagrams, data flow, phase roadmap).
**Purpose:** This document captures every **locked technical decision**, the reasoning behind it, and the explicit alternatives that were considered and rejected. Treat every decision below as stable unless a future entry explicitly says it was reopened.

---

## 1. Data Layer

### 1.1 Dataset
- **Source:** CUAD v1, canonical URL `https://github.com/TheAtticusProject/cuad/raw/main/data.zip` — **must use `main` branch, not `master`** (the `master` branch raw JSON path returns 404).
- **Structure after unzip:** `CUADv1.json` → top-level key `data` → 510 entries → `paragraphs[0]['context']` holds the full contract text.
- **Scale:** 510 expert-labeled contracts, 41 clause categories.
- **Native task:** span-extraction QA (question, contract, answer-span triples).
- **Reformulation:** multi-label classification at the segment level. This reformulation is why segmentation quality is a first-class architectural concern — the classification unit doesn't exist natively in CUAD and must be engineered.

### 1.2 Data Versioning
- **Tool:** DVC, initialized from Phase 0 (git-first, not retrofitted).
- **Known environment issue:** `pathspec` must be pinned `<1.0.0` for compatibility.
- **Known environment issue:** Chroma's persisted DB and DVC together caused file-lock conflicts when the repo lived under OneDrive Desktop — **repo was relocated to `C:\Dev\Covenant\`** to avoid cloud-sync file locking.

---

## 2. Segmentation Engine (Phase 2 — design locked, implementation pending)

### 2.1 Purpose
Serves **two consumers from one pipeline**: classifier training data (segment = classification unit) and RAG chunking (segment = retrieval unit). This dual-use requirement is why segmentation design was treated as a first-class phase rather than a preprocessing afterthought.

### 2.2 Cascading Pattern-Detection Logic (locked order)
1. **Boilerplate stripping** — SEC EDGAR footer/header noise removal. Folded into the segmenter as its **first internal step**, not a separate module.
2. **ARTICLE headers** (top-level cut)
3. **Section N.N numbering** (top-level cut)
4. **Bare numbering** (top-level cut)
5. **Fallback** (top-level cut, when no reliable numbering scheme is detected)

- Numbering creates new segments; blank lines alone do **not** create new segments.
- Hierarchy handling (parent-child relationships) applies **only** at oversized-segment sub-splitting time, not during initial cascade/top-level cut detection.

### 2.3 Thresholds (empirically grounded, not arbitrary)
Measured across all 510 CUAD contracts:
- **Pattern confirmation threshold:** 3+ pattern hits required to confirm a genuine numbering scheme (prevents false-positive cascade selection from incidental number-like text).
- **Oversized segment cutoff:** ~4000–5000 characters (~p75–p80 of measured segment length distribution) — triggers sub-splitting with parent-child metadata.
- **Undersized floor:** ~150–200 characters — triggers context injection (parent section title prepended) at embed time; **raw text is preserved separately for classifier labels** (the context-injected version is embed-only, not label-training data).

### 2.4 TOC Disambiguation
- Tables of contents frequently reuse the same numbering scheme as real section headers in long contracts — this is a known false-positive risk for the cascade.
- **Disambiguation rule:** TOC lines are reliably distinguished from real headers by the **absence of prose between consecutive header-pattern matches** (a real header is followed by clause text; a TOC entry is followed immediately by another TOC entry).
- This must be explicitly tested against real CUAD long-contract examples before Phase 2 is considered validated.

### 2.5 Output Schema
- Deliberately **sketched loosely for now** — locked only once Phase 3's eval harness reveals the exact fields required for retrieval-correctness scoring and classifier training. Avoids premature schema commitment before the consumer requirements (harness) are known.

---

## 3. Eval Harness (Phase 3 — built before classifier or RAG exist)

### 3.1 Why Eval-First
Ordering B was chosen explicitly over building the classifier/RAG first: every component built after Phase 3 is measured against a real yardstick from day one rather than having evaluation retrofitted.

### 3.2 Eval Set Construction
- Derived from CUAD's **native (question, contract, answer-span) triples** — not hand-written questions. This keeps the eval set grounded in expert-authored ground truth rather than the builder's own assumptions about what a good question looks like.
- **Scope claim (locked, must be stated explicitly in any reporting):** "validated against CUAD's contract distribution" — explicitly **not** "legal documents in general." This is a deliberate generalization-scope guardrail against overclaiming.

### 3.3 Two Decomposed Signals (never conflated)
1. **Retrieval correctness** — mechanical character-span overlap between the retrieved chunk and CUAD's labeled answer span. No LLM judge involved; fully non-gameable.
2. **Faithfulness** — LLM-as-judge, cloud-only, offline-only, run as a separate call after generation, **never the same model that generated the answer**.

**Rationale for splitting:** allows attribution of a bad answer to retrieval failure vs. generation/grounding failure, so fixes target the correct layer of the pipeline instead of guessing.

### 3.4 Judge Score Caveats (must be documented, not hidden)
- Judge score is a **trend indicator only**, known to bias toward fluent/longer answers.
- Always reported paired with the non-gameable mechanical retrieval-correctness metric — never presented alone as ground truth.

### 3.5 Logging Requirements
- Every harness log row records **backend identity** alongside prompt, retrieved chunks, and answer.
- This is required so that a regression can be traced to a backend swap (e.g. Ollama model version change) vs. a genuine retrieval/chunking change — without this, regressions are undiagnosable.

### 3.6 Build Sequencing Note
- The harness is built **after** the chunking strategy from Phase 2 is roughly settled (not before). Building a harness against a chunking approach likely to be discarded wastes effort — but the harness itself still precedes the classifier and RAG builds.

---

## 4. Classifier (Phase 4–5)

### 4.1 Scope
- Full 41-category multi-label task — the real CUAD benchmark, not a simplified subset.
- Severe class imbalance expected and must be reported honestly: weighted loss, per-category precision/recall/F1, macro-F1 + micro-F1. **A single blended accuracy number is never acceptable reporting for this task.**

### 4.2 Model Approach (baseline-first, locked)
- **Baseline:** TF-IDF + weighted logistic regression / XGBoost, per category.
- **Transformer fine-tuning is explicitly shelved.** Reopened only if the classical ceiling is proven insufficient with evidence — not preemptively.

### 4.3 Escalation Ladder (locked priority order — classical-only levers)
1. **N-grams** (bigrams/trigrams) instead of unigrams in TF-IDF — highest priority; directly attacks "similar words, different meaning" confusion between categories.
2. **Structural features** from the segmenter (section position, header type, clause length, trigger phrases) fed alongside TF-IDF — cheap, since the segmenter already extracts these as a byproduct.
3. **Per-category decision threshold tuning** (not a fixed 0.5) — especially important for rare/imbalanced categories; cheap, no retraining required.
4. **Imbalance handling beyond weighted loss** (e.g. SMOTE) for genuinely rare categories. **Caveat, must be actively watched for:** oversampling on TF-IDF vectors can generate synthetic vectors that don't correspond to any real sentence — an artifact risk, not a free win.
5. **Ensembling logreg + XGBoost** per category — last classical lever before transformer escalation would even be considered.

### 4.4 Zero-Shot LLM Baseline (diagnostic role only)
- Used as a **diagnostic tool, not a build gate.**
- If the zero-shot LLM also fails on the same examples the classical model fails on, that is evidence of **label ambiguity/noise in CUAD itself** — not a model architecture gap. Engineering effort should **not** be spent chasing those categories further.

### 4.5 Key Conceptual Distinction (governs Phase 2 vs Phase 4/5 effort allocation)
- **Chunking/segmentation quality sets the ceiling** on classifier performance — bad chunking corrupts input before any model ever sees it.
- **Feature representation quality** (n-grams, structural features) determines how close the classifier gets to that ceiling.
- These are **sequential, not interchangeable** levers — good chunking with poor representation still underperforms; the reverse is also true. This is why Phase 2 was treated as its own phase rather than folded into Phase 4.

---

## 5. RAG Architecture (Phase 6–8)

### 5.1 Default Retrieval Mode
- Full-document similarity search across the contract. Classifier filter is optional, lawyer-toggled, off by default.

### 5.2 Classifier-to-RAG Filter (locked — Interpretation A: hard pre-search filter)
- Classifier runs **at ingestion**, writes predicted categories as **metadata on each Chroma chunk** (multi-label).
- **Toggle OFF (default):** full-document search, no `where` clause.
- **Toggle ON:** lawyer selects one of 41 categories; retrieval adds `where={"category": selected}` **before** cosine similarity runs. The candidate pool is constrained **pre-ranking**, never post-ranking re-scored.
- **100% lawyer-driven** — never automatic, never inferred from the query text.

### 5.3 Rejected/Deferred Alternative: Soft Boost / Re-rank
- Documented as a **Phase 2 escalation candidate** (i.e., a future architectural revision, not current scope).
- **Trigger condition for reopening:** the hard-filter ablation (§5.4) shows evidence of failure — either precision@k gains don't materialize, or filtered-out false negatives (relevant chunks excluded by classifier mislabeling, especially on sparse/ambiguous categories) occur often enough to be a real problem.
- Not built speculatively — only evidence-gated.

### 5.4 Required Eval
- The eval harness **must run filter-on vs. filter-off comparisons** (not a single pass) to produce a precision@k delta as a concrete, defensible portfolio artifact (Phase 8).

### 5.5 Merge Semantics (locked — thin Interpretation B)
- `/ask` response includes `filter_applied` and `classifier_confidence_on_filter` directly in its JSON payload.
- These fields are populated **only when the filter toggle was actually ON** for that query; both are `null` when OFF.
- **Explicitly rejected:** an always-on "other relevant categories detected" field — this would create a standing dependency of RAG on classifier output existing for every query, which is architecturally undesirable. RAG must be able to run correctly with zero classifier involvement.

---

## 6. Inference Backends

### 6.1 Backend Set (locked)
- **Only two backends:** Ollama (local) and a cloud API. **vLLM is fully removed from consideration** — hardware has no CUDA/discrete GPU, so vLLM provides no benefit and adds deployment complexity for nothing.

### 6.2 Routing Rules (locked, code-path-determined — never user-facing)
| Call site | Backend |
|---|---|
| Interactive demo generation | Always Ollama |
| Batch eval runs | Ollama or cloud (per-run choice, not per-request toggle) |
| LLM-as-judge scoring | Always cloud, always offline, always a separate call after generation, **never the same model that generated the answer being judged** |

- Backend selection is **never a runtime/user-facing toggle** — it's determined by which code path (call site) is executing.
- **Three distinct `generate()` call sites** exist with different routing logic and must not be conflated in code or in logging: (1) generation, (2) judge, (3) — if ever built — risk severity scoring (Phase 10, shelved feature B2).

### 6.3 Accepted Tradeoff
- This design loses batched-throughput advantages available at larger scale. **Accepted** at portfolio eval-set scale; revisit only if the eval set size or iteration frequency grows significantly enough to make batching's throughput gains material.

---

## 7. Vector Database

- **Chosen:** Chroma, 1.x line (Rust-based core, prebuilt wheels — avoids requiring a Rust toolchain on the build machine). Zero-setup, pip-installable.
- **Deferred:** Qdrant — explicitly not adopted now. **Reopening condition:** evidence of a real filtering bottleneck under the current Chroma-based hard-filter design (§5.2). Not proactively adopted on the assumption that it might be needed.

---

## 8. Risk Rating Layer (Phase 10)

### 8.1 Ground Truth Status
- **No ground truth exists** for contract risk in CUAD or anywhere else usable here. Therefore **no precision/recall/F1 can ever be legitimately claimed** for this component. This must be stated explicitly in any documentation or portfolio presentation of this feature — it must never be presented with the same evidentiary status as the classifier.

### 8.2 Scope (locked — Interpretation A only)
- Clause **presence/absence** heuristics only.
- **Out of scope:** content-severity judgment (Interpretation B — "how bad is this specific clause's language") and cross-clause interaction risk (Interpretation C — "does clause X interacting with clause Y create risk").

### 8.3 Mechanics (locked)
- A **predefined, manually-built lookup table** (not learned) maps each risk-relevant CUAD category to a flag **direction**:
  - **Flag if ABSENT** — protective clauses (e.g. Limitation of Liability, Indemnification, Insurance) — their absence implies unbounded exposure.
  - **Flag if PRESENT** — one-sided clauses (e.g. Termination for Convenience, Uncapped Liability, Most Favored Nation) — their mere presence is the risk signal.
- Each table entry must carry a **citable legal-consensus reason**, decided upfront by the builder (not inferred or generated at runtime).
- **At inference time:** the classifier's existing per-category present/absent output (already produced upstream — **zero new inference calls**) is checked against the table; matches append to a `risk_flags` list.
- **Optional aggregate score:** count or weighted count of flags; weighting is manually decided and documented (not learned).
- **No clause text is read at this stage** — this is a pure lookup on classifier output, which is what keeps it cheap and explainable, but also what caps its scope.

### 8.4 Validation Approach (locked)
- **Method 1 — face validity:** each rule grounded in citable, standard contract-law guidance.
- **Method 3 — human spot-check calibration:** manual review of a contract sample against generated flags, with findings/adjustments documented.
- **Method 2 (internal consistency checks) explicitly dropped** — not part of the validation plan.

### 8.5 Shelved Escalation: Interpretation B2 (LLM-assisted severity scoring)
- **Status:** shelved, evidence-gated — not built until there's evidence the presence/absence layer is insufficient.
- **If ever revisited, constraints already locked in advance:**
  - Requires a **third distinct LLM call site**, separate from generation and judge — offline, cloud-only, cached at ingestion (same treatment as the faithfulness judge; **never** the generating model).
  - Has **no mechanical validation check** available (unlike retrieval correctness) — this is a structural weakness of the approach, acknowledged upfront.
  - **Loses most Method 1 (face-validity) grounding**, since severity judgments vary by jurisdiction and industry — this is a known cost of escalating beyond presence/absence.
  - Must **layer on top of** the presence/absence rules, never replace them — severity scoring would only ever apply to clauses already confirmed present by the existing rule set.
  - Scheduled, if built at all, for **Phase 10**, after classifier and RAG are already validated — positioned explicitly as a defensible bonus feature, not a fourth ML pillar of the project.

---

## 9. MLOps

### 9.1 Classifier Side
- **MLflow** — experiment tracking, starting Phase 4.
- **DVC** — data versioning, initialized Phase 0, formalized across all datasets in Phase 9.
- **Drift monitoring** — input distribution vs. training data distribution.

### 9.2 RAG Side
- Fixed eval set re-run on **every pipeline change** (not periodic — every change).
- Query log: retrieved chunks + generated answer + faithfulness score, per query.
- Embedding drift monitoring on incoming documents.
- **Explicitly diagnostic, not curative** — the instrumentation makes hallucination and drift *measurable*; it does not fix them. Fixes are judged against this evidence, not against intuition.

### 9.3 Structured/Relational Data Store
- **SQLite**, chosen over Postgres.
- **Rationale:** baseline-first discipline applied to infrastructure decisions, not just model decisions — there is no evidence yet that concurrent writes or complex joins are required for a solo portfolio project. Escalation to Postgres would need the same evidence-gating discipline as any other locked decision here.

---

## 10. Repository & Version Control

- **Structure:** monorepo (backend + frontend + infra together).
- **Git:** initialized **before any code exists** (Phase 0), not retrofitted after the ML core works.
- **DVC hooks into git from the start**, not added later.
- **`.gitignore`**, configured in the **first commit**, excludes: model artifacts, `chroma_db/` persisted data, `.env` secrets, MLflow run logs.
- **Commit cadence (locked):** at minimum, `git add` / `git commit` / `git push` at the end of every phase, same day the phase finishes. Local and remote must never diverge by more than one phase.
- **Force-push:** has been used once already, to resolve a diverged remote — noted here as a real event, not a hypothetical.

### 10.1 Environment Specifics (current, real)
- **Repo path:** `C:\Dev\Covenant\` (moved from OneDrive Desktop specifically to avoid DVC/Chroma file-lock conflicts under cloud sync).
- **Python:** 3.12.10 (deliberately not 3.14, due to ML wheel compatibility gaps at time of setup).
- **Virtual environment:** `venv`, created at `C:\Dev\Covenant\` with an explicit Python path specified at creation time (not relying on whatever `python` resolves to on PATH).
- **IDE:** Cursor, with **agent mode (autonomous file generation) deliberately disabled** — prior unwanted file creation from Agent ∞ mode is the documented reason.

---

## 11. Multi-Agent Workflow (process/tooling requirement)

- **Claude** — architect role: design decisions, this document set, phase planning.
- **Gemini** — debugger role.
- **Grok** — tester role.
- **`PROJECT_LEDGER.md`** — shared, living ground-truth handoff document used to keep all three agents synchronized on actual project state (what's built, what's locked, what's pending) without relying on any single agent's conversation memory.

This TRD, the PRD, and `ARCHITECTURE.md` are designed to be **pasted wholesale into a new session with any of these agents (or any other model)** to fully reconstruct context — this is why every decision here includes its rationale and rejected alternatives, not just the conclusion.