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
| **2** | Segmentation engine | ✅ **Complete** | Cascading pattern-detection segmenter, 16 tests, validated against real CUAD contracts. Generalization gap found and fixed during Phase 3 (see §6) |
| **3** | Eval harness skeleton | 🔶 **Steps 1–2 complete; step 3 deferred to Phase 7 (current phase)** | Eval set from CUAD's native QA triples + retrieval-correctness metric + baselines ✅. Faithfulness judge & logging schema → Phase 7 |
| **4** | Classical classifier baseline | ✅ **Complete** | Training-set construction from gold spans + contract-level split + TF-IDF/weighted-logreg baseline over all 41 categories, MLflow tracking live (see §6B) |
| **5** | Classifier feature improvements | 🔶 **Steps 1–3 complete + diagnostic + richer-features follow-up run; steps 4–5 explicitly deferred** | N-grams + structural features + tuned thresholds + category-similarity feature all adopted, macro-F1 0.430 → 0.503. Diagnostic (3 backends) + trigrams/similarity follow-up both point to a representation ceiling, not label noise or data quantity — see §6C |
| **6** | RAG retrieval path | ✅ **Complete** | Chroma ingestion (510 contracts → 45,254 windowed records), hybrid RRF retrieval + leading-segment prior, validated against the harness before generation added. Dense-vs-TF-IDF gate measured, a truncation confound in the first measurement found and fixed, segmentation ceiling established at 0.9985. **0.8351 hit_rate@5 on 102 held-out contracts**, vs 0.6934 for the Phase 3 baseline (see §6D) |
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
| hit_rate@5 | 0.2769 | **0.6934** |
| gold_recall | 0.1532 | 0.5848 |
| gold_density | 0.0097 | 0.0338 |
| mean_retrieved_chars | 8,481 | 9,311 |

**TF-IDF lift: +0.416** at comparable retrieval volume, so the gain is genuine retrieval quality rather than chunk-size inflation.

**Re-measured after the Phase 4 preamble fix (§6B.1).** The originally reported figures were random 0.2226 / tfidf 0.5768 at 7,401 / 8,741 mean chars. Making the pre-first-header preamble reachable lifted tfidf hit_rate@5 by **+0.117** while `gold_density` held flat (0.0332 → 0.0338) against a 6.5% rise in retrieved chars — density holding as volume rises is what separates a real gain from chunk-size inflation (§6A.2). The same segmentation defect was therefore suppressing *both* retrieval and classification; it was found from the classifier side only because Phase 4 measures capture rate explicitly.

**Phase 6 decision input (the reason these baselines exist):** plain lexical TF-IDF already reaches **0.693** hit_rate@5. Dense embedding retrieval must beat that by a worthwhile margin to justify Chroma + an embedding model + an ingestion pipeline. That bar is now measured, not assumed — and it is 0.693, not the 0.577 recorded before the preamble fix.

---

## 6B. Phase 4 (Classical Classifier Baseline) — Complete

Built in `backend/classifier/`. Three pieces: training-set construction, the contract-level split, and the baseline model.

### 6B.1 Label construction (`data/build_training_data.py`)

A segment carries label C if its character span **overlaps any** gold answer span for C in that contract. This reuses Phase 3's `spans_overlap` primitive rather than reimplementing it, so "what counts as a hit" is defined once across eval and training.

Full corpus, 510 contracts:

| Metric | Value |
|---|---|
| Segments | 20,874 |
| With ≥1 label | 5,732 (27.5%) |
| Unlabeled (negatives) | 15,142 (72.5%) |
| Gold spans | 13,823 |
| **Captured by some segment** | **13,801 (99.84%)** |

**Gold-span capture rate is the segmentation ceiling**, measured before any model is trained: a gold span no segment overlaps is a label the classifier can never learn and retrieval can never hit. At 99.84%, Phase 2's segmentation is not the binding constraint on Phase 4/5 numbers.

**Segmenter fix this exposed:** `cut_top_level` was discarding everything before the first header. That region is the title block, parties recital and execution date — the only place Document Name / Parties / Agreement Date / Effective Date ever appear. It is now emitted as a `[preamble]` segment. Measured corpus-wide over all 13,823 gold spans, **capture rate 79.45% → 99.84%** — one-fifth of all labels in the dataset were unreachable. Same pattern as §6.1: a defect all unit tests passed, caught by measuring against gold. It was suppressing retrieval too — Phase 3's baselines were re-run after the fix and moved materially (§6A.3).

**Class imbalance, reported not buried** (per TRD §4.1) — positives per category: min 22 (`Unlimited/All-You-Can-Eat-License`), median 200, max 653 (`License Grant`). 6/41 categories have <50 positive segments and 10/41 have <100, corpus-wide, before any split. Those categories cannot produce meaningful F1 regardless of model, and that is a data finding, not a model failure.

### 6B.2 Split discipline (`data/split.py`)

**By contract, never by segment**, deterministic given (contract ids, seed) and independent of row order. Segments within one contract share vocabulary, defined terms, party names and near-verbatim boilerplate; a segment-level split puts near-duplicates on both sides and inflates every downstream number *silently*. 7 tests, the load-bearing one asserting contract-disjointness. 408 train / 102 test contracts at seed 42.

### 6B.3 Baseline model (`models/train_baseline.py`)

TF-IDF unigram (`min_df=3`, `max_df=0.9`, sublinear tf) → one-vs-rest `LogisticRegression(class_weight="balanced")`, one binary model per category. 16,089 train / 4,785 test segments, 12,384 features, fits in ~12s.

| | Precision | Recall | F1 |
|---|---|---|---|
| **macro** | 0.338 | 0.632 | **0.430** |
| **micro** | 0.363 | 0.742 | **0.488** |

**No blended accuracy is computed anywhere in this path** — at ~2% positive rate per category an all-negative predictor scores >97% "accurate". A test asserts the results dict contains no `accuracy` key.

Range across the 41 categories: `No-Solicit Of Employees` 0.833 and `Governing Law` 0.818 at the top; `Most Favored Nation` and `Price Restrictions` at 0.000 (29 and 26 train positives respectively). 2/41 categories sit at F1 = 0.000.

**The shape of the failure, which sets Phase 5's priority:** recall is roughly 2× precision almost everywhere — `class_weight="balanced"` buys recall by over-predicting. This is a *threshold* problem before it is a representation problem, which is direct evidence for escalation-ladder step 3 (per-category threshold tuning, no retraining needed) alongside step 1 (n-grams). Deliberately **not** folded into this run: each ladder step needs its own MLflow run to count as evidence.

Unigrams are used deliberately as the weakest reasonable representation — this is the floor n-grams have to beat.

### 6B.4 MLflow

Tracking starts here, local file store (`file:./mlruns`, git-ignored), experiment `covenant-classifier`. Params, headline metrics, and all 123 per-category metrics logged per run.

---

## 6C. Phase 5 (Classifier Feature Improvements) — Complete

Built in `backend/classifier/`: `data/split.py` (three-way split), `features/structural.py` (structural feature block), `models/train_experiment.py` (the ladder runner). Runs TRD §4.3's locked priority order as **separate, individually-measured experiments** — no step is folded into another, and each is judged on its own delta.

### 6C.1 Evidence-gating discipline (the point of this phase)

Each lever is applied **on top of the current champion**, not the previous step, and kept only if it beats that champion's macro-F1. A lever that loses is discarded, not silently retained because it was next on the roadmap — a rejected lever must not keep influencing later steps. `test_train_experiment.py` asserts this directly: it forces a lever to lose and checks the next lever's candidate config shows no trace of it.

A three-way **train/val/test** split (`split.py`) replaces Phase 4's two-way split — threshold tuning needs data the model never trained on and that isn't the test set, or the reported F1 would just be measuring what the thresholds were tuned to produce. The test set is carved out with the identical call Phase 4 used at the same seed, so Phase 5's test-side numbers are directly comparable to Phase 4's, not confounded by a different held-out set.

**Control run, not Phase 4's printed figure, is the baseline for every delta.** Because validation is carved out of the training pool, every ladder step trains on fewer contracts than Phase 4 did (326 vs. 408). Step 0 re-runs Phase 4's exact recipe on Phase 5's smaller pool so each lever's delta is attributable to the lever alone, not to a shrunken training set.

### 6C.2 Results (510 contracts, seed 42: 326 train / 82 val / 102 test)

| step | macro-F1 | micro-F1 | Δ vs champion | verdict |
|---|---|---|---|---|
| control (Phase 4 recipe, Phase 5 data) | 0.4315 | 0.5000 | — | adopted |
| + bigrams | 0.4722 | 0.5645 | +0.0407 | **adopted** |
| + structural features | 0.4760 | 0.5822 | +0.0037 | **adopted** |
| + tuned thresholds | **0.5009** | **0.5995** | +0.0249 | **adopted** |

**All three levers won.** Champion: bigrams + structural features + tuned thresholds. macro-F1 **0.430 → 0.501** (+0.069), macro-P 0.348 → 0.538, macro-R 0.610 → 0.494 — precision and recall converged rather than recall dominating, confirming Phase 4's diagnosis that the gap was mostly a threshold problem.

**Seed stability, since step 2's gain is small:** re-run at seeds 7 and 123. Bigrams and threshold tuning won by a wide margin at every seed; structural features won at every seed too, by a small but never-negative margin (+0.0037 / +0.0001 / +0.0013). Small, but real and consistent — not adopted on a single lucky split.

Top categories: `Parties` 0.957, `Governing Law` 0.897, `Document Name` 0.883. Three categories sit at F1 = 0.000 (`Competitive Restriction Exception`, `Price Restrictions`, `Third Party Beneficiary`); `Price Restrictions` has **zero test-side positives** (18 train / 0 test), making its F1 = 0.000 undefined rather than a real failure — worth separating from genuine misses when this is reported. All 41 categories remain trainable (no untrainable categories, unlike some early smoke runs on small `--limit` samples).

### 6C.3 Structural features (`features/structural.py`)

14 columns: relative position in document, is-first/is-last, is-preamble, log-scaled char length, oversized/undersized/has-parent flags, header token count, header-has-digit, and a one-hot over the segmenter's numbering schemes. All derived from fields the segmenter already emits — `build_training_data.py` (§6B.1) now carries them on every row specifically so they don't require re-segmenting all 510 contracts to recover later.

Min-max scaled to `[0, 1]` using statistics fit on **train only**, then clipped on transform — an unscaled `char_len` of 4,500 would otherwise dominate the L2-normalized TF-IDF columns it's stacked next to, and fitting the scaler on all rows would leak test distribution into training the same way an unscaled feature would leak magnitude.

### 6C.4 Steps 4–5 (SMOTE, ensembling) — deliberately not run

TRD §4.3 step 4 (imbalance handling beyond weighted loss) and step 5 (logreg+XGBoost ensembling) are the most expensive levers on the ladder and are gated on evidence that the remaining gap is a capacity/imbalance problem rather than a representation or threshold problem. Steps 1–3 already moved macro-F1 +0.069, and the categories still failing are almost all low-support-by-data (`Price Restrictions` 18 train examples, `Most Favored Nation` 21, `Third Party Beneficiary` 29) rather than showing a precision/recall split that SMOTE or ensembling would fix — `class_weight="balanced"` (already in use) is the standard first lever for exactly this kind of imbalance, and it's already applied.

### 6C.5 Zero-shot LLM diagnostic (TRD §4.4) and the backend comparison

**Cloud backend changed: Anthropic → Gemini + Groq.** `requirements.txt` no longer pins `anthropic`; `google-genai==2.18.1` (model `gemini-3.6-flash`) and `groq==1.6.0` (model `qwen/qwen3.6-27b`) are the two cloud paths tested. Deliberate, explicit override of TRD's original `anthropic` choice — made when this diagnostic needed a working cloud key — not a silent re-litigation; TRD/CLAUDE.md never named a vendor explicitly, only "a cloud API," so no locked text required correction. `httpx==0.28.1` is pinned explicitly: `ollama`'s SDK needed bumping to 0.6.2 to accept it (0.4.4 pinned `httpx<0.28`, `google-genai` requires `>=0.28.1`).

**Local backend installed:** Ollama (`llama3.2:3b`, 2GB) via `winget install Ollama.Ollama`. None of the three backends existed in the environment this phase was otherwise built in — all three were set up specifically to run this diagnostic and to let the project owner compare them for future call sites (TRD §5.5: batch eval may use Ollama OR cloud, per-run config — any of the three is architecturally valid, this measures which is actually usable).

`backend/classifier/models/zero_shot_diagnostic.py` reuses CUAD's own per-category question (from `eval_set.jsonl`, Phase 3) rather than writing a new prompt — same "define it once" discipline as Phase 4 reusing `spans_overlap`. Ran all three backends over 16 balanced examples (8 positive / 8 negative) per each of the 3 categories the Phase 5 champion scores F1 = 0.000 on.

| backend | model | scored / sent | accuracy | errors | mean latency |
|---|---|---|---|---|---|
| Ollama | `llama3.2:3b` (local, 2GB) | 48 / 48 | 0.771 | 0 | 3.14s |
| Gemini | `gemini-3.6-flash` | 10 / 48 | 1.000* (n=10, not meaningful) | 38 (free-tier `429 RESOURCE_EXHAUSTED`) | 26.67s |
| **Groq** | **`qwen/qwen3.6-27b`** | **48 / 48** | **0.896** | **0** | **8.42s** |

**Groq note:** `qwen/qwen3.6-27b` is a reasoning model — by default it emits a `<think>...</think>` block before the answer, which the parser never sees (first run: 47/48 unparseable). Fixed by passing `reasoning_format="hidden"` to Groq's API, which strips the reasoning block server-side. A second Groq model, `openai/gpt-oss-120b`, was tried first and scored identically (0.896 accuracy, 48/48, 0 errors) at lower latency (3.28s) — kept `qwen/qwen3.6-27b` as the pinned model per project owner's explicit choice after confirming no Llama model is available on this Groq account (`llama-3.1-70b-versatile` and `llama-3.3-70b-versatile` are both decommissioned; Groq's current catalog is `openai/gpt-oss-*` and `qwen/qwen3.6-27b` as the general-purpose options).

**Backend verdict for this call site (batch, higher-volume): Groq.** It matched Ollama's reliability (48/48, zero errors) while beating its accuracy by 12.5 points (0.896 vs 0.771), and it's free with no rate-limit issue observed at this volume. Gemini's free tier could not complete a 48-call batch — it exhausted its quota after ~10 requests. Gemini's 1.000 accuracy is not meaningful evidence at n=10 with 38 uncounted failures. This doesn't indict Gemini's quality — a paid tier likely clears this — but the free tier is impractical for TRD §5.5's batch-eval-volume call site without paying. Gemini remains a fine fit for the low-volume, single-call-per-run faithfulness judge (Phase 7), where rate limits aren't a factor; Groq is now the stronger default for anything higher-volume, and Ollama remains the guaranteed-local, zero-network fallback the architecture already commits to for the interactive demo path.

**Diagnostic finding (the actual TRD §4.4 question — gap or noise?):** Groq per-category breakdown — `Competitive Restriction Exception` 16/16, `Third Party Beneficiary` 14/16, `Price Restrictions` 13/16. A zero-shot model clears ~90% on a balanced sample where the classical champion scores **F1 = 0.000**. That gap is real learnable signal a linear TF-IDF model isn't capturing — this reads as a **representation/capacity gap, not CUAD label noise**, and the finding is now corroborated by two independent models (Ollama 77%, Groq 90%) rather than one. (Caveat: this sample is small, class-balanced 50/50 unlike the real ~0.3–0.6% positive rate, and accuracy isn't F1 — not rigorous evidence, just enough to answer the diagnostic's one question per TRD §4.4's stated scope.)

**Consequence for steps 4–5:** the "low support, not fixable by SMOTE/ensembling" read in §6C.4 is now in tension with this finding — every backend tested finds signal these categories' classical features don't expose. Acted on in §6C.6 below.

### 6C.6 Richer-features follow-up (tried before steps 4–5; NOT on TRD's locked ladder)

Motivated directly by §6C.5's finding: a zero-shot LLM with **zero training examples** cleared ~80–90% on the categories the champion scores F1 = 0.000 on. If the champion's failure were a data-*quantity* problem, giving a model *less* data should make it worse, not better — so SMOTE/ensembling (both data-quantity levers) were set aside in favor of testing whether a *richer representation* could close the gap first. Same evidence-gating discipline as §6C.1: each candidate applied on top of the champion, kept only if it beats the champion's macro-F1.

**Candidate A — trigrams** (`ngram_range=(1,3)`, cheap sanity check): **macro-F1 0.5009 → 0.4777 (−0.023), rejected at all 3 seeds tested (42/7/123).** Feature explosion without added signal — extending the n-gram window doesn't help when the real problem is relational (see below), not vocabulary coverage.

**Candidate B — category-definition similarity** (`features/category_similarity.py`): one extra column per classifier — cosine similarity (via the shared TF-IDF space) between the clause and CUAD's own definition text for that category, e.g. *"is there a restriction on the ability of a party to raise or reduce prices."* Targets the relational gap directly, still classical (no transformer). **macro-F1 0.5009 → 0.5034 (+0.0025), consistently positive across all 3 seeds (+0.0025 / +0.0027 / +0.0057) — small but real, adopted.**

**Neither candidate cracked the 3 target categories at seed 42** (all three stayed at F1 = 0.000 with candidate B). But at other seeds, candidate B did occasionally move them — `Price Restrictions` 0.000 → 0.162 at seed 7, `Third Party Beneficiary` 0.000 → 0.200 at seed 123 — inconsistent because these categories have only 18–29 training examples, so which few examples land in train vs. test swings the result. This is itself informative: the signal candidate B adds is real (consistent small macro-F1 gain, occasional real per-category wins) but too weak on its own to reliably rescue categories this rare.

**Why a TF-IDF-based similarity feature is a partial fix, not the fix:** cosine similarity between two bag-of-words vectors is still just weighted word overlap between the clause and CUAD's question text — it doesn't capture the actual relational reasoning ("an exception carved out of a Non-Compete clause defined elsewhere in the document") that the zero-shot LLM's contextual understanding does. The natural non-speculative next step is **not** transformer fine-tuning (still shelved per TRD §4.2) — it's reusing the sentence-transformer embeddings Phase 6 is already building for RAG chunking as a classifier feature, once they exist. That's deferred to whenever Phase 6 lands, not run now.

**Decision on steps 4–5 (SMOTE/ensembling): still not run as ladder steps** — but step 4 (SMOTE) was tested directly against the similarity-feature follow-up on the project owner's request; see §6C.7.

### 6C.7 SMOTE, tested directly against the richer-features follow-up

Run as a scratch comparison, **not integrated or committed** — `imbalanced-learn` was pip-installed locally for this test only (not added to `requirements.txt`), and the test script itself was not added to the repo. Applied `SMOTE` per-category (oversampling that category's own training positives) on top of the champion's exact feature matrix (bigrams + structural features), for the 3 target categories only, at seed 42.

| category | real test positives | champion F1 | SMOTE F1 | similarity-feature F1 |
|---|---|---|---|---|
| `Competitive Restriction Exception` | 12 | 0.000 | **0.000** (0 predicted positive) | 0.000 at seed 42 |
| `Third Party Beneficiary` | 11 | 0.000 | **0.000** (0 predicted positive) | 0.000 at seed 42, **0.200 at seed 123** |
| `Price Restrictions` | **0** (vacuous at this seed) | 0.000 | 0.000 (undefined, not a real failure) | 0.000 at seed 42, **0.162 at seed 7** |

SMOTE manufactured 12,000+ synthetic training rows per category (real positives oversampled to match the negative class count) and still predicted **zero positives on every real test case** for both categories that actually had test-side positives to find. This is exactly the failure mode TRD §4.3 step 4's own caveat names: *"oversampling on TF-IDF vectors can generate synthetic vectors that don't correspond to any real sentence"* — the synthetic points apparently pulled the decision boundary in a direction that doesn't generalize to real clause text at all, worse than doing nothing.

**Verdict: the richer-features approach (§6C.6) clearly outperforms SMOTE.** The similarity feature never made things worse and occasionally produced a real per-category win; SMOTE never produced a single one across either real test case tried, despite manufacturing thousands of synthetic training examples. This closes the loop on TRD §4.3 step 4 for these categories with actual evidence rather than the earlier data-scarcity inference alone — **step 4 is now ruled out, not just deprioritized.**

---

## 6D. Phase 6 (RAG Retrieval Path) — Ingestion + retrieval built, harness-validated

Built in `backend/rag/` (`embedding.py`, `ingestion/ingest.py`, `retrieval/retrieve.py`) and `backend/eval/harness/run_dense_baseline.py`. Scope held to exactly what the roadmap names: Chroma ingestion, retrieval logic, and validation against the harness — **not** the classifier hard filter (Phase 8) and **not** generation (Phase 7). `ingestion/ingest.py` never imports `backend/classifier/` at all (asserted by a test, not just documented) — the concrete proof that RAG runs with zero classifier involvement, per TRD §5.5.

### 6D.1 Pipeline

- **Embedding model:** `all-MiniLM-L6-v2` (384-dim, 256 word-piece window), one shared wrapper (`embedding.py`) used by both ingestion and retrieval so they can never drift into different vector spaces. Baseline-first choice, same discipline as the classifier ladder — smallest reasonable model given no CUDA/discrete GPU (TRD §6.1), escalate only with evidence. **The model choice is locked in TRD §7.1**, which also records why it was missing from the doc set until after §6D.2 was measured.
- **Ingestion:** segments the RAG-path text (`embedding_text`, parent-context-injected — not raw `text`, which stays the classifier's uncontaminated path) via the Phase 2 segmenter, embeds, writes to Chroma (`hnsw:space: cosine`) with `contract_id` + span metadata. Full corpus: **510 contracts → 20,874 segments → 45,254 windowed records** (windowing per TRD §7.1; collections are namespaced per embedding model).
- **Retrieval:** `ChromaRetriever` queries scoped per contract (`where={"contract_id": ...}`) — matches TRD §5.1's "full-document search," never corpus-wide. Its `.retrieve()` method intentionally matches Phase 3's baseline retriever interface (`(start_char, end_char)` tuples) so it can run through the exact same harness for a fair comparison. Window records collapse back to distinct parent spans, best window first, so ranking is over segments and retrieved spans stay parent-sized.
- **Hybrid fusion:** `HybridRetriever` (`retrieval/hybrid.py`) fuses the TF-IDF and dense rankings by Reciprocal Rank Fusion (TRD §7.2). This is the default retrieval path into Phase 7.

### 6D.2 Harness validation — the actual Phase 6 decision, now measured

ARCHITECTURE.md §6A.3 framed this before any embedding code existed: *"Dense embedding retrieval must beat [TF-IDF's] 0.693 hit_rate@5 by a worthwhile margin to justify Chroma + an embedding model + an ingestion pipeline."* Full corpus, 510 contracts, same 6,702 scored rows as the TF-IDF baseline (apples-to-apples), k=5:

| | tfidf (§6A.3) | chroma_dense |
|---|---|---|
| hit_rate@5 | 0.6934 | **0.5818** |
| gold_recall | 0.5848 | 0.4836 |
| gold_density | 0.0338 | 0.0284 |
| mean_retrieved_chars | 9,311 | 9,164 |

**Dense retrieval lost, decisively (−0.1116 hit_rate@5), at comparable retrieval volume — not a chunk-size artifact.** This is the answer TRD's own framing said to watch for, and it came back negative. CUAD's questions are templated with exact domain vocabulary ("governing law," "audit rights," "change of control") that plays directly to TF-IDF's strength — literal keyword overlap — while a small general-purpose sentence embedding model has no legal-domain fine-tuning to close that gap.

**This does not mean Phase 6 failed** — it means the evidence-gate did its job before Phase 7 (generation) got built on an unvalidated assumption.

**⚠ This section's conclusion was subsequently found to be overstated. See §6D.3 — a silent truncation confound accounted for most of the −0.1116, and the numbers above compare a fully-read TF-IDF against a partially-read dense retriever.** The table is left standing as the record of what was measured and believed at the time; it is not the current result.

### 6D.3 Escalation — the truncation confound, and hybrid fusion

Two things were wrong with treating §6D.2 as final.

**1. The comparison wasn't apples-to-apples.** sentence-transformers silently truncates input past `max_seq_length` (256 word-pieces for MiniLM) instead of raising. Measured across the corpus, **39.9% of the 20,874 segments exceed that limit** — so for two of every five segments, dense retrieval was ranking on an opening fragment while TF-IDF read every word. `mean_retrieved_chars` was comparable, which correctly ruled out a chunk-size artifact on the *output* side, but nothing was checking the *input* side. Fix: windowed ingestion (TRD §7.1).

**2. "Which retriever wins" ≠ "what should ship."** Two rankers that fail on different queries fuse to something better than either. §6D.2 answered the first question and stopped.

Full corpus, 510 contracts, 6,702 scored rows, k=5 — same rows, same `scorer.score_query()`, same k as every number above. `tfidf` and `dense` were re-run fresh as controls and reproduced their prior values to the digit (`tfidf` = 0.6933751119068935, `dense` = 0.5818), so the harness is known not to have drifted under these results:

| variant | hit_rate@5 | vs tfidf | gold_recall | gold_density | mean_chars | chars vs tfidf |
|---|---|---|---|---|---|---|
| random (§6A.3) | 0.2769 | −0.4164 | 0.1532 | 0.0097 | 8,481 | −8.9% |
| tfidf (§6A.3, control) | 0.6934 | — | 0.5848 | 0.0338 | 9,311 | — |
| tfidf_bigram | 0.7050 | +0.0116 | 0.5985 | 0.0334 | 9,643 | +3.6% |
| dense (§6D.2, control) | 0.5818 | −0.1116 | 0.4836 | 0.0284 | 9,164 | −1.6% |
| dense_win | 0.6622 | −0.0312 | 0.5619 | 0.0275 | 10,988 | +18.0% |
| hybrid (RRF) | 0.7150 | +0.0216 | 0.6183 | 0.0317 | 10,482 | +12.6% |
| **hybrid_bigram** | **0.7195** | **+0.0261** | 0.6232 | 0.0315 | 10,654 | +14.4% |

**Removing truncation alone moved dense +0.0804 (0.5818 → 0.6622) — 72% of the deficit §6D.2 attributed to the model.** The stated explanation there (no legal-domain fine-tuning) was partly measuring a bug. **Dense still loses on its own** (−0.0312 vs TF-IDF), so §6D.2's direction survives; its magnitude and its reasoning do not.

**Hybrid RRF is the only configuration that beats the Phase 3 baseline**, and the best is `hybrid_bigram` at **0.7195 (+0.0261, +3.8% relative)**. Fusing a losing retriever with a winning one beating the winner alone is the concrete evidence that the dense half earned its build cost rather than being written off.

**Honest reading of that number (the anti-gaming pair, doing its job).** `hybrid_bigram` retrieves **14.4% more characters at 6.8% lower gold_density** (0.0315 vs 0.0338). Part of the gain is bought with retrieval volume, not purely better ranking — exactly what `gold_density` exists to expose, and it is not netted out of the headline. By the density standard, **`tfidf_bigram` is the cleanest gain in the table**: +0.0116 at essentially flat density (−1.2%) and +3.6% characters. Whether hybrid's trade is worth it is a Phase 7 question — feeding a generator ~14% more context to reach ~4% more answers is normally a good bargain, since context is cheap relative to a missed clause, but that is a judgment about the downstream task, not something this table settles.

**Two caveats that must travel with these numbers.**
- **Variant selection happened on the reported eval set.** All 510 contracts are in it, so choosing `hybrid_bigram` from this table is model selection on the same rows the number is quoted from, and the +0.0261 is therefore optimistic. The constants that could have been tuned were deliberately not (`RRF_C = 60` published default, fusion depth 20 fixed a priori, equal ranker weights — TRD §7.2), which limits but does not eliminate this. A held-out contract split for retrieval selection is the real fix, and is not yet built.
- **Cost:** hybrid runs ~193ms/query vs TF-IDF's ~10ms (Chroma returns 100 records per query at fusion depth 20). Fine for the interactive demo, material for batch eval.

**Phase 6 outcome:** default retrieval into Phase 7 is `hybrid_bigram` (TRD §7.2). Chroma is load-bearing, not vestigial — which also means Phase 8's metadata hard filter has a real index to filter.

### 6D.4 Where the remaining misses are, and the positional prior

§6D.3 left 28% of questions failing without saying why. Three measurements settled it.

**The segmenter is not the bottleneck.** `run_ceiling.py` asks how many gold spans overlap *no* segment at all — a hard ceiling on hit_rate, since a retriever can only return segments. **Ceiling: 0.9985.** Ten unreachable rows out of 6,702. Every remaining miss is a ranking failure, and "improve the segmenter" is therefore *not* a lever on retrieval accuracy, however strongly intuition suggests otherwise.

**The ranking is close, not lost.** `run_rank_diagnostic.py` reports the rank of the first gold-overlapping segment rather than just whether it made the top 5. Rank 1: 42.6%. Rank 2–5: 29.5%. Rank 6–20: 20.3%. Beyond 50: 1.8%. **The answer is in the top 20 for 92.3% of questions** — the ranker surfaces the right segment and misorders it, which is the tractable case. The k-curve says the same: 0.4260 @1, 0.7205 @5, 0.8409 @10, 0.9230 @20.

**28% of the eval set was structurally unreachable, not badly ranked.** Per-category results were bimodal — Insurance 0.952, Change Of Control 0.950, against Document Name 0.392, Parties 0.468, Agreement Date 0.474, Effective Date 0.613. Those four are 1,879 rows. Two follow-up measurements explain them:

| category | gold overlaps segment 0 | position in doc | query's words present in gold text |
|---|---|---|---|
| Document Name | 85.5% | 0.3% | 7.3% |
| Parties | 97.4% | 0.5% | — |
| Agreement Date | 87.4% | 0.7% | — |
| Effective Date | 76.9% | 0.9% | — |
| *(contrast)* Insurance | 2.4% | 58.1% | 97.6% |

The categories that score 0.95+ are the ones whose name is literally printed in the clause. The metadata categories' answers sit in the title block, where the query's own words appear ~7% of the time. Both rankers score on word overlap — literal for TF-IDF, semantic for the embedding model — so there was no signal to rank on at all. Not a tuning problem; a category of question whose answer is located structurally.

So it gets a structural answer: `retrieval/lead_prior.py` promotes the document's opening segment for metadata queries (TRD §7.3). The qualifying categories are **fitted on training contracts only**, via Phase 4's `split_contract_ids` — the diagnostic ran on the full corpus, so hardcoding the four names it surfaced would have been fitting to the reported set. Given only the train side, the fit independently recovers exactly those four.

**Held out — 102 unseen contracts, 1,377 scored rows** (`run_retrieval_variants.py --split test`):

| variant | hit_rate@5 | gold_recall | gold_density | mean_chars |
|---|---|---|---|---|
| tfidf | 0.6877 | 0.5786 | 0.0372 | 8,933 |
| tfidf_bigram | 0.7001 | 0.5905 | 0.0368 | 9,200 |
| hybrid_bigram | 0.6993 | 0.5873 | 0.0338 | 9,965 |
| tfidf_bigram_prior | 0.8264 | 0.6018 | 0.0345 | 10,018 |
| **hybrid_bigram_prior** | **0.8351** | 0.5974 | 0.0319 | 10,748 |

**+0.1358 out of sample**, roughly 11 standard errors, and the gain matches its predicted mechanism arithmetically: 28% of rows moving from ~0.48 to ~0.96 predicts +0.134. The prior is the identity function for the other 37 categories by construction (asserted by test), so nothing was traded away. `gold_density` falls slightly while `gold_recall` barely moves — the signature of short answers (a title, a party name), not of chunk-size gaming, but reported together per §3.3 regardless.

**Two findings this table forces into the open.**

- **The hybrid lead did not replicate.** Held out, `hybrid_bigram` (0.6993) does not separate from `tfidf_bigram` (0.7001); with the prior, 0.8351 vs 0.8264. Both gaps are inside one standard error. §6D.3's +0.0145 full-corpus lead was selection optimism — exactly the limit §6D.3 flagged, which turned out to be the whole effect. Hybrid is kept regardless, as an argued decision about the benchmark's bias rather than a measured win; see TRD §7.2 for the argument and its reopening condition.
- **Fuzzy/lexical tricks cannot reach the rest.** For the worst remaining categories the query's words are simply absent from the gold text — Volume Restriction 98.8%, Covenant Not To Sue 95.0%, Most Favored Nation 92.9%. Fuzzy matching finds near-spellings of words that exist; it cannot find words never written. Those need either semantics or **Phase 8's classifier filter**, which learns that a segment *is* a Volume Restriction clause without needing the words — quantitative evidence for Phase 8's value that the plan did not previously have.

**Untested levers, with measured expectations:** character n-grams as a third fusion ranker (~+1.0–1.5, bounded by 143 measured stem-only misses); cross-encoder reranking over the top 20 (ceiling 0.9230, ~1–2h per eval run on CPU); k=5→10 (+0.12, but that is showing more results, not better search, and must never be reported as the latter); TRD §7.1 ladder rungs 3–4.

### 6D.5 Tests

`test_embedding.py` (determinism, L2-normalization, similarity ordering), `test_ingest.py` (metadata correctness, the no-classifier-import boundary, idempotent reset, window coverage/overlap, parent-span collapse, per-model collection namespacing), `test_retrieve.py` (per-contract scoping — a query never returns another contract's spans — interface-shape compatibility with `scorer.score_query()`, window dedup, and batched-equals-unbatched), `test_hybrid.py` (RRF arithmetic against stubbed rankers: the agreement bonus, exact degradation to either single ranker at zero weight, depth limiting, k capping), `test_lead_prior.py` (the prior fires only for fitted categories and is otherwise the identity function, rejects categories that only sometimes lead or have too few rows to judge, and learns only from the rows it is handed). 43 tests in `backend/rag`, **105 tests total**.

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
- **Testing:** pytest, defaults (no `pytest.ini`/`pyproject.toml`). Test files sit beside the module they test. Current suite: **105 tests** (16 segmenter + 12 eval scorer + 11 split + 6 classifier baseline + 7 structural features + 6 ladder runner + 4 category-similarity + 43 RAG: 4 embedding + 12 ingestion + 7 retrieval + 7 hybrid + 13 lead prior)

### 7.1 Generated artifacts (not in git)

`data/processed/eval/` (`contracts.jsonl`, `eval_set.jsonl`, `metadata.json`, `baseline_results.json`) is git-ignored — regenerate with:

```powershell
python backend/eval/harness/build_eval_set.py          # rebuild eval set
python backend/eval/harness/run_baselines.py --limit 0 # re-run baselines (all 510)
```

`data/processed/classifier/` (`training_segments.jsonl`, `label_stats.json`),
`backend/classifier/models/artifacts/` (model + metrics) and `mlruns/` are likewise
git-ignored — regenerate with:

```powershell
python backend/classifier/data/build_training_data.py   # labels + capture rate
python backend/classifier/models/train_baseline.py      # train + report + MLflow
```

`chroma_db/` is likewise git-ignored. It holds one collection per embedding model — regenerate with:

```powershell
python backend/rag/ingestion/ingest.py --limit 0                              # 45,254 windowed records
python backend/eval/harness/run_retrieval_variants.py --variants all --limit 0 # re-measure §6D.3
```

Retrieval numbers should only ever be produced by the variant sweep, never by a one-off script: it
re-runs `tfidf` and `dense` as controls in the same pass, so harness drift surfaces immediately instead
of silently contaminating a comparison. To evaluate a different embedding model, set
`COVENANT_EMBED_MODEL` for **both** commands — collections are namespaced per model so two models'
vectors can never end up in one index.

DVC-tracking these is Phase 9 work (TRD §9.1), not yet done.

---

## 8. How to Use This Document Set With Another Model

Paste `PRD.md`, `TRD.md`, and this file (`ARCHITECTURE.md`) together into a new session. That gives any model:
- **PRD** → what the project is, who it's for, what's explicitly out of scope, why it's built the way it's built.
- **TRD** → every locked technical decision with rationale and rejected alternatives — enough to answer "why not X instead?" without re-deriving from scratch.
- **ARCHITECTURE** → how components connect, current build status, and exactly what phase is active right now.

Any agent picking this up should treat every decision marked "locked" as stable unless the user explicitly says they want to reopen it — this mirrors the working style already established with Claude.