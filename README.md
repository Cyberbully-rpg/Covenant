# Covenant

ML/RAG systems project for legal contract intelligence, built on CUAD
(Contract Understanding Atticus Dataset — 510 real contracts, labeled by
legal experts across 41 clause categories).

This is a solo skill-building and portfolio demonstration project — not a
product. The goal is to show production-grade ML engineering depth across
three pillars, roughly equally: classical ML, RAG/retrieval engineering,
and MLOps.

## What this is

- A **multi-label clause classifier** (TF-IDF + logistic regression /
  XGBoost, baseline-first, all 41 CUAD categories)
- A **RAG-based contract Q&A system** (Chroma + Ollama/cloud inference)
- A **heuristic risk-rating layer** (presence/absence lookup, no ML)
- **MLOps instrumentation** throughout (MLflow, DVC, drift monitoring)
- A thin **FastAPI** layer exposing `/classify`, `/ask`, `/eval`
- A **Next.js frontend** (last, thin layer over an already-working backend)

## Hardware constraints (accepted, documented)

AMD Ryzen 7 8840HS, integrated Radeon 780M, 32GB RAM — no discrete GPU,
no CUDA. This rules out vLLM entirely and keeps inference on Ollama
(local) or cloud API, never GPU-batched local serving.

## Status

Phase 0 — repo & environment scaffolding. See `ARCHITECTURE.md` for full
phase list and locked design decisions.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
dvc init
```

## Data

CUAD dataset — not included in this repo (DVC-tracked, git-ignored).
Source: [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad),
Hugging Face, or Kaggle. See arXiv 2103.06268 for the original benchmark
paper and eval metrics (Precision@N%Recall, AUPR, Jaccard).