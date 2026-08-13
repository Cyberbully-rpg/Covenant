# Covenant

ML/RAG systems project for legal contract intelligence, built on CUAD
(Contract Understanding Atticus Dataset — 510 real contracts, labeled by
legal experts across 41 clause categories).

Solo portfolio project for skill-building, not a commercial product.
Roughly equal learning depth across three areas: classical ML,
RAG/retrieval engineering, and MLOps/production practices.

## What it does

Two pipelines running in parallel:

1. **Multi-label clause classifier** — predicts which of 41 CUAD categories
   apply to a given contract segment. Classical ML baseline-first (TF-IDF +
   logistic regression / XGBoost), escalated only if evidence shows the
   classical ceiling is insufficient.
2. **RAG question-answering** — retrieves relevant contract sections before
   generating grounded answers to questions about a contract.

The two pipelines interact only through an optional, lawyer-toggled filter
(classifier output can narrow RAG's retrieval pool to one category) — never
automatically.

A heuristic risk-rating layer (clause presence/absence lookup table) is a
Phase 10 bonus feature, built after the classifier and RAG are validated.

## Stack

- **Vector DB**: Chroma
- **Data versioning**: DVC
- **Structured data / logs**: SQLite
- **Experiment tracking**: MLflow
- **Local inference**: Ollama
- **Cloud inference**: batch eval and LLM-as-judge scoring only
- **Dataset**: [CUAD](https://github.com/TheAtticusProject/cuad)

See `ARCHITECTURE.md` for full locked design decisions and rationale.

## Status

Phase 0 — repo and environment scaffolding.

## Hardware constraint

AMD Ryzen 7 8840HS, integrated Radeon 780M, 32GB RAM. No discrete GPU, no
CUDA. Rules out vLLM and GPU-dependent fine-tuning — documented as an
accepted tradeoff.
