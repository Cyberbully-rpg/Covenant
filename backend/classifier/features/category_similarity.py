"""
Covenant Phase 5 follow-up — category-description similarity feature.

NOT part of TRD §4.3's locked ladder (that names n-grams, structural
features, threshold tuning, SMOTE, ensembling). This is a targeted
response to the zero-shot diagnostic finding (§6C.5): the 3 categories the
champion scores F1 = 0.000 on are not a data-quantity problem (a zero-shot
LLM with NO training examples got ~90% of them right) — they're a
representation problem. TF-IDF's bag-of-words can't see the RELATIONSHIP
CUAD's own category definition asks about:

  "Price Restrictions"      — is there a restriction ON a party's ability
                               to raise/reduce prices (a restriction ON an
                               action, not just the word "price")
  "Third Party Beneficiary" — is there a non-contracting party who is a
                               beneficiary AND can enforce rights
  "Competitive Restriction
   Exception"                — is this clause an EXCEPTION carved out of a
                               Non-Compete/Exclusivity/No-Solicit clause

SMOTE (more copies of the same word-counted vectors) and ensembling
(combining two word-counters) don't address this — neither teaches the
model to weigh a relationship. What might: giving the model, as an extra
signal alongside raw TF-IDF, HOW SIMILAR a clause's wording is to CUAD's
own definition of the category. This is still classical (a cosine
similarity computed via TF-IDF, not a transformer) — consistent with
TRD §4.2's baseline-first rule, just a different lever than the locked
ladder anticipated.

Mechanism: `TfidfVectorizer` (default `norm="l2"`) row-normalizes every
vector, so cosine similarity between an L2-normalized clause vector and an
L2-normalized category-description vector is just their dot product — no
extra normalization needed at query time.
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy import sparse

EVAL_DIR = Path("data/processed/eval")


def load_category_questions() -> dict[str, str]:
    """One CUAD question/definition string per category (41 total)."""
    questions = {}
    with open(EVAL_DIR / "eval_set.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["category"] not in questions:
                questions[row["category"]] = row["question"]
    return questions


class CategorySimilarity:
    """
    Computes, for ONE category, a per-segment similarity-to-definition
    column using an already-fitted TfidfVectorizer (the same one used for
    the segment text, so both live in the same vector space).
    """

    def __init__(self, vectorizer, questions: dict[str, str]):
        self.vectorizer = vectorizer
        self.questions = questions

    def column_for(self, category: str, x_text: sparse.csr_matrix) -> sparse.csr_matrix:
        question = self.questions[category]
        q_vec = self.vectorizer.transform([question])  # (1, n_features), L2-normalized
        # dot product of L2-normalized vectors == cosine similarity
        sims = x_text @ q_vec.T  # (n_segments, 1)
        return sparse.csr_matrix(sims)


def hstack_similarity(x_text: sparse.csr_matrix, sim_col: sparse.csr_matrix) -> sparse.csr_matrix:
    return sparse.hstack([x_text, sim_col], format="csr")
