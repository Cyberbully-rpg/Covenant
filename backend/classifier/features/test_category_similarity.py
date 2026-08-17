"""
Tests for the category-definition similarity feature (Phase 5 follow-up).

The property that matters: the similarity score for a category must
actually track topical closeness — a clause using the category's own
vocabulary should score higher than an unrelated clause. This is a
weak/cheap signal by design (see module docstring for why it's not a
full fix), but if it can't even separate an obviously-relevant clause
from an obviously-irrelevant one, it isn't worth the extra column at all.
"""

import sys
from pathlib import Path

import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent))

from category_similarity import CategorySimilarity, hstack_similarity  # noqa: E402


QUESTIONS = {
    "Price Restrictions": "Is there a restriction on the ability of a party "
                           "to raise or reduce prices of goods or services?",
    "Governing Law": "Which state or country's law governs the contract?",
}


@pytest.fixture
def fitted():
    corpus = [
        "the party shall not raise or reduce the price of the goods without consent",
        "this agreement shall be governed by the laws of the state of delaware",
        "the parties acknowledge receipt of the attached schedules and exhibits",
    ]
    vectorizer = TfidfVectorizer(lowercase=True, min_df=1)
    x = vectorizer.fit_transform(corpus)
    return vectorizer, x


def test_relevant_clause_scores_higher_than_unrelated_clause(fitted):
    vectorizer, x = fitted
    sim = CategorySimilarity(vectorizer, QUESTIONS)
    col = sim.column_for("Price Restrictions", x).toarray().ravel()
    # row 0 is about price restriction, row 2 is unrelated boilerplate
    assert col[0] > col[2]


def test_similarity_column_is_bounded_zero_to_one(fitted):
    vectorizer, x = fitted
    sim = CategorySimilarity(vectorizer, QUESTIONS)
    col = sim.column_for("Governing Law", x).toarray().ravel()
    assert (col >= 0).all()
    assert (col <= 1.0 + 1e-9).all()


def test_different_categories_produce_different_columns(fitted):
    vectorizer, x = fitted
    sim = CategorySimilarity(vectorizer, QUESTIONS)
    price_col = sim.column_for("Price Restrictions", x).toarray().ravel()
    law_col = sim.column_for("Governing Law", x).toarray().ravel()
    assert not (price_col == law_col).all()


def test_hstack_similarity_preserves_row_count(fitted):
    _, x = fitted
    sim_col = sparse.csr_matrix([[0.1], [0.2], [0.3]])
    combined = hstack_similarity(x, sim_col)
    assert combined.shape[0] == x.shape[0]
    assert combined.shape[1] == x.shape[1] + 1
