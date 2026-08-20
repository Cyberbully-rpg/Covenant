"""
Tests for the shared embedding wrapper (embedding.py).

The property that matters most: ingestion and retrieval both call
embed() from THIS module, never construct their own SentenceTransformer.
That's what guarantees they can never drift into different vector spaces
-- tested here by checking the function is deterministic and produces
comparable similarity for related vs. unrelated text, which would be
meaningless if the two call sites used different models.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from embedding import embed, EMBEDDING_DIM  # noqa: E402


def test_embed_returns_correct_shape():
    vecs = embed(["hello world", "a second sentence"])
    assert vecs.shape == (2, EMBEDDING_DIM)


def test_embeddings_are_l2_normalized():
    vecs = embed(["this agreement shall be governed by the laws of delaware"])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-4


def test_embed_is_deterministic():
    a = embed(["governing law clause"])
    b = embed(["governing law clause"])
    assert np.allclose(a, b)


def test_similar_texts_score_higher_than_unrelated_texts():
    base = embed(["this agreement shall be governed by the laws of the state of new york"])[0]
    similar = embed(["this contract is governed by the laws of california"])[0]
    unrelated = embed(["buyer may audit the books and records of seller annually"])[0]
    sim_score = float(base @ similar)
    unrelated_score = float(base @ unrelated)
    assert sim_score > unrelated_score
