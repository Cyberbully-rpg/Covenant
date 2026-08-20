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


# --- asymmetric retrieval instruction prefixes -----------------------------

def test_the_default_model_takes_no_query_prefix():
    """MiniLM is a symmetric model -- prepending an instruction to queries
    would corrupt the vector space it was trained on."""
    from embedding import query_prefix
    assert query_prefix("all-MiniLM-L6-v2") == ""


def test_bge_models_declare_a_query_prefix():
    from embedding import query_prefix
    assert query_prefix("BAAI/bge-small-en-v1.5").startswith("Represent this sentence")


def test_prefix_is_applied_to_queries_and_never_to_documents(monkeypatch):
    """The asymmetry IS the feature: BGE wants the instruction on the query
    side only. Applying it to both, or to neither, degrades retrieval
    silently rather than raising -- so it is pinned by test."""
    import embedding

    seen = []

    class _Recorder:
        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return np.zeros((len(texts), 384))

    monkeypatch.setattr(embedding, "get_model", lambda *a, **kw: _Recorder())
    monkeypatch.setitem(embedding._QUERY_PREFIXES, "stub-model", "INSTRUCTION: ")

    embedding.embed(["a clause"], model_name="stub-model", is_query=False)
    embedding.embed(["a question"], model_name="stub-model", is_query=True)

    assert seen[0] == "a clause"                  # document: untouched
    assert seen[1] == "INSTRUCTION: a question"   # query: instructed


def test_retrieval_asks_for_the_query_side_encoding():
    """retrieve.py must pass is_query=True, or a BGE collection is queried
    with uninstructed vectors and quietly underperforms."""
    import inspect
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "retrieval"))
    import retrieve
    assert "is_query=True" in inspect.getsource(retrieve.ChromaRetriever._query_many)
