"""
Covenant Phase 6 — shared embedding model wrapper.

Ingestion (writes to Chroma) and retrieval (queries Chroma) MUST use the
exact same model and the exact same normalization — a vector space only
makes sense if both sides of a similarity comparison live in it. Housing
this in one module rather than duplicating a model name string in two
places is what keeps that guarantee true instead of just documented.

Model choice is TRD §7.1 (locked there, not here — this module reads
MODEL_NAME from that decision rather than owning it). Baseline-first
(TRD §4.2's discipline applied to retrieval too): start at the standard
small/fast sentence-transformers default, escalate only on measured
evidence. No CUDA/discrete GPU on this machine (TRD §6.1), so CPU embed
throughput is a real constraint on how large a model is viable.

The active model is overridable via the COVENANT_EMBED_MODEL environment
variable. That exists so the §7.1 escalation ladder can be *measured*
(ingest under model X, score it, compare) without editing source between
runs — not so it can be varied at serving time. Ingestion and retrieval
must always be run under the same value, which is why the collection name
carries the model tag (see ingestion/ingest.py).

`max_seq_tokens()` matters more than it looks: every model here silently
TRUNCATES input past its window rather than erroring. 39.9% of Covenant's
20,874 segments exceed all-MiniLM-L6-v2's 256-wordpiece limit, so under
that model a large minority of segments were only ever embedded from
their opening fragment. Windowed ingestion (ingest.py) exists to fix
exactly that.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = os.environ.get("COVENANT_EMBED_MODEL", DEFAULT_MODEL)

# Output dimensionality per model. Kept explicit rather than probed at
# import time so importing this module never triggers a model download.
_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}
EMBEDDING_DIM = _DIMS.get(MODEL_NAME, 384)

# Asymmetric-retrieval instruction prefixes. BGE models are trained with a
# short instruction prepended to the QUERY side only; omitting it costs
# real accuracy and the model gives no indication anything is wrong — the
# same silent-misuse shape as the truncation confound (§6D.3), so it is
# handled here rather than left to each call site to remember. Models not
# listed take no prefix; MiniLM is symmetric and wants none.
_QUERY_PREFIXES = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

# Short tag used to namespace Chroma collections per model, so two models'
# vectors can never land in the same collection.
def model_tag(model_name: str | None = None) -> str:
    name = model_name or MODEL_NAME
    return name.split("/")[-1].replace("-", "_").replace(".", "")


@lru_cache(maxsize=2)
def get_model(model_name: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name or MODEL_NAME)


def max_seq_tokens(model_name: str | None = None) -> int:
    """The model's input window, in word-pieces. Input past this is
    silently truncated by sentence-transformers, never raised."""
    return int(get_model(model_name).max_seq_length)


def query_prefix(model_name: str | None = None) -> str:
    return _QUERY_PREFIXES.get(model_name or MODEL_NAME, "")


def embed(texts: list[str], model_name: str | None = None,
          is_query: bool = False) -> np.ndarray:
    """Returns L2-normalized embeddings, shape (len(texts), model dim).

    L2-normalized so cosine similarity == dot product == what Chroma's
    cosine space computes without extra work at query time.

    `is_query=True` applies the model's retrieval instruction prefix if it
    has one. Ingestion must always leave this False and retrieval must
    always set it True — that asymmetry is the whole point, and getting it
    backwards degrades silently rather than raising.
    """
    model = get_model(model_name)
    prefix = query_prefix(model_name) if is_query else ""
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
