"""
Tests for ChromaRetriever (retrieve.py).

Uses the same temp-Chroma-directory pattern as test_ingest.py, with a
tiny fake corpus of two contracts covering different topics -- enough to
verify the two properties that actually matter for correctness:

  1. contract_id scoping is real -- a query never returns spans from a
     different contract, which would silently break TRD 5.1's
     "full-document search" semantics (per-contract, not corpus-wide).
  2. the returned shape is exactly what scorer.score_query() expects
     ((start_char, end_char) tuples), since that shape compatibility is
     the whole point of matching the baseline retriever interface.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # backend/rag/retrieval
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/rag
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingestion"))

import ingest  # noqa: E402
from retrieve import ChromaRetriever  # noqa: E402

CONTRACT_A = (
    "1. GOVERNING LAW\n\nThis Agreement shall be governed by the laws of "
    "the State of Delaware without regard to conflict of law principles."
)
CONTRACT_B = (
    "1. AUDIT RIGHTS\n\nBuyer may audit the books and records of Seller "
    "upon thirty days prior written notice, no more than once annually."
)


@pytest.fixture
def two_contract_index(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_test")
    contracts_file = tmp_path / "contracts.jsonl"
    with open(contracts_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"contract_id": "contract_a", "context": CONTRACT_A}) + "\n")
        f.write(json.dumps({"contract_id": "contract_b", "context": CONTRACT_B}) + "\n")
    monkeypatch.setattr(ingest, "EVAL_DIR", tmp_path)
    ingest.ingest(limit=None, batch_size=8, reset=True)

    import retrieve as retrieve_module
    monkeypatch.setattr(retrieve_module, "get_collection", ingest.get_collection)
    return tmp_path


def test_retrieve_only_returns_spans_from_the_requested_contract(two_contract_index):
    r = ChromaRetriever()
    result = r.retrieve_with_text("what governs this agreement", "contract_a", k=5)
    assert len(result) > 0
    assert all(x["contract_id"] == "contract_a" for x in result)


def test_retrieve_scoping_works_for_the_other_contract_too(two_contract_index):
    r = ChromaRetriever()
    result = r.retrieve_with_text("audit rights", "contract_b", k=5)
    assert len(result) > 0
    assert all(x["contract_id"] == "contract_b" for x in result)


def test_retrieve_returns_start_end_tuples_matching_scorer_interface(two_contract_index):
    r = ChromaRetriever()
    spans = r.retrieve("governing law", "contract_a", k=3)
    assert all(isinstance(s, tuple) and len(s) == 2 for s in spans)
    assert all(isinstance(s[0], int) and isinstance(s[1], int) for s in spans)
    assert all(s[0] < s[1] for s in spans)


def test_retrieve_finds_the_relevant_clause_for_an_on_topic_query(two_contract_index):
    r = ChromaRetriever()
    result = r.retrieve_with_text("what law governs this contract", "contract_a", k=1)
    assert "delaware" in result[0]["text"].lower() or "governing law" in result[0]["header"].lower()


# --- window deduplication ---------------------------------------------------

LONG_CONTRACT = (
    "1. INDEMNIFICATION\n\n" +
    ("Each party shall indemnify and hold harmless the other party from any "
     "and all claims, damages and losses arising out of this Agreement. ") * 15 +
    "\n\n2. GOVERNING LAW\n\nThis Agreement is governed by the laws of New York."
)


@pytest.fixture
def windowed_index(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_test")
    contracts_file = tmp_path / "contracts.jsonl"
    with open(contracts_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"contract_id": "long_c", "context": LONG_CONTRACT}) + "\n")
    monkeypatch.setattr(ingest, "EVAL_DIR", tmp_path)
    stats = ingest.ingest(limit=None, batch_size=8, reset=True, window=True)
    assert stats["n_records_written"] > stats["n_segments_written"]  # windows exist

    import retrieve as retrieve_module
    monkeypatch.setattr(retrieve_module, "get_collection", ingest.get_collection)
    return stats


def test_retrieve_returns_distinct_parent_spans_never_repeated_windows(windowed_index):
    """Without dedup, one long segment's several windows would occupy
    several of the k slots with different slices of the same clause and
    crowd out every other clause in the contract."""
    r = ChromaRetriever()
    spans = r.retrieve("indemnification for claims and damages", "long_c", k=5)
    assert len(spans) == len(set(spans))


def test_retrieve_never_returns_more_than_k_spans(windowed_index):
    r = ChromaRetriever()
    assert len(r.retrieve("indemnify", "long_c", k=1)) == 1


def test_batched_query_matches_one_at_a_time_query(windowed_index):
    """Per-contract batching is a speed optimization for the harness; it
    must not change a single retrieved span."""
    r = ChromaRetriever()
    questions = ["who indemnifies whom", "what law governs this agreement"]
    batched = r.retrieve_batch(questions, "long_c", k=3)
    one_at_a_time = [r.retrieve(q, "long_c", k=3) for q in questions]
    assert batched == one_at_a_time
