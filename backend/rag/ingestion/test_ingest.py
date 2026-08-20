"""
Tests for the Chroma ingestion pipeline.

Uses a temp Chroma directory (monkeypatched CHROMA_DIR) and a tiny fake
contract set -- never touches the real 510-contract collection, and never
downloads/embeds more than a few short sentences, so this stays fast.

The property that matters most, tested explicitly: ingestion never
imports or calls into backend/classifier/ (TRD 5.5 -- RAG must run with
zero classifier involvement). That's a real architectural boundary, not
just documentation, so it's worth asserting rather than trusting.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/rag

import ingest  # noqa: E402


FAKE_CONTRACT = (
    "This Agreement is entered into as of January 1, 2024, between Acme "
    "Corp and Beta LLC.\n\n"
    "1. GOVERNING LAW\n\nThis Agreement shall be governed by the laws of "
    "the State of Delaware.\n\n"
    "2. TERMINATION\n\nEither party may terminate this Agreement upon "
    "thirty days written notice to the other party."
)


@pytest.fixture
def temp_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_test")
    contracts_file = tmp_path / "contracts.jsonl"
    with open(contracts_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"contract_id": "fake_001", "context": FAKE_CONTRACT}) + "\n")
    monkeypatch.setattr(ingest, "EVAL_DIR", tmp_path)
    return tmp_path


def test_ingest_writes_segments_to_collection(temp_chroma):
    stats = ingest.ingest(limit=None, batch_size=8, reset=True)
    assert stats["n_contracts"] == 1
    assert stats["n_segments_written"] > 0
    # One record per window, so records >= parent segments; the collection
    # holds records, which is what n_records_written counts.
    assert stats["collection_count"] == stats["n_records_written"]
    assert stats["n_records_written"] >= stats["n_segments_written"]


def test_ingested_metadata_has_contract_id_and_spans(temp_chroma):
    ingest.ingest(limit=None, batch_size=8, reset=True)
    collection = ingest.get_collection(reset=False)
    result = collection.get(limit=1)
    meta = result["metadatas"][0]
    assert meta["contract_id"] == "fake_001"
    assert "start_char" in meta and "end_char" in meta
    assert "header" in meta and "scheme" in meta


def test_ingest_never_writes_classifier_category_metadata(temp_chroma):
    """TRD 5.5: no classifier metadata at ingestion -- that's Phase 8, and
    RAG must run correctly with zero classifier involvement."""
    ingest.ingest(limit=None, batch_size=8, reset=True)
    collection = ingest.get_collection(reset=False)
    result = collection.get()
    for meta in result["metadatas"]:
        assert "category" not in meta
        assert "predicted_category" not in meta


def test_ingestion_module_never_imports_classifier_package():
    import inspect
    source = inspect.getsource(ingest)
    assert "backend.classifier" not in source
    assert "from classifier" not in source


def test_reset_true_replaces_prior_collection_contents(temp_chroma):
    ingest.ingest(limit=None, batch_size=8, reset=True)
    first_count = ingest.get_collection(reset=False).count()
    ingest.ingest(limit=None, batch_size=8, reset=True)
    second_count = ingest.get_collection(reset=False).count()
    assert first_count == second_count  # re-ingesting the same fixture, not accumulating


# --- windowing (the fix for the 6D.2 truncation confound) -------------------

def test_split_windows_returns_the_text_unchanged_when_it_already_fits():
    assert ingest.split_windows("short clause", 870) == ["short clause"]


def test_split_windows_leaves_no_part_of_a_long_text_unembedded():
    # Distinct tokens, so "did this content survive windowing" is decidable
    # per token. A repeating filler string would make substring checks
    # match the wrong repeat and prove nothing.
    tokens = [f"tok{i:04d}" for i in range(700)]
    text = " ".join(tokens)
    windows = ingest.split_windows(text, 870)

    assert len(windows) > 1
    assert all(len(w) <= 870 for w in windows)
    # The property that actually matters: every token reaches the encoder
    # in at least one window. A gap here would mean part of a clause is
    # never embedded at all -- the exact failure windowing exists to fix.
    joined = "\n".join(windows)
    missing = [t for t in tokens if t not in joined]
    assert not missing, f"{len(missing)} tokens never embedded, e.g. {missing[:3]}"


def test_split_windows_overlap_so_a_clause_is_not_cut_at_a_boundary():
    text = " ".join(f"tok{i:04d}" for i in range(700))
    windows = ingest.split_windows(text, 870)
    for earlier, later in zip(windows, windows[1:]):
        assert later[:100] in earlier, "consecutive windows must overlap"


LONG_CLAUSE = (
    "1. INDEMNIFICATION\n\n" +
    ("Each party shall indemnify, defend and hold harmless the other party "
     "from and against any and all claims, damages, losses, liabilities, "
     "costs and expenses arising out of or resulting from any breach of "
     "this Agreement. ") * 12
)


@pytest.fixture
def temp_chroma_long(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma_test")
    contracts_file = tmp_path / "contracts.jsonl"
    with open(contracts_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"contract_id": "long_001", "context": LONG_CLAUSE}) + "\n")
    monkeypatch.setattr(ingest, "EVAL_DIR", tmp_path)
    return tmp_path


def test_long_segment_becomes_several_records_sharing_one_parent_span(temp_chroma_long):
    stats = ingest.ingest(limit=None, batch_size=8, reset=True, window=True)
    assert stats["n_records_written"] > stats["n_segments_written"], \
        "a segment longer than the model's input window must split"

    collection = ingest.get_collection(reset=False)
    metas = collection.get()["metadatas"]
    spans = {(m["start_char"], m["end_char"]) for m in metas}
    # Many records, but they collapse to the original segment spans --
    # this is what lets retrieval dedupe back to parent-sized spans and
    # keeps mean_retrieved_chars comparable across variants.
    assert len(spans) == stats["n_segments_written"]
    assert max(m["n_windows"] for m in metas) > 1
    assert {m["window_index"] for m in metas} >= {0, 1}


def test_no_window_mode_writes_exactly_one_record_per_segment(temp_chroma_long):
    stats = ingest.ingest(limit=None, batch_size=8, reset=True, window=False)
    assert stats["n_records_written"] == stats["n_segments_written"]
    assert stats["collection_name"] == ingest.collection_name(windowed=False)


def test_windowed_and_unwindowed_collections_never_share_a_name():
    assert ingest.collection_name(windowed=True) != ingest.collection_name(windowed=False)


def test_collection_name_is_namespaced_by_model():
    a = ingest.collection_name(model_name="all-MiniLM-L6-v2")
    b = ingest.collection_name(model_name="BAAI/bge-base-en-v1.5")
    assert a != b, "two models' vectors must never land in one collection"
