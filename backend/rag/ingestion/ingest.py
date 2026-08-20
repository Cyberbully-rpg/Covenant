"""
Covenant Phase 6 — Chroma ingestion pipeline.

Turns CUAD contracts into a persisted, queryable vector store: segment
(Phase 2) -> embed the RAG-path text (`embedding_text`, which has parent
section titles injected for undersized fragments — NOT raw `text`, which
is the classifier's path and must stay uncontaminated per Phase 2's design)
-> write to Chroma with metadata sufficient to reconstruct the original
contract span.

Scope boundary (locked, TRD §5.2 / roadmap): this is retrieval-only.
No classifier category metadata is written here — that's Phase 8's hard
filter integration, a separate, later, optional dependency. RAG must be
able to run with zero classifier involvement (TRD §5.5), and this
ingestion path proves that boundary by construction: it never imports or
calls anything from backend/classifier/.

One collection per (embedding model, windowing mode) — `contract_id` is
stored as metadata and filtered at query time, which matches TRD §5.1's
"full-document similarity search across the contract" (search is scoped to
one contract, not global across all 510) without needing 510 separate
collections. The name is namespaced by model because two models' vectors
sharing one collection would be silently meaningless rather than an error.

WINDOWING (the fix for the §6D.2 truncation confound)
-----------------------------------------------------
sentence-transformers silently truncates any input past the model's
`max_seq_length` — 256 word-pieces for all-MiniLM-L6-v2. Measured over
this corpus, 39.9% of the 20,874 segments exceed that, so under
unwindowed ingestion a large minority of segments were embedded from
their opening fragment alone while TF-IDF read every word of them. That
is not a fair comparison, and it is not a property anyone chose.

Windowed ingestion splits an oversized segment into overlapping windows
sized to the model's actual input limit, embeds each, and writes each as
its own record carrying the SAME parent span metadata. Retrieval then
dedupes back to parent spans (retrieve.py), which makes the scheme a
max-pool over windows: a segment is ranked by its best-matching part
rather than by a truncation of its opening. Retrieved spans stay
parent-sized, so `mean_retrieved_chars` remains comparable to the
unwindowed and TF-IDF runs and hit_rate cannot be inflated by chunk-size
gaming (scorer.py's anti-gaming pair).

Usage:
    python backend/rag/ingestion/ingest.py [--limit N] [--batch-size 64] [--no-window]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmenter"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmenter import segment_contract  # noqa: E402
from embedding import embed, MODEL_NAME, model_tag, max_seq_tokens  # noqa: E402

EVAL_DIR = Path("data/processed/eval")
CHROMA_DIR = Path("chroma_db")

# The Phase 6 original, unwindowed MiniLM collection. Kept under a fixed
# name so the §6D.2 measurement stays reproducible now that collections
# are namespaced per model.
LEGACY_COLLECTION_NAME = "covenant_contracts"

# Word-pieces -> characters. Legal English runs ~3.5-4.0 chars/token; 3.4
# is deliberately conservative so a window lands inside the model's limit
# rather than exactly on it. Used only to size windows, never to count
# tokens.
CHARS_PER_TOKEN = 3.4
WINDOW_OVERLAP_FRACTION = 0.20


def collection_name(model_name: str | None = None, windowed: bool = True) -> str:
    return f"covenant_{model_tag(model_name)}" + ("_win" if windowed else "")


def window_chars_for(model_name: str | None = None) -> int:
    return int(max_seq_tokens(model_name) * CHARS_PER_TOKEN)


def split_windows(text: str, window_chars: int) -> list[str]:
    """Overlapping character windows. Returns [text] when it already fits."""
    if len(text) <= window_chars:
        return [text]
    overlap = int(window_chars * WINDOW_OVERLAP_FRACTION)
    step = max(1, window_chars - overlap)
    out = []
    for start in range(0, len(text), step):
        chunk = text[start:start + window_chars]
        if chunk.strip():
            out.append(chunk)
        if start + window_chars >= len(text):
            break
    return out


def load_contracts(limit: int | None) -> dict[str, str]:
    contracts = {}
    with open(EVAL_DIR / "contracts.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            contracts[row["contract_id"]] = row["context"]
            if limit and len(contracts) >= limit:
                break
    return contracts


def get_collection(reset: bool = False, name: str | None = None):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = name or collection_name()
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 — collection may not exist yet
            pass
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def ingest(
    limit: int | None = None,
    batch_size: int = 64,
    reset: bool = True,
    window: bool = True,
    name: str | None = None,
) -> dict:
    contracts = load_contracts(limit)
    name = name or collection_name(windowed=window)
    collection = get_collection(reset=reset, name=name)
    win_chars = window_chars_for() if window else None

    ids, texts, metadatas = [], [], []
    n_segments = 0
    n_parents_written = 0
    for cid, raw_text in contracts.items():
        segments = segment_contract(raw_text, doc_id=cid)
        n_segments += len(segments)
        for seg in segments:
            # Undersized segments' embedding_text can be empty-ish after
            # context injection edge cases; Chroma rejects empty documents.
            text = seg.embedding_text.strip() or seg.text.strip() or seg.header
            if not text:
                continue
            n_parents_written += 1
            chunks = split_windows(text, win_chars) if window else [text]
            for j, chunk in enumerate(chunks):
                ids.append(f"{seg.segment_id}#w{j}" if len(chunks) > 1 else seg.segment_id)
                texts.append(chunk)
                metadatas.append({
                    "contract_id": cid,
                    "segment_id": seg.segment_id,
                    # Parent span, identical across every window of a
                    # segment — what retrieval dedupes on and what gets
                    # scored against CUAD's gold spans.
                    "start_char": seg.start_char,
                    "end_char": seg.end_char,
                    "header": seg.header or "",
                    "scheme": seg.scheme,
                    "is_oversized_split": seg.is_oversized_split,
                    "is_undersized": seg.is_undersized,
                    "window_index": j,
                    "n_windows": len(chunks),
                })

    n_written = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        vectors = embed(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=vectors.tolist(),
            documents=batch_texts,
            metadatas=batch_meta,
        )
        n_written += len(batch_ids)

    return {
        "n_contracts": len(contracts),
        "n_segments_seen": n_segments,
        "n_segments_written": n_parents_written,
        "n_records_written": n_written,
        "collection_count": collection.count(),
        "collection_name": name,
        "model": MODEL_NAME,
        "windowed": window,
        "window_chars": win_chars,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="contracts to ingest (0 = all 510)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-reset", action="store_true",
                    help="append to existing collection instead of rebuilding it")
    ap.add_argument("--no-window", action="store_true",
                    help="embed each segment whole (silently truncated past the model's "
                         "input limit) — reproduces the original 6D.2 ingestion")
    ap.add_argument("--name", default=None, help="override collection name")
    args = ap.parse_args()

    window = not args.no_window
    stats = ingest(args.limit or None, args.batch_size, reset=not args.no_reset,
                   window=window, name=args.name)
    print(f"model                : {stats['model']}")
    print(f"collection           : {stats['collection_name']}")
    print(f"windowed             : {stats['windowed']} (window_chars={stats['window_chars']})")
    print(f"contracts ingested   : {stats['n_contracts']}")
    print(f"segments seen        : {stats['n_segments_seen']}")
    print(f"parent segments      : {stats['n_segments_written']}")
    print(f"records written      : {stats['n_records_written']}")
    print(f"collection count now : {stats['collection_count']}")
    print(f"\nChroma DB at {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()
