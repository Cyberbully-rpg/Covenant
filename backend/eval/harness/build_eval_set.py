"""
Covenant Phase 3, Step 1 — Eval set construction from CUAD's native QA triples.

SCOPE CLAIM (locked, must be stated wherever this eval set's results are
reported): this eval set is derived entirely from CUAD's own 41 templated
per-category probe questions run over CUAD's 510 contracts. It is
"validated against CUAD's contract distribution using CUAD's 41 templated
category probes" — it does NOT represent diverse user questions, and
results here must never be described as validating against "legal
documents in general."

Source: CUADv1.json (SQuAD 2.0-style format). Each contract has exactly 41
QA pairs, one per clause category. A category with no matching clause in
the contract is an "impossible" question with zero answer spans
(is_impossible=True, answers=[]); a category present in the contract
carries one or more gold answer spans (CUAD allows multiple valid spans
per category per contract — e.g. an Audit Rights clause mentioned in two
separate places — so "correct" for retrieval purposes means overlap with
ANY of them, never just the first).

Output: two JSONL files plus a metadata sidecar, joined by contract_id,
consumable by Step 2's scorer with no further transformation:
  - contracts.jsonl : {contract_id, title, context}
  - eval_set.jsonl  : {contract_id, category, question, gold_spans,
                        has_gold_span}
    gold_spans is a list of {text, start, end} (end exclusive,
    start + len(text)), empty when has_gold_span is False.
  - metadata.json   : scope claim + row-count breakdown.

Nothing is dropped: every (contract, category) row from CUAD — present or
absent — is preserved in eval_set.jsonl, tagged with has_gold_span.
Whether absent-category rows become a separate abstention/no-answer eval
slice or get excluded is a Step 2 scoring decision, not made here.

Also sanity-checks CUAD's own offsets: context[start:end] must equal the
gold span's text for every present span. Step 2's scorer trusts these
offsets without re-deriving them, so any mismatch here would silently
corrupt every downstream retrieval-correctness number.
"""

import json
from pathlib import Path

CUAD_JSON_PATH = Path("data/raw/CUADv1.json")
OUTPUT_DIR = Path("data/processed/eval")

SCOPE_CLAIM = (
    "validated against CUAD's contract distribution using CUAD's 41 "
    "templated category probes"
)


def build() -> dict:
    with open(CUAD_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contracts_path = OUTPUT_DIR / "contracts.jsonl"
    eval_set_path = OUTPUT_DIR / "eval_set.jsonl"
    meta_path = OUTPUT_DIR / "metadata.json"

    n_contracts = 0
    n_rows = 0
    n_empty = 0
    n_multi = 0
    n_offset_mismatches = 0
    offset_mismatch_examples = []

    with open(contracts_path, "w", encoding="utf-8") as cf, \
         open(eval_set_path, "w", encoding="utf-8") as ef:
        for entry in data["data"]:
            title = entry["title"]
            n_contracts += 1
            for para in entry["paragraphs"]:
                context = para["context"]
                cf.write(json.dumps({
                    "contract_id": title,
                    "title": title,
                    "context": context,
                }) + "\n")

                id_prefix = title + "__"
                for qa in para["qas"]:
                    assert qa["id"].startswith(id_prefix), (
                        f"Unexpected qa id format: {qa['id']!r} for title {title!r}"
                    )
                    category = qa["id"][len(id_prefix):]

                    gold_spans = []
                    for a in qa["answers"]:
                        start = a["answer_start"]
                        end = start + len(a["text"])
                        if context[start:end] != a["text"]:
                            n_offset_mismatches += 1
                            if len(offset_mismatch_examples) < 5:
                                offset_mismatch_examples.append({
                                    "contract_id": title,
                                    "category": category,
                                    "expected": a["text"],
                                    "actual": context[start:end],
                                })
                        gold_spans.append({"text": a["text"], "start": start, "end": end})

                    has_gold_span = len(gold_spans) > 0
                    n_rows += 1
                    if not has_gold_span:
                        n_empty += 1
                    if len(gold_spans) > 1:
                        n_multi += 1

                    ef.write(json.dumps({
                        "contract_id": title,
                        "category": category,
                        "question": qa["question"],
                        "gold_spans": gold_spans,
                        "has_gold_span": has_gold_span,
                    }) + "\n")

    metadata = {
        "scope_claim": SCOPE_CLAIM,
        "source": "CUADv1.json",
        "n_contracts": n_contracts,
        "n_rows": n_rows,
        "n_rows_with_gold_span": n_rows - n_empty,
        "n_rows_empty_category": n_empty,
        "n_rows_multi_span": n_multi,
        "n_offset_mismatches": n_offset_mismatches,
        "offset_mismatch_examples": offset_mismatch_examples,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    return metadata


if __name__ == "__main__":
    meta = build()
    print(json.dumps(meta, indent=2))
