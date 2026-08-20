"""
Covenant Phase 6 — leading-segment positional prior.

WHY
---
The rank diagnostic (run_rank_diagnostic.py) found that four CUAD
categories — Document Name, Parties, Agreement Date, Effective Date —
account for 1,879 of 6,702 scored rows (28%) and score 0.39-0.61
hit_rate@5, while categories with distinctive vocabulary score 0.95+.
Checking where their answers physically sit explained it: 77-97% of them
overlap **segment index 0**, the title/preamble block, at ~0.5% into the
document.

Neither ranker can find those. Both score on word overlap with the query,
literal for TF-IDF and semantic for the embedding model, and a title
block shares no vocabulary with the phrase "Document Name" — there is no
signal to rank on. This isn't a ranking failure to be tuned away; it's a
category of question whose answer is located structurally rather than
lexically. So it gets a structural answer: when the query asks for
document metadata, put the document's opening segment at the top.

The category is read from the query text, which for CUAD's probes states
it verbatim (`related to "Parties"`). That is query parsing, not label
leakage — but it does lean on CUAD's templated question format, which
this project's scope claim already fences off ("41 templated category
probes, not diverse user questions"). A free-text `/ask` would need the
lawyer's category selection (TRD §5.2) to trigger the same path, and
without one the prior simply doesn't fire.

FIT ON TRAIN, NOT ON THE REPORTED SET
--------------------------------------
Which categories qualify is LEARNED, from the training contracts only,
via the Phase 4 contract-level split (`classifier/data/split.py`) — the
same seed, the same function, the same contract-disjointness guarantee.
The diagnostic that suggested this idea ran on the full corpus, so
hardcoding the four category names it surfaced would be fitting to the
evaluation set. Fitting the list on train contracts and reporting on
held-out ones is what makes the resulting number honest.

`n_lead` and `threshold` are fixed a priori (1 segment, 0.6) rather than
searched, for the same reason the RRF constants are left at their
published defaults (TRD §7.2).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "harness"))

from scorer import spans_overlap  # noqa: E402

# CUAD probes state the category verbatim: 'related to "Parties" that ...'
_CATEGORY_RE = re.compile(r'related to\s+"([^"]+)"')

DEFAULT_THRESHOLD = 0.6
DEFAULT_N_LEAD = 1
DEFAULT_MIN_ROWS = 20


def category_from_question(question: str) -> str | None:
    """The category a CUAD probe names, or None for free-text questions."""
    m = _CATEGORY_RE.search(question)
    return m.group(1) if m else None


class LeadPrior:
    """Promotes a document's opening segment(s) for metadata-type queries."""

    def __init__(self, categories: set[str] | None = None, n_lead: int = DEFAULT_N_LEAD):
        self.categories = set(categories or ())
        self.n_lead = n_lead

    # --- fitting ---------------------------------------------------------

    @classmethod
    def fit(
        cls,
        rows,
        segments_by_contract,
        threshold: float = DEFAULT_THRESHOLD,
        n_lead: int = DEFAULT_N_LEAD,
        min_rows: int = DEFAULT_MIN_ROWS,
    ) -> "LeadPrior":
        """Learn which categories keep their answer in the first `n_lead`
        segments. `rows` must be TRAIN rows only, with gold spans."""
        hits, totals = defaultdict(int), defaultdict(int)
        for row in rows:
            if not row.get("has_gold_span"):
                continue
            cat = row.get("category") or category_from_question(row["question"])
            if not cat:
                continue
            segments = segments_by_contract.get(row["contract_id"]) or []
            if not segments:
                continue
            gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
            lead = [(s.start_char, s.end_char) for s in segments[:n_lead]]
            totals[cat] += 1
            if any(spans_overlap(l, g) for l in lead for g in gold):
                hits[cat] += 1

        chosen = {
            cat for cat, n in totals.items()
            if n >= min_rows and hits[cat] / n >= threshold
        }
        return cls(categories=chosen, n_lead=n_lead)

    def rates(self, rows, segments_by_contract) -> dict[str, tuple[int, float]]:
        """Diagnostic helper: per-category (n, lead-overlap rate)."""
        hits, totals = defaultdict(int), defaultdict(int)
        for row in rows:
            if not row.get("has_gold_span"):
                continue
            cat = row.get("category") or category_from_question(row["question"])
            segments = segments_by_contract.get(row["contract_id"]) or []
            if not cat or not segments:
                continue
            gold = [(g["start"], g["end"]) for g in row["gold_spans"]]
            lead = [(s.start_char, s.end_char) for s in segments[:self.n_lead]]
            totals[cat] += 1
            if any(spans_overlap(l, g) for l in lead for g in gold):
                hits[cat] += 1
        return {c: (totals[c], hits[c] / totals[c]) for c in sorted(totals)}

    # --- applying --------------------------------------------------------

    def applies_to(self, question: str) -> bool:
        cat = category_from_question(question)
        return cat is not None and cat in self.categories

    def apply(self, ranked_spans, segments, question: str, k: int):
        """Promote the leading segment(s) to the front for a matching query.

        Returns exactly k spans, deduped, order preserved otherwise. When
        the prior doesn't fire this is the identity function on the first
        k spans — so a query it knows nothing about is never degraded.
        """
        if not self.applies_to(question) or not segments:
            return ranked_spans[:k]
        lead = [(s.start_char, s.end_char) for s in segments[:self.n_lead]]
        out, seen = [], set()
        for span in lead + list(ranked_spans):
            if span in seen:
                continue
            seen.add(span)
            out.append(span)
            if len(out) == k:
                break
        return out

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"categories": sorted(self.categories), "n_lead": self.n_lead}

    @classmethod
    def from_dict(cls, d: dict) -> "LeadPrior":
        return cls(categories=set(d.get("categories", ())), n_lead=d.get("n_lead", DEFAULT_N_LEAD))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LeadPrior":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class PriorRetriever:
    """Wraps any retriever, applying a LeadPrior to its ranking."""

    def __init__(self, base, prior: LeadPrior, name: str = "prior"):
        self.base = base
        self.prior = prior
        self.name = name

    def retrieve_batch(self, questions, segments, contract_id, k: int = 5):
        # Ask for extra depth so promoting a lead segment displaces the
        # weakest hit rather than silently shortening the result.
        base_k = k + self.prior.n_lead
        try:
            batch = self.base.retrieve_batch(questions, segments, contract_id, base_k)
        except TypeError:  # dense-only retrievers take no `segments`
            batch = self.base.retrieve_batch(questions, contract_id, base_k)
        return [
            self.prior.apply(ranked, segments, q, k)
            for q, ranked in zip(questions, batch)
        ]
