"""
Covenant Phase 6 — query boilerplate stripping.

Every CUAD probe is wrapped in the same instruction template:

    Highlight the parts (if any) of this contract related to "Governing Law"
    that should be reviewed by a lawyer. Details: Which state's law governs
    the interpretation of the contract?

Only the quoted category and the Details clause carry information. The
wrapper is byte-identical across all 41 probes and is the majority of the
query's tokens, which hurts every ranker in a different way:

  - TF-IDF matches "contract", "parts", "reviewed" against contract text
    where such words are ubiquitous — pure noise competing with the one
    discriminative term;
  - the bi-encoder mean-pools over the whole query, so most of the vector
    is a constant shared by every question ever asked;
  - a cross-encoder trained on short web queries (MS MARCO) has never seen
    an instruction-shaped query at all, and scores the pair on something
    other than what was intended.

`strip_boilerplate` reduces the probe to `<category>. <details>`, keeping
every informative token and dropping the constant. It is a no-op on any
string that doesn't match the template, so a free-text `/ask` question
passes through untouched.

This is query preprocessing, not label use: the category is stated in the
question text itself, and the gold spans are unaffected. It does lean on
CUAD's templated format, which the scope claim (TRD §3.3) already fences
off — see lead_prior.py for the same caveat.
"""

from __future__ import annotations

import re

_TEMPLATE = re.compile(
    r'^\s*Highlight the parts \(if any\) of this contract related to\s+'
    r'"(?P<category>[^"]+)"\s*'
    r'that should be reviewed by a lawyer\.\s*'
    r'(?:Details:\s*(?P<details>.*))?$',
    re.IGNORECASE | re.DOTALL,
)


def strip_boilerplate(question: str) -> str:
    """`<category>. <details>` for a CUAD probe; the input unchanged otherwise."""
    m = _TEMPLATE.match(question or "")
    if not m:
        return question
    category = (m.group("category") or "").strip()
    details = (m.group("details") or "").strip()
    if not details:
        return category
    return f"{category}. {details}"


class CleanQueryRetriever:
    """Strips template boilerplate before handing questions to `base`.

    Deliberately a wrapper rather than a change inside each retriever: the
    lead prior parses the category out of the ORIGINAL question text, so
    cleaning has to happen strictly below it in the stack, never above.
    """

    name = "cleanq"

    def __init__(self, base, name: str | None = None):
        self.base = base
        if name:
            self.name = name

    def retrieve_batch(self, questions, segments, contract_id, k: int = 5):
        cleaned = [strip_boilerplate(q) for q in questions]
        try:
            return self.base.retrieve_batch(cleaned, segments, contract_id, k)
        except TypeError:  # dense-only retrievers take no `segments`
            return self.base.retrieve_batch(cleaned, contract_id, k)

    def retrieve(self, question: str, segments: list, contract_id: str, k: int = 5):
        return self.retrieve_batch([question], segments, contract_id, k)[0]
