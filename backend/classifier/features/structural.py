"""
Covenant Phase 5, ladder step 2 — structural features from the segmenter.

TRD §4.3 step 2: "section position, header type, clause length, trigger
phrases fed alongside TF-IDF — cheap, since the segmenter already extracts
these as a byproduct."

The premise is that where a clause sits and how it is shaped carries signal
that its words do not. A governing-law clause is short and lives near the
end; a license grant is long and lives early; document name and parties
live in the preamble at position 0. TF-IDF cannot see any of that.

Feature block (11 columns, all derived from fields the segmenter already
emits — nothing here requires re-reading the contract):

  relative_position    where in the document, 0.0-1.0
  is_first / is_last   endpoints, which carry outsized signal in contracts
  is_preamble          the title/parties/date block (see segmenter §preamble)
  log_char_len         clause length, log-scaled because the raw range spans
                       three orders of magnitude
  is_oversized_split   this segment came out of a sub-split
  is_undersized        short fragment that got context injection downstream
  has_parent           nested under a top-level segment
  header_n_tokens      header shape
  header_has_digit     numbered header vs. bare title
  scheme_*             one-hot over the segmenter's numbering schemes

Scaling matters here: these are hstacked onto L2-normalized TF-IDF columns,
where every value sits in [0, 1]. An unscaled char_len of 4,500 would
dominate the whole feature space and effectively delete the text signal, so
the numeric columns are min-max scaled to [0, 1] using statistics fit on
TRAIN ONLY (fitting on all rows would leak test distribution into training).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

SCHEMES = ["article", "section_nn", "bare_nn", "fallback"]

FEATURE_NAMES = [
    "relative_position",
    "is_first",
    "is_last",
    "is_preamble",
    "log_char_len",
    "is_oversized_split",
    "is_undersized",
    "has_parent",
    "header_n_tokens",
    "header_has_digit",
] + [f"scheme_{s}" for s in SCHEMES]


def _row(example: dict) -> list[float]:
    header = example.get("header") or ""
    pos = example.get("position", 0)
    n_segs = example.get("n_segments_in_contract", 1) or 1
    scheme = example.get("scheme", "")
    # A scheme the segmenter grew later (e.g. an inline variant) must not
    # silently collapse into "article"; unknown schemes get an all-zero
    # one-hot block, which is honest rather than wrong.
    scheme_onehot = [1.0 if scheme == s else 0.0 for s in SCHEMES]
    return [
        float(example.get("relative_position", 0.0)),
        1.0 if pos == 0 else 0.0,
        1.0 if pos == n_segs - 1 else 0.0,
        1.0 if header == "[preamble]" else 0.0,
        float(np.log1p(example.get("char_len", len(example.get("text", ""))))),
        1.0 if example.get("is_oversized_split") else 0.0,
        1.0 if example.get("is_undersized") else 0.0,
        1.0 if example.get("has_parent") else 0.0,
        float(len(header.split())),
        1.0 if any(ch.isdigit() for ch in header) else 0.0,
    ] + scheme_onehot


def build_matrix(examples) -> np.ndarray:
    """Dense (n_examples, len(FEATURE_NAMES)) float matrix — unscaled."""
    return np.asarray([_row(e) for e in examples], dtype=np.float64)


class StructuralFeatures:
    """Fit min-max scaling on train, apply to any split."""

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, examples) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.scaler.fit_transform(build_matrix(examples)))

    def transform(self, examples) -> sparse.csr_matrix:
        # clip: a test segment longer than anything in train would otherwise
        # scale above 1.0 and re-dominate the TF-IDF columns
        scaled = np.clip(self.scaler.transform(build_matrix(examples)), 0.0, 1.0)
        return sparse.csr_matrix(scaled)


def hstack_with_text(text_matrix, structural_matrix) -> sparse.csr_matrix:
    return sparse.hstack([text_matrix, structural_matrix], format="csr")
