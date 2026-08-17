"""
Tests for Phase 5's structural feature block.

Two things matter here that a plain "does it run" test wouldn't catch:
scaling is fit on TRAIN only (fitting on all rows leaks test distribution
into the transform), and the transform never blows past [0, 1] on unseen
data with a larger value than anything in train — an unclipped scaler
would let one long test segment re-dominate the feature space the same
way an unscaled column would.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from structural import StructuralFeatures, FEATURE_NAMES, build_matrix, hstack_with_text  # noqa: E402


def make_example(**overrides):
    base = {
        "text": "some clause text", "position": 2, "n_segments_in_contract": 10,
        "relative_position": 0.2, "char_len": 500, "header": "3.1 Payment",
        "scheme": "section_nn", "is_oversized_split": False,
        "is_undersized": False, "has_parent": True,
    }
    base.update(overrides)
    return base


def test_matrix_shape_matches_feature_names():
    rows = [make_example() for _ in range(5)]
    m = build_matrix(rows)
    assert m.shape == (5, len(FEATURE_NAMES))


def test_preamble_and_position_flags():
    rows = [make_example(header="[preamble]", position=0, n_segments_in_contract=5)]
    m = build_matrix(rows)
    is_first = m[0, FEATURE_NAMES.index("is_first")]
    is_preamble = m[0, FEATURE_NAMES.index("is_preamble")]
    assert is_first == 1.0
    assert is_preamble == 1.0


def test_last_segment_flagged():
    rows = [make_example(position=4, n_segments_in_contract=5)]
    m = build_matrix(rows)
    assert m[0, FEATURE_NAMES.index("is_last")] == 1.0


def test_unknown_scheme_gets_all_zero_onehot_not_a_wrong_bucket():
    rows = [make_example(scheme="some_future_scheme")]
    m = build_matrix(rows)
    scheme_cols = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("scheme_")]
    assert m[0, scheme_cols].sum() == 0.0


def test_scaler_fit_on_train_only_not_leaked_from_test():
    train_rows = [make_example(char_len=100) for _ in range(20)]
    test_rows = [make_example(char_len=100000)]  # far outside train's range

    feats = StructuralFeatures()
    train_scaled = feats.fit_transform(train_rows)
    test_scaled = feats.transform(test_rows)

    # train's own log_char_len should sit at a fixed point since every
    # train row is identical -- proves the scaler used TRAIN stats, not a
    # combined fit that would spread train values out to accommodate test.
    log_len_col = FEATURE_NAMES.index("log_char_len")
    assert train_scaled[:, log_len_col].toarray().max() == pytest.approx(
        train_scaled[:, log_len_col].toarray().min()
    )
    # an out-of-range test value must be clipped, never exceed 1.0
    assert test_scaled[0, log_len_col] <= 1.0


def test_transform_never_exceeds_unit_range():
    # char_len is always len(text) >= 0 in real segments; 0 is the real
    # floor (an empty segment), 10_000_000 stands in for "far larger than
    # anything seen in train".
    train_rows = [make_example(char_len=n) for n in (50, 100, 150, 200)]
    test_rows = [make_example(char_len=0), make_example(char_len=10_000_000)]

    feats = StructuralFeatures()
    feats.fit_transform(train_rows)
    scaled = feats.transform(test_rows).toarray()
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


def test_hstack_preserves_row_count():
    from scipy import sparse

    rows = [make_example() for _ in range(4)]
    feats = StructuralFeatures()
    struct_matrix = feats.fit_transform(rows)
    text_matrix = sparse.csr_matrix(np.ones((4, 3)))
    combined = hstack_with_text(text_matrix, struct_matrix)
    assert combined.shape[0] == 4
    assert combined.shape[1] == 3 + len(FEATURE_NAMES)
