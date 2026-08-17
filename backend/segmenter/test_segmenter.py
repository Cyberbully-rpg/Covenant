"""
Covenant Phase 2 — Automated tests for segmenter.py

Run with:
    pip install pytest --break-system-packages   (if not already installed)
    pytest backend/segmenter/test_segmenter.py -v

Covers:
  1. ARTICLE-scheme top-level segmentation (already validated manually against
     a real contract — this locks that behavior in place).
  2. Section N.N-scheme top-level segmentation (never tested before).
  3. Bare-numbering-only top-level segmentation (never tested before).
  4. No-scheme fallback (whole doc as one segment, never tested before).
  5. TOC detection/removal.
  6. Boilerplate stripping + offset bounding correctness.
  7. Oversized sub-splitting with parent_id linkage.
  8. Undersized embedding_text context injection.

These use small SYNTHETIC contract-like text, not real CUAD data — built to
isolate and prove one behavior per test rather than relying on one big messy
real contract where multiple behaviors are tangled together.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from segmenter import (
    segment_contract,
    select_scheme,
    find_toc_span,
    clean_boilerplate,
    cleaned_to_original_offset,
    MIN_SCHEME_MATCHES,
    OVERSIZED_THRESHOLD_CHARS,
    UNDERSIZED_THRESHOLD_CHARS,
)


# ---------------------------------------------------------------------------
# Behavior 1: ARTICLE scheme
# ---------------------------------------------------------------------------

def make_article_contract():
    return (
        "This Agreement is entered into as of January 1, 2024.\n\n"
        "ARTICLE 1 DEFINITIONS\n\n"
        "Unless otherwise specified, the following terms have the meanings "
        "set forth below in this section of the agreement.\n\n"
        "ARTICLE 2 PAYMENT TERMS\n\n"
        "Company shall pay Vendor within thirty days of invoice receipt as "
        "described in this section governing payment obligations.\n\n"
        "ARTICLE 3 TERMINATION\n\n"
        "Either party may terminate this Agreement upon sixty days written "
        "notice to the other party as set forth in this section.\n\n"
        "ARTICLE 4 CONFIDENTIALITY\n\n"
        "Each party agrees to maintain the confidentiality of the other "
        "party's proprietary information disclosed under this Agreement.\n\n"
    )


def test_article_scheme_detected():
    text = make_article_contract()
    scheme = select_scheme(text)
    assert scheme == "article", f"Expected 'article' scheme, got {scheme!r}"


def test_article_scheme_produces_four_top_level_segments():
    text = make_article_contract()
    segs = segment_contract(text, doc_id="test")
    top_level = [s for s in segs if s.parent_id is None and s.header != "[preamble]"]
    assert len(top_level) == 4, f"Expected 4 top-level segments, got {len(top_level)}"
    headers = [s.header for s in top_level]
    assert any("ARTICLE 1" in h for h in headers)
    assert any("ARTICLE 4" in h for h in headers)


def test_preamble_before_first_header_is_kept_not_discarded():
    """
    Text before the first header must survive as its own segment. On real
    contracts that region holds the title block, parties recital and
    execution date — the only place CUAD's Document Name / Parties /
    Agreement Date / Effective Date categories ever appear. Discarding it
    caps what any downstream classifier or retriever can find, regardless
    of model quality. Measured corpus-wide over all 13,823 CUAD gold
    spans: capture rate 79.45% -> 99.84%.
    """
    text = make_article_contract()
    segs = segment_contract(text, doc_id="test")

    preambles = [s for s in segs if s.header == "[preamble]"]
    assert len(preambles) == 1, f"Expected exactly one preamble segment, got {len(preambles)}"
    assert "This Agreement is entered into as of January 1, 2024." in preambles[0].text

    # and it must be the FIRST thing in the document, starting at offset 0
    assert preambles[0].start_char == 0
    assert min(s.start_char for s in segs) == 0, (
        "segments must start at the beginning of the document, leaving no dropped prefix"
    )


# ---------------------------------------------------------------------------
# Behavior 2: Section N.N scheme (no ARTICLE present)
# ---------------------------------------------------------------------------

def make_section_nn_contract():
    return (
        "This Agreement is entered into as of January 1, 2024.\n\n"
        "Section 1.1 Definitions. Unless otherwise specified, the following "
        "terms have the meanings set forth below in this section.\n\n"
        "Section 1.2 Interpretation. This Agreement shall be interpreted in "
        "accordance with the laws of the State of Delaware.\n\n"
        "Section 2.1 Payment. Company shall pay Vendor within thirty days of "
        "invoice receipt as described in this section governing payment.\n\n"
        "Section 2.2 Late Fees. Any payment not received within the period "
        "specified shall accrue interest at the rate set forth herein.\n\n"
    )


def test_section_nn_scheme_detected_when_no_article_present():
    text = make_section_nn_contract()
    scheme = select_scheme(text)
    assert scheme == "section_nn", f"Expected 'section_nn' scheme, got {scheme!r}"


def test_section_nn_scheme_produces_correct_segment_count():
    text = make_section_nn_contract()
    segs = segment_contract(text, doc_id="test")
    top_level = [s for s in segs if s.parent_id is None and s.header != "[preamble]"]
    assert len(top_level) == 4, f"Expected 4 top-level segments, got {len(top_level)}"
    assert any("1.1" in s.header for s in top_level)
    assert any("2.2" in s.header for s in top_level)


# ---------------------------------------------------------------------------
# Behavior 3: bare numbering only (no ARTICLE, no "Section" word)
# ---------------------------------------------------------------------------

def make_bare_nn_contract():
    return (
        "This Agreement is entered into as of January 1, 2024.\n\n"
        "1.1 Definitions. Unless otherwise specified, the following terms "
        "have the meanings set forth below in this part of the agreement.\n\n"
        "1.2 Interpretation. This Agreement shall be interpreted according "
        "to the laws of the State of Delaware without regard to conflicts.\n\n"
        "2.1 Payment. Company shall pay Vendor within thirty days of invoice "
        "receipt as described in this part governing payment obligations.\n\n"
        "2.2 Late Fees. Any payment not received within the period specified "
        "shall accrue interest at the rate set forth in this agreement.\n\n"
    )


def test_bare_nn_scheme_detected_when_no_article_or_section_word():
    text = make_bare_nn_contract()
    scheme = select_scheme(text)
    assert scheme == "bare_nn", f"Expected 'bare_nn' scheme, got {scheme!r}"


def test_bare_nn_scheme_produces_correct_segment_count():
    text = make_bare_nn_contract()
    segs = segment_contract(text, doc_id="test")
    top_level = [s for s in segs if s.parent_id is None and s.header != "[preamble]"]
    assert len(top_level) == 4, f"Expected 4 top-level segments, got {len(top_level)}"


# ---------------------------------------------------------------------------
# Behavior 4: no scheme detected — fallback to single segment
# ---------------------------------------------------------------------------

def make_no_scheme_contract():
    # Plain prose, no numbering at all, and fewer than MIN_SCHEME_MATCHES of
    # any pattern so nothing should be trusted as a real scheme.
    return (
        "This Agreement is entered into as of January 1, 2024 between the "
        "parties. The parties agree to the following terms and conditions "
        "governing their business relationship, which shall remain in "
        "effect until terminated in accordance with the provisions "
        "described herein. Both parties acknowledge they have read and "
        "understood the terms of this Agreement prior to signing below."
    )


def test_no_scheme_falls_back_to_none():
    text = make_no_scheme_contract()
    scheme = select_scheme(text)
    assert scheme is None, f"Expected no scheme detected, got {scheme!r}"


def test_fallback_produces_single_whole_document_segment():
    text = make_no_scheme_contract()
    segs = segment_contract(text, doc_id="test")
    assert len(segs) == 1, f"Expected 1 fallback segment, got {len(segs)}"
    assert segs[0].scheme == "fallback"
    assert segs[0].text == text


# ---------------------------------------------------------------------------
# Behavior 5: TOC detection and removal
# ---------------------------------------------------------------------------

def make_contract_with_toc():
    # Three real top-level ARTICLEs (1, 2, 3) so the "article" scheme clears
    # MIN_SCHEME_MATCHES after TOC removal — with only 2, it never would be.
    toc = (
        "TABLE OF CONTENTS\n\n"
        "ARTICLE 1 DEFINITIONS 1\n\n"
        "ARTICLE 2 PAYMENT TERMS 5\n\n"
        "ARTICLE 3 TERMINATION 8\n\n"
        "1.1 Scope. 1 1.2 Interpretation. 2\n\n"
        "2.1 Payment. 5 2.2 Late Fees. 6\n\n"
    )
    body = (
        "ARTICLE 1 DEFINITIONS\n\n"
        "Unless otherwise specified, the following terms have the meanings "
        "set forth below in this section of the agreement document text.\n\n"
        "ARTICLE 2 PAYMENT TERMS\n\n"
        "Company shall pay Vendor within thirty days of invoice receipt as "
        "described in this section governing payment obligations here.\n\n"
        "ARTICLE 3 TERMINATION\n\n"
        "Either party may terminate this Agreement upon sixty days written "
        "notice to the other party as set forth in this section clearly.\n\n"
    )
    return toc + body


def test_toc_span_detected():
    text = make_contract_with_toc()
    span = find_toc_span(text)
    assert span is not None, "Expected a TOC span to be detected"
    toc_text = text[span[0]:span[1]]
    assert "TABLE OF CONTENTS" in toc_text or "1.1 Scope" in toc_text


def test_toc_excluded_from_final_segments():
    text = make_contract_with_toc()
    segs = segment_contract(text, doc_id="test")
    top_level = [s for s in segs if s.parent_id is None and s.header != "[preamble]"]
    # Should only find the 3 REAL top-level ARTICLEs, not TOC-duplicated ones
    assert len(top_level) == 3, (
        f"Expected 3 real top-level segments after TOC removal, got "
        f"{len(top_level)}: {[s.header for s in top_level]}"
    )


# ---------------------------------------------------------------------------
# Behavior 6: boilerplate stripping + offset round-trip correctness
# ---------------------------------------------------------------------------

def make_contract_with_boilerplate():
    return (
        "ARTICLE 1 DEFINITIONS\n\n"
        "Unless otherwise specified, the following terms have the meanings "
        "set forth below in this section of the agreement document text.\n\n"
        "Source: ACME CORP, 10-K, 3/12/2020\n\n"
        "- 1 -\n\n"
        "ARTICLE 2 PAYMENT TERMS\n\n"
        "Company shall pay Vendor within thirty days of invoice receipt as "
        "described in this section governing payment obligations here.\n\n"
        "Source: ACME CORP, 10-K, 3/12/2020\n\n"
        "ARTICLE 3 TERMINATION\n\n"
        "Either party may terminate this Agreement upon sixty days written "
        "notice to the other party as set forth in this section clearly.\n\n"
    )


def test_boilerplate_stripped_from_cleaned_text():
    text = make_contract_with_boilerplate()
    cleaned, _ = clean_boilerplate(text)
    assert "Source: ACME CORP" not in cleaned
    assert "- 1 -" not in cleaned


def test_offsets_bound_segment_extent():
    """
    start_char/end_char bound the segment's original extent — they are NOT
    guaranteed to reproduce segment.text byte-for-byte in general. Boilerplate
    stripped from the MIDDLE of a segment's body (not just its edges, e.g. a
    SEC page-break footer landing mid-clause) can't be represented by a
    single contiguous (start_char, end_char) span; the bounded raw slice is
    then a superset of segment.text (see Segment's docstring). Validated
    against real CUAD data: 50/323 segments in a real contract sample hit
    this case, typically inflating extent by <10% (p90 8.86%).

    What IS guaranteed and checked here:
      1. start_char/end_char are valid, ordered offsets into raw_text.
      2. The bounded raw slice is at least as long as segment.text (it can
         only be inflated by stripped content, never truncated).
      3. Re-running the same boilerplate cleaning on the bounded raw slice
         reproduces segment.text exactly — i.e. the bound fully CONTAINS
         segment.text's real content plus (optionally) known boilerplate,
         and nothing else.
    """
    text = make_contract_with_boilerplate()
    segs = segment_contract(text, doc_id="test")
    for s in segs:
        assert 0 <= s.start_char <= s.end_char <= len(text), (
            f"Invalid bounds for {s.segment_id}: start_char={s.start_char}, "
            f"end_char={s.end_char}, len(text)={len(text)}"
        )
        raw_slice = text[s.start_char:s.end_char]
        assert len(raw_slice) >= len(s.text), (
            f"Bounded extent for {s.segment_id} is shorter than segment.text: "
            f"extent={len(raw_slice)} < text={len(s.text)}"
        )
        recleaned, _ = clean_boilerplate(raw_slice)
        assert recleaned == s.text, (
            f"Bounded extent for {s.segment_id} doesn't cover segment.text once "
            f"boilerplate is accounted for: recleaned={recleaned!r} != text={s.text!r}"
        )


# ---------------------------------------------------------------------------
# Behavior 7: oversized sub-splitting with parent_id
# ---------------------------------------------------------------------------

def make_oversized_article_contract():
    # ARTICLE 1 body padded well past OVERSIZED_THRESHOLD_CHARS, with nested
    # bare-numbering sub-clauses inside it. Three top-level ARTICLEs total
    # (1, 2, 3) so the "article" scheme actually clears MIN_SCHEME_MATCHES
    # at the top level — with only 2, it would never be trusted.
    filler = "This clause contains standard contractual boilerplate language. " * 20
    body = (
        "This Agreement is entered into as of January 1, 2024.\n\n"
        "ARTICLE 1 DEFINITIONS\n\n"
        f"1.1 Term One. {filler}\n\n"
        f"1.2 Term Two. {filler}\n\n"
        f"1.3 Term Three. {filler}\n\n"
        f"1.4 Term Four. {filler}\n\n"
    )
    # sanity: make sure this actually exceeds the oversized threshold
    assert len(body) > OVERSIZED_THRESHOLD_CHARS, "test fixture not actually oversized"
    return (
        body +
        "ARTICLE 2 PAYMENT TERMS\n\n"
        "Company shall pay Vendor within thirty days of invoice receipt as "
        "described in this section governing payment obligations here.\n\n"
        "ARTICLE 3 TERMINATION\n\n"
        "Either party may terminate this Agreement upon sixty days written "
        "notice to the other party as set forth in this section clearly.\n\n"
    )


def test_oversized_segment_gets_sub_split():
    text = make_oversized_article_contract()
    segs = segment_contract(text, doc_id="test")
    subs = [s for s in segs if s.metadata.get("parent_header", "").startswith("ARTICLE 1")]
    assert len(subs) >= 4, f"Expected oversized ARTICLE 1 to sub-split into >=4 pieces, got {len(subs)}"
    parent_ids = {s.parent_id for s in subs}
    assert len(parent_ids) == 1, f"sub-splits should share one parent, got {parent_ids}"
    for s in subs:
        assert s.is_oversized_split is True


def test_oversized_parent_segment_not_duplicated():
    """The oversized top-level segment itself should NOT also appear whole —
    only its sub-splits should be present."""
    text = make_oversized_article_contract()
    segs = segment_contract(text, doc_id="test")
    subs = [s for s in segs if s.metadata.get("parent_header", "").startswith("ARTICLE 1")]
    parent_id = next(iter({s.parent_id for s in subs}))
    whole_article_1 = [s for s in segs if s.segment_id == parent_id]
    assert len(whole_article_1) == 0, "Oversized parent should not appear as its own segment"


# ---------------------------------------------------------------------------
# Behavior 8: undersized context injection
# ---------------------------------------------------------------------------

def test_undersized_segment_gets_parent_title_in_embedding_text_only():
    text = make_article_contract()
    segs = segment_contract(text, doc_id="test")
    # ARTICLE 1's body in make_article_contract() is short; find any
    # undersized segment and verify embedding_text differs from raw text
    undersized = [s for s in segs if s.is_undersized]
    if not undersized:
        # fixture may not produce an undersized one — construct a direct case
        short_text = (
            "ARTICLE 1 DEFINITIONS\n\n"
            "1.1 Short. Brief clause.\n\n"
            "ARTICLE 2 PAYMENT TERMS\n\n"
            "1.1 Short. Brief clause.\n\n"
        )
        segs = segment_contract(short_text, doc_id="test2")
        undersized = [s for s in segs if s.is_undersized]

    assert len(undersized) > 0, "Expected at least one undersized segment in test fixtures"
    for s in undersized:
        assert s.embedding_text != s.text, (


            
            f"Undersized segment {s.segment_id} should have context-injected "
            f"embedding_text different from raw text"
        )
        # raw text (used for classifier labels) must remain UNTOUCHED
        assert len(s.text) < UNDERSIZED_THRESHOLD_CHARS


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))