"""
Covenant Phase 2 — Contract Segmenter

Cuts a raw contract's text into labeled segments for two downstream consumers:
  1. Classifier training data (Phase 4+)
  2. RAG chunking (Phase 6+)

Design (locked):
  Step 0 - Strip recurring extraction boilerplate (e.g. SEC EDGAR footers),
           tracking a cleaned->original character offset mapping so CUAD
           answer-span positions can still be resolved later (Phase 3 eval).
  Step 1 - Detect and remove table-of-contents blocks. A header candidate is
           TOC if another header candidate follows within a short character
           window with no sentence-ending prose in between.
  Step 2 - Pick ONE top-level numbering scheme for the whole document, in
           priority order: ARTICLE -> Section N.N -> bare N.N -> fallback
           (single segment). A scheme is trusted only if it has >=3 matches
           after TOC removal. Only the winning scheme cuts top-level segments.
  Step 3 - Any top-level segment longer than OVERSIZED_THRESHOLD_CHARS is
           re-cut internally using whatever numbering pattern is nested
           inside it. Sub-segments carry parent_id back to the top-level
           segment.
  Step 4 - Any segment shorter than UNDERSIZED_THRESHOLD_CHARS is flagged;
           its embedding_text gets the parent section title prepended.
           Raw `text` is left untouched (used for classifier labels).

Schema is intentionally loose (per Phase 2 decision) - locked for real once
Phase 3's eval harness defines what it needs for span-overlap matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Thresholds (grounded in measurement across 510 CUAD contracts, see chat log)
# ---------------------------------------------------------------------------

MIN_SCHEME_MATCHES = 3          # matches needed to trust a numbering scheme as real
OVERSIZED_THRESHOLD_CHARS = 4500   # segments longer than this get sub-split
UNDERSIZED_THRESHOLD_CHARS = 175   # segments shorter than this get context injection
TOC_HEADER_LOOKAHEAD_CHARS = 400   # window to check "is another header right after this one"
TOC_MIN_PROSE_CHARS = 120          # min non-header chars between headers to NOT count as TOC

# Safety net for documents with no detectable numbering scheme at all. Without
# this, an unstructured contract becomes ONE segment spanning the whole
# document, which makes retrieval meaningless — Phase 3's harness measured a
# random retriever scoring 0.74 hit-rate purely because "the chunk" was the
# entire contract. A contract shorter than the floor is left whole: a 1KB
# filing agreement genuinely is a single unit, and chopping it adds nothing.
FALLBACK_WINDOW_CHARS = 3000       # target size when window-splitting a fallback doc
FALLBACK_MIN_DOC_CHARS = 6000      # below this, an unstructured doc stays one segment


# ---------------------------------------------------------------------------
# Patterns for each numbering scheme
# ---------------------------------------------------------------------------

PATTERNS = {
    # Each pattern is anchored on blank-line OR start-of-string: a header is
    # often the first thing in the text being scanned (top-level cut on a
    # TOC-stripped document, or a nested header at the very start of an
    # oversized segment's body), and \n\n alone would never match at
    # position 0.
    #
    # ARTICLE headers are reliably followed by a blank-line-delimited title.
    "article": re.compile(r"(?:\A|\n\n)(ARTICLE\s+[IVXLCDM0-9]+)\s+([A-Z][A-Za-z0-9 ,/\-']{2,80}?)(?=\n)", re.IGNORECASE),
    # "Section 4.2 Title." style — same blank-line + short-title-ending-in-period
    # discriminator as bare_nn, just with the word "Section" present. Real headers
    # of this style are rare in practice (most contracts use bare numbering once
    # inside a section), but the pattern is kept distinct from bare_nn so a
    # document that consistently uses the word "Section" in headers is still
    # detected correctly rather than silently falling through to bare_nn.
    "section_nn": re.compile(r"(?:\A|\n\n)Section\s+(\d+(?:\.\d+){1,3})\s+([A-Z][A-Za-z0-9 ,/\-']{2,60}?)\.\s", re.IGNORECASE),
    # Bare numbering headers: blank-line-preceded, followed by EITHER
    #   (a) a short Title-Case label ending in a period
    #       e.g. "2.1 Joint Governance Committee. "
    #   (b) a quoted or redacted defined term immediately followed by prose
    #       e.g. '1.1 "AbbVie" has the meaning set forth...' or
    #            '1.2 [***] has the meaning set forth...'
    # (b) is common in definitions-style ARTICLEs and has no title-ending
    # period before the prose begins. Both require the blank-line anchor,
    # which is what separates a real header from a mid-sentence cross-
    # reference like "as set forth in Section 12.2.1." (no preceding blank
    # line, no standalone title).
    "bare_nn": re.compile(
        r'(?:\A|\n\n)(\d+(?:\.\d+){1,3})\s+'
        r'(?:([A-Z][A-Za-z0-9 ,/\-\']{2,60}?)\.\s'
        r'|(["\u201c][^"\u201d\n]{1,60}["\u201d]|\[\*\*\*\]))'
    ),

    # --- Inline variants (lower priority than everything above) --------------
    # Measured against 120 CUAD contracts: only 23% expose their headers after
    # a blank line. The rest run headers INLINE inside flowing text, e.g.
    # "...prevail. 1.2 If the provisions of the agreement are inconsistent..."
    # \u2014 real headers, just not blank-line-delimited. These variants anchor on
    # a preceding sentence-end/whitespace instead of "\n\n".
    #
    # They are deliberately last in the cascade: anchoring on whitespace rather
    # than a blank line is weaker evidence, and can catch a mid-sentence
    # cross-reference ("as set forth in Section 12.2"). The 3+ match threshold
    # plus requiring a Title-Case label terminated by "." or ":" is what keeps
    # the false-positive rate tolerable. A document whose headers ARE
    # blank-line-delimited never reaches these, because a stricter scheme wins
    # first.
    "section_inline": re.compile(
        r'(?:(?<=[.\s])|\A)(SECTION\s+\d+(?:\.\d+)*)\b', re.IGNORECASE
    ),
    "article_inline": re.compile(
        r'(?:(?<=[.\s])|\A)(ARTICLE\s+[IVXLCDM0-9]+)\b', re.IGNORECASE
    ),
    "bare_nn_inline": re.compile(
        r'(?:(?<=[.\s])|\A)(\d+(?:\.\d+){1,3})\s+([A-Z][A-Za-z0-9 ,/\-\']{2,60}?)[\.\:]\s'
    ),
    "int_dot_inline": re.compile(
        r'(?:(?<=[.\s])|\A)(\d{1,2})\.\s+([A-Z][A-Za-z0-9 ,/\-\']{2,60}?)[\.\:]\s'
    ),
}

# Strict (blank-line-anchored) schemes are tried first; inline variants only
# get a look once every stricter scheme has failed to clear MIN_SCHEME_MATCHES.
SCHEME_PRIORITY = [
    "article", "section_nn", "bare_nn",
    "section_inline", "article_inline", "bare_nn_inline", "int_dot_inline",
]

# Boilerplate: SEC EDGAR-style extraction footer, e.g.
# "Source: HARPOON THERAPEUTICS, INC., 10-K, 3/12/2020"
# and bare page-number markers like "- 12 -"
BOILERPLATE_PATTERNS = [
    re.compile(r"Source:\s*[A-Z0-9 ,.\-&]+?,\s*(?:10-K|10-Q|8-K|S-1|EX-[\d.]+)[^\n]*", re.IGNORECASE),
    re.compile(r"(?m)^\s*-\s*\d+\s*-\s*$"),
]

SENTENCE_END = re.compile(r"[.;:][\"')]?\s")


def _header_label(m: re.Match) -> str:
    """Builds a readable header label from a match's capture groups."""
    groups = [g for g in m.groups() if g]
    return " ".join(g.strip() for g in groups) if groups else m.group().strip()


@dataclass
class Segment:
    """
    start_char/end_char bound the segment's original extent; for segments
    with mid-body boilerplate removed, this is a superset — the range may
    contain stripped content not present in segment.text. Empirically
    (50-segment sample), this inflation is typically <10% of extent (p90
    8.86%) and driven by page-break boilerplate; segments spanning multiple
    page breaks accumulate proportionally more stripped content. Exact
    sub-span exclusion is deferred until Phase 3 defines whether it's needed.
    """
    segment_id: str
    text: str
    embedding_text: str
    start_char: int          # offset into ORIGINAL (pre-clean) contract text
    end_char: int
    scheme: str               # "article" | "section_nn" | "bare_nn" | "fallback"
    header: str
    parent_id: str | None = None
    is_oversized_split: bool = False
    is_undersized: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 0 — boilerplate cleaning with offset tracking
# ---------------------------------------------------------------------------

# offset_map entries are (cleaned_start, cleaned_end, original_start), sorted
# by cleaned_start. original_start is None for a synthetic separator (see
# clean_boilerplate) — those characters don't correspond to any raw_text
# position at all.
OffsetMap = list[tuple[int, int, int | None]]


def clean_boilerplate(raw_text: str) -> tuple[str, OffsetMap]:
    """
    Strips recurring extraction boilerplate from raw_text.

    Returns (cleaned_text, offset_map) describing contiguous kept spans,
    used to translate a position in cleaned_text back to raw_text.
    """
    cuts = []  # (start, end) spans in ORIGINAL text to remove
    for pattern in BOILERPLATE_PATTERNS:
        for m in pattern.finditer(raw_text):
            cuts.append((m.start(), m.end()))

    cuts.sort()
    merged = []
    for start, end in cuts:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Expand each cut to also swallow adjacent blank-line whitespace, so
    # what's kept on either side always ends/starts at real (non-whitespace)
    # content — never mid-blank-line. Left as-is, a lone leftover newline
    # split across a cut boundary (e.g. two boilerplate lines separated by
    # a blank line) fragments the offset map in a way that makes a later
    # segment boundary impossible to resolve to a single contiguous
    # original slice.
    expanded = []
    for start, end in merged:
        while start > 0 and raw_text[start - 1] in " \t\n":
            start -= 1
        while end < len(raw_text) and raw_text[end] in " \t\n":
            end += 1
        if expanded and start <= expanded[-1][1]:
            expanded[-1] = (expanded[-1][0], max(expanded[-1][1], end))
        else:
            expanded.append((start, end))
    merged = expanded

    cleaned_parts = []
    offset_map: OffsetMap = []
    cursor = 0
    cleaned_cursor = 0
    for start, end in merged:
        if start > cursor:
            kept = raw_text[cursor:start]
            cleaned_parts.append(kept)
            offset_map.append((cleaned_cursor, cleaned_cursor + len(kept), cursor))
            cleaned_cursor += len(kept)
        # A cut swallowed its entire surrounding blank line (see expansion
        # above), so a header pattern immediately following it would no
        # longer see the "\n\n" it needs. Put back a canonical separator —
        # but only between two real kept regions; a cut touching the very
        # start/end of the document has no content on that side to
        # separate, and inserting one there would just add stray leading/
        # trailing whitespace.
        if start > 0 and end < len(raw_text):
            cleaned_parts.append("\n\n")
            offset_map.append((cleaned_cursor, cleaned_cursor + 2, None))
            cleaned_cursor += 2
        cursor = end
    if cursor < len(raw_text):
        kept = raw_text[cursor:]
        cleaned_parts.append(kept)
        offset_map.append((cleaned_cursor, cleaned_cursor + len(kept), cursor))

    cleaned_text = "".join(cleaned_parts)
    return cleaned_text, offset_map


def cleaned_to_original_offset(pos: int, offset_map: OffsetMap) -> int:
    """
    START-style: translate a cleaned position to raw_text. A position
    inside (or exactly at the start of) a synthetic gap — see
    clean_boilerplate — has no real backing, so it projects forward to
    wherever real content next resumes.
    """
    for c_start, c_end, o_start in offset_map:
        if o_start is not None and c_start <= pos < c_end:
            return o_start + (pos - c_start)
    for c_start, c_end, o_start in offset_map:
        if o_start is not None and pos <= c_start:
            return o_start
    for c_start, c_end, o_start in reversed(offset_map):
        if o_start is not None:
            return o_start + (c_end - c_start)
    return pos


def cleaned_to_original_end_offset(pos: int, offset_map: OffsetMap) -> int:
    """
    END-style (exclusive): translate a cleaned position to raw_text. A
    boundary position can sit exactly where one kept span ends and the
    next begins (with a removed span between their original positions, or
    a synthetic gap in between) — projecting forward like the start-style
    lookup would pull the removed span back into the segment, so this
    instead projects backward to wherever real content last ended.
    """
    for c_start, c_end, o_start in offset_map:
        if o_start is not None and c_start < pos <= c_end:
            return o_start + (pos - c_start)
    prev_end = 0
    for c_start, c_end, o_start in offset_map:
        if o_start is None:
            continue
        if c_end <= pos:
            prev_end = o_start + (c_end - c_start)
        else:
            break
    return prev_end


def skip_leading_gap(offset_map: OffsetMap, pos: int) -> int:
    """
    If pos falls inside a synthetic gap (see clean_boilerplate), advances it
    to where real content resumes. A segment's owned text must never start
    on a gap — those characters exist only to keep a header pattern
    recognizable across a fully-swallowed cut and have no raw_text backing,
    so a segment starting on one could never round-trip to a raw_text slice.
    """
    for c_start, c_end, o_start in offset_map:
        if o_start is None and c_start <= pos < c_end:
            return c_end
    return pos


def remove_span(text: str, offset_map: OffsetMap, start: int, end: int) -> tuple[str, OffsetMap]:
    """
    Removes text[start:end] and returns the shortened text plus offset_map
    remapped to match. Entries that straddle the removed span are split into
    their kept left/right portions rather than dropped, since a straddling
    entry is the common case (e.g. a TOC span with no boilerplate match
    inside it lives entirely within one contiguous offset_map entry).
    """
    removed_len = end - start
    new_map: OffsetMap = []
    for c_start, c_end, o_start in offset_map:
        if c_end <= start:
            new_map.append((c_start, c_end, o_start))
        elif c_start >= end:
            new_map.append((c_start - removed_len, c_end - removed_len, o_start))
        else:
            if c_start < start:
                new_map.append((c_start, start, o_start))
            if c_end > end:
                seg_start = max(c_start, end)
                new_o_start = o_start if o_start is None else o_start + (seg_start - c_start)
                new_map.append((seg_start - removed_len, c_end - removed_len, new_o_start))
    return text[:start] + text[end:], new_map


# ---------------------------------------------------------------------------
# Step 1 — TOC detection and removal
# ---------------------------------------------------------------------------

# A single packed TOC entry: "N.N Title text. <page_num> " (page number is
# what distinguishes this from a real header, which is never followed
# immediately by a bare number).
TOC_ENTRY = re.compile(
    r"(?:ARTICLE\s+[IVXLCDM0-9]+\s+[A-Z][^\n]{0,80}?|\d+(?:\.\d+){1,3}\s+[A-Z][^\n]{2,80}?\.)\s+\d{1,4}\s+",
    re.IGNORECASE,
)
TOC_ANCHOR = re.compile(r"TABLE\s+OF\s+CONTENTS", re.IGNORECASE)


def find_toc_span(text: str) -> tuple[int, int] | None:
    """
    Finds the leading table-of-contents block and returns its (start, end)
    span to exclude from segmentation, or None if no TOC detected.

    TOC entries have a distinctive, unambiguous shape that body text never
    produces: a numbered title immediately followed by a bare 1-4 digit page
    number, e.g. "13.2 Change in Control of Harpoon. 68 13.3 Export...".
    Real section headers in body text are never followed by a bare number.

    We scan for a run of TOC_ENTRY matches (allowing small gaps between
    consecutive matches, since ARTICLE-level TOC lines sometimes have a
    line break where sub-entries don't) and take the span from the first
    match to the end of the last match in that run.
    """
    entries = list(TOC_ENTRY.finditer(text))
    if not entries:
        return None

    anchor_start = None
    am = TOC_ANCHOR.search(text)
    if am:
        anchor_start = am.start()

    # keep only entries reasonably close to the anchor / start of document,
    # then find the longest contiguous run (small gaps tolerated)
    MAX_GAP = 60
    run_start = entries[0].start()
    run_end = entries[0].end()
    for prev, curr in zip(entries, entries[1:]):
        gap = curr.start() - prev.end()
        if gap <= MAX_GAP:
            run_end = curr.end()
        else:
            break

    toc_start = anchor_start if anchor_start is not None and anchor_start < run_start else run_start
    return (toc_start, run_end)


# ---------------------------------------------------------------------------
# Step 2 — top-level scheme selection and cutting
# ---------------------------------------------------------------------------

def select_scheme(text: str) -> str | None:
    """Returns the winning scheme name, or None if no scheme clears the threshold."""
    for scheme in SCHEME_PRIORITY:
        matches = list(PATTERNS[scheme].finditer(text))
        if len(matches) >= MIN_SCHEME_MATCHES:
            return scheme
    return None


def cut_top_level(text: str, scheme: str) -> list[tuple[int, int, str]]:
    """
    Returns list of (start, end, header_text) for top-level segments using
    the given scheme. Segments run from one header to the start of the next
    (or end of document).

    The text BEFORE the first header is emitted as a "[preamble]" segment
    rather than discarded. On real contracts that region is the title
    block, parties recital and execution date — the only place several
    CUAD categories (Document Name, Parties, Agreement Date, Effective
    Date) ever appear. Dropping it silently caps what any downstream
    classifier or retriever can possibly find: measured across all 13,823
    CUAD gold spans, gold-span capture rate is 79.45% without it and
    99.84% with it.
    """
    pattern = PATTERNS[scheme]
    matches = list(pattern.finditer(text))
    if not matches:
        return [(0, len(text), "")]

    boundaries = [m.start() for m in matches]
    headers = [_header_label(m) for m in matches]
    boundaries.append(len(text))

    segments = []
    if boundaries[0] > 0:
        segments.append((0, boundaries[0], "[preamble]"))
    for i in range(len(matches)):
        start = boundaries[i]
        end = boundaries[i + 1]
        segments.append((start, end, headers[i]))
    return segments


def window_split_fallback(text: str) -> list[tuple[int, int, str]]:
    """
    Splits an unstructured document into ~FALLBACK_WINDOW_CHARS windows,
    breaking on paragraph boundaries where possible and sentence boundaries
    otherwise, so a fallback document still yields usable retrieval units.

    Returns (start, end, header) triples like cut_top_level. Headers are
    positional ("[window 2]") since there are no real headers to name —
    the whole reason this document hit fallback.
    """
    if len(text) < FALLBACK_MIN_DOC_CHARS:
        return [(0, len(text), "")]

    # candidate break points, best first: paragraph breaks, then line breaks,
    # then sentence ends
    windows = []
    start = 0
    idx = 0
    while start < len(text):
        target = start + FALLBACK_WINDOW_CHARS
        if target >= len(text):
            windows.append((start, len(text), f"[window {idx}]"))
            break

        # look for a clean break within the back quarter of the window
        search_from = start + (FALLBACK_WINDOW_CHARS * 3) // 4
        cut = -1
        for sep in ("\n\n", "\n", ". "):
            found = text.rfind(sep, search_from, target)
            if found != -1:
                cut = found + len(sep)
                break
        if cut == -1:
            cut = target  # no clean break available; hard-cut

        windows.append((start, cut, f"[window {idx}]"))
        start = cut
        idx += 1

    return windows


# ---------------------------------------------------------------------------
# Step 3 — oversized sub-splitting
# ---------------------------------------------------------------------------

def sub_split_oversized(segment_text: str) -> list[tuple[int, int, str]]:
    """
    Attempts to sub-split an oversized segment using nested numbering found
    within it. Tries nested numbering schemes in cascade order; if none
    clears the threshold, window-splits instead of returning the oversized
    segment whole — an un-split 100KB segment is useless as a retrieval unit
    regardless of why its numbering couldn't be parsed.

    Returns list of (start, end, header) offsets RELATIVE to segment_text.
    """
    # article-level numbering shouldn't recur inside an article, so it's
    # excluded here; everything else is fair game as nested structure.
    for scheme in ("section_nn", "bare_nn", "section_inline",
                   "bare_nn_inline", "int_dot_inline"):
        matches = list(PATTERNS[scheme].finditer(segment_text))
        if len(matches) >= MIN_SCHEME_MATCHES:
            boundaries = [m.start() for m in matches]
            headers = [_header_label(m) for m in matches]
            boundaries.append(len(segment_text))
            sub_segments = []
            # keep any text before the first nested header as part of segment intro
            if boundaries[0] > 0:
                sub_segments.append((0, boundaries[0], "[preamble]"))
            for i in range(len(matches)):
                sub_segments.append((boundaries[i], boundaries[i + 1], headers[i]))

            # A sub-segment can itself still be oversized (a nested clause with
            # no further numbering inside it). Window-split those rather than
            # leaving a 25KB "chunk" in the output — sub-splitting one level
            # deep isn't enough on real contracts.
            expanded = []
            for s_start, s_end, s_header in sub_segments:
                if (s_end - s_start) <= OVERSIZED_THRESHOLD_CHARS:
                    expanded.append((s_start, s_end, s_header))
                    continue
                for w_start, w_end, w_header in window_split_fallback(segment_text[s_start:s_end]):
                    expanded.append((s_start + w_start, s_start + w_end,
                                     f"{s_header} {w_header}".strip()))
            return expanded

    return window_split_fallback(segment_text)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def segment_contract(raw_text: str, doc_id: str = "doc") -> list[Segment]:
    cleaned_text, offset_map = clean_boilerplate(raw_text)

    toc_span = find_toc_span(cleaned_text)
    if toc_span:
        cleaned_text, offset_map = remove_span(cleaned_text, offset_map, toc_span[0], toc_span[1])

    scheme = select_scheme(cleaned_text)
    segments: list[Segment] = []

    if scheme is None:
        # fallback: no numbering scheme detected. Window-split rather than
        # emitting the whole document as one segment (see
        # FALLBACK_WINDOW_CHARS); short documents still come back whole.
        for idx, (start, end, header) in enumerate(window_split_fallback(cleaned_text)):
            start = skip_leading_gap(offset_map, start)
            win_text = cleaned_text[start:end]
            undersized = len(win_text) < UNDERSIZED_THRESHOLD_CHARS
            segments.append(Segment(
                segment_id=f"{doc_id}_seg{idx}",
                text=win_text,
                embedding_text=f"{header}\n{win_text}" if undersized else win_text,
                start_char=cleaned_to_original_offset(start, offset_map),
                end_char=cleaned_to_original_end_offset(end, offset_map),
                scheme="fallback",
                header=header,
                is_undersized=undersized,
            ))
        return segments

    top_level = cut_top_level(cleaned_text, scheme)

    for idx, (start, end, header) in enumerate(top_level):
        start = skip_leading_gap(offset_map, start)
        seg_text = cleaned_text[start:end]
        top_id = f"{doc_id}_top{idx}"

        if len(seg_text) > OVERSIZED_THRESHOLD_CHARS:
            sub_spans = sub_split_oversized(seg_text)
            if len(sub_spans) > 1:
                for sub_idx, (s_start, s_end, s_header) in enumerate(sub_spans):
                    abs_start = start + s_start
                    abs_end = start + s_end
                    real_abs_start = skip_leading_gap(offset_map, abs_start)
                    if real_abs_start != abs_start:
                        s_start += real_abs_start - abs_start
                        abs_start = real_abs_start
                    sub_text = seg_text[s_start:s_end]
                    undersized = len(sub_text) < UNDERSIZED_THRESHOLD_CHARS
                    embed_text = f"{header} > {s_header}\n{sub_text}" if undersized else sub_text
                    segments.append(Segment(
                        segment_id=f"{top_id}_sub{sub_idx}",
                        text=sub_text,
                        embedding_text=embed_text,
                        start_char=cleaned_to_original_offset(abs_start, offset_map),
                        end_char=cleaned_to_original_end_offset(abs_end, offset_map),
                        scheme=scheme,
                        header=s_header,
                        parent_id=top_id,
                        is_oversized_split=True,
                        is_undersized=undersized,
                        metadata={"parent_header": header},
                    ))
                continue  # oversized segment handled via sub-splits, skip adding the parent itself

        # not oversized (or no internal pattern found) — keep as single segment
        undersized = len(seg_text) < UNDERSIZED_THRESHOLD_CHARS
        embed_text = f"{header}\n{seg_text}" if undersized else seg_text
        segments.append(Segment(
            segment_id=top_id,
            text=seg_text,
            embedding_text=embed_text,
            start_char=cleaned_to_original_offset(start, offset_map),
            end_char=cleaned_to_original_end_offset(end, offset_map),
            scheme=scheme,
            header=header,
            parent_id=None,
            is_undersized=undersized,
        ))

    return segments


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/cuad/harpoon_sample.txt"
    with open(path) as f:
        raw = f.read()
    segs = segment_contract(raw, doc_id="harpoon")
    print(f"Produced {len(segs)} segments\n")
    for s in segs[:15]:
        print(f"[{s.segment_id}] scheme={s.scheme} parent={s.parent_id} "
              f"len={len(s.text)} undersized={s.is_undersized} header={s.header[:60]!r}")