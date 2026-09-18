"""Unit tests for the cleaning pass.

Each test pins a behaviour the prototype got wrong: furniture that hid under
a ``Title`` category, equations shattered into fragments, and the
digit-collapse trap where numbered-but-identical sentences looked like
repeated headers.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.cleaning import (
    clean_documents,
    clean_text,
    dedupe_local,
    detect_furniture,
    reassemble_fragments,
)
from rag.config import get_settings
from helpers import make_doc as _doc


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_normalises_quotes_and_dashes() -> None:
    assert clean_text("“hello” – world—") == '"hello" - world-'


def test_clean_text_collapses_horizontal_whitespace_keeps_newlines() -> None:
    assert clean_text("a\t b   c\n\nd") == "a b c\n\nd"


def test_clean_text_strips_inline_bullets_into_lines() -> None:
    result = clean_text("one. • two. • three.")
    assert "- two." in result


# ---------------------------------------------------------------------------
# furniture
# ---------------------------------------------------------------------------


def test_running_header_detected_by_repetition_not_category() -> None:
    docs = [_doc(f"rlhfbook.com {p}", category="Title", page=p) for p in range(1, 12)]
    furniture = detect_furniture(docs, page_ratio=get_settings().furniture_page_ratio)
    assert furniture, "page-numbered header must be detected despite category=Title"


def test_identical_sentences_with_different_numbers_are_not_furniture() -> None:
    docs = [
        _doc(f"Chapter {p} discusses alignment.", page=p) for p in range(1, 12)
    ]
    furniture = detect_furniture(docs, page_ratio=get_settings().furniture_page_ratio)
    assert not furniture, "digit-only differences in long sentences are content"


def test_furniture_needs_multiple_pages() -> None:
    docs = [_doc("short repeated line", page=1)] * 3
    assert detect_furniture(docs, page_ratio=0.3) == set()


# ---------------------------------------------------------------------------
# fragment reassembly
# ---------------------------------------------------------------------------


def test_fragment_runs_merge_into_one_element() -> None:
    docs = [
        _doc("θ", category="Title", page=5),
        _doc("∞ X", category="Title", page=5),
        _doc("PK−1", category="Title", page=5),
    ]
    out, merged = reassemble_fragments(docs)
    assert merged == 3
    assert len(out) == 1
    assert out[0].metadata["category"] == "Formula"


def test_lone_short_element_stays_separate() -> None:
    docs = [
        _doc("A genuinely long narrative paragraph. " * 5, page=1),
        _doc("Abstract", category="Title", page=1),
        _doc("Another genuinely long narrative paragraph. " * 5, page=2),
    ]
    out, merged = reassemble_fragments(docs)
    assert merged == 0
    assert len(out) == 3


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------


def test_local_dedupe_removes_nearby_duplicates() -> None:
    docs = [_doc("same text"), _doc("Same   text!"), _doc("different")]
    out, removed = dedupe_local(docs, window=10)
    assert removed == 1
    assert len(out) == 2


def test_local_dedupe_keeps_digits_distinct() -> None:
    docs = [_doc("accuracy improved to 71%"), _doc("accuracy improved to 92%")]
    out, removed = dedupe_local(docs, window=10)
    assert removed == 0
    assert len(out) == 2


def test_local_dedupe_window_expiry() -> None:
    # Duplicate outside the window survives; inside it is removed.
    docs = [
        _doc("repeated phrase"),
        *[_doc(f"filler {i}", page=i) for i in range(10)],
        _doc("repeated phrase"),
    ]
    out, removed = dedupe_local(docs, window=3)
    assert removed == 0
    assert len(out) == 12


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------


def test_clean_documents_end_to_end() -> None:
    docs = []
    for page in range(1, 12):
        docs.append(_doc(f"rlhfbook.com {page}", category="Title", page=page))
        docs.append(_doc(f"Chapter {page} body text about alignment.", page=page))

    cleaned = clean_documents(docs)
    assert all("rlhfbook" not in d.page_content for d in cleaned)
    assert sum(1 for d in cleaned if "Chapter" in d.page_content) == 11
