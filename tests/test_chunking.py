"""Unit tests for chunking.

Uses a fake tokenizer (monkeypatched) so the tests exercise chunking logic
without downloading model tokenizers.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.chunking import chunk_documents, token_length
from rag.config import get_settings
from helpers import make_doc as _doc


@pytest.fixture(autouse=True)
def fast_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Estimate tokens as words -- fast, deterministic, no downloads."""
    monkeypatch.setattr(
        "rag.chunking.token_length", lambda text, model_name=None: len(text.split())
    )


# Bodies long enough to clear min_chunk_tokens (default 24 word-tokens in
# these tests, where token_length is monkeypatched to word count).
_BODY = (
    "Reinforcement learning from human feedback trains models to behave in "
    "ways humans judge preferable, using preference comparisons as reward "
    "signal across many training rounds and evaluation steps."
)


def test_small_element_becomes_single_chunk() -> None:
    chunks = chunk_documents([_doc(_BODY)])
    assert len(chunks) == 1
    assert chunks[0].page_content == _BODY


def test_long_element_splits_within_budget() -> None:
    settings = get_settings()
    long_text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_documents([_doc(long_text)])
    assert len(chunks) > 1
    assert all(
        c.metadata["token_count"] <= settings.chunk_size + 10 for c in chunks
    )  # splitter may slightly overshoot on the overlap


def test_table_never_split() -> None:
    big_table_html = "<table>" + "<tr><td>cell</td></tr>" * 500 + "</table>"
    doc = Document(
        page_content=big_table_html,
        metadata={
            "category": "Table",
            "page_number": 7,
            "source": "test.pdf",
            "text_as_html": big_table_html,
        },
    )
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].metadata["is_table"] is True
    assert chunks[0].metadata["page_start"] == 7


def test_table_chunk_embeds_the_html_rendering() -> None:
    # Unstructured gives a table both a flattened text rendering and an HTML
    # one. Only the HTML preserves cell structure, so it must be the text that
    # reaches the chunk -- computing it and then discarding it silently
    # degrades every table to the flattened form.
    html = "<table><tr><td>Model</td><td>Score</td></tr></table>"
    doc = Document(
        page_content="Model Score",  # the flattened fallback
        metadata={
            "category": "Table",
            "page_number": 3,
            "source": "test.pdf",
            "text_as_html": html,
        },
    )
    chunks = chunk_documents([doc])
    assert chunks[0].page_content == html


def test_table_falls_back_to_plain_text_without_html() -> None:
    doc = Document(
        page_content="Model Score",
        metadata={"category": "Table", "page_number": 3, "source": "test.pdf"},
    )
    chunks = chunk_documents([doc])
    assert chunks[0].page_content == "Model Score"


def test_section_breadcrumb_inherited() -> None:
    # No element in this document reports a heading depth, so it is flat and
    # its headings are siblings: the second replaces the first rather than
    # nesting under it. See the freeze test below for why.
    docs = [
        _doc("Chapter 11", category="Title", page=1),
        _doc("Policy Gradients", category="Title", page=1),
        _doc(f"GRPO is a group-based method. {_BODY}", page=2),
    ]
    chunks = chunk_documents(docs)
    assert chunks[-1].metadata["section_path"] == "Policy Gradients"


def test_flat_document_headings_do_not_freeze_the_breadcrumb() -> None:
    """Depth-less headings must not accumulate into a frozen prefix.

    Guessing a depth for every unlabelled heading grows the stack until it
    passes _MAX_SECTION_DEPTH; every heading after that is discarded instead
    of recorded, so the breadcrumb never changes again. On the 241-page RLHF
    PDF that left 1812 of 1816 chunks sharing one breadcrumb taken from page
    5 -- including the chapter on direct alignment algorithms, labelled
    "Instruction Fine-Tuning".
    """
    docs = [_doc(f"Heading {i}", category="Title", page=i) for i in range(1, 8)]
    docs.append(_doc(f"Body text under the last heading. {_BODY}", page=8))
    chunks = chunk_documents(docs)

    assert chunks[-1].metadata["section_path"] == "Heading 7"


def test_a_document_that_reports_any_depth_is_not_treated_as_flat() -> None:
    # Flatness is a property of the whole document, decided once. One element
    # reporting a depth is enough to keep the nesting logic, so a document
    # that sizes only some of its headings still builds a hierarchy -- which
    # is the case the depth-guessing fallback exists for.
    docs = [
        _doc("Chapter 1", category="Title", category_depth=0, page=1),
        _doc("Unsized subheading", category="Title", page=1),
        _doc(f"Body under the subheading. {_BODY}", page=2),
    ]
    chunks = chunk_documents(docs)
    assert chunks[-1].metadata["section_path"] == "Chapter 1 > Unsized subheading"


def test_top_level_heading_depth_resets_the_breadcrumb() -> None:
    # Unstructured reports heading depth 0-based where it reports it at all,
    # so category_depth=0 is a top-level heading -- not "unknown". Treating it
    # as unknown nests the new chapter under the previous one and cites the
    # wrong parent section.
    docs = [
        _doc("Chapter 1", category="Title", category_depth=0, page=1),
        _doc(f"First chapter body. {_BODY}", page=1),
        _doc("Chapter 2", category="Title", category_depth=0, page=2),
        _doc(f"Second chapter body. {_BODY}", page=2),
    ]
    chunks = chunk_documents(docs)
    # The heading-only chunks fall below min_chunk_tokens and are dropped, so
    # the surviving bodies are the observable part of the breadcrumb.
    assert len(chunks) == 2
    assert chunks[0].metadata["section_path"] == "Chapter 1"
    assert chunks[1].metadata["section_path"] == "Chapter 2"


def test_nested_heading_depth_builds_a_hierarchy() -> None:
    docs = [
        _doc("Chapter 1", category="Title", category_depth=0, page=1),
        _doc("Group Relative Policy Optimization", category="Title", category_depth=1, page=1),
        _doc(f"Nested body text. {_BODY}", page=1),
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].metadata["section_path"] == (
        "Chapter 1 > Group Relative Policy Optimization"
    )


def test_chunk_ids_stable_across_runs() -> None:
    # Same text at the same position must produce the same id, so an
    # unchanged re-ingest is a true upsert.
    first = chunk_documents([_doc(_BODY)])
    second = chunk_documents([_doc(_BODY)])
    assert first[0].metadata["chunk_id"] == second[0].metadata["chunk_id"]


def test_repeated_paragraph_gets_distinct_chunk_ids() -> None:
    # Identical text at two positions (beyond the dedupe window) must not
    # collide on one point id -- that would silently drop one copy and
    # tangle the neighbour links. The filler is long enough to force the two
    # copies into separate chunks, which is where a collision would show up.
    filler = " ".join(f"filler{i}" for i in range(400))
    docs = [_doc(_BODY), _doc(filler), _doc(_BODY)]
    chunks = chunk_documents(docs)
    ids = [c.metadata["chunk_id"] for c in chunks if c.page_content == _BODY]
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_neighbour_links_set() -> None:
    # Each paragraph is large enough that packing cannot merge two of them,
    # so the links below join five distinct chunks.
    filler = " ".join(f"word{i}" for i in range(300))
    docs = [_doc(f"Paragraph {i}. {filler}") for i in range(5)]
    chunks = chunk_documents(docs)
    assert len(chunks) == 5
    assert chunks[0].metadata.get("prev_id") is None
    assert chunks[0].metadata["next_id"] == chunks[1].metadata["chunk_id"]
    assert chunks[-1].metadata.get("next_id") is None
    assert chunks[-1].metadata["prev_id"] == chunks[-2].metadata["chunk_id"]


def test_fragment_chunks_dropped() -> None:
    docs = [_doc("yes")]  # one word: below min_chunk_tokens
    assert chunk_documents(docs) == []


def test_page_span_recorded() -> None:
    doc = _doc(f"On page nine: {_BODY}", page=9)
    chunks = chunk_documents([doc])
    assert chunks[0].metadata["page_start"] == 9
    assert chunks[0].metadata["page_end"] == 9


# ---------------------------------------------------------------------------
# packing into full-size chunks
# ---------------------------------------------------------------------------


def test_consecutive_small_elements_pack_into_one_chunk() -> None:
    # The fast PDF strategy emits sentence-sized elements. Emitting one chunk
    # each collapsed the effective chunk size to a sentence -- a 241-page book
    # produced 1816 chunks at a median of 57 tokens against a configured 384
    # -- so top_k=5 handed the answer stage five clauses and it reported,
    # correctly, that the context did not cover the question.
    docs = [
        _doc(f"Fragment {i} of the passage explains the method in detail.", page=1)
        for i in range(3)
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content.count("Fragment") == 3


def test_packing_stops_at_the_chunk_budget() -> None:
    settings = get_settings()
    docs = [_doc(" ".join(f"word{j}" for j in range(100))) for _ in range(10)]
    chunks = chunk_documents(docs)
    # Three 100-word elements fit under 384; the fourth would not.
    assert len(chunks) == 4
    assert all(c.metadata["token_count"] <= settings.chunk_size for c in chunks)


def test_packing_stops_at_a_section_boundary() -> None:
    docs = [
        _doc("Chapter 1", category="Title", category_depth=0, page=1),
        _doc(f"The first chapter body text. {_BODY}", page=1),
        _doc("Chapter 2", category="Title", category_depth=0, page=2),
        _doc(f"The second chapter body text. {_BODY}", page=2),
    ]
    chunks = chunk_documents(docs)
    assert [c.metadata["section_path"] for c in chunks] == ["Chapter 1", "Chapter 2"]
    # The heading opens the text it introduces, as its breadcrumb line.
    assert chunks[0].page_content.startswith("Chapter 1\n\n")


def test_packed_chunk_reports_the_pages_it_covers() -> None:
    docs = [
        _doc(f"Text starting on this page. {_BODY}", page=12),
        _doc(f"Text continuing overleaf. {_BODY}", page=13),
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].metadata["page_start"] == 12
    assert chunks[0].metadata["page_end"] == 13


def test_table_is_never_packed_with_its_neighbours() -> None:
    table = Document(
        page_content="Model Score",
        metadata={
            "category": "Table",
            "page_number": 3,
            "source": "test.pdf",
            "text_as_html": "<table><tr><td>Model</td><td>Score</td></tr></table>",
        },
    )
    docs = [
        _doc(f"Prose before the table. {_BODY}", page=3),
        table,
        _doc(f"Prose after the table. {_BODY}", page=3),
    ]
    chunks = chunk_documents(docs)
    assert [c.metadata["is_table"] for c in chunks] == [False, True, False]


# ---------------------------------------------------------------------------
# the section breadcrumb in the chunk text
# ---------------------------------------------------------------------------


def test_chunk_text_opens_with_its_section_breadcrumb() -> None:
    # The breadcrumb has to be *text*, not metadata: page_content is the only
    # thing the retrieval stack sees. A passage reading "the probability that i
    # is preferred to j is the ratio of their strengths" contains none of the
    # words "Bradley-Terry", so with the heading left in metadata nothing in
    # the dense embedding, the BM25 index or the reranker could ever connect it
    # to a question about the Bradley-Terry equation.
    docs = [
        _doc("The Bradley-Terry model", category="Title", category_depth=0),
        _doc(_BODY),
    ]
    chunks = chunk_documents(docs)

    assert chunks[0].page_content == f"The Bradley-Terry model\n\n{_BODY}"
    assert chunks[0].metadata["section_path"] == "The Bradley-Terry model"


def test_the_heading_is_not_also_packed_as_content() -> None:
    # Both halves of the same fix: the heading becomes the opening line and is
    # then left out of the body. Packing it as an element as well printed the
    # heading twice in every chunk and spent a duplicated line's tokens on
    # every chunk in the document.
    docs = [_doc("Chapter 1", category="Title", category_depth=0), _doc(_BODY)]
    chunks = chunk_documents(docs)

    assert chunks[0].page_content.count("Chapter 1") == 1


def test_a_nested_breadcrumb_is_written_in_full() -> None:
    docs = [
        _doc("Chapter 1", category="Title", category_depth=0),
        _doc("Policy Gradients", category="Title", category_depth=1),
        _doc(_BODY),
    ]
    chunks = chunk_documents(docs)

    assert chunks[0].page_content.startswith("Chapter 1 > Policy Gradients\n\n")
    assert chunks[0].metadata["section_path"] == "Chapter 1 > Policy Gradients"


def test_the_breadcrumb_is_charged_against_the_chunk_budget() -> None:
    # The breadcrumb is part of the stored chunk, so it has to be part of the
    # budget. Charging only the body would let a chunk exceed chunk_size by
    # whatever the heading cost.
    settings = get_settings()
    body = " ".join(f"word{j}" for j in range(settings.chunk_size - 20))
    docs = [
        _doc("A heading long enough to matter here", category="Title", category_depth=0),
        _doc(body),
    ]
    chunks = chunk_documents(docs)

    # The chunk reports what it actually holds -- heading included -- rather
    # than the body alone, so the budget the answer stage spends is the budget
    # that was configured.
    assert chunks[0].metadata["token_count"] == len(
        chunks[0].page_content.split()
    )
    assert chunks[0].page_content.startswith("A heading long enough")


def test_a_heading_too_deep_to_be_a_section_stays_content() -> None:
    # Beyond _MAX_SECTION_DEPTH the element is extraction noise mistaken for a
    # heading, so it is neither a breadcrumb nor consumed as one -- it stays in
    # the body where a reader can see it for what it is.
    docs = [_doc("Emphasised words", category="Title", category_depth=7), _doc(_BODY)]
    chunks = chunk_documents(docs)

    assert "Emphasised words" in chunks[0].page_content
    assert "section_path" not in chunks[0].metadata


def test_token_length_fallback_on_missing_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("rag.chunking._get_tokenizer", boom)
    # 8 chars / 4 = 2 estimated tokens
    assert token_length("12345678") == 2
