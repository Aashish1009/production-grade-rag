"""Unit tests for loading, discovery and provenance repair.

Nothing here parses a real document: ``_restore_provenance`` is pure metadata
arithmetic and ``discover_files`` is filesystem filtering, so both are
exercised directly rather than through Unstructured (which would drag in
layout models, poppler and tesseract).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.loaders import _restore_provenance, compute_doc_id, discover_files
from rag.state import MetadataKeys as MK


def _element(page: int | None = None, **extra: object) -> Document:
    meta: dict[str, object] = {MK.CATEGORY: "NarrativeText", "filename": "batch_00020_00040.pdf"}
    if page is not None:
        meta[MK.PAGE] = page
    meta.update(extra)
    return Document(page_content="some narrative text", metadata=meta)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_page_offset_is_added_to_batch_relative_pages() -> None:
    # Unstructured numbers pages relative to the file it was handed, so batch
    # 2 restarts at 1. Without the offset every citation past the first batch
    # points at the wrong page.
    docs = _restore_provenance(
        [_element(page=1), _element(page=2)],
        source_path=Path("/corpus/rlhf book.pdf"),
        doc_id="abc123",
        page_offset=20,
    )
    assert [d.metadata[MK.PAGE] for d in docs] == [21, 22]


def test_zero_offset_batch_keeps_pages_unchanged() -> None:
    docs = _restore_provenance(
        [_element(page=5)],
        source_path=Path("/corpus/rlhf book.pdf"),
        doc_id="abc123",
        page_offset=0,
    )
    assert docs[0].metadata[MK.PAGE] == 5


def test_pageless_element_in_a_batch_gets_the_batch_lower_bound() -> None:
    docs = _restore_provenance(
        [_element(page=None)],
        source_path=Path("/corpus/rlhf book.pdf"),
        doc_id="abc123",
        page_offset=40,
    )
    assert docs[0].metadata[MK.PAGE] == 41


def test_pageless_element_without_an_offset_gets_no_page() -> None:
    # A Markdown file has no pages. Inventing "page 1" would put a fabricated
    # page number in a citation, which is worse than citing the section.
    docs = _restore_provenance(
        [_element(page=None)],
        source_path=Path("/corpus/notes.md"),
        doc_id="abc123",
        page_offset=None,
    )
    assert MK.PAGE not in docs[0].metadata


def test_source_and_doc_id_are_rewritten_to_the_original_document() -> None:
    docs = _restore_provenance(
        [_element(page=1)],
        source_path=Path("/corpus/rlhf book.pdf"),
        doc_id="abc123",
        page_offset=0,
    )
    meta = docs[0].metadata
    assert meta[MK.SOURCE] == str(Path("/corpus/rlhf book.pdf"))
    assert meta[MK.SOURCE_NAME] == "rlhf book.pdf"
    assert meta[MK.DOC_ID] == "abc123"
    # The batch temp file must not survive into the citation.
    assert "filename" not in meta
    assert "file_directory" not in meta


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_skips_dependency_and_store_directories(tmp_path: Path) -> None:
    # Ingesting a project root must not descend into the virtualenv or the
    # vector store: doing so ingests every README and JSON file in
    # site-packages, and the store's own files.
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "real.pdf").write_bytes(b"%PDF-1.4 fake")
    for noise in (".venv", "node_modules", "__pycache__", "qdrant_db", ".git"):
        nested = tmp_path / noise / "lib"
        nested.mkdir(parents=True)
        (nested / "ignored.md").write_text("dependency docs")
        (nested / "ignored.json").write_text("{}")

    found = discover_files(tmp_path)
    assert [p.name for p in found] == ["real.pdf"]


def test_discover_recurses_into_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "deep.md").write_text("deep content")
    (tmp_path / "top.txt").write_text("top content")

    names = {p.name for p in discover_files(tmp_path)}
    assert names == {"deep.md", "top.txt"}


def test_discover_ignores_unsupported_extensions(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("kept")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04")
    (tmp_path / "binary.exe").write_bytes(b"MZ")

    assert [p.name for p in discover_files(tmp_path)] == ["notes.md"]


def test_discover_single_unsupported_file_returns_empty(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"\x00\x01")

    assert discover_files(target) == []


def test_discover_single_file_is_returned_as_is(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("content")

    assert discover_files(target) == [target.resolve()]


def test_discover_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope")


# ---------------------------------------------------------------------------
# doc ids
# ---------------------------------------------------------------------------


def test_compute_doc_id_is_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("identical")
    b.write_text("identical")
    c = tmp_path / "c.md"
    c.write_text("different")

    # Content-addressed, not path-addressed: identical bytes must hash the
    # same regardless of filename, or a renamed file re-indexes from scratch.
    assert compute_doc_id(a) == compute_doc_id(b)
    assert compute_doc_id(a) != compute_doc_id(c)
    assert len(compute_doc_id(a)) == 16
