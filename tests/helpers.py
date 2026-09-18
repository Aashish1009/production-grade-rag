"""Shared test fixtures and helpers."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.state import MetadataKeys as MK


def make_doc(
    text: str, category: str = "NarrativeText", page: int = 1, **extra: object
) -> Document:
    """Build an element-level Document with canonical metadata keys.

    All tests go through this so metadata keys always come from
    ``MetadataKeys`` -- a typo in a test literal would otherwise mask a
    production typo it should have caught.
    """
    return Document(
        page_content=text,
        metadata={
            MK.CATEGORY: category,
            MK.PAGE: page,
            MK.SOURCE: "test.pdf",
            **extra,
        },
    )
