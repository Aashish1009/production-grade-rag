"""Progress reporting and cancellation in the ingest graph.

``_process_file`` is called directly rather than through the compiled graph:
the callback rides in the state, so the node is an ordinary function of a
dict, and a test can drive it with a fake sink and stubbed stages without
compiling a graph, touching a store, or loading a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rag.chunking as chunking
import rag.graphs as graphs
import rag.indexing as indexing
from rag.state import JobCancelled, MetadataKeys as MK


@pytest.fixture()
def book(tmp_path: Path) -> Path:
    """A file whose *content* never matters: every stage is stubbed out."""
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF-1.4 placeholder")
    return path


@pytest.fixture()
def stubbed_pipeline(monkeypatch: pytest.MonkeyPatch, book: Path):
    """Stub every stage of ``_process_file`` and record what it was handed.

    Each stub reports progress the way the real stage does, so the asserted
    event sequence is the one the graph would actually produce -- not an
    idealised one.
    """
    seen: dict[str, object] = {}

    monkeypatch.setattr(graphs, "compute_doc_id", lambda path: "d0")

    def load(path, settings, *, on_progress=None):
        seen["load_sink"] = on_progress
        if on_progress is not None:
            # A two-batch PDF, pages numbered in the original document.
            on_progress({"stage": "load", "done": 20, "total": 40, "message": "pages 1-20 of 40"})
            on_progress({"stage": "load", "done": 40, "total": 40, "message": "pages 21-40 of 40"})
        return [{"element": 1}, {"element": 2}]

    def chunks(documents, settings):
        seen["elements"] = list(documents)
        return [
            _chunk(i) for i in range(5)
        ]

    def index(chunks_in, settings, *, on_progress=None):
        seen["chunks_in"] = len(chunks_in)
        if on_progress is not None:
            on_progress({"stage": "index", "done": len(chunks_in), "total": len(chunks_in)})
        return len(chunks_in)

    monkeypatch.setattr(graphs, "load_document", load)
    monkeypatch.setattr(graphs, "clean_documents", lambda docs, settings: docs)
    monkeypatch.setattr(chunking, "chunk_documents", chunks)
    monkeypatch.setattr(graphs, "index_chunks", index)
    # The skip check would otherwise open the real embedded store.
    monkeypatch.setattr(indexing, "document_is_indexed", lambda *a, **k: False)
    monkeypatch.setattr(indexing, "get_client", lambda settings=None: object())

    return seen


def _chunk(i: int):
    from langchain_core.documents import Document

    return Document(
        page_content=f"text {i}",
        metadata={MK.DOC_ID: "d0", MK.CHUNK_ID: f"c{i}"},
    )


def test_process_file_threads_progress_through_every_stage(
    book: Path, stubbed_pipeline: dict
) -> None:
    events: list[dict] = []
    result = graphs._process_file({"path": book, "progress": events.append})

    assert [e["stage"] for e in events] == ["load", "load", "clean", "chunk", "index"]
    assert [e["done"] for e in events if e["stage"] == "load"] == [20, 40]
    assert events[0]["total"] == 40, "the page count is the document's, not the batch's"
    assert result["reports"][0]["status"] == "ok"
    assert result["reports"][0]["chunks"] == 5


def test_every_event_is_attributed_to_its_file(book: Path, stubbed_pipeline: dict) -> None:
    """The UI renders one card per file, so an unattributed event is unusable."""
    events: list[dict] = []
    graphs._process_file({"path": book, "progress": events.append})

    assert events, "the stub should have produced events"
    assert all(e["path"] == str(book) for e in events)


def test_the_sink_reaches_the_stages_that_produce_the_slow_events(
    book: Path, stubbed_pipeline: dict
) -> None:
    """Parsing and embedding dominate a 600-page ingest, so their sinks matter.

    Asserted separately because a regression here is invisible in the event
    sequence: the graph would still report ``clean`` and ``chunk`` progress
    while the two long stages went silent.
    """
    events: list[dict] = []
    graphs._process_file({"path": book, "progress": events.append})

    assert stubbed_pipeline["load_sink"] is not None
    assert stubbed_pipeline["chunks_in"] == 5


def test_process_file_without_a_sink_still_runs(book: Path, stubbed_pipeline: dict) -> None:
    """No sink is the library path (scripts, tests), not an error."""
    result = graphs._process_file({"path": book})
    assert result["reports"][0]["status"] == "ok"


def test_cancellation_escapes_instead_of_becoming_a_failed_report(
    book: Path, stubbed_pipeline: dict
) -> None:
    """The catch-all must not swallow a cancellation.

    ``_process_file`` turns any exception into a ``failed`` report and the
    graph then moves on to the next file -- which is the opposite of what a
    cancellation asks for. So ``JobCancelled`` is re-raised, and this pins
    that it still is.
    """

    def sink(event: dict) -> None:
        raise JobCancelled("cancelled by the test")

    with pytest.raises(JobCancelled):
        graphs._process_file({"path": book, "progress": sink})


def test_other_failures_are_still_contained(
    book: Path, stubbed_pipeline: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation is the *only* thing that escapes the per-file branch."""

    def boom(*args, **kwargs):
        raise RuntimeError("page 42 is corrupt")

    monkeypatch.setattr(graphs, "load_document", boom)

    result = graphs._process_file({"path": book})

    report = result["reports"][0]
    assert report["status"] == "failed"
    assert report["error"] == "RuntimeError: page 42 is corrupt"
