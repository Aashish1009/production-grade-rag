"""Unit tests for the indexing layer.

Uses a deterministic fake embedder so no model downloads are needed and the
tests exercise the real Qdrant embedded engine against a temporary path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseVector

from rag.config import Settings
from rag.state import MetadataKeys as MK

# Width of the fake dense vectors; must match the fake collection schema.
FAKE_DIM = 8


class FakeDense(Embeddings):
    """Deterministic embeddings: hash of the text mod a small prime."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        h = abs(hash(text))
        return [((h >> i) % 97) / 97.0 for i in range(FAKE_DIM)]


class FakeSparse:
    """Single-token sparse vector; satisfies the SparseEmbeddings protocol.

    The protocol returns ``langchain_qdrant.SparseVector`` models, not dicts:
    the vector store reads ``.indices``/``.values`` off each result when it
    builds the Qdrant payload, so a dict-shaped double raises
    ``AttributeError: 'dict' object has no attribute 'indices'`` inside
    ``add_documents`` -- a failure in the double that looks like a failure in
    the indexing layer.
    """

    def embed_documents(self, texts):
        return [SparseVector(indices=[0], values=[1.0]) for _ in texts]

    def embed_query(self, text):
        return SparseVector(indices=[0], values=[1.0])


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Isolated Settings pointing the whole pipeline at a temp directory."""
    monkeypatch.setenv("RAG_QDRANT_PATH", str(tmp_path / "qdrant"))
    s = Settings(
        data_dir=tmp_path / "data",
        qdrant_path=tmp_path / "qdrant",
        embedding_cache_dir=tmp_path / "cache",
        _env_file=None,
    )

    import rag.indexing as indexing

    # Swap the model-backed embedders for deterministic fakes and pin the
    # dense dimension the fake produces (the real path asks the model). The
    # dense patch targets ``get_dense_embeddings`` because that is the seam
    # the store and the dimension probe both go through -- patching the class
    # instead would leave the real (memoised) embedder in play.
    monkeypatch.setattr(indexing, "get_dense_embeddings", lambda s=None: FakeDense())
    monkeypatch.setattr(indexing, "get_sparse_embeddings", lambda s=None: FakeSparse())
    monkeypatch.setattr(indexing, "_dense_dimension", lambda s, d=None: FAKE_DIM)

    # Reset the module-level client singleton so each test gets its own
    # embedded database under tmp_path.
    indexing._CLIENT = None
    yield s
    if indexing._CLIENT is not None:
        indexing._CLIENT.close()  # release the embedded storage lock
    indexing._CLIENT = None


def _chunks(n: int, doc_id: str = "d0") -> list[Document]:
    return [
        Document(
            page_content=f"chunk text {i} about reinforcement learning",
            metadata={
                MK.DOC_ID: doc_id,
                MK.CHUNK_ID: f"cid{i}",
                MK.SOURCE: "test.pdf",
                MK.SOURCE_NAME: "test.pdf",
                MK.PAGE_START: i,
                MK.PAGE_END: i,
            },
        )
        for i in range(n)
    ]


def test_point_id_is_deterministic_and_uuid_shaped() -> None:
    from rag.indexing import point_id

    a, b = point_id("cid1"), point_id("cid1")
    assert a == b
    assert a != point_id("cid2")
    import uuid

    uuid.UUID(a)  # raises if not a valid UUID string


def test_index_then_document_is_indexed(settings: Settings) -> None:
    import rag.indexing as indexing

    chunks = _chunks(3)
    assert indexing.index_chunks(chunks, settings) == 3

    client = indexing.get_client(settings)
    assert indexing.document_is_indexed(client, "d0", settings)
    assert not indexing.document_is_indexed(client, "missing", settings)


def test_reindex_replaces_points(settings: Settings) -> None:
    import rag.indexing as indexing

    indexing.index_chunks(_chunks(3, doc_id="d0"), settings)
    # Same doc, now only 2 chunks (edited document lost a paragraph).
    indexing.index_chunks(_chunks(2, doc_id="d0"), settings)

    client = indexing.get_client(settings)
    info = client.get_collection(settings.collection_name)
    assert info.points_count == 2, "stale third point must be deleted on re-index"


def test_index_empty_is_noop(settings: Settings) -> None:
    import rag.indexing as indexing

    assert indexing.index_chunks([], settings) == 0
    assert not indexing.get_client(settings).collection_exists(settings.collection_name)


def test_slices_report_progress_and_stamp_completion(settings: Settings) -> None:
    import rag.indexing as indexing

    events: list[dict] = []
    assert indexing.index_chunks(
        _chunks(10), settings, on_progress=events.append, slice_size=4
    ) == 10

    assert [e["done"] for e in events] == [4, 8, 10]
    assert all(e["stage"] == "index" for e in events)
    assert all(e["total"] == 10 for e in events)
    assert indexing.document_is_indexed(indexing.get_client(settings), "d0", settings)


def test_interrupted_write_is_not_reported_as_indexed(settings: Settings) -> None:
    """A cancelled ingest keeps its slices but must not look finished.

    This is the whole reason the completion stamp exists. Without it the
    interrupted document would count as indexed, the skip path in
    ``_process_file`` would fire on the next run, and the corpus would stay
    silently truncated at wherever the cancellation landed.
    """
    import rag.indexing as indexing
    from rag.state import JobCancelled

    def sink(event: dict) -> None:
        if event["stage"] == "index" and event["done"] == 4:
            raise JobCancelled("cancelled by the test")

    with pytest.raises(JobCancelled):
        indexing.index_chunks(_chunks(6), settings, on_progress=sink, slice_size=2)

    client = indexing.get_client(settings)
    assert indexing.count_document_chunks("d0", settings) == 4, "slices already written stay"
    assert not indexing.document_is_indexed(client, "d0", settings)


def test_completion_stamp_preserves_chunk_metadata(settings: Settings) -> None:
    """The stamp must be a payload *update*, never a payload rewrite.

    Measured against the embedded store: ``set_payload`` merges only at the
    top level, so a stamp written as ``{"metadata": {...}}`` replaces the
    whole metadata object and silently destroys doc_id, chunk_id and every
    page number. Provenance and re-ingest idempotency both depend on those
    surviving, so they are asserted here rather than trusted.
    """
    import rag.indexing as indexing

    chunks = _chunks(2)
    indexing.index_chunks(chunks, settings)
    client = indexing.get_client(settings)
    points, _ = client.scroll(settings.collection_name, limit=10, with_payload=True)

    assert len(points) == 2
    for point in points:
        meta = point.payload["metadata"]
        assert meta[MK.DOC_ID] == "d0"
        assert meta[MK.CHUNK_ID]
        assert meta[MK.PAGE_START] is not None, "page provenance must survive the stamp"
        assert meta[MK.SOURCE_NAME] == "test.pdf"


def test_list_documents_folds_chunks_and_flags_incomplete(settings: Settings) -> None:
    import rag.indexing as indexing
    from rag.state import JobCancelled

    indexing.index_chunks(_chunks(3, doc_id="d0"), settings)
    documents = indexing.list_documents(settings)
    assert len(documents) == 1
    assert documents[0]["doc_id"] == "d0"
    assert documents[0]["chunks"] == 3
    assert documents[0]["complete"] is True
    assert (documents[0]["page_start"], documents[0]["page_end"]) == (0, 2)

    def sink(event: dict) -> None:
        raise JobCancelled("stop after the first slice")

    with pytest.raises(JobCancelled):
        indexing.index_chunks(_chunks(4, doc_id="d1"), settings, on_progress=sink, slice_size=1)

    by_id = {d["doc_id"]: d for d in indexing.list_documents(settings)}
    assert by_id["d1"]["complete"] is False, "a half-written document is flagged, not hidden"
    assert by_id["d0"]["complete"] is True


def test_collection_stats_missing(settings: Settings) -> None:
    from rag.indexing import collection_stats

    stats = collection_stats(settings)
    assert stats["exists"] is False
    assert stats["points"] == 0
