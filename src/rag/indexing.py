"""Qdrant vector store management.

Runs Qdrant **embedded** (``QdrantClient(path=...)``) -- a full search engine
inside this process, no Docker, no server, free. Collection layout mirrors the
hybrid design: one named dense vector plus one named sparse vector (BM25),
both in a single collection, fused at query time by Qdrant's RRF.

Idempotent ingest is the other half of this module. Chunk ids are content
hashes, so upserting an unchanged corpus overwrites identical points rather
than duplicating them. Re-ingesting an *edited* document would leave stale
chunks behind (content hashes of deleted text never arrive again), so before
indexing a document's new chunks, :func:`delete_document_chunks` removes
everything previously written under its ``doc_id``. And because ``doc_id``
itself is a content hash, :func:`document_is_indexed` can skip the whole
parse-embed-index round for unchanged documents at negligible cost.

**Thread safety.** Embedded Qdrant takes an exclusive lock on its *directory*
(portalocker, held for the life of the process) but nothing stops two threads
in this process from using one client concurrently, and the client is not
documented as thread-safe. :func:`store_guard` is the single lock every
store-touching function here and in :mod:`rag.retrieval` enters, so the
server can ingest in a worker thread while answering queries in others.

**Writes are sliced and marked.** A 600-page book is a multi-minute job.
Upserting it in one call would mean no progress until it finished and a crash
near the end would discard the lot, so :func:`index_chunks` writes a slice at
a time and only stamps the document complete after the last one. A document
without the stamp is not treated as indexed, so an interrupted run is
re-done rather than silently half-searchable.
"""

from __future__ import annotations

import atexit
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from rag.config import Settings, get_settings
from rag.embedding import CachedHuggingFaceEmbeddings, get_dense_embeddings, get_sparse_embeddings
from rag.logging_utils import get_logger, timed
from rag.state import MetadataKeys as MK, ProgressFn, ProgressStage

logger = get_logger(__name__)

# Where the completion stamp lives: a payload key of its own, a *sibling* of
# ``metadata`` rather than a field inside it. Both of the obvious alternatives
# were measured against the embedded store and both are broken, each in a way
# that fails silently:
#
#   set_payload({"metadata.ingest_status": "complete"})  ->  the local store
#       does not expand dot notation on a payload *key*, so it writes a
#       literal top-level "metadata.ingest_status" and the filter on
#       metadata.ingest_status still matches nothing. Every document stays
#       permanently incomplete and is re-embedded on every ingest.
#
#   set_payload({"metadata": {"ingest_status": "complete"}})  ->  set_payload
#       merges at the top level only, so "metadata" is replaced wholesale and
#       every other field -- doc_id, chunk_id, page numbers -- is destroyed.
#       That breaks idempotency and provenance on the whole document.
#
# A brand-new root key avoids both: it is inserted without touching any
# existing sibling, and the nested path filters correctly.
INGEST_PAYLOAD_KEY = "ingest"
INGEST_STATUS_FIELD = f"{INGEST_PAYLOAD_KEY}.status"
INGEST_COMPLETE = "complete"


def _ingest_status(payload: dict | None) -> str | None:
    """Read the completion stamp off a point payload, or ``None``."""
    if not payload:
        return None
    status = payload.get(INGEST_PAYLOAD_KEY)
    return status.get("status") if isinstance(status, dict) else None

# One lock for every store read and write in the process. Re-entrant because
# index_chunks holds it per slice and nests delete_document_chunks inside the
# run, and because a caller may reasonably wrap several store calls in one
# transaction-like block.
_STORE_LOCK = threading.RLock()


@contextmanager
def store_guard() -> Iterator[None]:
    """Serialise access to the shared embedded Qdrant client.

    Held for the duration of a single store operation, never for a whole
    ingest: a query arriving mid-ingest waits for the slice in flight (a few
    seconds of embedding) rather than for the entire book.
    """
    with _STORE_LOCK:
        yield


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id for a chunk id.

    Qdrant point ids must be UUIDs or integers; our chunk ids are 16-hex
    content hashes. The UUIDv5 derivation is a cross-module contract --
    neighbour expansion resolves stored ``prev_id``/``next_id`` values
    through this same function -- so it lives here, once, as the single
    definition both writers and readers use.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-chunk:{chunk_id}"))


def get_client(settings: Settings | None = None) -> QdrantClient:
    """Embedded Qdrant client over the on-disk database, one per process.

    Embedded Qdrant takes an exclusive storage lock on its path; a second
    client on the same path raises rather than sharing. The module-level
    singleton guarantees ingest, retrieval, and stats all reuse one
    connection, and ``qdrant_path`` changing mid-process is not supported
    (the first call wins; later ``settings`` arguments are ignored).
    """
    global _CLIENT
    if _CLIENT is None:
        settings = settings or get_settings()
        _CLIENT = QdrantClient(path=str(settings.qdrant_path))
    return _CLIENT


_CLIENT: QdrantClient | None = None


def close_client() -> None:
    """Release the embedded store's storage lock by closing the client.

    Embedded Qdrant takes an exclusive lock on its directory, so a process
    that exits without closing leaves the next run unable to open the
    database. Registered with ``atexit`` for normal exits, and public so
    long-lived callers (a test fixture, an embedding host) can close early.
    """
    global _CLIENT
    if _CLIENT is None:
        return
    try:
        _CLIENT.close()
    except Exception as exc:  # pragma: no cover - close is best-effort
        logger.warning("closing the Qdrant client failed: %s: %s", type(exc).__name__, exc)
    finally:
        _CLIENT = None


atexit.register(close_client)


def _doc_filter(doc_id: str) -> models.Filter:
    """Qdrant filter matching every point of one document.

    Built in one place because three separate operations must agree on it
    exactly -- the "is it indexed" probe, the delete-before-rewrite, and the
    completion stamp. A near-miss in any of them fails silently: a filter
    that matches nothing reports a document as unindexed (harmless, just
    re-work), but a delete filter that matches nothing leaves stale chunks
    behind, and a stamp that matches nothing marks nothing complete.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"metadata.{MK.DOC_ID}",
                match=models.MatchValue(value=doc_id),
            )
        ]
    )


def _completed_filter(doc_id: str) -> models.Filter:
    """Filter matching only the *fully written* points of one document."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"metadata.{MK.DOC_ID}",
                match=models.MatchValue(value=doc_id),
            ),
            models.FieldCondition(
                key=INGEST_STATUS_FIELD,
                match=models.MatchValue(value=INGEST_COMPLETE),
            ),
        ]
    )


def document_is_indexed(
    client: QdrantClient, doc_id: str, settings: Settings
) -> bool:
    """Whether this (content-hashed) document is stored *and* complete.

    ``doc_id`` is a hash of the file's bytes, so an indexed doc_id means the
    file is byte-identical to what was ingested before -- the expensive
    parse/clean/chunk/embed work can be skipped entirely.

    "Stored" alone is not enough. Writes are sliced, so a run interrupted
    partway leaves real, searchable points behind with no completion stamp;
    counting those as indexed would make the skip permanent, and the corpus
    would stay silently truncated at wherever the crash happened. Requiring
    the stamp costs one extra condition and makes an interrupted document
    simply re-ingest.
    """
    if not client.collection_exists(settings.collection_name):
        return False
    with store_guard():
        result = client.count(
            collection_name=settings.collection_name,
            exact=False,
            count_filter=_completed_filter(doc_id),
        )
    return result.count > 0


def ensure_collection(
    client: QdrantClient,
    settings: Settings,
    *,
    dense: CachedHuggingFaceEmbeddings | None = None,
) -> None:
    """Create the hybrid collection if it does not exist.

    The explicit schema (rather than letting the vector store auto-create)
    pins the vector names that both index and query sides rely on, and lets
    us declare payload indexes on the filterable metadata fields -- inert in
    the embedded store, see the note at the call site.
    """
    if client.collection_exists(settings.collection_name):
        return

    dense_dim = _dense_dimension(settings, dense)

    with store_guard():
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                settings.dense_vector_name: models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                settings.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

        # Metadata fields declared as payload indexes. The embedded store
        # ignores payload indexes outright -- qdrant-client says as much in a
        # UserWarning at creation time -- so these are inert here and would
        # only buy a seek if the collection were pointed at a Qdrant server.
        # Every filter therefore scans and still returns correct results; do
        # not read this list as "the fast filters".
        #
        # SOURCE_NAME is here because it is the key a user reaches for first
        # ("only this book"), and it is much friendlier than the absolute path
        # in SOURCE.
        #
        # Attached at collection creation only, so an already-created
        # collection keeps whatever it was built with.
        for field in (MK.DOC_ID, MK.SOURCE, MK.SOURCE_NAME, MK.SECTION_PATH, MK.CHUNK_INDEX):
            try:
                client.create_payload_index(
                    collection_name=settings.collection_name,
                    field_name=f"metadata.{field}",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # pragma: no cover - index creation is advisory
                logger.warning("payload index on %s failed: %s", field, exc)

    logger.info(
        "Created collection '%s' (dense dim=%d, sparse=%s)",
        settings.collection_name,
        dense_dim,
        settings.sparse_vector_name,
    )


def _dense_dimension(
    settings: Settings,
    dense: CachedHuggingFaceEmbeddings | None = None,
) -> int:
    """Dense vector width, taken from the model itself (not hardcoded).

    bge-small-en-v1.5 is 384-dimensional; deriving it keeps the schema
    correct whenever the model is swapped via ``RAG_DENSE_MODEL``. When an
    embedder is supplied it is reused, so ingest never loads the model twice.
    """
    dense = dense or get_dense_embeddings(settings)
    # HuggingFaceEmbeddings exposes the underlying SentenceTransformer only
    # as its private ``_client``; there is no public accessor for the dim.
    dim = dense.model._client.get_embedding_dimension()
    if dim is None:  # pragma: no cover
        raise ValueError(
            f"Cannot determine embedding dimension for {settings.dense_model}"
        )
    return int(dim)


def get_vector_store(
    settings: Settings | None = None,
    *,
    dense: CachedHuggingFaceEmbeddings | None = None,
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    """Assemble the hybrid vector store used by both ingest and query."""
    settings = settings or get_settings()
    client = client or get_client(settings)
    dense = dense or get_dense_embeddings(settings)
    sparse = get_sparse_embeddings(settings)

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )


# ---------------------------------------------------------------------------
# Idempotent writes
# ---------------------------------------------------------------------------


def delete_document_chunks(client: QdrantClient, doc_id: str, settings: Settings) -> None:
    """Remove every chunk previously indexed under ``doc_id``.

    Called before re-indexing a document so an edited source cannot leave
    orphaned chunks behind. A no-op when the document was never indexed.
    """
    with store_guard():
        client.delete(
            collection_name=settings.collection_name,
            points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
        )
        _record_write()


def mark_document_complete(
    client: QdrantClient, doc_id: str, settings: Settings
) -> None:
    """Stamp every stored chunk of ``doc_id`` as fully written.

    One payload-only update over a filter, issued after the final slice
    lands. Re-embedding the chunks to add the flag would cost the same as
    indexing them again, which is the whole thing slicing exists to avoid.
    See :data:`INGEST_PAYLOAD_KEY` for why the stamp sits beside
    ``metadata`` rather than inside it.
    """
    with store_guard():
        client.set_payload(
            collection_name=settings.collection_name,
            payload={INGEST_PAYLOAD_KEY: {"status": INGEST_COMPLETE}},
            points=models.FilterSelector(filter=_doc_filter(doc_id)),
        )
        _record_write()


def index_chunks(
    chunks: list[Document],
    settings: Settings | None = None,
    *,
    on_progress: ProgressFn | None = None,
    slice_size: int | None = None,
) -> int:
    """Index one document's chunks into the hybrid collection, in slices.

    Old chunks for the same ``doc_id`` are deleted first, so re-ingesting an
    edited document replaces its points instead of accumulating stale ones.
    Returns the number of points written. Empty input is a clean no-op.

    ``on_progress`` receives one :data:`~rag.state.ProgressEvent` per slice
    with ``stage="index"`` and the point count done so far. The store lock is
    taken per slice rather than around the whole loop, so a query interleaves
    between slices and waits ~one slice's worth of embedding instead of the
    whole document. ``on_progress`` is called *outside* the lock -- it writes
    to the API's job registry, and doing that while holding the store lock
    would let a progress update block a reader.

    The completion stamp is written last and only if every slice landed; if
    the sink raises (a cancellation) or an upsert fails, the slices already
    written stay, and the missing stamp makes the next run redo the document.
    """
    settings = settings or get_settings()
    if not chunks:
        return 0

    doc_id = chunks[0].metadata[MK.DOC_ID]
    client = get_client(settings)
    slice_size = slice_size or settings.ingest_slice_size
    total = len(chunks)

    # One embedder for both the schema probe and the upserts: each fresh
    # instance would load its own copy of the model into memory.
    dense = get_dense_embeddings(settings)
    ensure_collection(client, settings, dense=dense)

    delete_document_chunks(client, doc_id, settings)

    store = get_vector_store(settings, dense=dense, client=client)

    with timed(logger, f"Upserting {total} chunk(s) for doc {doc_id}"):
        for start in range(0, total, slice_size):
            batch = chunks[start : start + slice_size]
            with store_guard():
                store.add_documents(
                    batch,
                    ids=[point_id(c.metadata[MK.CHUNK_ID]) for c in batch],
                )
                _record_write()
            done = min(start + len(batch), total)
            logger.info("  indexed %d/%d chunk(s)", done, total)
            if on_progress is not None:
                on_progress(
                    {
                        "stage": ProgressStage.INDEX.value,
                        "done": done,
                        "total": total,
                        "message": f"indexing chunks {done}/{total}",
                    }
                )

    mark_document_complete(client, doc_id, settings)
    return total


def count_document_chunks(doc_id: str, settings: Settings | None = None) -> int:
    """How many points are stored for one document, complete or not.

    Exists so the delete endpoint can distinguish "removed it" from "there
    was nothing there" without scrolling the whole collection to find out.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    with store_guard():
        if not client.collection_exists(settings.collection_name):
            return 0
        return client.count(
            collection_name=settings.collection_name,
            count_filter=_doc_filter(doc_id),
            exact=True,
        ).count


def collection_stats(settings: Settings | None = None) -> dict:
    """Collection existence and point count, for the stats panel."""
    settings = settings or get_settings()
    client = get_client(settings)

    with store_guard():
        if not client.collection_exists(settings.collection_name):
            return {"collection": settings.collection_name, "exists": False, "points": 0}

        info = client.get_collection(settings.collection_name)
    return {
        "collection": settings.collection_name,
        "exists": True,
        "points": info.points_count or 0,
    }


# ---------------------------------------------------------------------------
# Document inventory
# ---------------------------------------------------------------------------

# Bumped by every mutation below. Used as the inventory cache key.
#
# A point-count key is tempting and wrong: deleting a three-chunk document and
# indexing a different three-chunk document leaves the count identical, so the
# cached summary would keep describing the document that is no longer there.
# The store admits one process, so this counter sees every write that can
# change the inventory -- which makes it exact rather than a heuristic.
_WRITE_GENERATION = 0
_DOCS_CACHE: tuple[int, list[dict]] | None = None


def _record_write() -> int:
    global _WRITE_GENERATION
    _WRITE_GENERATION += 1
    return _WRITE_GENERATION


def list_documents(settings: Settings | None = None) -> list[dict]:
    """Summarise what is currently indexed, one entry per source document.

    The payload carries no document-level record, so the summary is folded
    out of the chunks: one scroll, aggregated per ``doc_id`` in Python. That
    is a real cost on a large corpus, hence the cache -- free while the
    pipeline is idle, recomputed whenever a write moves the generation.

    A document whose stamp is missing is reported with ``complete: False``
    rather than hidden: it is genuinely in the store and partially
    searchable, and the user's next move (re-ingest it) is easier to make if
    they can see what happened.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    global _DOCS_CACHE

    generation = _WRITE_GENERATION
    if _DOCS_CACHE is not None and _DOCS_CACHE[0] == generation:
        return [dict(entry) for entry in _DOCS_CACHE[1]]

    with store_guard():
        if not client.collection_exists(settings.collection_name):
            _DOCS_CACHE = None
            return []

        grouped: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=settings.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                meta = payload.get("metadata") or {}
                doc_id = meta.get(MK.DOC_ID)
                if not doc_id:
                    continue
                entry = grouped.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "source_name": meta.get(MK.SOURCE_NAME) or "unknown",
                        "source": meta.get(MK.SOURCE),
                        "chunks": 0,
                        "page_start": None,
                        "page_end": None,
                        "complete": True,
                    },
                )
                entry["chunks"] += 1
                start, end = meta.get(MK.PAGE_START), meta.get(MK.PAGE_END)
                if isinstance(start, int):
                    entry["page_start"] = (
                        start if entry["page_start"] is None else min(entry["page_start"], start)
                    )
                if isinstance(end, int):
                    entry["page_end"] = (
                        end if entry["page_end"] is None else max(entry["page_end"], end)
                    )
                if _ingest_status(payload) != INGEST_COMPLETE:
                    entry["complete"] = False
            if offset is None:
                break

    documents = sorted(grouped.values(), key=lambda d: (d["source_name"], d["doc_id"]))
    _DOCS_CACHE = (generation, documents)
    return [dict(entry) for entry in documents]
