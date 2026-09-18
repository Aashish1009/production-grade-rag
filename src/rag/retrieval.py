"""Hybrid retrieval from the local Qdrant collection.

Two layers, both LangChain-native:

**Primary — Qdrant server-side hybrid.** Dense and sparse (BM25) searches
run as prefetches of a single Qdrant Query API call and are fused
server-side with Reciprocal Rank Fusion. This is what
``QdrantVectorStore.as_retriever`` does under ``RetrievalMode.HYBRID``, so
fusion, dedup of shared hits, and score normalisation stay inside the
library rather than reimplemented here.

**Fallback — ``EnsembleRetriever``.** If the hybrid query fails (corrupt
index, schema drift, sparse-model load error), LangChain's
``EnsembleRetriever`` combines a dense-only Qdrant retriever with a pure
lexical BM25 retriever over the stored chunks, weighting each by its
reliability. Query answering degrades instead of dying.

On top of retrieval, **context widening** pulls the chunks either side of each
*winning* passage (linked at chunking time via ``prev_id``/``next_id``) so the
answer is written from a whole claim rather than a fragment of one. It runs
after reranking, not before -- :func:`widen_context` is where that argument is
made.

Every call into the shared embedded client is wrapped in
:func:`rag.indexing.store_guard` -- the client is one per process and is not
documented as thread-safe, so a query may not read while ingest is writing.
The guard is per call rather than per query, so the wait is one store
operation, not one unrelated job.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from rag.config import Settings, get_settings
from rag.indexing import get_client, store_guard
from rag.logging_utils import get_logger
from rag.state import MetadataKeys as MK

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Translate a metadata equality mapping into a Qdrant filter.

    ``{"doc_id": "ab12…"}`` becomes ``metadata.doc_id == "ab12…"``, which
    covers the realistic cases: restrict to one document, one source file, or
    one section. Anything richer -- ranges, boolean trees -- belongs in a
    hand-built ``qdrant_client.models.Filter`` passed by the caller rather
    than in a query language grown here.

    ``doc_id``, ``source``, ``source_name``, ``section_path`` and
    ``chunk_index`` are declared as payload indexes (see
    :func:`rag.indexing.ensure_collection`), but the embedded store ignores
    payload indexes, so in practice every key scans -- and every key still
    returns correct results.
    """
    if not filters:
        return None
    conditions = [
        models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
        for key, value in filters.items()
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


def build_retriever(
    settings: Settings | None = None,
    vector_store=None,
    *,
    filters: dict[str, Any] | None = None,
):
    """LangChain retriever over the hybrid collection.

    ``k`` must travel inside ``search_kwargs``: ``VectorStoreRetriever`` is a
    pydantic model with ``extra="ignore"``, so top-level ``k=...`` /
    ``retrieval_mode=...`` kwargs are silently dropped and retrieval would
    quietly run at the default k=4. The retrieval mode is therefore set on
    the vector store itself, which ``similarity_search`` consults per call.

    ``filters`` reach the vector store's ``similarity_search`` as a Qdrant
    filter, and are pushed into both the dense and the sparse prefetch of the
    hybrid query, so a filtered hybrid search stays a single round trip.
    """
    from langchain_qdrant import RetrievalMode

    settings = settings or get_settings()

    if vector_store is None:
        from rag.indexing import get_vector_store

        vector_store = get_vector_store(settings)

    # Settings.retrieval_mode is a StrEnum whose values match RetrievalMode's.
    vector_store.retrieval_mode = RetrievalMode(settings.retrieval_mode.value)

    search_kwargs: dict[str, Any] = {"k": settings.fetch_k}
    query_filter = build_filter(filters)
    if query_filter is not None:
        search_kwargs["filter"] = query_filter

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


# ---------------------------------------------------------------------------
# Context widening
# ---------------------------------------------------------------------------


def _stored_chunk_ids(documents: list[Document]) -> set[str]:
    return {d.metadata[MK.CHUNK_ID] for d in documents if d.metadata.get(MK.CHUNK_ID)}


def _read_chunks(
    chunk_ids: set[str], client: QdrantClient, settings: Settings
) -> dict[str, Document]:
    """Read stored chunks by chunk id, in one batched call.

    Raises on a store failure rather than swallowing it: :func:`widen_context`
    is the only caller and it decides what a failure means.
    """
    from rag.indexing import point_id

    with store_guard():
        response = client.retrieve(
            collection_name=settings.collection_name,
            ids=[point_id(cid) for cid in chunk_ids],
            with_payload=True,
        )

    by_id: dict[str, Document] = {}
    for point in response:
        payload = point.payload or {}
        if not payload.get("page_content"):
            continue
        meta = dict(payload.get("metadata") or {})
        by_id[meta.get(MK.CHUNK_ID, "")] = Document(
            page_content=payload["page_content"], metadata=meta
        )
    return by_id


def _as_context(doc: Document, anchor_id: str, role: str) -> Document:
    """Mark ``doc`` as context for ``anchor_id`` rather than a result of its own."""
    meta = dict(doc.metadata)
    meta[MK.CONTEXT_OF] = anchor_id
    meta[MK.CONTEXT_ROLE] = role
    return Document(page_content=doc.page_content, metadata=meta)


def widen_context(
    documents: list[Document],
    client: QdrantClient,
    settings: Settings,
) -> list[Document]:
    """Widen each reranked passage with the chunks around it -- after rerank.

    **This replaces pre-rerank neighbour expansion, and the ordering is the
    whole point.** Expanding *before* reranking (which is what this pipeline
    used to offer behind ``enable_neighbour_expansion``) makes a neighbour a
    candidate in its own right: it competes for the same ``top_k`` slots as the
    passage it sits next to, usually scores lower than it -- a cross-encoder
    asked about "the Bradley-Terry equation" likes the paragraph that states it
    more than the one that discusses its criticisms -- and so *displaces* it.
    The anchor survives only if its neighbour also happens to score well, and
    ``top_k`` slots are spent on whatever was adjacent to a hit rather than on
    other hits. It also triples the rerank bill, since the cross-encoder scores
    every neighbour as a pair.

    Widening after the rerank inverts both. The reranker chooses winners on
    precise, individually-scored passages -- which is what it is good at -- and
    the winners then bring their surroundings along as *context*, which is what
    an answer needs: a claim stated on one page and qualified on the next is
    half a claim from either alone, and the model that reports "the context
    does not cover this" is usually right about the fragment it was handed.

    Neighbours are marked with ``context_of``/``context_role`` rather than
    given citation numbers, so ``top_k`` still counts passages answered *for*
    and the citation map does not inflate. See
    :func:`rag.generation.format_context`, which renders them inside their
    anchor's block.

    A failure here is not a failed query: the store may be busy or the links
    may dangle, and the passages are still a complete answer on their own.
    """
    radius = settings.context_neighbour_radius
    if not documents or radius < 1:
        return documents

    heads, tails = _walk_neighbours(documents, client, settings, radius)
    if not any(heads.values()) and not any(tails.values()):
        return documents

    widened: list[Document] = []
    for doc in documents:
        anchor_id = doc.metadata.get(MK.CHUNK_ID)
        for side in heads.get(anchor_id, []):
            widened.append(_as_context(side, anchor_id, "prev"))
        widened.append(doc)
        for side in tails.get(anchor_id, []):
            widened.append(_as_context(side, anchor_id, "next"))
    return widened


def _walk_neighbours(
    documents: list[Document],
    client: QdrantClient,
    settings: Settings,
    radius: int,
) -> tuple[dict[str, list[Document]], dict[str, list[Document]]]:
    """Follow ``prev_id``/``next_id`` out to ``radius`` chunks from each anchor.

    One batched read per step rather than per anchor. Returns ``(heads, tails)``
    keyed by anchor chunk id, with ``heads`` ordered farthest-to-nearest and
    ``tails`` nearest-to-farthest, so both read correctly once concatenated
    around the anchor.

    Already-retrieved chunks are excluded from the walk: if a passage's
    neighbour is itself one of the reranked winners, it is going to be rendered
    as its own block, and pulling it in here as well would print it twice.
    """
    known = _stored_chunk_ids(documents)
    heads: dict[str, list[Document]] = {cid: [] for cid in known}
    tails: dict[str, list[Document]] = {cid: [] for cid in known}

    frontier: dict[str, tuple[str | None, str | None]] = {
        doc.metadata[MK.CHUNK_ID]: (
            doc.metadata.get(MK.PREV_ID),
            doc.metadata.get(MK.NEXT_ID),
        )
        for doc in documents
        if doc.metadata.get(MK.CHUNK_ID)
    }

    for _ in range(radius):
        wanted = {cid for pair in frontier.values() for cid in pair if cid} - known
        if not wanted:
            break
        try:
            fetched = _read_chunks(wanted, client, settings)
        except Exception as exc:
            logger.warning(
                "context widening failed (%s: %s); using the passages alone",
                type(exc).__name__,
                exc,
            )
            break

        known |= set(fetched)
        following: dict[str, tuple[str | None, str | None]] = {}
        for anchor_id, (prev_id, next_id) in frontier.items():
            prev_doc = fetched.get(prev_id or "")
            next_doc = fetched.get(next_id or "")
            if prev_doc is not None:
                # Inserted at the front because the walk moves outward: the
                # second-round neighbour is farther from the anchor than the
                # first, and has to end up ahead of it in the rendered block.
                heads[anchor_id].insert(0, prev_doc)
            if next_doc is not None:
                tails[anchor_id].append(next_doc)
            following[anchor_id] = (
                prev_doc.metadata.get(MK.PREV_ID) if prev_doc is not None else None,
                next_doc.metadata.get(MK.NEXT_ID) if next_doc is not None else None,
            )
        frontier = following

    return heads, tails


# ---------------------------------------------------------------------------
# BM25 lexical retriever
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokenisation for BM25, digits kept.

    Word boundaries must survive. An earlier revision reused the cleaning
    pass's dedupe normaliser, which strips *all* non-alphanumerics -- spaces
    included -- so ``"policy gradient methods"`` became the single token
    ``"policygradientmethods"``. Both the corpus and the queries were mangled
    the same way, so scoring degraded to whole-string equality and the
    lexical fallback silently retrieved nothing for every realistic query.
    Splitting here is the fix; digits are kept so "GPT-4" still matches "4".
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25FallbackRetriever(BaseRetriever):
    """Lexical-only retriever over the chunks already stored in Qdrant.

    The corpus, its tokenisation and the BM25 index are built once and cached
    at module level, keyed by collection point count: a fallback that
    re-scrolled *and* re-tokenised *and* re-indexed the whole corpus on every
    query would collapse exactly when the system is already degraded -- which
    is the situation the fallback exists for.
    """

    k: int = 10
    filters: dict[str, Any] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        settings = get_settings()
        client = get_client(settings)

        bm25, corpus = _cached_bm25(client, settings)
        if bm25 is None:
            return []

        if self.filters:
            wanted = {k: v for k, v in self.filters.items() if v is not None}
            corpus = [
                doc for doc in corpus if all(doc.metadata.get(k) == v for k, v in wanted.items())
            ]
            if not corpus:
                return []

        scores = bm25.get_scores(_tokenise(query))
        ranked = sorted(zip(scores, corpus), key=lambda pair: pair[0], reverse=True)
        return [doc for score, doc in ranked[: self.k] if score > 0]


_BM25_CACHE: tuple[int, BM25Okapi, list[Document]] | None = None


def _cached_bm25(
    client: QdrantClient, settings: Settings
) -> tuple[BM25Okapi | None, list[Document]]:
    """BM25 index and corpus over every stored chunk.

    Cached until the collection's point count moves, which is a cheap and
    sufficient proxy for content change: ingest adds points and re-index
    swaps them, and both move the count. Rebuilding is therefore paid once
    per ingest, not once per query.

    The cache write is a single assignment of an already-built tuple, so two
    threads racing here duplicate work at worst; neither can observe a
    half-built entry. Deliberately *not* done under the store guard: building
    the index tokenises the whole corpus, which can take seconds, and holding
    the store lock across it would stall an ingest for that whole time. Only
    the reads inside are guarded.
    """
    global _BM25_CACHE

    with store_guard():
        if not client.collection_exists(settings.collection_name):
            return None, []

        info = client.get_collection(settings.collection_name)
        count = info.points_count or 0
        if _BM25_CACHE is not None and _BM25_CACHE[0] == count:
            return _BM25_CACHE[1], _BM25_CACHE[2]

    corpus = _scroll_corpus(client, settings)
    if not corpus:
        _BM25_CACHE = None
        return None, []

    bm25 = BM25Okapi([_tokenise(doc.page_content) for doc in corpus])
    _BM25_CACHE = (count, bm25, corpus)
    return bm25, corpus


def _scroll_corpus(client: QdrantClient, settings: Settings) -> list[Document]:
    """Read every stored chunk's text, in id order."""
    corpus: list[Document] = []
    offset = None
    with store_guard():
        while True:
            points, offset = client.scroll(
                collection_name=settings.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                payload = p.payload or {}
                if payload.get("page_content"):
                    corpus.append(
                        Document(
                            page_content=payload["page_content"],
                            metadata=dict(payload.get("metadata") or {}),
                        )
                    )
            if offset is None:
                break
    return corpus


# ---------------------------------------------------------------------------
# Top-level search with ensemble fallback
# ---------------------------------------------------------------------------


def _fallback_retriever(settings: Settings, filters: dict[str, Any] | None = None):
    """Dense-only Qdrant retriever + BM25 lexical, fused by EnsembleRetriever.

    The dense side drops to ``RetrievalMode.DENSE`` on the vector store
    itself (the retriever-level ``retrieval_mode`` kwarg is ignored by
    ``VectorStoreRetriever``), so the fallback really does bypass the
    hybrid path that just failed.

    Filters are applied on both sides -- as a Qdrant filter on the dense
    retriever and as a metadata predicate on the lexical one -- so the
    fallback cannot smuggle in hits the primary path would have excluded.
    """
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_qdrant import RetrievalMode

    from rag.indexing import get_vector_store

    dense_store = get_vector_store(settings)
    dense_store.retrieval_mode = RetrievalMode.DENSE

    search_kwargs: dict[str, Any] = {"k": settings.fetch_k}
    query_filter = build_filter(filters)
    if query_filter is not None:
        search_kwargs["filter"] = query_filter

    dense = dense_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    lexical = BM25FallbackRetriever(k=settings.fetch_k, filters=filters)

    return EnsembleRetriever(
        retrievers=[dense, lexical],
        weights=[0.5, 0.5],
    )


def search(
    question: str,
    settings: Settings | None = None,
    *,
    filters: dict[str, Any] | None = None,
    client: QdrantClient | None = None,
) -> list[Document]:
    """Run the configured hybrid search, falling back to a dense+BM25 ensemble.

    ``filters`` is a metadata equality mapping applied to both the primary and
    the fallback path -- see :func:`build_filter`.

    An empty or missing collection returns ``[]`` -- "no passages found" is
    a valid answer, not an error, and must not escape as a traceback.

    This returns lean candidates only. Neighbour context is added *after*
    reranking, by :func:`widen_context` -- the ordering is the point, and that
    function explains why.
    """
    settings = settings or get_settings()
    client = client or get_client(settings)

    with store_guard():
        if not client.collection_exists(settings.collection_name):
            logger.info(
                "Collection '%s' does not exist yet; nothing to search",
                settings.collection_name,
            )
            return []

    try:
        retriever = build_retriever(settings, filters=filters)
        with store_guard():
            hits = retriever.invoke(question)
    except Exception as exc:
        logger.warning(
            "Hybrid search failed (%s: %s); falling back to dense+BM25 ensemble",
            type(exc).__name__,
            exc,
        )
        try:
            # Held across the ensemble because both halves of it touch the
            # shared client. The lexical half may build the whole BM25 index
            # inside this window, which is why the cache exists: that build
            # happens once per corpus change, not once per fallback query.
            with store_guard():
                hits = _fallback_retriever(settings, filters).invoke(question)
        except Exception as fallback_exc:
            # The fallback exists so a query degrades rather than dies; if
            # the fallback itself is broken, an empty result beats a crash.
            logger.error(
                "Fallback retrieval also failed (%s: %s); returning no hits",
                type(fallback_exc).__name__,
                fallback_exc,
            )
            return []

    return hits

