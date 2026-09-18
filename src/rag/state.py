"""LangGraph state definitions.

Both graphs use ``langchain_core.documents.Document`` as the single data
carrier, so there is no parallel hierarchy of custom models to keep in sync
with LangChain's own interfaces. The metadata conventions each stage relies on
are documented in ``MetadataKeys`` below.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class MetadataKeys:
    """Canonical ``Document.metadata`` keys used across the pipeline.

    Centralised because these strings are the contract between the loader,
    the cleaner, the chunker, and the retriever. A typo in any one of them
    degrades retrieval silently rather than raising, so they are never
    written as bare literals at the call sites.
    """

    # --- provenance (set by loaders) ---------------------------------------
    SOURCE = "source"  # absolute path of the *original* document
    SOURCE_NAME = "source_name"  # display filename, never a batch temp file
    DOC_ID = "doc_id"  # stable hash of the source document
    PAGE = "page_number"  # ABSOLUTE page in the original document
    CATEGORY = "category"  # Unstructured element type
    ELEMENT_ID = "element_id"
    PARENT_ID = "parent_id"
    CATEGORY_DEPTH = "category_depth"
    TEXT_AS_HTML = "text_as_html"  # tables only

    # --- chunk-level (set by chunking) -------------------------------------
    CHUNK_ID = "chunk_id"  # content hash -> stable across re-ingest
    CONTENT_HASH = "content_hash"
    SECTION_PATH = "section_path"  # "Chapter 11 > Policy Gradients > GRPO"
    PAGE_START = "page_start"
    PAGE_END = "page_end"
    TOKEN_COUNT = "token_count"
    CHUNK_INDEX = "chunk_index"
    PREV_ID = "prev_id"
    NEXT_ID = "next_id"
    IS_TABLE = "is_table"

    # --- retrieval-level (set by retriever/reranker) -----------------------
    DENSE_SCORE = "dense_score"
    SPARSE_SCORE = "sparse_score"
    FUSED_SCORE = "fused_score"
    RERANK_SCORE = "rerank_score"

    # --- context widening (set after rerank) -------------------------------
    # A neighbouring chunk pulled in to complete a passage the reranker chose.
    # ``CONTEXT_OF`` names the winning chunk it belongs to and ``CONTEXT_ROLE``
    # says which side of it, so the renderer can put it in the right place and
    # -- more importantly -- *not* give it a citation number of its own.
    CONTEXT_OF = "context_of"
    CONTEXT_ROLE = "context_role"


# Metadata keys a caller may filter on -- the ones that actually exist in a
# stored payload and mean something as an equality match. Exposed as a set so
# the API can reject a typo ("docID") at the request boundary: an unknown key
# would otherwise translate into a filter matching nothing, and the user would
# see "no passages found" for a corpus that is right there.
#
# The completion stamp is deliberately absent: it is a fact about the *write*,
# not about the document, so it lives beside ``metadata`` in the payload
# rather than inside it -- see ``rag.indexing.INGEST_PAYLOAD_KEY``.
FILTERABLE_KEYS: frozenset[str] = frozenset(
    {
        MetadataKeys.DOC_ID,
        MetadataKeys.SOURCE,
        MetadataKeys.SOURCE_NAME,
        MetadataKeys.SECTION_PATH,
        MetadataKeys.CHUNK_INDEX,
        MetadataKeys.PAGE_START,
        MetadataKeys.PAGE_END,
        MetadataKeys.IS_TABLE,
    }
)


class ProgressStage(StrEnum):
    """Stages a document passes through during ingest.

    String values because they travel to the browser verbatim -- the UI
    renders the stage name directly, so the wire format and the enum cannot
    drift apart the way a separate mapping table would.
    """

    DISCOVER = "discover"
    LOAD = "load"
    CLEAN = "clean"
    CHUNK = "chunk"
    INDEX = "index"


class ProgressEvent(TypedDict, total=False):
    """One progress report from the ingest pipeline.

    ``done``/``total`` are optional because not every stage has countable
    units: page batches and index slices do (``total`` known, so the UI draws
    a determinate bar), while cleaning and chunking do not (the UI draws an
    indeterminate one). ``total`` is present only when it was known *before*
    the first event of that stage, so a bar never jumps backwards.
    """

    stage: str
    done: int
    total: int
    message: str
    path: str


ProgressFn = Callable[[ProgressEvent], None]
"""Sink an ingest run reports progress to. May raise to abort the run."""


class QueryStage(StrEnum):
    """Stages a question passes through, in order.

    Separate from :class:`ProgressStage` because the two lifecycles share no
    step: an ingest parses and embeds, a query retrieves and generates. One
    enum covering both would be a list of nine names where any given run uses
    half of them, and the UI's stepper would have to know which half.

    String values because they travel to the browser verbatim; the UI renders
    the stage name directly, so the wire format and the enum cannot drift the
    way a separate mapping table would.
    """

    GUARD = "guard"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    VERIFY = "verify"
    # Only reached on a web search, which is the one path that does not start
    # from the index. They sit after the corpus stages rather than before them
    # because that is where they happen in time: the corpus answers first, and
    # the web is what a user asks for once it has come back empty-handed.
    SEARCH = "search"
    FETCH = "fetch"


class QueryEvent(TypedDict, total=False):
    """One progress report from the query pipeline.

    ``elapsed_ms`` is measured server-side from the start of the query, so the
    UI can show a per-stage timing that is a fact about the pipeline rather
    than about when a line happened to arrive over the wire. The ingest side
    has no equivalent because its stages are minutes long, where milliseconds
    are noise.
    """

    stage: str
    message: str
    elapsed_ms: float


QueryFn = Callable[[QueryEvent], None]
"""Sink a query run reports progress to. Never raises: a query has no
checkpoint to cancel at, and a broken observer must not fail the answer."""


class JobCancelled(Exception):
    """Raised by a progress sink to abort an ingest at the next checkpoint.

    Defined here rather than in the API layer because the pipeline itself has
    to recognise it: ``_process_file`` catches ``Exception`` to turn a broken
    document into a failed report, which would otherwise swallow a
    cancellation and report the cancelled file as merely "failed" -- leaving
    the run to carry on with the next file the user just asked it to stop.
    The raise happens inside the sink, so this travels *up* through every
    frame between the sink and the graph runner.
    """


class FileReport(TypedDict):
    """Per-document outcome, aggregated into the final ingest summary."""

    path: str
    doc_id: str
    status: str  # "ok" | "failed" | "skipped"
    elements: int
    chunks: int
    error: str | None


class IngestState(TypedDict, total=False):
    """State threaded through the ingestion graph.

    ``reports`` uses an ``operator.add`` reducer so that parallel per-file
    branches merge cleanly instead of overwriting one another. Only channels
    the graph actually writes are declared: LangGraph rejects a node update
    for an undeclared key, so a vestigial field here is a trap rather than a
    harmless leftover.
    """

    # inputs
    input_path: str
    force_reindex: bool
    # Progress travels in the state rather than through LangGraph's
    # ``configurable`` so a node can be called directly in a test with a fake
    # sink -- no compiled graph and no config plumbing required.
    progress: ProgressFn

    # discovered work
    files: list[Path]

    # accumulated across fan-out branches
    reports: Annotated[list[FileReport], operator.add]

    # outputs
    indexed_count: int
    stats: dict[str, Any]


class PerFileState(TypedDict, total=False):
    """State for one fanned-out document branch.

    Carries only that branch's inputs; results travel back to ``IngestState``
    through the ``reports`` channel.
    """

    path: Path
    force_reindex: bool
    doc_id: str
    progress: ProgressFn


class QueryState(TypedDict, total=False):
    """State threaded through the query graph.

    ``filters`` is an optional metadata equality filter, e.g.
    ``{"doc_id": "ab12…"}`` or ``{"source_name": "rlhf book.pdf"}`` -- see
    :func:`rag.retrieval.search` for the accepted shape.

    ``top_k`` overrides ``Settings.top_k`` for this one query. It is a channel
    on the state rather than a mutation of the cached settings object so that
    two concurrent requests cannot see each other's value.

    ``progress`` rides in the state for the same reason it does on
    :class:`IngestState`: a node can then be called directly in a test with a
    fake sink, and no compiled graph or LangGraph config plumbing is needed to
    observe one stage.
    """

    # inputs
    question: str
    filters: dict[str, Any] | None
    top_k: int | None
    progress: QueryFn

    # working
    search_query: str
    candidates: list[Document]
    reranked: list[Document]

    # outputs
    context: str
    citations: list[dict[str, Any]]
    answer: str | None
    refused: bool


@dataclass(frozen=True, slots=True)
class WebPage:
    """One page of search results, as the answer stage sees it.

    Not a ``Document``, unlike everything else that travels through this
    pipeline. A web page has no page number, no section path and no rerank
    score, and giving it a ``Document`` with those keys absent would invite
    every consumer of ``metadata`` to guess at them. It is its own shape
    because it is its own kind of thing.

    ``snippet`` is the search engine's own summary of the page and ``text``
    is what was actually read out of it -- empty on the keyless path when the
    page would not load, which is why the two are kept apart rather than
    merged into one "content" field.
    """

    title: str
    url: str
    snippet: str
    text: str


class WebSearchState(TypedDict, total=False):
    """State threaded through the web-search fallback.

    The mirror of :class:`QueryState` for the path that does not touch the
    index. ``error`` is an ordinary outcome rather than an exception: asking
    the open web can fail in ways that are nobody's fault -- no key, no
    results, a rate limit -- and the section reports what happened instead of
    the request failing.
    """

    # inputs
    question: str
    progress: QueryFn

    # working
    provider: str
    pages: list[WebPage]

    # outputs
    sources: list[dict[str, Any]]
    context: str
    answer: str | None
    refused: bool
    error: str | None
