"""LangGraph orchestration: the ingest and query graphs.

Two graphs, one per lifecycle. Both use ``Document`` as the single data
carrier and read their structure from :mod:`rag.state`.

**Ingest graph** -- sequential with a per-file fan-out::

    discover -> (per file: load -> clean -> chunk -> index) -> summarise

The fan-out uses LangGraph's ``Send`` API so each document is processed as
its own branch and branch results merge through the ``operator.add``
reducers in ``IngestState``. One file failing never kills the run; failures
land in that file's ``FileReport``.

**Query graph** -- a straight line, because latency matters more than
parallelism here (the user is waiting)::

    guard -> retrieve -> rerank -> widen -> generate -> guard

Input guardrails reject injection before anything expensive runs; output
guardrails (citation, grounding, PII) check the answer before it leaves.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph
from langgraph.types import Send

from rag.cleaning import clean_documents
from rag.config import Settings, get_settings
from rag.generation import describe_model, generate_answer
from rag.guardrails import apply_output_guardrails, check_input, is_refusal
from rag.indexing import get_client, index_chunks
from rag.loaders import compute_doc_id, discover_files, load_document
from rag.logging_utils import get_logger
from rag.reranking import rerank_documents
from rag.retrieval import search, widen_context
from rag.state import (
    FileReport,
    IngestState,
    JobCancelled,
    PerFileState,
    ProgressFn,
    ProgressStage,
    QueryFn,
    QueryStage,
    QueryState,
)

logger = get_logger(__name__)


def _file_sink(progress: ProgressFn | None, path: Path) -> ProgressFn | None:
    """Wrap the run's sink so every event carries the file it came from.

    The API shows one progress card per uploaded file, so an event with no
    path cannot be attributed. Stamping it here means the stages below never
    have to remember to do it.
    """
    if progress is None:
        return None

    def sink(event) -> None:
        progress({**event, "path": str(path)})

    return sink


# ---------------------------------------------------------------------------
# Ingest graph
# ---------------------------------------------------------------------------


def _discover(state: IngestState) -> dict[str, Any]:
    """Turn ``input_path`` into the list of files to process."""
    files = discover_files(state["input_path"])
    if not files:
        logger.warning("No supported documents found under %s", state["input_path"])
    progress = state.get("progress")
    if progress is not None:
        progress(
            {
                "stage": ProgressStage.DISCOVER.value,
                "done": 0,
                "total": len(files),
                "message": f"found {len(files)} document(s)",
            }
        )
    return {"files": files}


def _fan_out_files(state: IngestState) -> list[Send | str]:
    """One branch per file; already-indexed docs are skipped unless forced.

    When discovery finds nothing, the route is ``"summarise"`` rather than an
    empty send list: returning ``[]`` from a conditional edge ends the run at
    that node, so the graph would finish with no ``stats`` key at all and the
    caller would print a summary of zeros as though the ingest had succeeded.

    ``progress`` rides along in the payload because a ``Send`` payload is a
    fresh state for the branch -- anything not copied here is simply absent
    in ``process_file``.
    """
    files = state.get("files", [])
    if not files:
        return ["summarise"]
    progress = state.get("progress")
    return [
        Send(
            "process_file",
            {
                "path": path,
                "force_reindex": state.get("force_reindex", False),
                "progress": progress,
            },
        )
        for path in files
    ]


def _process_file(state: PerFileState) -> dict[str, Any]:
    """load -> clean -> chunk -> index for one document.

    Any failure is contained: the branch returns a ``failed`` report and the
    run continues with the next file. Unchanged documents are skipped before
    parsing (their ``doc_id`` is a content hash) unless ``force_reindex``.

    Cancellation is the one thing deliberately *not* contained -- see the
    ``JobCancelled`` clause below.
    """
    settings = get_settings()
    path: Path = state["path"]
    force = state.get("force_reindex", False)
    sink = _file_sink(state.get("progress"), path)
    report: FileReport = {
        "path": str(path),
        "doc_id": "",
        "status": "failed",
        "elements": 0,
        "chunks": 0,
        "error": None,
    }

    try:
        report["doc_id"] = compute_doc_id(path)

        from rag.indexing import document_is_indexed, get_client

        if not force and document_is_indexed(get_client(settings), report["doc_id"], settings):
            report["status"] = "skipped"
            logger.info("  %s unchanged since last ingest; skipping", path.name)
            return {"reports": [report]}

        documents = load_document(path, settings, on_progress=sink)
        report["elements"] = len(documents)

        if sink is not None:
            sink(
                {
                    "stage": ProgressStage.CLEAN.value,
                    "message": f"cleaning {len(documents)} element(s)",
                }
            )
        cleaned = clean_documents(documents, settings)

        from rag.chunking import chunk_documents

        if sink is not None:
            sink({"stage": ProgressStage.CHUNK.value, "message": "chunking"})
        chunks = chunk_documents(cleaned, settings)
        report["chunks"] = len(chunks)

        indexed = index_chunks(chunks, settings, on_progress=sink)
        report["status"] = "ok"
        logger.info(
            "  indexed %d chunk(s) from %s", indexed, path.name
        )
    except JobCancelled:
        # Re-raised, never reported: the branch's catch-all below would file
        # the cancelled document as merely "failed" and the graph would then
        # carry on with the next file -- which is the opposite of what a
        # cancellation is asking for.
        logger.info("  cancelled while processing %s", path.name)
        raise
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("  FAILED %s: %s", path.name, report["error"])

    return {"reports": [report]}


def _summarise(state: IngestState) -> dict[str, Any]:
    """Aggregate per-file reports into the final ingest summary."""
    reports = state.get("reports", [])
    ok = sum(1 for r in reports if r["status"] == "ok")
    skipped = sum(1 for r in reports if r["status"] == "skipped")
    failed = sum(1 for r in reports if r["status"] == "failed")
    chunks = sum(r["chunks"] for r in reports if r["status"] == "ok")

    stats: dict[str, Any] = {
        "files_total": len(reports),
        "files_ok": ok,
        "files_skipped": skipped,
        "files_failed": failed,
        "chunks_indexed": chunks,
        "errors": [r["error"] for r in reports if r["error"]],
    }
    logger.info(
        "Ingest complete: %d/%d file(s) ok, %d skipped, %d chunk(s) indexed",
        ok,
        len(reports),
        skipped,
        chunks,
    )
    return {"indexed_count": chunks, "stats": stats}


def build_ingest_graph() -> StateGraph:
    """Assemble the ingest graph; compile with :func:`ingest_graph`."""
    graph = StateGraph(IngestState)
    graph.add_node("discover", _discover)
    graph.add_node("process_file", _process_file)
    graph.add_node("summarise", _summarise)

    graph.set_entry_point("discover")
    graph.add_conditional_edges(
        "discover",
        _fan_out_files,
        path_map=["process_file", "summarise"],
    )
    graph.add_edge("process_file", "summarise")
    graph.set_finish_point("summarise")
    return graph


_INGEST_GRAPH = None


def ingest_graph():
    """Compiled ingest graph, built once per process."""
    global _INGEST_GRAPH
    if _INGEST_GRAPH is None:
        _INGEST_GRAPH = build_ingest_graph().compile()
    return _INGEST_GRAPH


# ---------------------------------------------------------------------------
# Query graph
# ---------------------------------------------------------------------------


def _report(state: QueryState, stage: QueryStage, message: str) -> None:
    """Announce a query stage, if anyone is listening.

    Unlike the ingest sink this never raises: a query has no checkpoint to
    cancel at, and an observer that breaks must not be able to fail the answer
    it is only watching. ``elapsed_ms`` is stamped by :func:`run_query`, which
    wraps the sink once -- so a node never has to know when the run began.
    """
    progress = state.get("progress")
    if progress is not None:
        progress({"stage": stage.value, "message": message})


def _guard(state: QueryState) -> dict[str, Any]:
    """Reject injected questions before anything expensive runs."""
    question = check_input(state["question"])
    # Emitted unconditionally and first, so the UI has something to show
    # immediately: the gap before retrieval is where the user decides the
    # click registered.
    _report(state, QueryStage.GUARD, "question accepted")
    return {"search_query": question}


def _with_top_k(settings: Settings, top_k: int | None) -> Settings:
    """Settings carrying this query's ``top_k`` override.

    A ``model_copy``, deliberately, and not the two obvious alternatives:

    * re-validating a ``model_dump()`` would drop both API keys, because the
      secret fields are declared ``exclude=True`` -- generation would then
      silently turn itself off on any query that passed a top_k;
    * mutating the cached singleton and clearing its cache (what the CLI did
      for ``--top-k``) leaks the override into every concurrent request, so
      two users asking at once would silently get each other's ``top_k``.

    The range is re-checked here because ``model_copy`` skips validation, and
    ``top_k=0`` would otherwise mean "answer every question with no passages"
    rather than an error at the boundary.
    """
    if top_k is None or top_k == settings.top_k:
        return settings
    if top_k < 1 or top_k > settings.fetch_k:
        raise ValueError(
            f"top_k must be between 1 and fetch_k ({settings.fetch_k}); got {top_k}"
        )
    return settings.model_copy(update={"top_k": top_k})


def _retrieve(state: QueryState) -> dict[str, Any]:
    """Hybrid retrieval (+ ensemble fallback + optional neighbour expansion)."""
    _report(state, QueryStage.RETRIEVE, "searching the index")
    candidates = search(state["search_query"], filters=state.get("filters"))
    if not candidates:
        _report(state, QueryStage.RETRIEVE, "no candidates found")
    else:
        plural = "" if len(candidates) == 1 else "s"
        _report(state, QueryStage.RETRIEVE, f"{len(candidates)} candidate chunk{plural}")
    return {"candidates": candidates}


def _rerank(state: QueryState) -> dict[str, Any]:
    """Cross-encoder rerank down to ``top_k``, scoring each kept chunk."""
    settings = _with_top_k(get_settings(), state.get("top_k"))
    candidates = state["candidates"]
    _report(
        state,
        QueryStage.RERANK,
        f"scoring {len(candidates)} candidate{'' if len(candidates) == 1 else 's'}",
    )
    reranked = rerank_documents(state["search_query"], candidates, settings)
    if not reranked:
        logger.warning("No chunks survived reranking for: %s", state["search_query"])
        _report(state, QueryStage.RERANK, "nothing survived the score threshold")
    else:
        _report(state, QueryStage.RERANK, f"kept {len(reranked)} passages")
    return {"reranked": reranked}


def _widen(state: QueryState) -> dict[str, Any]:
    """Attach each surviving passage's neighbours to it, as context.

    Sits *after* rerank and before generate, which is the whole design: the
    reranker picks winners on precise, individually-scored passages, and the
    winners then bring their surroundings along. Expanding before the rerank
    instead -- which is what this pipeline used to offer -- made neighbours
    compete for the same ``top_k`` slots as the passage they sit beside, and a
    neighbour usually loses that contest and takes the anchor's place with it.
    See :func:`rag.retrieval.widen_context`.

    Failure is not an option to worry about: widening degrades to the passages
    alone, because a passage that survived reranking is a complete answer to
    cite even when the chunks around it cannot be read.

    The step reports under :data:`QueryStage.RERANK` rather than a stage of its
    own. It is the tail of assembling the ranked context, it costs a batched
    read rather than a model call, and the trace folds consecutive events of
    one stage into one row -- so a row of its own would add a line to every
    trace and a stage's worth of apparent latency to a step that has none.
    """
    reranked = state["reranked"]
    settings = get_settings()
    if not reranked or settings.context_neighbour_radius < 1:
        return {"reranked": reranked}

    widened = widen_context(reranked, get_client(settings), settings)
    added = len(widened) - len(reranked)
    if added:
        plural = "" if len(reranked) == 1 else "s"
        neighbours = "" if added == 1 else "s"
        _report(
            state,
            QueryStage.RERANK,
            f"kept {len(reranked)} passage{plural}, "
            f"with {added} neighbouring chunk{neighbours} as context",
        )
    return {"reranked": widened}


def _generate(state: QueryState) -> dict[str, Any]:
    """Optional LLM answer over the ranked context, with model fallbacks.

    The context string comes back from :func:`generate_answer` rather than
    being re-rendered here: the block the model was shown and the text the
    grounding guardrail measures overlap against must be identical, and
    rendering it twice is a standing invitation for the two to drift.
    """
    settings = get_settings()
    if settings.enable_generation:
        _report(state, QueryStage.GENERATE, f"asking {describe_model(settings)}")
    else:
        _report(state, QueryStage.GENERATE, "generation is off; using passages only")

    answer, citations, context = generate_answer(state["search_query"], state["reranked"])

    if settings.enable_generation:
        _report(
            state,
            QueryStage.GENERATE,
            "answer ready" if answer else "no answer; every model in the chain failed",
        )
    return {"answer": answer, "citations": citations, "context": context}


def _guard_output(state: QueryState) -> dict[str, Any]:
    """Citation/grounding/PII checks over the generated answer."""
    answer = apply_output_guardrails(
        state.get("answer"), state.get("citations", []), state.get("context", "")
    )
    refused = is_refusal(answer)
    # Reported even when it passes: "checked and clean" is the fact that makes
    # the citations worth trusting, and it is invisible if only failures speak.
    _report(
        state,
        QueryStage.VERIFY,
        "answer withheld by the guardrails" if refused else "citations verified",
    )
    return {"answer": answer, "refused": refused}


def build_query_graph() -> StateGraph:
    """Assemble the query graph; compile with :func:`query_graph`."""
    graph = StateGraph(QueryState)
    graph.add_node("guard", _guard)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rerank", _rerank)
    graph.add_node("widen", _widen)
    graph.add_node("generate", _generate)
    graph.add_node("guard_output", _guard_output)

    graph.set_entry_point("guard")
    graph.add_edge("guard", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "widen")
    graph.add_edge("widen", "generate")
    graph.add_edge("generate", "guard_output")
    graph.set_finish_point("guard_output")
    return graph


_QUERY_GRAPH = None


def query_graph():
    """Compiled query graph, built once per process."""
    global _QUERY_GRAPH
    if _QUERY_GRAPH is None:
        _QUERY_GRAPH = build_query_graph().compile()
    return _QUERY_GRAPH


def run_ingest(
    input_path: str | Path,
    *,
    force_reindex: bool = False,
    progress: ProgressFn | None = None,
) -> IngestState:
    """Run the ingest graph over a file or directory and return its state.

    ``progress`` is called as the run advances and may raise
    :class:`~rag.state.JobCancelled` to abort: the exception travels out
    through the graph, leaving whatever was already written in place but
    unstamped, so the next run redoes the document.
    """
    graph = ingest_graph()
    return graph.invoke(
        {
            "input_path": str(input_path),
            "force_reindex": force_reindex,
            "progress": progress,
        }
    )


def run_query(
    question: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    progress: QueryFn | None = None,
) -> QueryState:
    """Run the query graph and return its final state.

    ``filters`` restricts retrieval to chunks whose metadata matches, e.g.
    ``{"doc_id": "ab12…"}``; ``top_k`` overrides the configured number of
    passages for this query alone.

    ``progress`` is called as each stage is entered and left, so a caller can
    show what the pipeline is doing rather than a spinner. The sink is wrapped
    here to stamp every event with the elapsed time since the run began --
    one place, so no node has to carry the origin, exactly as the ingest
    path's ``_file_sink`` stamps the originating file.

    Raises :class:`~rag.guardrails.InputRejectedError` when input guardrails
    reject the question; callers surface that as a user-facing message.
    """
    started = time.perf_counter()
    sink: QueryFn | None = None
    if progress is not None:

        def sink(event) -> None:
            progress({**event, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})

    graph = query_graph()
    return graph.invoke(
        {"question": question, "filters": filters, "top_k": top_k, "progress": sink}
    )
