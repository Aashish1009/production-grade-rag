"""FastAPI application: the HTTP surface over the pipeline, plus the UI.

**Why a server at all.** The embedded Qdrant store takes an exclusive lock on
its directory, so exactly one process may own it. A browser cannot import
Python, so something has to sit between them, and that something must be the
process holding the store -- hence an in-process server rather than a CLI the
UI shells out to (two processes, one deadlock).

**Threading model.** Handlers that touch the store are declared ``def``, not
``async def``, so Starlette runs them in its threadpool. That matters: a
handler waiting on the store guard would otherwise block the event loop, and
the whole UI would freeze for the duration of an ingest slice -- including the
``GET /api/jobs`` polls that exist to show the ingest progressing. Only the
upload handler is async, because it awaits the request body.

**Security posture.** Loopback only (``api_host`` defaults to 127.0.0.1), no
authentication, and it can read and delete the index. That is the right trade
for a local single-user tool and the wrong one for anything else, so binding
it to a public interface is an explicit ``RAG_API_HOST`` change rather than a
default, and ``/api/config`` is safe to expose because the key fields are
``exclude=True`` on the Settings model -- a guarantee that is now load-bearing
and is pinned by a test.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag.api.jobs import JobRegistry
from rag.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    DeleteResponse,
    DocumentSummary,
    HealthResponse,
    Job,
    StatsResponse,
    WebAskResponse,
    WebSearchRequest,
    WebSource,
)
from rag.config import PROJECT_ROOT, Settings, get_settings
from rag.graphs import run_query
from rag.guardrails import InputRejectedError, is_not_covered
from rag.indexing import (
    close_client,
    collection_stats,
    count_document_chunks,
    delete_document_chunks,
    get_client,
    list_documents,
)
from rag.loaders import SUPPORTED_SUFFIXES
from rag.logging_utils import get_logger, setup_logging
from rag.state import QueryFn
from rag.websearch import run_web_search

logger = get_logger(__name__)

# Read size while streaming an upload. FastAPI spools the body to a temp file
# past 1MB, so this loop copies between two files rather than holding a
# 600-page book in memory.
_UPLOAD_CHUNK_BYTES = 1 << 20

# How a streamed pipeline hands a line to the response generator. ``None``
# ends the stream, which is why the payload is optional.
EmitFn = Callable[[dict[str, Any] | None], None]

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _package_version() -> str:
    try:
        return version("rag")
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        return "unknown"


def _validation_detail(exc: RequestValidationError) -> str:
    """Flatten FastAPI's validation errors into one readable sentence."""
    parts = []
    for error in exc.errors():
        # Skip the leading "body"/"query" loc segment: it is an implementation
        # detail of how the value arrived, not something the user typed.
        location = ".".join(str(item) for item in error["loc"][1:]) or "request"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or "Malformed request."


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. A factory so tests get their own registry."""
    settings = settings or get_settings()
    registry = JobRegistry(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        # Both of these used to live in cli.py's bootstrap, which died with
        # the CLI. Without them a fresh checkout logs nowhere and the first
        # ingest fails creating the store directory.
        setup_logging(settings.log_level)
        settings.ensure_directories()
        logger.info(
            "RAG API listening on http://%s:%d (store: %s)",
            settings.api_host,
            settings.api_port,
            settings.qdrant_path,
        )
        try:
            yield
        finally:
            # Release the embedded store's storage lock before the process
            # exits, so the next start is not refused by a stale lock.
            registry.shutdown(wait=False)
            close_client()

    app = FastAPI(
        title="Local RAG Pipeline",
        version=_package_version(),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Report malformed input as 400 rather than FastAPI's default 422.

        One error code for "you sent something I cannot use" keeps the
        contract small enough for the UI to act on without a lookup table of
        which 4xx means what. The detail is unchanged and still names the
        offending field.
        """
        return JSONResponse(status_code=400, content={"detail": _validation_detail(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error in the API")
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}"},
        )

    # --- liveness ----------------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness only: no store, no model, no lock. Safe to poll."""
        return HealthResponse(version=_package_version())

    # --- ingest jobs -------------------------------------------------------

    @app.post("/api/documents", response_model=Job, status_code=202)
    async def upload_document(file: UploadFile = File(...)) -> Job:
        """Accept a file, stream it to disk, and queue it for ingest.

        The name is reduced to its basename before use: a client is free to
        send ``../../.ssh/authorized_keys`` as a filename, and nothing here
        should ever honour a path it was handed. Each upload gets its own
        directory so two files with the same name cannot overwrite each other,
        while the stored name stays the one the user recognises -- it becomes
        ``source_name`` in every citation.
        """
        name = Path(file.filename or "").name.strip()
        suffix = Path(name).suffix.lower()
        if not name:
            raise HTTPException(400, detail="The upload has no filename.")
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                400,
                detail=(
                    f"Unsupported file type '{suffix or name}'. Supported: "
                    f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
                ),
            )

        dest = settings.uploads_dir / uuid.uuid4().hex[:12] / name
        dest.parent.mkdir(parents=True, exist_ok=True)

        limit = settings.max_upload_mb * 1024 * 1024
        written = 0
        oversized = False
        with dest.open("wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    oversized = True
                    break
                out.write(chunk)

        # The UI checks the size before it starts, so this is the backstop for
        # a caller that did not (curl, a script). It stops reading at the
        # limit rather than draining the rest of an oversized body: finishing
        # the read to deliver a nicer error would mean writing the gigabytes
        # the limit exists to refuse.
        if oversized:
            shutil.rmtree(dest.parent, ignore_errors=True)
            raise HTTPException(
                413,
                detail=(
                    f"File exceeds the {settings.max_upload_mb} MB limit "
                    f"(RAG_MAX_UPLOAD_MB)."
                ),
            )
        if not written:
            shutil.rmtree(dest.parent, ignore_errors=True)
            raise HTTPException(400, detail="The uploaded file is empty.")

        logger.info("received %s (%.1f MB); queueing ingest", name, written / (1 << 20))
        return registry.submit(name, dest)

    @app.get("/api/jobs", response_model=list[Job])
    def list_jobs() -> list[Job]:
        return registry.list()

    @app.get("/api/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        job = registry.get(job_id)
        if job is None:
            raise HTTPException(404, detail=f"No such job: {job_id}")
        return job

    @app.post("/api/jobs/{job_id}/cancel", response_model=Job)
    def cancel_job(job_id: str) -> Job:
        """Request cancellation; the job stops at its next checkpoint.

        Returns the job as it stands, with ``cancel_requested`` set. It is
        not immediately ``cancelled`` because nothing can interrupt a page
        being parsed -- the honest answer is "asked", and the UI shows that
        until the status actually changes.
        """
        job = registry.cancel(job_id)
        if job is None:
            raise HTTPException(404, detail=f"No such job: {job_id}")
        return job

    # --- query -------------------------------------------------------------

    def _execute_ask(payload: AskRequest, on_event: QueryFn | None) -> AskResponse:
        """Run one query, optionally reporting each stage to ``on_event``.

        Shared by the plain and streamed endpoints so the two can only ever
        differ in how the result travels, never in what it says.
        """
        started = time.perf_counter()

        # Checked here rather than in the request model because the ceiling is
        # a live setting: the reranker can only narrow fetch_k candidates, so
        # a larger top_k would promise passages that cannot exist.
        if payload.top_k is not None and payload.top_k > settings.fetch_k:
            raise HTTPException(
                400,
                detail=(
                    f"top_k ({payload.top_k}) exceeds fetch_k "
                    f"({settings.fetch_k}), the number of candidates retrieved."
                ),
            )

        try:
            state = run_query(
                payload.question,
                filters=payload.filters,
                top_k=payload.top_k,
                progress=on_event,
            )
        except InputRejectedError as exc:
            # A guardrail rejection is a statement about the request, not a
            # server fault: 400, with the guardrail's own wording.
            raise HTTPException(400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        # "" and None both mean "no LLM answer" -- generation disabled, every
        # fallback model down, or a refusal that already said its piece.
        answer = (state.get("answer") or "").strip() or None
        refused = bool(state.get("refused"))

        # The three ways a corpus answer comes back empty-handed: generation
        # produced nothing, a guardrail withheld it, or the model read the
        # passages and reported that none of them cover the question. The last
        # is the common one, and the only one the prose alone cannot identify
        # -- which is why the prompt requires the sentence and is_not_covered
        # matches on it.
        #
        # Generation is required for the suggestion even though the web path
        # runs without it: the button promises an *answer* from the web, and
        # with no model there is none to give, only pages to read.
        web_ready = settings.resolved_web_provider is not None and settings.enable_generation

        return AskResponse(
            question=payload.question,
            answer=answer,
            refused=refused,
            context=state.get("context") or "",
            citations=[Citation(**c) for c in state.get("citations", [])],
            generation_enabled=settings.enable_generation,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            web_search_enabled=web_ready,
            search_suggested=web_ready
            and (answer is None or refused or is_not_covered(answer)),
        )

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        """Retrieve, rerank, and optionally answer with citations.

        The non-streaming twin of ``/api/ask/stream``: same pipeline, same
        result, no stage events. It stays because it is what a script or a
        ``curl`` wants -- and because a client that cannot read a stream
        should still be able to ask a question.
        """
        return _execute_ask(payload, None)

    def _ndjson(run: Callable[[EmitFn], None]) -> StreamingResponse:
        """Relay a blocking pipeline over a newline-delimited JSON stream.

        Shared by both streaming endpoints, which differ only in what they
        compute. The pipeline is blocking CPU work, so it runs in a thread and
        hands events back through a queue while the generator stays on the
        event loop -- so a slow rerank cannot stall the rest of the server,
        the same reason every store-touching handler in this module is a plain
        ``def``.

        ``run`` does the work and emits its own ``result`` line. Failures are
        turned into ``error`` lines here, because the status line went out
        before the pipeline ever ran.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def emit(item: dict[str, Any] | None) -> None:
            # Reachable from the worker thread, where touching the queue
            # directly would be a data race: asyncio.Queue is not thread-safe.
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def work() -> None:
            try:
                run(emit)
            except HTTPException as exc:
                # A rejected question is an answer about the request, delivered
                # on the stream: the status line is already sent by now, so the
                # client learns it from this event instead.
                emit({"type": "error", "detail": exc.detail, "status": exc.status_code})
            except Exception as exc:  # pragma: no cover - the handler's backstop
                logger.exception("streamed query failed")
                emit(
                    {
                        "type": "error",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "status": 500,
                    }
                )
            finally:
                emit(None)

        async def lines() -> AsyncGenerator[str, None]:
            running = loop.run_in_executor(None, work)
            try:
                while (item := await queue.get()) is not None:
                    yield json.dumps(item) + "\n"
            finally:
                # Lets the worker finish and log its own failure rather than
                # leaving a thread running against a client that has gone.
                await running

        return StreamingResponse(
            lines(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/ask/stream")
    async def ask_stream(payload: AskRequest) -> StreamingResponse:
        """Answer a question, reporting each pipeline stage as it happens.

        NDJSON over a POST rather than SSE over an EventSource, for two
        reasons: the question is a JSON body (an EventSource can only GET, so
        it would have to be crammed into a query string, URL-encoded), and the
        browser already talks to this API with ``fetch``. One JSON object per
        line; the stages come first and the last line is the same
        ``AskResponse`` that ``POST /api/ask`` returns -- so a client that
        ignores the stages sees no difference between the two endpoints.
        """

        def run(emit: EmitFn) -> None:
            response = _execute_ask(
                payload, lambda event: emit({"type": "stage", **event})
            )
            emit({"type": "result", **response.model_dump()})

        return _ndjson(run)

    def _execute_web_search(
        payload: WebSearchRequest, on_event: QueryFn | None
    ) -> WebAskResponse:
        """Run one web search, optionally reporting each stage to ``on_event``.

        The twin of :func:`_execute_ask`, minus the HTTP plumbing.
        """
        started = time.perf_counter()
        try:
            state = run_web_search(payload.question, progress=on_event)
        except InputRejectedError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        return WebAskResponse(
            question=payload.question,
            answer=(state.get("answer") or "").strip() or None,
            refused=bool(state.get("refused")),
            sources=[WebSource(**s) for s in state.get("sources", [])],
            provider=state.get("provider"),
            error=state.get("error"),
            generation_enabled=settings.enable_generation,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    @app.post("/api/web/stream")
    async def web_stream(payload: WebSearchRequest) -> StreamingResponse:
        """Search the web for a question, and answer from what it finds.

        The fallback for a question the corpus could not answer, and the only
        route in this module that sends anything off the machine. It is a
        separate endpoint rather than a flag on ``/api/ask`` for exactly that
        reason: nothing reaches the network unless a user asked for it in so
        many words, and no request body can turn it on by accident.

        Streamed for the same reason the corpus endpoint is -- a search, three
        page fetches and a generation is tens of seconds, and the stage events
        are what keep that from reading as a hang.
        """

        def run(emit: EmitFn) -> None:
            response = _execute_web_search(
                payload, lambda event: emit({"type": "stage", **event})
            )
            emit({"type": "result", **response.model_dump()})

        return _ndjson(run)

    # --- corpus ------------------------------------------------------------

    @app.get("/api/documents", response_model=list[DocumentSummary])
    def get_documents() -> list[DocumentSummary]:
        return [DocumentSummary(**doc) for doc in list_documents(settings)]

    @app.delete("/api/documents/{doc_id}", response_model=DeleteResponse)
    def delete_document(doc_id: str) -> DeleteResponse:
        """Drop every chunk of one document.

        Deletes the source file too, so a later ingest of the uploads
        directory cannot resurrect what the user just removed.
        """
        removed = count_document_chunks(doc_id, settings)
        if not removed:
            raise HTTPException(404, detail=f"No indexed chunks for '{doc_id}'.")

        delete_document_chunks(get_client(settings), doc_id, settings)

        for document in list_documents(settings):
            if document["doc_id"] == doc_id:
                source = document.get("source")
                if source and Path(source).is_file():
                    # Only ever inside the uploads directory: a document
                    # indexed from the user's own tree must not have its
                    # original deleted as a side effect of unindexing it.
                    try:
                        Path(source).resolve().relative_to(settings.uploads_dir)
                    except ValueError:
                        logger.info("left %s in place; it is outside the uploads dir", source)
                    else:
                        shutil.rmtree(Path(source).parent, ignore_errors=True)
                break

        logger.info("deleted %d chunk(s) for document %s", removed, doc_id)
        return DeleteResponse(doc_id=doc_id, deleted_chunks=removed)

    @app.get("/api/stats", response_model=StatsResponse)
    def get_stats() -> StatsResponse:
        info = collection_stats(settings)
        return StatsResponse(
            **info,
            dense_model=settings.dense_model,
            sparse_model=settings.sparse_model,
            reranker_model=settings.active_reranker_model,
            device=settings.resolved_device,
            retrieval_mode=settings.retrieval_mode.value,
            fetch_k=settings.fetch_k,
            top_k=settings.top_k,
            max_upload_mb=settings.max_upload_mb,
            generation_enabled=settings.enable_generation,
            llm_provider=settings.resolved_llm_provider,
            web_search_provider=settings.resolved_web_provider,
        )

    @app.get("/api/config")
    def get_config() -> Response:
        """Every effective setting, secrets excluded.

        ``model_dump_json`` is used directly, so FastAPI never re-serialises
        the model and cannot reintroduce a field the Settings model excludes.
        """
        return Response(
            content=settings.model_dump_json(indent=2),
            media_type="application/json",
        )

    # --- the UI ------------------------------------------------------------

    # Mounted last so it cannot shadow an /api route. ``html=True`` serves
    # index.html for "/" but 404s an unknown path, which is deliberate: the
    # app has no client-side routes, so a catch-all would only turn a
    # mistyped API path into a page of HTML that a JSON client cannot parse.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
        logger.info("serving the web UI from %s", FRONTEND_DIST)
    else:
        logger.warning(
            "No built UI at %s -- the API works, but there is nothing to "
            "serve at /. Run `npm install && npm run build` in frontend/.",
            FRONTEND_DIST,
        )

    return app


app = create_app()
