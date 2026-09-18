"""In-process ingest job registry.

Ingest is a multi-minute CPU job on a 600-page book, so it cannot run inside
the request that uploaded the file: the upload returns a job id immediately
and the UI polls. Two structural facts shape the rest of this module.

**One writer.** The embedded Qdrant store is single-process, and ingest is CPU
bound anyway, so jobs run on a ``ThreadPoolExecutor(max_workers=1)``: queued
uploads wait their turn rather than interleaving their embeddings and fighting
over the same lock. Queries are *not* serialised behind them -- they use the
store guard per operation, so a question asked mid-ingest waits a slice, not a
book.

**Cancellation is cooperative.** Nothing can interrupt a PDF page being parsed
inside Unstructured. What can be done is to check at every progress
checkpoint, which is where :class:`~rag.state.JobCancelled` is raised from.
A cancel therefore takes effect within one page batch (~seconds), and the
request is recorded as ``cancel_requested`` immediately so the UI can show
that it was heard while it waits.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rag.api.schemas import Job, StageRecord
from rag.config import Settings, get_settings
from rag.graphs import run_ingest
from rag.logging_utils import get_logger
from rag.state import JobCancelled, ProgressEvent

logger = get_logger(__name__)


def interpret(stats: dict[str, Any] | None) -> tuple[str, str | None]:
    """Turn an ingest summary into ``(status, error)`` for one uploaded file.

    The three outcomes are not interchangeable and each has a distinct
    remedy, so the summary is read rather than reduced to "failed or not":

    * ``skipped`` -- byte-identical to an earlier ingest. A success: the
      document is in the index, and ``chunks_indexed`` is legitimately 0.
    * ``ok`` with zero chunks -- parsing worked, nothing survived cleaning and
      chunking. Usually a scanned PDF with no text layer and no OCR. Calling
      this a success would leave the user with an indexed "document" that can
      never be retrieved, so it is reported as a failure with that
      explanation.
    * anything else -- the per-file error from the report.
    """
    if not stats:
        return "failed", "The ingest produced no summary."

    if stats.get("files_skipped"):
        return "done", None
    if stats.get("files_ok"):
        if stats.get("chunks_indexed"):
            return "done", None
        return "failed", (
            "No ingestable text was found in this file. If it is a scanned "
            "PDF, its pages are images with no text layer -- try "
            "RAG_PDF_STRATEGY=ocr_only."
        )
    errors = [e for e in stats.get("errors", []) if e]
    return "failed", errors[0] if errors else "The document could not be ingested."


class JobRegistry:
    """Submit, poll and cancel ingest jobs.

    Every mutation happens under one lock, and reads return deep copies.
    The worker thread writes progress while request threads serialise
    snapshots, so without the copy a pydantic model could be mid-mutation
    as it was being dumped to JSON.
    """

    def __init__(self, settings: Settings | None = None, *, max_workers: int = 1) -> None:
        self.settings = settings or get_settings()
        self._jobs: dict[str, Job] = {}
        self._cancelled: set[str] = set()
        # When the stage each job is currently in began, so leaving it can be
        # given a duration. Kept here rather than on the Job because it is
        # bookkeeping about a transition, not part of what a client is told.
        self._stage_started: dict[str, float] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rag-ingest"
        )

    # --- reads -------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job is not None else None

    def list(self) -> list[Job]:
        """Jobs, newest first."""
        with self._lock:
            jobs = [job.model_copy(deep=True) for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    # --- writes ------------------------------------------------------------

    def submit(self, filename: str, path: Path) -> Job:
        """Queue an ingest of ``path`` and return the job to poll."""
        job = Job(
            id=uuid.uuid4().hex[:12],
            filename=filename,
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job.id, Path(path))
        return job.model_copy(deep=True)

    def cancel(self, job_id: str) -> Job | None:
        """Ask the running job to stop at its next checkpoint.

        Returns the job with ``cancel_requested`` set, or ``None`` if there is
        no such job. A job that has already finished is left as it is: flipping
        a completed job to cancelled would misreport work that really did land
        in the index.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"queued", "running"}:
                self._cancelled.add(job_id)
                job.cancel_requested = True
            return job.model_copy(deep=True)

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop the worker pool. Called on app shutdown and by tests."""
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # --- worker ------------------------------------------------------------

    def _run(self, job_id: str, path: Path) -> None:
        self._update(job_id, status="running", started_at=time.time())
        try:
            state = run_ingest(path, progress=self._sink(job_id))
        except JobCancelled:
            logger.info("ingest job %s cancelled", job_id)
            self._update(job_id, status="cancelled", stage=None, message="cancelled")
        except Exception as exc:
            logger.error("ingest job %s failed: %s: %s", job_id, type(exc).__name__, exc)
            self._update(
                job_id,
                status="failed",
                stage=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            stats = state.get("stats") or {}
            status, error = interpret(stats)
            self._update(
                job_id,
                status=status,
                stage=None,
                chunks=stats.get("chunks_indexed", 0),
                stats=stats,
                error=error,
            )
        finally:
            self._update(job_id, finished_at=time.time())
            with self._lock:
                self._cancelled.discard(job_id)

    def _close_stage(self, job: Job, job_id: str) -> None:
        """Give the stage being left a duration. The caller holds the lock.

        Popped rather than read, so a stage can never be closed twice -- which
        would otherwise let the terminal ``stage=None`` update bill the whole
        remaining run to whichever stage happened to be last.
        """
        started = self._stage_started.pop(job_id, None)
        if started is None or not job.stages:
            return
        job.stages[-1].seconds = round(time.time() - started, 1)

    def _sink(self, job_id: str):
        """Progress sink for one job: record the event, or raise to cancel."""

        def sink(event: ProgressEvent) -> None:
            with self._lock:
                if job_id in self._cancelled:
                    raise JobCancelled(f"ingest job {job_id} was cancelled")
                job = self._jobs.get(job_id)
                if job is None:  # pragma: no cover - the registry owns the job
                    return

                stage = event.get("stage")
                if stage and stage != job.stage:
                    self._close_stage(job, job_id)
                    job.stage = stage
                    self._stage_started[job_id] = time.time()
                    job.stages.append(StageRecord(stage=stage))

                if "done" in event:
                    job.done = int(event["done"])
                # ``total`` is only overwritten when the event carries one, so
                # a stage that reports no total does not blank the previous
                # stage's denominator and make the bar jump.
                if event.get("total"):
                    job.total = int(event["total"])
                if "message" in event:
                    message = event.get("message")
                    job.message = message
                    # The live line and the recorded one are the same fact kept
                    # in two places, so they are written together: the last
                    # message a stage reported before it was left *is* its
                    # outcome, and updating them separately is how the card
                    # ends up disagreeing with its own history.
                    if job.stages:
                        job.stages[-1].message = message

        return sink

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:  # pragma: no cover - the registry owns the job
                return
            # Closed before the field loop, while ``job.stage`` is still the
            # stage being left -- the loop is about to set it to ``None``.
            closing = "stage" in fields and fields["stage"] is None
            if closing:
                self._close_stage(job, job_id)
            for key, value in fields.items():
                setattr(job, key, value)
            if closing:
                job.done = 0
                job.total = None
