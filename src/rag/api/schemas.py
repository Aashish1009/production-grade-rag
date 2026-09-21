"""Request and response bodies for the HTTP API.

Pydantic models rather than dicts for one reason that matters here: the
request models *are* the validation boundary the CLI used to be. A filter with
an unknown metadata key or a ``top_k`` outside the reranker's range must fail
with a clear 400 before it reaches retrieval, where it would quietly return
the wrong passages instead of an error.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from rag.state import FILTERABLE_KEYS

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]

# Values a metadata filter may compare against. Scalar only: the stored
# payload fields are scalars, so a list value would build a filter that
# matches nothing rather than an error, which is the failure mode this module
# exists to prevent.
FilterValue = str | int | float | bool


class HealthResponse(BaseModel):
    """Liveness. Deliberately touches neither the store nor any model."""

    status: Literal["ok"] = "ok"
    version: str


class StageRecord(BaseModel):
    """One ingest stage that has been entered, and how it went.

    Appended as the job moves between stages, so the card can show what
    happened rather than only what is happening. ``seconds`` is filled in when
    the stage is *left* -- recording it on entry would mean writing a duration
    that is still growing.
    """

    model_config = ConfigDict(validate_assignment=True)

    stage: str
    message: str | None = None
    seconds: float | None = None


class Job(BaseModel):
    """One ingest job's live state, as the UI polls it.

    Mutable and updated under the registry's lock; the registry hands out
    ``model_copy(deep=True)`` snapshots so a response being serialised cannot
    be mutated underneath the serializer.

    ``done``/``total`` belong to the *current* stage, not the whole run, and
    mean different units per stage -- pages while parsing, chunks while
    indexing. ``total`` is ``None`` for the stages with nothing countable
    (cleaning, chunking), which is what tells the UI to draw an indeterminate
    bar rather than a fake one.

    ``stages`` accumulates one :class:`StageRecord` per stage entered. Without
    it a job that has reached Indexing can no longer say that it spent four
    minutes parsing -- every stage before the current one is overwritten as it
    is left, so the card could only ever show a stepper with a moving
    highlight and no account of what the work actually was.
    """

    model_config = ConfigDict(validate_assignment=True)

    id: str
    filename: str
    status: JobStatus = "queued"
    stage: str | None = None
    done: int = 0
    total: int | None = None
    message: str | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    chunks: int = 0
    error: str | None = None
    stats: dict[str, Any] | None = None
    cancel_requested: bool = False
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def percent(self) -> float | None:
        """Percent complete *of the current stage*, or ``None`` if unknown.

        ``None`` is a meaningful value here, not a missing one: it is what
        makes the bar indeterminate during cleaning and chunking instead of
        sitting at zero while work is plainly happening.

        A running job never reports 100. The loader announces the batch it is
        *about* to parse, so a one-page PDF sits at 1/1 from the moment the
        parse begins -- and a full bar parked there for the minutes that
        `hi_res` layout detection actually takes reads as "finished" or
        "hung", never as "working". Capping just below 100 until the job
        itself is done keeps the bar honest for every stage.
        """
        if not self.total:
            return None
        if self.status == "done":
            return 100.0
        return round(min(max(self.done, 0) / self.total, 1.0) * 99, 1)


class _QuestionBody(BaseModel):
    """A request body that is nothing but a question.

    Shared by both ask endpoints so the two cannot drift on what a question
    is. The length bound and the blank check are the same question whether it
    is being put to the corpus or to the web.
    """

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class AskRequest(_QuestionBody):
    """A question, optionally scoped to part of the corpus."""

    top_k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Passages to return after reranking. Upper bound is fetch_k, "
            "checked against the live settings rather than here, since the "
            "reranker can only narrow the candidate set, never widen it."
        ),
    )
    filters: dict[str, FilterValue] | None = Field(
        default=None,
        description="Metadata equality filter, e.g. {\"doc_id\": \"ab12cd34\"}.",
    )

    @field_validator("filters")
    @classmethod
    def _known_keys(
        cls, value: dict[str, FilterValue] | None
    ) -> dict[str, FilterValue] | None:
        """Reject unknown filter keys instead of silently matching nothing.

        ``build_filter`` turns any key into ``metadata.<key> == value``, and a
        key that does not exist matches zero points. The user then sees "no
        passages found" for a corpus that is sitting right there, with no hint
        that the filter was the problem. A misspelt ``docID`` should be a 400.
        """
        if not value:
            return None
        unknown = sorted(set(value) - FILTERABLE_KEYS)
        if unknown:
            raise ValueError(
                f"unknown filter key(s): {', '.join(unknown)}. "
                f"Known keys: {', '.join(sorted(FILTERABLE_KEYS))}"
            )
        return value


class WebSearchRequest(_QuestionBody):
    """A question to put to the web instead of the corpus.

    A question and nothing else. ``top_k`` and ``filters`` are absent rather
    than optional: there is no reranker to narrow and no index to scope, so
    accepting them would offer an effect they cannot have -- the same silent
    no-op this module rejects an unknown filter key for.
    """


class Citation(BaseModel):
    """One numbered source behind an answer."""

    marker: str
    source: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    rerank_score: float | None = None
    context_chunks: int = Field(
        default=0,
        description=(
            "Neighbouring chunks rendered inside this citation's block, which "
            "is how much of the passage's surroundings the answer was written "
            "from. They carry no marker of their own: the reranker scored this "
            "passage, so this is the one that is cited. Surfaced because an "
            "answer built from a widened passage reads differently from one "
            "built from a bare fragment, and the difference is otherwise "
            "invisible."
        ),
    )


class AskResponse(BaseModel):
    """The result of one query.

    ``answer`` is ``None`` -- not an empty string -- when no LLM answer exists,
    whether because generation is disabled, every fallback model failed, or
    the question was answered with no context at all. ``refused`` then
    distinguishes the case where the guardrails withheld an answer from the
    ordinary retrieval-only result, so the UI can say which happened.
    """

    question: str
    answer: str | None = None
    refused: bool = False
    context: str = ""
    citations: list[Citation] = Field(default_factory=list)
    generation_enabled: bool = False
    elapsed_ms: float
    web_search_enabled: bool = Field(
        default=False,
        description=(
            "Whether a web search could run at all on this server -- a "
            "backend is configured and generation is on."
        ),
    )
    search_suggested: bool = Field(
        default=False,
        description=(
            "Whether this particular answer came back empty-handed, which is "
            "what the UI offers the search button on. Computed here rather "
            "than in the browser because part of it is a comparison against a "
            "string the *server* just produced -- the model's report that the "
            "sources do not cover the question -- and the frontend would need "
            "its own copy of that constant to reach the same verdict."
        ),
    )


class WebSource(BaseModel):
    """One web page behind a web answer.

    ``snippet`` is the search engine's summary and may be absent; the page
    itself is not returned -- only what the answer was built from, which is
    the citation, not the text.
    """

    marker: str
    title: str
    url: str
    snippet: str | None = None


class WebAskResponse(BaseModel):
    """The result of one web search.

    ``error`` is an ordinary outcome, not an exception: a search that finds
    nothing, a rate limit, a backend that is not configured. The section says
    what happened rather than the request failing, because there is already a
    corpus answer on screen and this is an addition to it.

    ``answer`` is ``None`` alongside a non-empty ``sources`` when generation
    is off or every model failed -- the same graceful degradation as the
    corpus path, where the pages are still worth showing.
    """

    question: str
    answer: str | None = None
    refused: bool = False
    sources: list[WebSource] = Field(default_factory=list)
    provider: str | None = None
    error: str | None = None
    generation_enabled: bool = False
    elapsed_ms: float


class DocumentSummary(BaseModel):
    """One indexed source document, folded out of its chunks."""

    doc_id: str
    source_name: str
    source: str | None = None
    chunks: int
    page_start: int | None = None
    page_end: int | None = None
    complete: bool = True


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int


class WarmResponse(BaseModel):
    """What is resident after a warm-up request.

    ``loaded`` is a list of names rather than a bare boolean so the trace and
    the logs say *which* model was rebuilt: a partial warm-up (one model
    loaded, one failed) is a different problem from a cold start, and a
    boolean would report both as "not warm yet".
    """

    loaded: list[str] = Field(default_factory=list)
    models_loaded: bool = False


class StatsResponse(BaseModel):
    """Collection and model state, for the status bar.

    No document count: folding one out of the store means a full scroll, and
    this endpoint is polled. The document list is its own endpoint and the UI
    already has it.
    """

    collection: str
    exists: bool
    points: int
    dense_model: str
    sparse_model: str
    reranker_model: str
    device: str
    models_loaded: bool = Field(
        default=False,
        description=(
            "Whether the models are in memory right now. False means the next "
            "question pays a one-time model load of roughly 90 seconds, which "
            "is what the UI reports instead of appearing to hang -- and why "
            "it starts the load on page open, so the wait overlaps the "
            "visitor reading or uploading. See rag/model_lifecycle.py."
        ),
    )
    retrieval_mode: str
    fetch_k: int
    top_k: int
    max_upload_mb: int
    generation_enabled: bool
    llm_provider: str | None = None
    web_search_provider: str | None = Field(
        default=None,
        description=(
            "The web-search backend in use, or ``None`` if none is available. "
            "Reported so the UI can say why the search button never appears "
            "rather than leaving the user to guess."
        ),
    )
