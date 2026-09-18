"""Web search: the fallback for a question the corpus cannot answer.

This is the one path in the pipeline that leaves the machine. Everything else
-- parsing, embedding, retrieval, reranking -- is local, and the app says so on
its own header. So the shape here is deliberately narrow: nothing in this
module runs unless a user clicked a button offering it, and the corpus answer
it supplements is already on screen before it does.

**Two backends, one interface.** :func:`collect_web_sources` returns
:class:`~rag.state.WebPage` objects either way, so the answer stage and the
guardrails never learn which one ran:

* *Tavily* runs the search and fetches each result server-side, returning the
  page text in the same response. One request for the whole fallback. It goes
  through LangChain's ``TavilySearch`` tool, the same way retrieval goes
  through ``EnsembleRetriever`` and generation through ``ChatLiteLLM``.
* *DuckDuckGo* (through ``ddgs``) is keyless and returns snippets only, so the
  top results are fetched here and their main content extracted. There is no
  LangChain tool for it that is not the same scraper behind another name, so
  this half speaks to the library directly.

Neither is allowed to raise into the answer path. Asking the open web fails in
ways that are nobody's fault -- a rate limit, a page that 404s, a host that
refuses a non-browser user agent -- and every one of them degrades to fewer
sources, or to an error line the UI can show, rather than to a failed request.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx
from langchain_core.tools import ToolException

from rag.config import Settings, get_settings
from rag.generation import describe_model, generate_web_answer
from rag.guardrails import (
    InputRejectedError,
    apply_output_guardrails,
    check_input,
    is_refusal,
)
from rag.logging_utils import get_logger
from rag.state import QueryEvent, QueryFn, QueryStage, WebPage, WebSearchState

logger = get_logger(__name__)

# Honest about what this is rather than impersonating a browser. Some sites
# refuse a non-browser agent, and those pages are simply skipped -- pretending
# otherwise would work more often, at the cost of the one claim this module
# makes about itself. The snippet fallback in :func:`_duckduckgo_search` is
# what keeps a refusal from costing the answer entirely.
_USER_AGENT = "rag-local/0.1 (local document assistant)"

_LABEL = {"tavily": "Tavily", "duckduckgo": "DuckDuckGo"}

_TRUNCATION = "\n[... page truncated ...]"

_NOT_CONFIGURED = (
    "Web search is not available: no Tavily key is set and the keyless "
    "backend's library (ddgs) is not installed."
)

_NO_RESULTS = (
    "The search returned nothing that could be read. Try rephrasing the "
    "question, or search for it directly."
)


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    """Cap a page's text, cutting at a word and saying that it was cut.

    The marker matters: without it the model sees a passage that stops
    mid-thought and may treat the break as the end of the source's claim.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip() + _TRUNCATION


def extract_readable_text(html: str, url: str | None = None) -> str:
    """Pull the main content out of an HTML page, dropping the chrome.

    Public because it is the one piece of the fetch path worth testing on its
    own, against a saved page rather than the network.

    ``trafilatura`` rather than ``unstructured``'s HTML partitioner, which is
    already a dependency: the partitioner keeps navigation, cookie banners and
    footers, and those tokens would go straight into the prompt as though the
    page had said them.
    """
    try:
        import trafilatura
    except ImportError:  # pragma: no cover - declared dependency
        logger.warning(
            "trafilatura is not installed; the keyless backend cannot read "
            "pages and will fall back to search snippets"
        )
        return ""
    try:
        return (
            trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
            )
            or ""
        ).strip()
    except Exception as exc:  # pragma: no cover - malformed markup
        logger.info("could not extract text from %s (%s)", url, type(exc).__name__)
        return ""


def _fetch_text(url: str, settings: Settings) -> str:
    """Fetch one URL and return its readable text, or ``""`` if it will not come."""
    try:
        response = httpx.get(
            url,
            timeout=settings.web_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except Exception as exc:
        logger.info(
            "web fetch: %s did not load (%s: %s)", url, type(exc).__name__, exc
        )
        return ""

    # Checked before parsing: the point is to not hand a decompression bomb
    # or a mislabelled disk image to an HTML parser.
    if len(response.content) > settings.web_page_max_bytes:
        logger.info(
            "web fetch: %s is %d bytes, over the %d cap; skipping",
            url,
            len(response.content),
            settings.web_page_max_bytes,
        )
        return ""

    return extract_readable_text(response.text, url=url)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

Report = Callable[[QueryStage, str], None]
"""Sink a step uses to announce itself. Wrapped by :func:`run_web_search`."""


def _tavily_tool(settings: Settings) -> Any:
    """Build the Tavily search tool for this pipeline.

    A fresh instance per call rather than a cached one, because the tool takes
    its configuration at *construction* -- ``max_results``, ``search_depth``
    and ``include_raw_content`` are rejected as invocation arguments -- and a
    cached tool would pin the settings of whichever query happened to run
    first. The object is a key and four scalars, so building it is free.

    ``include_raw_content`` is what makes the whole fallback one request: the
    tool asks Tavily to fetch and parse each result server-side, so the pages
    arrive with the search results and the keyless path's separate fetch loop
    has nothing to do.
    """
    from langchain_tavily import TavilySearch

    return TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        api_base_url=settings.tavily_api_base_url,
        max_results=settings.web_search_max_results,
        search_depth=settings.web_search_depth,
        include_raw_content=True,
    )


def _tavily_error(error: Any) -> str:
    """Turn the tool's failure payload into something a reader can act on.

    The tool does not raise on an API error -- it catches every exception and
    returns ``{"error": <exception>}`` in the result. So a rate limit and a
    rejected key both arrive here as the string of an exception raised three
    layers down, and the only place the distinction can be recovered is here.
    """
    text = str(error)
    lowered = text.lower()
    if "429" in text or "rate limit" in lowered:
        return (
            "Tavily is rate limiting this key. The free tier allows 1,000 "
            "searches a month. Try again later, or set "
            "RAG_WEB_SEARCH_PROVIDER=duckduckgo to use the keyless backend."
        )
    if "401" in text or "unauthorized" in lowered or "invalid api key" in lowered:
        return "Tavily rejected the API key. Check TAVILY_API_KEY."
    return f"Tavily could not complete the search: {text}"


def _tavily_search(question: str, settings: Settings, report: Report) -> list[WebPage]:
    """Search with Tavily, which returns each result's text alongside it."""
    try:
        payload = _tavily_tool(settings).invoke({"query": question})
    except ToolException:
        # The tool raises this for an empty result set, which is an ordinary
        # outcome of asking the open web and not a failure to report.
        report(QueryStage.FETCH, "the search returned nothing to read")
        return []

    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(_tavily_error(payload["error"]))

    results = (payload or {}).get("results") or []

    # Tavily has already fetched every page it returns, so this reports work
    # that happened rather than work about to happen. It is still its own
    # stage: the user is watching a trace, and "read 5 pages" is the fact that
    # explains why the answer took as long as it did.
    report(QueryStage.FETCH, f"read {len(results)} page{_plural(results)}")

    pages: list[WebPage] = []
    for result in results:
        # raw_content is the page; content is the search engine's summary of
        # it. Falling back keeps a result Tavily could not fetch usable as a
        # source in its own right, which is what the snippet is.
        text = _as_str(result.get("raw_content")) or _as_str(result.get("content"))
        if not text:
            continue
        url = _as_str(result.get("url"))
        pages.append(
            WebPage(
                title=_as_str(result.get("title")) or url or "untitled",
                url=url,
                snippet=_as_str(result.get("content")),
                text=_truncate(text, settings.web_max_page_chars),
            )
        )
    return pages


def _duckduckgo_search(
    question: str, settings: Settings, report: Report
) -> list[WebPage]:
    """Search DuckDuckGo for snippets, then read the top results.

    Imported inside the function because ``ddgs`` is optional: a checkout
    configured for Tavily never loads it, and the module has to stay
    importable without it.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.text(question, max_results=settings.web_search_max_results))

    results = [
        (
            _as_str(item.get("title")),
            _as_str(item.get("href")),
            _as_str(item.get("body")),
        )
        for item in raw
    ]
    results = [item for item in results if item[1]]
    if not results:
        return []

    wanted = results[: settings.web_fetch_pages]
    report(
        QueryStage.FETCH,
        f"reading {len(wanted)} of {len(results)} result page{_plural(results)}",
    )

    pages: list[WebPage] = []
    for title, url, snippet in wanted:
        text = _fetch_text(url, settings)
        if not text:
            continue
        pages.append(
            WebPage(
                title=title or url,
                url=url,
                snippet=snippet,
                text=_truncate(text, settings.web_max_page_chars),
            )
        )

    if pages:
        return pages

    # Every page refused to load. The snippets are still search-result text
    # describing the very question that was asked, and answering from them
    # beats answering from nothing -- which is what returning an empty list
    # here would amount to.
    logger.info("no result pages could be read; falling back to search snippets")
    return [
        WebPage(
            title=title or url,
            url=url,
            snippet=snippet,
            text=_truncate(snippet, settings.web_max_page_chars),
        )
        for title, url, snippet in wanted
        if snippet
    ]


def _as_str(value: Any) -> str:
    """Coerce a JSON field to a stripped string.

    Search APIs are loose about types -- a title that is ``null`` or a count
    that is a number in one response and a string in the next -- and a
    ``None`` reaching ``str.strip`` deep inside the answer path would be a
    crash in the one place that must not have one.
    """
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _plural(items: list[Any] | int) -> str:
    count = items if isinstance(items, int) else len(items)
    return "" if count == 1 else "s"


# ---------------------------------------------------------------------------
# the fallback, end to end
# ---------------------------------------------------------------------------


def _no_answer(provider: str | None, error: str) -> WebSearchState:
    """A failed search, in the same shape as a successful one.

    Every key the successful path returns is present here as well -- as
    ``None`` or empty -- so a caller never has to ask which kind of outcome it
    is holding before reading it. The API does not care, but a caller that
    reached for ``state["answer"]`` on one branch and ``state.get("answer")``
    on another is exactly how a missing key becomes an AttributeError in
    production.
    """
    return {
        "provider": provider,
        "pages": [],
        "sources": [],
        "context": "",
        "answer": None,
        "refused": False,
        "error": error,
    }


def collect_web_sources(
    question: str,
    settings: Settings | None = None,
    report: Report | None = None,
) -> list[WebPage]:
    """Search and read, returning whatever could be read.

    Returns an empty list rather than raising when the search fails: the
    caller has an answer on screen already, and a dead search is a message
    under it, not a failed request.
    """
    settings = settings or get_settings()
    provider = settings.resolved_web_provider
    if provider is None:  # pragma: no cover - guarded by run_web_search
        return []

    emit = report or (lambda stage, message: None)

    if provider == "tavily":
        return _tavily_search(question, settings, emit)
    return _duckduckgo_search(question, settings, emit)


def run_web_search(
    question: str,
    *,
    progress: QueryFn | None = None,
) -> WebSearchState:
    """Answer a question from the web: search, read, generate, verify.

    The mirror of :func:`rag.graphs.run_query`, down to wrapping the progress
    sink so ``elapsed_ms`` is stamped here rather than by each step. There is
    no graph to hand that to -- this is a straight line of four stages -- but
    the timings the UI draws must mean the same thing on both paths.
    """
    settings = get_settings()
    started = time.perf_counter()

    sink: QueryFn | None = None
    if progress is not None:

        def sink(event: QueryEvent) -> None:
            progress(
                {**event, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
            )

    def report(stage: QueryStage, message: str) -> None:
        if sink is not None:
            sink({"stage": stage.value, "message": message})

    # The question is screened even though it was screened on the corpus path,
    # because this is a second, independent request from the browser: it can be
    # sent without one ever having been asked. The text also reaches a model
    # with fetched page content beside it, which is the last place to relax.
    question = check_input(question)
    report(QueryStage.GUARD, "question accepted")

    provider = settings.resolved_web_provider
    if provider is None:
        return _no_answer(None, _NOT_CONFIGURED)

    report(QueryStage.SEARCH, f"searching {_LABEL[provider]} for: {question}")
    try:
        pages = collect_web_sources(question, settings, report)
    except InputRejectedError:
        raise
    except Exception as exc:
        logger.warning("web search failed (%s: %s)", type(exc).__name__, exc)
        return _no_answer(
            provider,
            f"The search could not be completed ({type(exc).__name__}: {exc}). "
            "The corpus answer above is unaffected.",
        )

    if not pages:
        report(QueryStage.SEARCH, "no readable results")
        return _no_answer(provider, _NO_RESULTS)

    report(QueryStage.SEARCH, f"{len(pages)} source{_plural(pages)} to answer from")

    if settings.enable_generation:
        report(QueryStage.GENERATE, f"asking {describe_model(settings)}")
    else:
        report(QueryStage.GENERATE, "generation is off; showing the pages instead")

    answer, sources, context = generate_web_answer(question, pages, settings)

    if settings.enable_generation:
        report(
            QueryStage.GENERATE,
            "answer ready" if answer else "no answer; every model in the chain failed",
        )

    report(QueryStage.VERIFY, "checking the answer against the pages")
    answer = apply_output_guardrails(answer, sources, context)
    refused = is_refusal(answer)
    # Reported even when it passes, for the same reason the corpus path does:
    # "checked and clean" is what makes the citations worth trusting, and it is
    # invisible if only failures ever speak.
    report(
        QueryStage.VERIFY,
        "answer withheld by the guardrails" if refused else "citations verified",
    )

    return {
        "provider": provider,
        "pages": pages,
        "sources": sources,
        "context": context,
        "answer": answer,
        "refused": refused,
        "error": None,
    }
