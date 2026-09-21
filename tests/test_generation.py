"""Unit tests for the generation stage's pure pieces.

Only the context renderers (corpus and web), the response normaliser and the
disabled-generation path are covered -- everything here runs without an API
key, a network call or a model download. Where the real pipeline builds a chat
client, the test replaces it with one that records what it was asked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document

from rag.config import Settings
from rag.generation import (
    _SYSTEM_PROMPT,
    _WEB_SYSTEM_PROMPT,
    _as_text,
    _share_char_budget,
    build_chat_model,
    format_context,
    format_web_context,
    generate_answer,
    generate_web_answer,
)
from rag.state import MetadataKeys as MK, WebPage


def _chunk(text: str, **meta: object) -> Document:
    return Document(page_content=text, metadata=dict(meta))


@pytest.fixture(autouse=True)
def fast_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Estimate tokens as words, so the context budget is deterministic.

    ``format_context`` measures the rendered context against
    ``context_token_budget``, and that measurement must not depend on a
    tokenizer download or on how the real model segments text.
    """
    monkeypatch.setattr(
        "rag.generation.token_length", lambda text, model_name=None: len(text.split())
    )


def _settings(**over: object) -> Settings:
    return Settings(_env_file=None, **over)


@pytest.fixture()
def no_generation(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings with generation explicitly off and no ambient API keys."""
    for var in ("OPENAI_API_KEY", "GROQ_API_KEY", "RAG_OPENAI_API_KEY", "RAG_GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None, enable_generation=False)


# ---------------------------------------------------------------------------
# context rendering
# ---------------------------------------------------------------------------


def test_format_context_numbers_blocks_and_maps_citations() -> None:
    docs = [
        _chunk(
            "GRPO baselines against group means.",
            **{MK.SOURCE_NAME: "rlhf book.pdf", MK.PAGE_START: 123, MK.PAGE_END: 124},
        ),
        _chunk(
            "PPO clips the surrogate objective.",
            **{MK.SOURCE_NAME: "rlhf book.pdf", MK.PAGE_START: 88, MK.PAGE_END: 88},
        ),
    ]
    context, citations = format_context(docs)

    assert "[1]" in context and "[2]" in context
    assert "pp. 123-124" in context
    assert "p. 88" in context  # a single page is not rendered as a range
    assert [c["marker"] for c in citations] == ["[1]", "[2]"]
    assert citations[0]["source"] == "rlhf book.pdf"
    assert citations[0]["page_start"] == 123


def test_format_context_falls_back_to_section_when_pageless() -> None:
    docs = [_chunk("Body text.", **{MK.SOURCE_NAME: "notes.md", MK.SECTION_PATH: "Setup > Config"})]
    context, citations = format_context(docs)

    assert "Setup > Config" in context
    assert citations[0]["page_start"] is None


# ---------------------------------------------------------------------------
# widened context
# ---------------------------------------------------------------------------


def _widened(text: str, anchor: str, role: str, **meta: object) -> Document:
    return _chunk(
        text, **{MK.CHUNK_ID: f"n-{text}", MK.CONTEXT_OF: anchor, MK.CONTEXT_ROLE: role, **meta}
    )


def test_a_neighbours_render_inside_their_passages_block() -> None:
    # Widening is what turns a fragment into a claim: a statement made in one
    # chunk and qualified in the next is half a claim from either alone.
    docs = [
        _widened("what precedes", "c2", "prev"),
        _chunk("the passage", **{MK.CHUNK_ID: "c2", MK.SOURCE_NAME: "book.pdf", MK.PAGE_START: 4}),
        _widened("what follows", "c2", "next"),
    ]
    context, citations = format_context(docs, _settings())

    assert (
        context.index("what precedes")
        < context.index("the passage")
        < context.index("what follows")
    )


def test_a_widened_passage_still_cites_once() -> None:
    # top_k counts passages and the reranker scored passages. A neighbour given
    # a marker of its own would be a claim nobody scored, cited as though
    # someone had -- and top_k would silently mean three times what it says.
    docs = [
        _widened("what precedes", "c2", "prev"),
        _chunk("the passage", **{MK.CHUNK_ID: "c2", MK.SOURCE_NAME: "book.pdf", MK.PAGE_START: 4}),
        _widened("what follows", "c2", "next"),
    ]
    context, citations = format_context(docs, _settings())

    assert len(citations) == 1
    assert citations[0]["marker"] == "[1]"
    assert citations[0]["context_chunks"] == 2
    assert "[2]" not in context


def test_context_is_trimmed_before_any_passage_is_dropped() -> None:
    # The budget is what keeps top_k a budget rather than a multiplier, and it
    # is spent from the outside in: the chunk furthest from the passage that
    # was actually matched is the least likely to be needed to complete it.
    docs = [
        _chunk(" ".join(["passage"] * 10), **{MK.CHUNK_ID: "c2", MK.SOURCE_NAME: "b.pdf", MK.PAGE_START: 1}),
        _widened(" ".join(["tail"] * 40), "c2", "next"),
    ]
    context, citations = format_context(docs, _settings(context_token_budget=20))

    assert len(citations) == 1
    assert "passage" in context
    assert "tail" not in context


def test_a_passage_is_never_trimmed_away_to_fit_the_budget() -> None:
    # Neighbours are dropped and only neighbours. An anchor survived reranking
    # on its own merits, so it is worth its tokens whatever the budget says --
    # and an empty context is a worse outcome than an over-long one, because a
    # model handed nothing can only report that nothing covers the question.
    docs = [
        _chunk(" ".join(["passage"] * 50), **{MK.CHUNK_ID: "c1", MK.SOURCE_NAME: "b.pdf"}),
        _chunk(" ".join(["other"] * 50), **{MK.CHUNK_ID: "c2", MK.SOURCE_NAME: "b.pdf"}),
    ]
    context, citations = format_context(docs, _settings(context_token_budget=5))

    assert len(citations) == 2
    assert "passage" in context and "other" in context


def test_the_block_header_does_not_repeat_a_breadcrumb_the_passage_carries() -> None:
    # Chunks now open with their own section line, so repeating it in the
    # header would show the model the same heading twice.
    docs = [
        _chunk(
            "Setup > Config\n\nBody text.",
            **{MK.SOURCE_NAME: "notes.md", MK.SECTION_PATH: "Setup > Config", MK.PAGE_START: 2},
        )
    ]
    context, _ = format_context(docs, _settings())

    assert context.count("Setup > Config") == 1


def test_context_markers_are_what_the_citation_guardrail_checks() -> None:
    # The prompt, the rendered context and the guardrail all have to agree on
    # "[N]". When they drifted, an answer citing invented source names passed
    # validation untouched.
    from rag.guardrails import CitationGuardrail

    docs = [_chunk("GRPO baselines against group means.", **{MK.SOURCE_NAME: "book.pdf", MK.PAGE_START: 3})]
    _, citations = format_context(docs)

    answer = "GRPO baselines against group means [1]."
    assert CitationGuardrail().check(answer, citations) == answer
    assert "[1], [2], [3]" in _SYSTEM_PROMPT


def test_prompt_asks_for_display_math_in_the_delimiters_the_ui_renders() -> None:
    # The answer card's math renderer keys on \[ ... \]. A prompt that asked
    # for some other delimiter would hand the browser raw backslashes, which
    # is precisely the bug this replaces.
    assert "\\[" in _SYSTEM_PROMPT
    assert "\\]" in _SYSTEM_PROMPT


def test_prompt_no_longer_forbids_elaboration() -> None:
    # An earlier revision banned preamble and filler, and the model complied
    # literally: asked for the Bradley-Terry equation it returned the formula
    # and two markers and nothing else. Those instructions must not come back.
    lowered = _SYSTEM_PROMPT.lower()
    assert "no preamble" not in lowered
    assert "never pad" not in lowered
    assert "explain" in lowered


# ---------------------------------------------------------------------------
# response normalisation
# ---------------------------------------------------------------------------


def test_as_text_passes_strings_through() -> None:
    assert _as_text("hello") == "hello"


def test_as_text_flattens_content_blocks() -> None:
    # Some providers behind the LiteLLM gateway return a list of content
    # blocks rather than a string; returning that raw breaks the output
    # guardrails' string operations.
    assert _as_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_as_text_ignores_non_text_blocks() -> None:
    assert _as_text([{"type": "tool_use", "id": "x"}, {"text": "kept"}]) == "kept"


# ---------------------------------------------------------------------------
# disabled / unavailable generation
# ---------------------------------------------------------------------------


def test_generate_answer_without_generation_returns_context(
    no_generation: Settings,
) -> None:
    docs = [_chunk("Some grounded text.", **{MK.SOURCE_NAME: "book.pdf", MK.PAGE_START: 1})]
    answer, citations, context = generate_answer("q", docs, no_generation)

    assert answer is None
    assert citations
    # The context is returned, not re-rendered by the caller, so the text the
    # grounding guardrail inspects is exactly what a model would have seen.
    assert context == format_context(docs)[0]


def test_generate_answer_with_no_documents(no_generation: Settings) -> None:
    answer, citations, context = generate_answer("q", [], no_generation)
    assert answer is None
    assert citations == []
    assert context == ""


def test_build_chat_model_without_a_key_raises(no_generation: Settings) -> None:
    with pytest.raises(RuntimeError, match="no API key"):
        build_chat_model(no_generation)


# ---------------------------------------------------------------------------
# the web context budget -- the fix for a request that never reached a model
# ---------------------------------------------------------------------------

# Chosen as the filler for pages under test because neither character occurs
# in any of the titles, urls or snippets below: a page's kept text can then be
# counted out of the rendered context exactly.
_SHORT_FILLER = "z"
_LONG_FILLER = "q"


def _page(text: str, **over: object) -> WebPage:
    """A fetched page whose ``text`` is what the budget is spent on."""
    fields: dict[str, object] = {
        "title": "The Bradley-Terry model",
        "url": "https://example.org/bt",
        "snippet": "A probability model for paired comparisons.",
        "text": text,
    }
    fields.update(over)
    return WebPage(**fields)  # type: ignore[arg-type]


def _generating(**over: object) -> Settings:
    """Settings with generation on, so both answer paths run past the guard."""
    return _settings(enable_generation=True, **over)


class _RecordingChat:
    """A chat client that records the messages it was handed.

    Stands in for the LiteLLM fallback chain: these tests are about the size
    of the request the stage builds, which is decided before any provider is
    contacted, so nothing here is a model or a network call.
    """

    def __init__(self, reply: str = "Answer [1].") -> None:
        self.reply = reply
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)


def _capture_build(chat: _RecordingChat) -> tuple[Any, dict[str, Any]]:
    """A stand-in for ``build_chat_model`` that records its keyword arguments."""
    seen: dict[str, Any] = {}

    def _build(settings: Settings | None = None, *, max_tokens: int | None = None) -> Any:
        seen["max_tokens"] = max_tokens
        return chat

    return _build, seen


def test_web_page_texts_are_trimmed_to_the_context_budget() -> None:
    # Five pages at the per-page ceiling is 30,000 characters -- ~7,674 tokens
    # by the provider's own count, which with the declared output reservation
    # went over the free tier's 8,000-token-per-minute limit and failed every
    # web answer before a model was called.
    pages = [_page(_LONG_FILLER * 6000) for _ in range(5)]

    context, _ = format_web_context(pages, _settings(web_max_context_chars=16000))

    assert context.count(_LONG_FILLER) == 16000


def test_the_budget_is_shared_out_rather_than_spent_in_page_order() -> None:
    # Every page here is also a source the reader is shown. A context that let
    # the first two pages take everything would have the answer citing a list
    # of sources it was never given the text of.
    pages = [_page(_LONG_FILLER * 6000) for _ in range(5)]

    context, sources = format_web_context(pages, _settings(web_max_context_chars=16000))

    assert len(sources) == 5
    # Each page keeps its own 3,200-character share. Spending in order would
    # have left pages 3 to 5 with a header and no text at all.
    assert context.count(_LONG_FILLER * 3200) == 5


def test_a_short_page_passes_its_unused_share_to_a_long_one() -> None:
    # Share is 500 characters each. The short page uses 100 and its spare 400
    # go to the long one, so the budget is spent on pages that have something
    # to spend it on rather than on the accident of their order.
    short = _page(_SHORT_FILLER * 100)
    long = _page(_LONG_FILLER * 10000)

    context, _ = format_web_context([short, long], _settings(web_max_context_chars=1000))

    assert context.count(_SHORT_FILLER) == 100
    assert context.count(_LONG_FILLER) == 900


def test_the_share_a_page_gets_does_not_depend_on_where_it_was_ranked() -> None:
    # The same two pages in the other order must keep the same amounts: a
    # budget that favoured whichever page the search engine happened to list
    # first would make the citation depth a property of the search ranking.
    short = _page(_SHORT_FILLER * 100)
    long = _page(_LONG_FILLER * 10000)

    context, _ = format_web_context([long, short], _settings(web_max_context_chars=1000))

    assert context.count(_SHORT_FILLER) == 100
    assert context.count(_LONG_FILLER) == 900


def test_trimming_keeps_the_top_of_a_page_rather_than_sampling_it() -> None:
    # Pages are cut, not excerpted. A summary sampled from the middle of a
    # page would drop the opening paragraph, which is where a page usually
    # says what it is about.
    pages = [_page("heading" + _LONG_FILLER * 10000) for _ in range(2)]

    context, _ = format_web_context(pages, _settings(web_max_context_chars=1000))

    assert context.count("heading") == 2


def test_every_page_keeps_its_marker_url_and_delimiters_under_a_tight_budget() -> None:
    # The markers are what the citations mean and the delimiters are what the
    # prompt's "everything between these is quoted" claim rests on, so
    # trimming may remove page text but never a page.
    pages = [
        _page(
            _LONG_FILLER * 6000,
            title=f"Page {n}",
            url=f"https://example.org/{n}",
        )
        for n in range(1, 6)
    ]

    context, sources = format_web_context(pages, _settings(web_max_context_chars=2000))

    for n in range(1, 6):
        assert f"[{n}] Page {n}" in context
        assert f"https://example.org/{n}" in context
        assert f"<<<BEGIN WEB PAGE {n}>>>" in context
        assert f"<<<END WEB PAGE {n}>>>" in context
    assert [s["marker"] for s in sources] == ["[1]", "[2]", "[3]", "[4]", "[5]"]


def test_a_page_that_was_never_read_still_gets_its_block() -> None:
    # The keyless path can return a page whose body would not load, leaving
    # only the search engine's snippet. Dropping that block would renumber
    # every citation after it.
    pages = [
        _page(_LONG_FILLER * 100),
        _page("", snippet="Only a summary here."),
        _page(_LONG_FILLER * 100),
    ]

    context, sources = format_web_context(pages, _settings(web_max_context_chars=1000))

    assert "[2] " in context
    assert sources[1]["snippet"] == "Only a summary here."


def test_the_source_map_is_not_trimmed_along_with_the_text() -> None:
    # The budget bounds what the model reads, not what the reader is shown:
    # the urls and snippets are the citation, and are what is left to display
    # when generation fails outright.
    pages = [_page(_LONG_FILLER * 6000, title=f"Page {n}") for n in range(1, 6)]

    _, sources = format_web_context(pages, _settings(web_max_context_chars=2000))

    assert [s["title"] for s in sources] == [f"Page {n}" for n in range(1, 6)]
    assert all(s["url"] for s in sources)


def test_the_char_budget_helper_leaves_text_it_can_afford() -> None:
    assert _share_char_budget([], 500) == []
    assert _share_char_budget(["short"], 500) == ["short"]


def test_a_zero_budget_empties_every_page_rather_than_erroring() -> None:
    # ``web_max_context_chars`` is validated to be greater than zero, so this
    # is the helper's own contract rather than a reachable configuration --
    # but it must not divide by zero or quietly hand back whole pages.
    assert _share_char_budget(["a" * 100, "b" * 100], 0) == ["", ""]


def test_the_default_web_context_cannot_overshoot_the_free_tier_ceiling() -> None:
    """The constraint the web path failed on, asserted as arithmetic.

    Groq charges the *declared* ``max_tokens`` against the per-minute limit
    whether or not the model uses it, so the request that has to fit is
    context + reserved output + prompts. The observed failure was
    "Limit 8000, Requested 8698" for five 6,000-character pages, and no
    fallback model could have rescued it: the whole chain sits under the same
    ceiling, which is why trimming is the only fix. Characters per token is
    taken as 3.5 rather than the 4 English tends to average, so this estimate
    errs high.
    """
    settings = _settings()

    request = (
        settings.web_max_context_chars / 3.5
        + len(_WEB_SYSTEM_PROMPT) / 3.5
        + settings.web_llm_max_tokens
    )

    assert request < 8000
    # And the old shape of the bug stays out. Five pages at the per-page
    # ceiling is more context than one request can carry, so the per-page
    # ceiling must never be the number that bounds a request -- that is what
    # web_max_context_chars exists for.
    overhead = 6000 * 5 / 3.5 + settings.web_llm_max_tokens
    assert overhead > 8000


def test_generate_web_answer_reserves_the_smaller_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reservation is part of the request size on a metered plan, and a
    # summary of fetched pages is not a long answer. Leaving the corpus path's
    # 1,024 in place here spends budget the answer never gets back.
    chat = _RecordingChat()
    build, seen = _capture_build(chat)
    monkeypatch.setattr("rag.generation.build_chat_model", build)

    answer, sources, context = generate_web_answer(
        "what is Bradley-Terry", [_page(_LONG_FILLER * 20)], _generating(web_llm_max_tokens=384)
    )

    assert seen["max_tokens"] == 384
    assert answer == "Answer [1]."
    assert sources and context
    assert chat.calls, "the web answer path never reached a model"


def test_generate_answer_leaves_the_output_budget_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``None`` means "whatever llm_max_tokens says". A corpus answer may be
    # quoting passages at length and is not the path being trimmed.
    chat = _RecordingChat()
    build, seen = _capture_build(chat)
    monkeypatch.setattr("rag.generation.build_chat_model", build)
    docs = [_chunk("Some grounded text.", **{MK.SOURCE_NAME: "book.pdf"})]

    answer, _, _ = generate_answer("q", docs, _generating())

    assert seen["max_tokens"] is None
    assert answer == "Answer [1]."


def test_web_answer_with_no_generation_returns_the_pages(no_generation: Settings) -> None:
    # Same graceful degradation as the corpus path: the pages are still worth
    # showing, and the source map is what renders them.
    pages = [_page(_LONG_FILLER * 20)]

    answer, sources, context = generate_web_answer("q", pages, no_generation)

    assert answer is None
    assert [s["marker"] for s in sources] == ["[1]"]
    assert context == format_web_context(pages, no_generation)[0]
