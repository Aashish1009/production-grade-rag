"""Unit tests for the generation stage's pure pieces.

Only ``format_context``, the response normaliser and the disabled-generation
path are covered -- everything here runs without an API key, a network call or
a model download.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.config import Settings
from rag.generation import _SYSTEM_PROMPT, _as_text, build_chat_model, format_context, generate_answer
from rag.state import MetadataKeys as MK


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
