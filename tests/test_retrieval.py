"""Unit tests for the retrieval layer's pure pieces.

Filters and the lexical fallback are exercised without a live collection:
``build_filter`` is pure translation, and the BM25 retriever is driven through
a stubbed corpus and client so no embedded database and no model are needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from rag.retrieval import BM25FallbackRetriever, _tokenise, build_filter, widen_context


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_no_filters_builds_no_filter() -> None:
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_filters_become_metadata_field_conditions() -> None:
    query_filter = build_filter({"doc_id": "ab12", "source_name": "book.pdf"})
    assert query_filter is not None
    assert len(query_filter.must) == 2
    keys = {condition.key for condition in query_filter.must}
    assert keys == {"metadata.doc_id", "metadata.source_name"}


def test_none_valued_filters_are_skipped() -> None:
    # A None means "not specified" rather than "match null", so it must not
    # prune the candidate set to nothing.
    assert build_filter({"doc_id": None}) is None
    query_filter = build_filter({"doc_id": "ab12", "source_name": None})
    assert query_filter is not None
    assert len(query_filter.must) == 1


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------


def test_tokenise_keeps_words_separate_and_drops_punctuation() -> None:
    assert _tokenise("GPT-4 vs. Llama 3.1!") == ["gpt", "4", "vs", "llama", "3", "1"]


def test_tokenise_does_not_fuse_a_sentence_into_one_token() -> None:
    # Regression: routing this through the dedupe normaliser (which strips
    # spaces) produced a single token per chunk, so BM25 could only ever match
    # a query that equalled the entire chunk verbatim.
    tokens = _tokenise("Policy gradient methods optimise a surrogate objective")
    assert len(tokens) == 7
    assert "gradient" in tokens


# ---------------------------------------------------------------------------
# lexical fallback
# ---------------------------------------------------------------------------


class _FakeBM25:
    """Scores each document by the number of query tokens it contains."""

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        assert query_tokens, "BM25 must never be called with an empty query"
        return [1.0, 2.0, 0.0]


@pytest.fixture()
def corpus(monkeypatch: pytest.MonkeyPatch) -> list[Document]:
    docs = [
        Document(page_content="grpo", metadata={"doc_id": "d1"}),
        Document(page_content="ppo", metadata={"doc_id": "d2"}),
        Document(page_content="other", metadata={"doc_id": "d1"}),
    ]
    monkeypatch.setattr(
        "rag.retrieval.get_client", lambda settings=None: object()
    )
    monkeypatch.setattr(
        "rag.retrieval._cached_bm25", lambda client, settings: (_FakeBM25(), docs)
    )
    return docs


def test_lexical_fallback_returns_scored_hits(corpus: list[Document]) -> None:
    hits = BM25FallbackRetriever(k=10)._get_relevant_documents("grpo")
    # The zero-scoring document is dropped: a lexical score of 0 means "no
    # shared token", not "weakly relevant".
    assert [d.page_content for d in hits] == ["ppo", "grpo"]


def test_lexical_fallback_honours_filters(corpus: list[Document]) -> None:
    hits = BM25FallbackRetriever(k=10, filters={"doc_id": "d2"})._get_relevant_documents("grpo")
    assert [d.page_content for d in hits] == ["ppo"]


def test_lexical_fallback_with_unmatched_filter_returns_nothing(
    corpus: list[Document],
) -> None:
    hits = BM25FallbackRetriever(k=10, filters={"doc_id": "missing"})._get_relevant_documents("grpo")
    assert hits == []


def test_lexical_fallback_on_empty_corpus_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag.retrieval.get_client", lambda settings=None: object())
    monkeypatch.setattr(
        "rag.retrieval._cached_bm25", lambda client, settings: (None, [])
    )
    assert BM25FallbackRetriever(k=10)._get_relevant_documents("anything") == []


# ---------------------------------------------------------------------------
# search() guards
# ---------------------------------------------------------------------------


class _EmptyClient:
    def collection_exists(self, name: str) -> bool:
        return False


def test_search_on_missing_collection_returns_empty() -> None:
    from rag.config import Settings
    from rag.retrieval import search

    # "No passages found" is a valid answer, not an error; a fresh install
    # with no ingest yet must not raise out of the query path. The client is
    # injected, so no real store is opened and no model is loaded.
    assert search("what is GRPO?", Settings(_env_file=None), client=_EmptyClient()) == []


# ---------------------------------------------------------------------------
# context widening
# ---------------------------------------------------------------------------


class _Point:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class _StubStore:
    """Serves stored chunks by point id and records every read it was asked for.

    Stands in for the Qdrant client so widening can be exercised without an
    embedded database, exactly as the BM25 tests do. ``links`` supplies each
    stored chunk's ``prev_id``/``next_id``, which is what lets the walk take a
    second step.
    """

    def __init__(
        self,
        stored: dict[str, str],
        *,
        links: dict[str, tuple[str | None, str | None]] | None = None,
        fail: bool = False,
    ) -> None:
        from rag.indexing import point_id

        self.fail = fail
        self.reads: list[list[str]] = []
        self.by_point: dict[str, _Point] = {}
        for chunk_id, text in stored.items():
            prev_id, next_id = (links or {}).get(chunk_id, (None, None))
            self.by_point[point_id(chunk_id)] = _Point(
                {
                    "page_content": text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "prev_id": prev_id,
                        "next_id": next_id,
                    },
                }
            )

    def retrieve(self, *, collection_name: str, ids: list[str], with_payload: bool):
        self.reads.append(list(ids))
        if self.fail:
            raise RuntimeError("the store is busy")
        wanted = set(ids)
        return [p for key, p in self.by_point.items() if key in wanted]


def _settings(**over: Any):
    from rag.config import Settings

    return Settings(_env_file=None, **over)


def _hit(chunk_id: str, text: str, **meta: Any) -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "rerank_score": 1.0, **meta},
    )


def test_widen_context_marks_neighbours_as_context_not_results() -> None:
    from rag.retrieval import widen_context

    hits = [_hit("c2", "the middle", prev_id="c1", next_id="c3")]
    store = _StubStore({"c1": "before", "c3": "after"})

    widened = widen_context(hits, store, _settings())

    assert [d.page_content for d in widened] == ["before", "the middle", "after"]
    assert widened[0].metadata["context_of"] == "c2"
    assert widened[0].metadata["context_role"] == "prev"
    assert widened[2].metadata["context_role"] == "next"
    # The anchor keeps its score and gains no context marks. The citation map
    # has to be able to tell the passage that was scored from the chunks pulled
    # in to support it, or top_k stops meaning "passages answered for".
    assert "context_of" not in widened[1].metadata
    assert widened[1].metadata["rerank_score"] == 1.0


def test_widen_context_is_a_no_op_without_a_radius() -> None:
    from rag.retrieval import widen_context

    hits = [_hit("c2", "text", prev_id="c1")]
    store = _StubStore({"c1": "before"})

    assert widen_context(hits, store, _settings(context_neighbour_radius=0)) == hits
    assert store.reads == []


def test_a_neighbour_that_is_itself_a_winner_is_not_pulled_in_twice() -> None:
    # c3 was reranked too, so it is going to be rendered in its own right.
    # Reading it again as c2's next neighbour would print the same text twice
    # and cite it under two different markers.
    from rag.retrieval import widen_context

    hits = [_hit("c2", "middle", prev_id="c1", next_id="c3"), _hit("c3", "after")]
    store = _StubStore({"c1": "before", "c3": "after"})

    widened = widen_context(hits, store, _settings())

    assert [d.page_content for d in widened] == ["before", "middle", "after"]
    assert [d.metadata.get("context_of") for d in widened] == ["c2", None, None]


def test_a_dangling_neighbour_link_is_simply_skipped() -> None:
    from rag.retrieval import widen_context

    hits = [_hit("c2", "middle", prev_id="gone", next_id="c3")]
    store = _StubStore({"c3": "after"})

    widened = widen_context(hits, store, _settings())

    assert [d.page_content for d in widened] == ["middle", "after"]


def test_a_store_failure_leaves_the_passages_alone() -> None:
    # Widening is an improvement, not a dependency: a passage that survived
    # reranking is a complete answer to cite even when its surroundings cannot
    # be read, so the failure degrades rather than propagates.
    from rag.retrieval import widen_context

    hits = [_hit("c2", "middle", prev_id="c1")]
    store = _StubStore({"c1": "before"}, fail=True)

    assert widen_context(hits, store, _settings()) == hits


def test_the_walk_reaches_further_out_at_a_higher_radius() -> None:
    # Ordering is the whole point of the two lists: a second-round neighbour is
    # farther from the anchor than a first-round one and has to end up further
    # from it in the rendered block.
    from rag.retrieval import widen_context

    hits = [_hit("c3", "middle", prev_id="c2", next_id="c4")]
    store = _StubStore(
        {"c1": "first", "c2": "second", "c4": "fourth", "c5": "fifth"},
        links={
            "c1": (None, "c2"),
            "c2": ("c1", "c3"),
            "c4": ("c3", "c5"),
            "c5": ("c4", None),
        },
    )

    widened = widen_context(hits, store, _settings(context_neighbour_radius=2))

    assert [d.page_content for d in widened] == [
        "first",
        "second",
        "middle",
        "fourth",
        "fifth",
    ]


def test_build_retriever_passes_k_and_filter_into_search_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag.config import Settings
    from rag.retrieval import build_retriever

    captured: dict[str, Any] = {}

    class _Store:
        retrieval_mode = None

        def as_retriever(self, *, search_type: str, search_kwargs: dict) -> str:
            captured["search_type"] = search_type
            captured["search_kwargs"] = search_kwargs
            return "retriever"

    settings = Settings(_env_file=None, fetch_k=17)
    build_retriever(settings, _Store(), filters={"doc_id": "ab12"})

    # k must ride inside search_kwargs: VectorStoreRetriever ignores
    # top-level kwargs, so a bare k= would silently leave the default k=4.
    assert captured["search_kwargs"]["k"] == 17
    assert captured["search_kwargs"]["filter"] is not None
    assert captured["search_type"] == "similarity"
