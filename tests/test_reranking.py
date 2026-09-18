"""Unit tests for reranking.

The cross-encoder is stubbed, so nothing here loads a model or touches the
network. What is under test is the ranking, the truncation to ``top_k``, and
the threshold's guarantee that it can trim noise but never empty the context.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.config import Settings
from rag.reranking import rerank_documents


class _StubEncoder:
    """Scores a document by the number its text ends with, so the ordering a
    test asserts is the ordering it wrote down."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(text.split()[-1]) for _, text in pairs]


@pytest.fixture(autouse=True)
def stub_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag.reranking.get_cross_encoder", lambda settings=None: _StubEncoder()
    )


def _doc(score: float) -> Document:
    return Document(page_content=f"chunk {score}", metadata={})


def _settings(**over: object) -> Settings:
    return Settings(_env_file=None, **over)


def test_candidates_are_ranked_and_truncated_to_top_k() -> None:
    result = rerank_documents(
        "q", [_doc(1), _doc(9), _doc(5)], _settings(top_k=2, rerank_score_threshold=0.0)
    )

    assert [d.metadata["rerank_score"] for d in result] == [9.0, 5.0]


def test_the_input_list_is_not_mutated() -> None:
    # The graph holds the retrieved candidates as state; reranking has to rank
    # them, not reorder the caller's list underneath it.
    candidates = [_doc(1), _doc(9)]
    rerank_documents("q", candidates, _settings(top_k=2))

    assert [d.page_content for d in candidates] == ["chunk 1", "chunk 9"]


def test_a_threshold_drops_the_passages_below_it() -> None:
    result = rerank_documents(
        "q", [_doc(9), _doc(-1), _doc(-7)], _settings(top_k=5, rerank_score_threshold=2.0)
    )

    assert [d.metadata["rerank_score"] for d in result] == [9.0]


def test_the_best_passage_survives_a_threshold_it_fails() -> None:
    # A threshold exists to remove noise, not to remove everything. An emptied
    # context makes the model report that nothing covers the question -- which
    # is the exact signal the UI reads before offering a search of the web. A
    # badly chosen threshold would therefore manufacture the symptom the
    # fallback exists to diagnose, and send the user to the open web for a
    # question their own documents answer.
    result = rerank_documents(
        "q", [_doc(-2), _doc(-9)], _settings(top_k=5, rerank_score_threshold=5.0)
    )

    assert [d.metadata["rerank_score"] for d in result] == [-2.0]


def test_no_threshold_keeps_everything_that_fits() -> None:
    # 0.0 is the documented "off", so a negative-scoring passage is still
    # handed over rather than silently filtered by a default nobody chose.
    result = rerank_documents(
        "q", [_doc(1), _doc(-9)], _settings(top_k=5, rerank_score_threshold=0.0)
    )

    assert len(result) == 2


def test_no_candidates_short_circuits_before_scoring() -> None:
    # An empty batch raises inside sentence-transformers' predict, so a caller
    # that already filtered its candidates down to nothing would get a
    # traceback instead of "no passages found".
    assert rerank_documents("q", [], _settings()) == []
