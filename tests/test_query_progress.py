"""Stage reporting in the query graph.

The query nodes are called directly rather than through the compiled graph,
for the same reason ``test_ingest_progress`` calls ``_process_file`` directly:
the sink rides in the state, so a node is an ordinary function of a dict and a
test can drive it with a fake sink and stubbed stages -- no store, no model,
no provider.

What is under test is that each stage *speaks*, because the failure mode is
silence rather than a wrong answer: a stage that stops reporting still returns
the right documents, and the UI simply goes quiet for however long it takes.
"""

from __future__ import annotations

import pytest

import rag.graphs as graphs
from rag.config import Settings
from rag.state import QueryStage

GUARD = QueryStage.GUARD.value
RETRIEVE = QueryStage.RETRIEVE.value
RERANK = QueryStage.RERANK.value
GENERATE = QueryStage.GENERATE.value
VERIFY = QueryStage.VERIFY.value


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Settings that ignore the developer's ``.env``.

    Without ``_env_file=None`` a machine with ``RAG_ENABLE_GENERATION=true``
    would take the other branch of ``_generate`` and the test would assert
    different behaviour here than in CI. The key is a placeholder: nothing in
    these tests builds a client, it only has to satisfy the validator that
    refuses generation without a provider.
    """
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        enable_generation=True,
        groq_api_key="test-key-not-used",
    )


def _stub_retrieval(monkeypatch: pytest.MonkeyPatch, found: list) -> None:
    monkeypatch.setattr(graphs, "search", lambda query, filters=None: found)


def _stub_rerank(monkeypatch: pytest.MonkeyPatch, kept: list) -> None:
    monkeypatch.setattr(graphs, "rerank_documents", lambda q, c, s: kept)


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


def test_retrieve_reports_before_and_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two events, because the wait before the first one is the visible one.

    A single event on completion would leave the UI silent for the whole
    search -- which is exactly the gap the trace exists to fill.
    """
    _stub_retrieval(monkeypatch, ["a", "b", "c"])
    events: list[dict] = []

    result = graphs._retrieve({"search_query": "grpo", "progress": events.append})

    assert [e["stage"] for e in events] == [RETRIEVE, RETRIEVE]
    assert events[0]["message"] == "searching the index"
    assert events[1]["message"] == "3 candidate chunks"
    assert result["candidates"] == ["a", "b", "c"]


def test_retrieve_says_so_when_nothing_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty result is the answer, and should not read as a blank line."""
    _stub_retrieval(monkeypatch, [])
    events: list[dict] = []

    graphs._retrieve({"search_query": "grpo", "progress": events.append})

    assert events[1]["message"] == "no candidates found"


def test_retrieve_counts_one_candidate_in_the_singular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_retrieval(monkeypatch, ["only"])
    events: list[dict] = []

    graphs._retrieve({"search_query": "grpo", "progress": events.append})

    assert events[1]["message"] == "1 candidate chunk"


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


def test_rerank_reports_the_funnel(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Both numbers, because "30 in, 5 out" is the whole point of the stage."""
    monkeypatch.setattr(graphs, "get_settings", lambda: settings)
    monkeypatch.setattr(graphs, "_with_top_k", lambda s, k: s)
    _stub_rerank(monkeypatch, ["kept"])
    events: list[dict] = []

    graphs._rerank({"search_query": "grpo", "candidates": ["a"] * 30, "progress": events.append})

    assert [e["stage"] for e in events] == [RERANK, RERANK]
    assert events[0]["message"] == "scoring 30 candidates"
    assert events[1]["message"] == "kept 1 passages"


def test_rerank_reports_an_empty_outcome(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Nothing surviving the threshold explains an otherwise empty answer."""
    monkeypatch.setattr(graphs, "get_settings", lambda: settings)
    monkeypatch.setattr(graphs, "_with_top_k", lambda s, k: s)
    _stub_rerank(monkeypatch, [])
    events: list[dict] = []

    graphs._rerank({"search_query": "grpo", "candidates": ["a"], "progress": events.append})

    assert events[1]["message"] == "nothing survived the score threshold"


# ---------------------------------------------------------------------------
# context widening
# ---------------------------------------------------------------------------


def test_widen_reports_how_much_context_was_added(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """The passage count and the neighbour count are different facts.

    Widening multiplies the prompt, so a user watching the trace should be able
    to see it happen rather than wondering why the answer step got slower.
    """
    monkeypatch.setattr(graphs, "get_settings", lambda: settings)
    monkeypatch.setattr(graphs, "get_client", lambda s: object())
    monkeypatch.setattr(
        graphs, "widen_context", lambda docs, client, s: docs + ["n1", "n2"]
    )
    events: list[dict] = []

    result = graphs._widen({"reranked": ["p1"], "progress": events.append})

    assert [e["stage"] for e in events] == [RERANK]
    assert events[0]["message"] == "kept 1 passage, with 2 neighbouring chunks as context"
    assert result["reranked"] == ["p1", "n1", "n2"]


def test_widen_stays_silent_when_there_is_nothing_to_add(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """A "widened by nothing" line on every query is noise.

    The rerank row above already says what was kept, and a stage that speaks
    without having done anything is the kind of trace that trains a reader to
    stop reading it.
    """
    monkeypatch.setattr(graphs, "get_settings", lambda: settings)
    monkeypatch.setattr(graphs, "get_client", lambda s: object())
    monkeypatch.setattr(graphs, "widen_context", lambda docs, client, s: docs)
    events: list[dict] = []

    graphs._widen({"reranked": ["p1"], "progress": events.append})

    assert events == []


def test_widen_is_skipped_entirely_at_radius_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    off = Settings(_env_file=None, data_dir=tmp_path, context_neighbour_radius=0)
    monkeypatch.setattr(graphs, "get_settings", lambda: off)
    monkeypatch.setattr(
        graphs, "widen_context", lambda *a: pytest.fail("the store must not be read")
    )

    result = graphs._widen({"reranked": ["p1"]})

    assert result["reranked"] == ["p1"]


def test_widen_does_nothing_with_no_passages(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """No passages is the answer for a question the corpus does not cover.

    It is also the state the web fallback reads, so it must reach the answer
    stage intact rather than being turned into a store read that returns
    nothing.
    """
    monkeypatch.setattr(graphs, "get_settings", lambda: settings)
    monkeypatch.setattr(graphs, "widen_context", lambda *a: pytest.fail("nothing to widen"))

    assert graphs._widen({"reranked": []})["reranked"] == []


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_names_the_model(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """The model is the one fact a user cannot infer from the answer's shape."""
    monkeypatch.setattr(graphs, "get_settings", lambda: settings)
    monkeypatch.setattr(
        graphs, "generate_answer", lambda q, docs: ("an answer [1]", [], "ctx")
    )
    events: list[dict] = []

    graphs._generate({"search_query": "grpo", "reranked": [], "progress": events.append})

    assert events[0]["stage"] == GENERATE
    assert settings.groq_model in events[0]["message"]
    assert events[1]["message"] == "answer ready"


def test_generate_reports_an_exhausted_fallback_chain(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """``None`` here means every model failed, which is worth saying out loud."""
    monkeypatch.setattr(graphs, "get_settings", lambda: settings)
    monkeypatch.setattr(graphs, "generate_answer", lambda q, docs: (None, [], "ctx"))
    events: list[dict] = []

    graphs._generate({"search_query": "grpo", "reranked": [], "progress": events.append})

    assert "every model in the chain failed" in events[1]["message"]


def test_generate_says_so_when_generation_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """With generation disabled there is no model to name and no answer to wait
    for, so the stage reports one event rather than a start and a finish."""
    off = Settings(_env_file=None, data_dir=tmp_path, enable_generation=False)
    monkeypatch.setattr(graphs, "get_settings", lambda: off)
    monkeypatch.setattr(graphs, "generate_answer", lambda q, docs: (None, [], "ctx"))
    events: list[dict] = []

    graphs._generate({"search_query": "grpo", "reranked": [], "progress": events.append})

    assert len(events) == 1
    assert events[0]["message"] == "generation is off; using passages only"


# ---------------------------------------------------------------------------
# output guardrails
# ---------------------------------------------------------------------------


def test_verify_reports_the_clean_case_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A check that passes is the fact that makes the citations trustworthy.

    Reported rather than silent because a trace that only ever shows failures
    is indistinguishable from a trace where the check never ran.
    """
    events: list[dict] = []

    result = graphs._guard_output(
        {
            "answer": "PPO clips the ratio [1].",
            "citations": [{"marker": "[1]"}],
            "context": "PPO clips the ratio to bound the update.",
            "progress": events.append,
        }
    )

    assert events[-1]["stage"] == VERIFY
    assert events[-1]["message"] == "citations verified"
    assert result["refused"] is False


def test_verify_reports_a_withheld_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag.guardrails import CITATION_REFUSAL

    events: list[dict] = []

    result = graphs._guard_output(
        {"answer": CITATION_REFUSAL, "citations": [], "context": "", "progress": events.append}
    )

    assert events[-1]["message"] == "answer withheld by the guardrails"
    assert result["refused"] is True


# ---------------------------------------------------------------------------
# the sink itself
# ---------------------------------------------------------------------------


def test_a_node_without_a_sink_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sink is the library path (scripts, tests), not an error."""
    _stub_retrieval(monkeypatch, ["a"])

    assert graphs._retrieve({"search_query": "grpo"})["candidates"] == ["a"]


def test_run_query_stamps_elapsed_time_on_every_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timing is added once, at the run boundary.

    A node does not know when the run began, so stamping it there would mean
    threading the origin through every stage -- and the numbers would then be
    per-stage rather than a single timeline the UI can subtract along.
    """

    class FakeGraph:
        def invoke(self, state):
            state["progress"]({"stage": GUARD, "message": "question accepted"})
            state["progress"]({"stage": RETRIEVE, "message": "searching the index"})
            return {}

    monkeypatch.setattr(graphs, "query_graph", lambda: FakeGraph())
    events: list[dict] = []

    graphs.run_query("what is grpo?", progress=events.append)

    assert [e["stage"] for e in events] == [GUARD, RETRIEVE]
    assert all("elapsed_ms" in e for e in events)
    assert events[1]["elapsed_ms"] >= events[0]["elapsed_ms"]


def test_run_query_without_a_sink_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing downstream should have to guard against a missing sink twice."""
    seen: dict = {}

    class FakeGraph:
        def invoke(self, state):
            seen["progress"] = state["progress"]
            return {}

    monkeypatch.setattr(graphs, "query_graph", lambda: FakeGraph())

    graphs.run_query("what is grpo?")

    assert seen["progress"] is None
