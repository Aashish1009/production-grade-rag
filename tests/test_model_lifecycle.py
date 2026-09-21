"""Unit tests for idle model unloading.

No model is loaded here, and none can be: the singletons are replaced with
stubs that only own a ``_model`` attribute, which is all the release path is
allowed to touch. What is under test is the policy -- when the models are
dropped, when they are refused, and the two invariants the module exists to
protect (never unload under a live request, never lose the singleton).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import suppress

import pytest

import rag.embedding as embedding
import rag.model_lifecycle as lifecycle
import rag.reranking as reranking
from rag.config import Settings


class _StubHolder:
    """Stands in for the embedder/reranker singleton: it owns ``_model``."""

    def __init__(self, loaded: bool = True) -> None:
        self._model = object() if loaded else None


@pytest.fixture(autouse=True)
def fresh_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own idle clock and refcount.

    Both are module state, and a test that leaked either would make the next
    one pass or fail for reasons that have nothing to do with it.
    """
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic())
    monkeypatch.setattr(lifecycle, "_in_flight", 0)


@pytest.fixture()
def stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[_StubHolder, _StubHolder]:
    """Install a loaded stub for each singleton, and return them."""
    dense, reranker = _StubHolder(), _StubHolder()
    monkeypatch.setattr(embedding, "_DENSE_EMBEDDINGS", dense)
    monkeypatch.setattr(reranking, "_CROSS_ENCODER", reranker)
    return dense, reranker


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------


def test_a_fresh_process_is_not_idle() -> None:
    # Just after import nothing has been used and nothing has been idle
    # either: a reaper that fired immediately on a cold process would fight
    # the very first request for the models.
    assert lifecycle.idle_seconds() < 5.0
    assert lifecycle.should_release(10) is False


def test_idle_seconds_grows_after_the_last_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic() - 120)

    assert lifecycle.idle_seconds() >= 120


def test_use_refreshes_the_clock_on_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long call must not look idle just because it started long ago.

    An ingest slice can embed for minutes; if the clock were only set on
    entry, the reaper would consider the models idle while a batch was still
    running through them.
    """
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic() - 3600)

    with lifecycle.use():
        pass

    assert lifecycle.idle_seconds() < 5.0


# ---------------------------------------------------------------------------
# the refcount -- the invariant that stops a reload landing mid-request
# ---------------------------------------------------------------------------


def test_should_release_refuses_while_a_call_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic() - 3600)

    with lifecycle.use():
        # Idle by the clock, yet something is using the models right now.
        assert lifecycle.in_flight() == 1
        assert lifecycle.should_release(10) is False

    assert lifecycle.in_flight() == 0
    # Still refused, because finishing refreshed the clock: the idle window
    # runs from the end of the work rather than from its start.
    assert lifecycle.should_release(10) is False

    # A window later, with nothing using them, they go.
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic() - 3600)
    assert lifecycle.should_release(10) is True


def test_the_refcount_is_released_even_when_the_call_raises() -> None:
    # An exception in a query must not leave the models permanently
    # unloadable -- the count has to unwind on the failure path too.
    with pytest.raises(ValueError):
        with lifecycle.use():
            raise ValueError("boom")

    assert lifecycle.in_flight() == 0


def test_should_release_is_false_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic() - 10_000)

    # 0 is the documented "off", and it must mean off rather than "immediately
    # idle" -- the reading a plain comparison would have given it.
    assert lifecycle.should_release(0) is False


# ---------------------------------------------------------------------------
# releasing
# ---------------------------------------------------------------------------


def test_release_drops_both_models_and_names_them(
    stubs: tuple[_StubHolder, _StubHolder],
) -> None:
    released = lifecycle.release_all()

    assert released == ["dense embedder", "reranker"]
    assert lifecycle.models_loaded() is False


def test_release_is_idempotent(stubs: tuple[_StubHolder, _StubHolder]) -> None:
    lifecycle.release_all()

    # Called again by the next reaper tick, and by every later tick until
    # something loads a model again. It must be a no-op, not an error.
    assert lifecycle.release_all() == []


def test_release_on_a_process_that_loaded_nothing_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding, "_DENSE_EMBEDDINGS", None)
    monkeypatch.setattr(reranking, "_CROSS_ENCODER", None)

    assert lifecycle.release_all() == []


def test_an_unloaded_singleton_reports_nothing_to_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding, "_DENSE_EMBEDDINGS", _StubHolder(loaded=False))
    monkeypatch.setattr(reranking, "_CROSS_ENCODER", _StubHolder(loaded=False))

    assert lifecycle.loaded_models() == []
    assert lifecycle.release_all() == []


def test_release_keeps_the_singleton_itself(stubs: tuple[_StubHolder, _StubHolder]) -> None:
    """The invariant that stops a second copy of the weights being built.

    The release clears the *model*, never the holder. If it dropped the
    singleton, a stale reference -- a vector store built a moment earlier, an
    embedder a retriever is still holding -- would keep the old weights alive
    while ``get_dense_embeddings()`` built a fresh object beside them, which is
    exactly the duplicate-model failure the singletons were introduced to
    prevent.
    """
    dense, reranker = stubs

    lifecycle.release_all()

    assert embedding._DENSE_EMBEDDINGS is dense
    assert reranking._CROSS_ENCODER is reranker
    assert dense._model is None
    assert reranker._model is None


def test_only_one_singleton_loaded_releases_only_that_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding, "_DENSE_EMBEDDINGS", _StubHolder(loaded=True))
    monkeypatch.setattr(reranking, "_CROSS_ENCODER", _StubHolder(loaded=False))

    assert lifecycle.release_all() == ["dense embedder"]


def test_release_does_not_import_a_module_that_was_never_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that never loaded a model must not pay torch's import to free one.

    ``release_all`` reads ``sys.modules`` before each import for this reason;
    importing ``rag.embedding`` unconditionally would drag the whole torch
    stack in, once every reaper tick, to release nothing.
    """
    monkeypatch.delitem(sys.modules, "rag.embedding")
    monkeypatch.delitem(sys.modules, "rag.reranking")
    monkeypatch.setattr(embedding, "_DENSE_EMBEDDINGS", None)
    monkeypatch.setattr(reranking, "_CROSS_ENCODER", None)

    assert lifecycle.release_all() == []


# ---------------------------------------------------------------------------
# warming
# ---------------------------------------------------------------------------


def test_warm_loads_every_model_and_marks_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm-up must actually touch ``.model``, not merely build the holders.

    Constructing the holder is free and loads nothing -- that is the whole
    point of the lazy property -- so a warm-up that only called
    ``get_dense_embeddings()`` would report success while leaving the ~90s
    load to the first question.
    """
    touched: list[str] = []

    class _Loader:
        def __init__(self, name: str) -> None:
            self._name = name
            self._model = None

        @property
        def model(self) -> object:
            touched.append(self._name)
            self._model = object()
            return self._model

    dense, reranker = _Loader("dense"), _Loader("reranker")
    monkeypatch.setattr(embedding, "_DENSE_EMBEDDINGS", dense)
    monkeypatch.setattr(reranking, "_CROSS_ENCODER", reranker)
    monkeypatch.setattr(embedding, "get_dense_embeddings", lambda settings=None: dense)
    monkeypatch.setattr(reranking, "get_cross_encoder", lambda settings=None: reranker)
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic() - 3600)

    loaded = lifecycle.warm()

    assert touched == ["dense", "reranker"]
    assert loaded == ["dense embedder", "reranker"]
    assert lifecycle.idle_seconds() < 5.0


def test_warm_is_a_no_op_when_already_warm(
    monkeypatch: pytest.MonkeyPatch, stubs: tuple[_StubHolder, _StubHolder]
) -> None:
    def _explode(settings=None):  # pragma: no cover - must never be reached
        raise AssertionError("warm() rebuilt models that were already loaded")

    monkeypatch.setattr(embedding, "get_dense_embeddings", _explode)
    monkeypatch.setattr(reranking, "get_cross_encoder", _explode)

    assert lifecycle.warm() == ["dense embedder", "reranker"]


def test_a_warm_up_holds_the_refcount_for_the_whole_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load is the one long model operation with no request behind it.

    Without the refcount the reaper sees a stale clock and nothing in flight,
    and releases the model the warm-up is in the middle of building. The window
    sampled is the shortest one that can be enabled: at the default ten minutes
    a 90s load usually wins that race, which is the version of this bug that
    reaches production.
    """
    monkeypatch.setattr(lifecycle, "_last_used", time.monotonic() - 3600)
    sampled: list[tuple[int, bool]] = []

    class _Loader:
        _model = None

        @property
        def model(self) -> object:
            sampled.append((lifecycle.in_flight(), lifecycle.should_release(1)))
            self._model = object()
            return self._model

    dense, reranker = _Loader(), _Loader()
    monkeypatch.setattr(embedding, "get_dense_embeddings", lambda settings=None: dense)
    monkeypatch.setattr(reranking, "get_cross_encoder", lambda settings=None: reranker)

    lifecycle.warm()

    # Sampled from inside both loads, with the clock an hour stale.
    assert sampled == [(1, False), (1, False)]
    assert lifecycle.in_flight() == 0
    # And finishing refreshed the clock, so the window starts now.
    assert lifecycle.idle_seconds() < 5.0


# ---------------------------------------------------------------------------
# building the weights -- one lock per model, taken by the lazy property
# ---------------------------------------------------------------------------


def _run_together(load: object, callers: int = 2) -> list[object]:
    """Call ``load`` from ``callers`` threads released at the same instant.

    A barrier rather than a sleep: the point is to put every caller inside the
    property before any of them has built anything, so a double build is
    provoked deterministically instead of hopefully.
    """
    barrier = threading.Barrier(callers + 1)
    seen: list[object] = []
    lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()
        result = load()  # type: ignore[operator]
        with lock:
            seen.append(result)

    threads = [threading.Thread(target=_worker) for _ in range(callers)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    return seen


def test_two_callers_loading_the_embedder_build_the_weights_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The race the warm-up made likely.

    The page that starts the warm-up is the page that offers the upload box, so
    an ingest's first embed arrives mid-load. Two builds would put two copies
    of the weights in memory at once, which is the duplicate the singletons
    exist to prevent -- and it is the expensive one.
    """
    built: list[str] = []

    class _SlowEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            built.append("built")
            time.sleep(0.05)  # a load, abridged to what a test can wait for
            self.kwargs = kwargs

    monkeypatch.setattr(embedding, "HuggingFaceEmbeddings", _SlowEmbeddings)
    embedder = embedding.CachedHuggingFaceEmbeddings(
        Settings(_env_file=None, embedding_cache_dir=tmp_path)
    )

    seen = _run_together(lambda: embedder.model)

    assert built == ["built"]
    assert seen[0] is seen[1]


def test_two_callers_loading_the_reranker_build_the_weights_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    class _SlowCrossEncoder:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            built.append(model_name)
            time.sleep(0.05)

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _SlowCrossEncoder)
    encoder = reranking.LocalCrossEncoder(Settings(_env_file=None))

    seen = _run_together(lambda: encoder.model)

    assert built == [encoder.model_name]
    assert seen[0] is seen[1]


# ---------------------------------------------------------------------------
# the reaper
# ---------------------------------------------------------------------------


async def _drive_reaper(polls: int = 4) -> None:
    """Run the real reaper task briefly, then cancel it as shutdown does."""
    task = asyncio.create_task(lifecycle.reaper(10, poll_seconds=0.01))
    for _ in range(polls):
        await asyncio.sleep(0.01)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def test_the_reaper_releases_once_the_window_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(lifecycle, "should_release", lambda minutes: True)
    monkeypatch.setattr(
        lifecycle, "release_all", lambda: (calls.append(1), ["dense embedder"])[1]
    )

    asyncio.run(_drive_reaper())

    assert calls, "the reaper never fired on an idle process"


def test_the_reaper_does_nothing_while_the_models_are_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(lifecycle, "should_release", lambda minutes: False)
    monkeypatch.setattr(
        lifecycle, "release_all", lambda: (calls.append(1), ["dense embedder"])[1]
    )

    asyncio.run(_drive_reaper())

    assert calls == []


def test_the_reaper_cancels_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown cancels it, so it must not swallow or raise CancelledError.

    ``asyncio.run`` would surface a task that refused to die as an unfinished
    task warning, and the lifespan awaits it for the same reason.
    """
    monkeypatch.setattr(lifecycle, "should_release", lambda minutes: False)

    asyncio.run(_drive_reaper())
