"""Idle unloading of the two models that make this app expensive to keep warm.

The pipeline holds two models for the whole life of the process: the dense
embedder (bge-small's weights, ~130MB) and the cross-encoder reranker (~90MB), and
nothing in the process ever releases them.

The weights are the smaller half of what this costs, and it is worth being exact
about which half returning them buys. Importing torch and sentence-transformers
commits ~870MB that outlives any release, so the live server measured 1,280MB
private with the models resident and 1,051MB with them dropped: **a release frees
~229MB, and the ~1GB of imports is a floor only the process exiting can lower.**
An earlier version of this docstring quoted ~1.6GB as the models' price; that was
the process's peak after it had served questions, which is a different
measurement. The trade is 229MB against a cold load inside a visitor's first
question -- worth making for a link that is idle almost all of the time, but not
the whole footprint.

That is the right trade for a tool someone is using. It is the wrong one for a
link that sits on a resume and is idle almost all of the time -- the models are
resident whether the link was opened ten seconds or ten days ago, and with
two visitors in 66 hours the standing cost is nearly all waste.

So the models are dropped after ``model_idle_unload_minutes`` of no use and
rebuilt on demand. The rebuild is the cost a cold start has always paid (~90s,
almost all of it reading weights back off disk), which is why the API reports
``models_loaded`` and offers a warm-up endpoint: the UI starts the reload when
the page opens, so it overlaps someone reading or uploading rather than being
waited on.

Two invariants this module exists to protect:

* **Never unload under a live request.** The refcount in :func:`use` is held
  for the duration of every call that can touch a model, and the reaper
  refuses to fire while it is non-zero. Dropping a model mid-query would not
  corrupt anything -- Python keeps an object alive while a caller holds a
  reference to it -- but it *would* start a second full load inside a request
  that is already running.
* **Never lose the singleton.** :func:`release_all` clears the loaded weights
  and leaves the holder object itself in place, so a stale reference (a vector
  store built a moment ago, an embedding object a retriever is holding) cannot
  keep a dead model alive while a fresh one is built beside it. Duplicate
  weights are the exact failure the singletons were introduced to prevent.

And one this module had to take over the moment the UI started warming models
on page open -- see :data:`EMBEDDER_BUILD_LOCK`.

* **Never build the same weights twice at once.** A warm-up runs beside
  anything else the visitor is doing.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import sys
import threading
import time
from collections.abc import Iterator

from rag.logging_utils import get_logger

logger = get_logger(__name__)

# How often the reaper wakes. Deliberately coarse: this only decides how late
# the release can be, and a shorter interval buys nothing but wakeups.
_POLL_SECONDS = 30.0

_LOCK = threading.Lock()
# Monotonic, not wall clock: an NTP correction or a DST jump must not be able
# to make the models look idle when they are not, or look used forever.
_last_used: float = time.monotonic()
_in_flight = 0

# Serialises warm-up so concurrent callers produce one model load, not one each.
_WARM_LOCK = threading.Lock()

# Taken by the lazy ``.model`` properties themselves, not only by :func:`warm`.
#
# Warming on page open turned a rare race into the likely one. The page that
# starts the warm-up is the same page that invites the visitor to upload a
# document, so an ingest's first embed can arrive in the middle of the ~90s
# load: both callers find ``_model is None`` and both build. Nothing would be
# corrupted -- the attribute assignment is atomic and the loser's model is
# collected -- but two copies of the weights exist at once, ~460MB of them, on
# the machine this whole feature exists to keep idle. Under this lock the
# second caller waits and then finds the model already built.
#
# One lock per model rather than one for both: the embedder and the reranker
# share no weights, so a query needing only the embedder has no reason to wait
# behind the reranker's load. Each is a plain ``Lock``, not an ``RLock`` --
# nothing on the build path touches ``.model`` again, and a re-entrant call
# added later would deadlock here rather than silently making a second copy.
EMBEDDER_BUILD_LOCK = threading.Lock()
RERANKER_BUILD_LOCK = threading.Lock()


def mark_used() -> None:
    """Record that a model was just used."""
    global _last_used
    with _LOCK:
        _last_used = time.monotonic()


def idle_seconds() -> float:
    """Seconds since the last recorded model use."""
    with _LOCK:
        return time.monotonic() - _last_used


def in_flight() -> int:
    """How many model-using calls are running right now."""
    with _LOCK:
        return _in_flight


@contextlib.contextmanager
def use() -> Iterator[None]:
    """Hold the models open for the duration of a call that uses them.

    Every path that can touch a model wraps itself in this. Two things depend
    on it: the clock the reaper reads, and the refcount that stops the reaper
    firing mid-request. The clock is refreshed on the way *out* as well as in,
    so a long call -- an ingest slice, a 600-page parse -- leaves the idle
    window starting from when the work finished rather than when it began.
    """
    global _in_flight, _last_used
    with _LOCK:
        _in_flight += 1
        _last_used = time.monotonic()
    try:
        yield
    finally:
        with _LOCK:
            _in_flight -= 1
            _last_used = time.monotonic()


def release_all() -> list[str]:
    """Drop every loaded model; returns the names actually released.

    The imports are deferred *and* guarded by ``sys.modules``. This can be
    called by a process that never loaded a model -- a server that has served
    only ``/api/health``, say -- and importing ``rag.embedding`` in order to
    release nothing would pull torch in for the privilege of not using it.
    """
    released: list[str] = []

    if "rag.embedding" in sys.modules:
        from rag.embedding import release_dense_embeddings

        if release_dense_embeddings():
            released.append("dense embedder")

    if "rag.reranking" in sys.modules:
        from rag.reranking import release_cross_encoder

        if release_cross_encoder():
            released.append("reranker")

    if released:
        # Refcounting frees the weights the moment the last reference goes;
        # the collect is for the cycles torch's module graph is full of. Not
        # every allocator hands the pages back to the OS, so this recovers
        # most of the peak rather than all of it -- which is why the number to
        # trust is the process's own footprint, not this function's return.
        gc.collect()

    return released


def loaded_models() -> list[str]:
    """Which models are resident right now, by human-readable name."""
    loaded: list[str] = []

    if "rag.embedding" in sys.modules:
        from rag.embedding import dense_model_loaded

        if dense_model_loaded():
            loaded.append("dense embedder")

    if "rag.reranking" in sys.modules:
        from rag.reranking import cross_encoder_loaded

        if cross_encoder_loaded():
            loaded.append("reranker")

    return loaded


def models_loaded() -> bool:
    """Whether any model is currently held in memory."""
    return bool(loaded_models())


def should_release(idle_minutes: float) -> bool:
    """Whether the idle window has elapsed with nothing using the models."""
    if idle_minutes <= 0:  # 0 is the documented "off"
        return False
    if in_flight():
        # Something is mid-query or mid-ingest. Waiting is not a compromise
        # here: the refcount is what keeps a reload from landing inside a
        # request that is already running.
        return False
    return idle_seconds() >= idle_minutes * 60.0


def warm() -> list[str]:
    """Load every model, and report what is resident afterwards.

    Idempotent and serialised, because the work it does is ~90s of disk and
    CPU: concurrent callers must produce one load between them, not one each.
    Blocking on purpose -- the caller decides which thread it runs on, and the
    API hands it to the threadpool rather than the event loop.

    The whole load runs inside :func:`use`. That is not tidiness: a warm-up is
    the one long model operation with no request behind it, so without the
    refcount the reaper sees an idle process -- stale clock, nothing in flight
    -- and releases the model this call is in the middle of building. At the
    default ten-minute window a 90s load would usually win that race and at
    the shortest enabled one (a minute) it would usually lose, which is
    exactly the kind of bug that only appears in someone else's hands.
    """
    with _WARM_LOCK, use():
        if models_loaded():
            return loaded_models()

        logger.info("warming models on request (this takes ~90s the first time)")
        # Importing here rather than at module level keeps this module cheap to
        # import for everything that only needs the clock.
        from rag.embedding import get_dense_embeddings
        from rag.reranking import get_cross_encoder

        # Touching ``.model`` is what loads it; the property is lazy so that
        # /api/stats can report the model *name* without paying for the model.
        get_dense_embeddings().model
        get_cross_encoder().model

        # No ``mark_used()`` here: ``use`` refreshes the clock on the way out,
        # which is the same clock and one fewer thing to keep in step.
        loaded = loaded_models()
        logger.info("models warm: %s", ", ".join(loaded) or "nothing")
        return loaded


async def reaper(idle_minutes: float, poll_seconds: float = _POLL_SECONDS) -> None:
    """Release the models whenever they have been idle long enough.

    Runs as a task on the event loop, and does nothing but sleep and -- rarely
    -- free memory. The release is safe from here because :func:`use` keeps it
    off any request that is still holding the models.
    """
    logger.info(
        "model idle unload armed: dropping after %s min idle (%.0fs poll)",
        idle_minutes,
        poll_seconds,
    )
    while True:
        await asyncio.sleep(poll_seconds)
        if not should_release(idle_minutes):
            continue
        released = release_all()
        if released:
            logger.info(
                "released %s after %.1f min idle (they reload on next use)",
                " and ".join(released),
                idle_seconds() / 60.0,
            )
