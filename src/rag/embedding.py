"""Embedding models: dense (BGE-small) and sparse (BM25).

Both models run fully locally through the libraries already in the stack:

* dense — ``langchain_huggingface.HuggingFaceEmbeddings`` wrapping
  ``BAAI/bge-small-en-v1.5`` (33M params, 384-dim, 512-token context,
  MIT licensed, top-tier on MTEB for its size). ~130MB downloads once from
  Hugging Face, then everything is offline and free. Swap to
  ``BAAI/bge-m3`` via ``RAG_DENSE_MODEL`` for multilingual corpora or
  8192-token contexts at ~2.3GB.
* sparse — ``langchain_qdrant.FastEmbedSparse`` wrapping ``Qdrant/bm25``.
  BM25 runs as lexical scoring (a vocabulary of stopwords, no neural
  inference), so it costs effectively nothing on CPU.

The dense side has a **disk cache** keyed by (model, text) content hash. A
re-ingest of an unchanged corpus would otherwise recompute identical vectors
for thousands of chunks; with the cache it is a dictionary of pickle hits.

Device selection honours ``RAG_DEVICE`` and degrades to CPU automatically.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import Settings, get_settings
from rag.logging_utils import get_logger, timed
from rag.model_lifecycle import EMBEDDER_BUILD_LOCK, use

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dense embeddings with disk cache
# ---------------------------------------------------------------------------


class CachedHuggingFaceEmbeddings(Embeddings):
    """``HuggingFaceEmbeddings`` with a persistent on-disk vector cache.

    Cache entries are keyed by sha256(model + text), so identical text under
    the same model always hits. The cache never invalidates the truth -- on a
    miss the model is called, and the result is written back.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: HuggingFaceEmbeddings | None = None
        self._cache_dir = self.settings.embedding_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # -- lazy model load -----------------------------------------------------

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """The underlying model, loaded on first use.

        Lazy so that ``GET /api/stats`` -- which reports the model *name*
        without needing it -- never triggers a multi-GB download.

        The build is serialised and re-checked under
        :data:`~rag.model_lifecycle.EMBEDDER_BUILD_LOCK`. Two callers can
        arrive together -- the warm-up the UI starts on page open and the first
        embed of an upload made from that same page -- and both would find
        ``_model`` empty, leaving two copies of the weights in memory for as
        long as the loser takes to be collected.
        """
        if self._model is None:
            with EMBEDDER_BUILD_LOCK:
                if self._model is None:
                    with timed(logger, f"Loading dense model {self.settings.dense_model}"):
                        self._model = HuggingFaceEmbeddings(
                            model_name=self.settings.dense_model,
                            model_kwargs={"device": self.settings.resolved_device},
                            encode_kwargs={"normalize_embeddings": True},
                        )
        return self._model

    # -- cache helpers -------------------------------------------------------

    def _key(self, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.settings.dense_model.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _load(self, key: str) -> list[float] | None:
        path = self._cache_dir / f"{key[:2]}/{key}.pkl"
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)  # noqa: S301 - our own cache files
        except Exception:
            # A truncated or corrupt cache entry is re-embeddable; never fatal.
            path.unlink(missing_ok=True)
            return None

    def _store(self, key: str, value: list[float]) -> None:
        """Write a cache entry atomically.

        A plain ``open(..., "wb")`` interrupted mid-write (Ctrl-C, disk full,
        power loss) leaves a half-written pickle that poisons the cache for
        that text: the reader would keep loading a truncated vector, or
        repeatedly delete and re-embed it. Writing to a temp file in the same
        directory and renaming makes the entry appear all-or-nothing, because
        ``os.replace`` is atomic on a POSIX filesystem and on Windows NTFS.
        """
        path = self._cache_dir / f"{key[:2]}/{key}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{key}.", suffix=".tmp", delete=False
        )
        try:
            with handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(handle.name, path)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise

    # -- Embeddings interface --------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents, serving cached texts from disk.

        The call is held open with :func:`~rag.model_lifecycle.use` for its
        whole duration, so the idle reaper cannot drop the model out from under
        a batch that is already running.
        """
        with use():
            return self._embed_documents(texts)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[int] = []

        for i, text in enumerate(texts):
            cached = self._load(self._key(text))
            if cached is not None:
                results[i] = cached
                self.hits += 1
            else:
                to_embed.append(i)
                self.misses += 1

        if to_embed:
            fresh = self.model.embed_documents([texts[i] for i in to_embed])
            if len(fresh) != len(to_embed):
                # A silent short return would leave holes in the result and
                # misalign vectors against their chunks downstream.
                raise RuntimeError(
                    f"Embedding model returned {len(fresh)} vector(s) for "
                    f"{len(to_embed)} input text(s)"
                )
            for i, vec in zip(to_embed, fresh):
                results[i] = vec
                self._store(self._key(texts[i]), vec)

        missing = [i for i, result in enumerate(results) if result is None]
        if missing:
            # Defensive: unreachable unless the model returned None for a slot.
            raise RuntimeError(f"No embedding produced for {len(missing)} input(s)")
        return [result for result in results if result is not None]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query, cached like documents.

        Guarded like :meth:`embed_documents`: this is the call a query makes
        first, so it is also the call most likely to be the one that loads the
        model after an idle release.
        """
        with use():
            return self._embed_query(text)

    def _embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        cached = self._load(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vec = self.model.embed_query(text)
        self._store(key, vec)
        return vec

    def cache_stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


_DENSE_EMBEDDINGS: CachedHuggingFaceEmbeddings | None = None


def get_dense_embeddings(settings: Settings | None = None) -> CachedHuggingFaceEmbeddings:
    """Process-wide dense embedder, loaded once.

    Constructing a fresh ``CachedHuggingFaceEmbeddings`` per call site looks
    harmless because each one loads its model lazily, but every instance ends
    up holding its own ``SentenceTransformer`` -- a second copy of the weights
    in RAM, and a second model load. Ingest builds a vector store for the
    dimension probe and again for the upsert, and a query builds one for the
    primary path and another if the ensemble fallback fires, so without this
    singleton the bill arrives as duplicated memory rather than as an error.

    The disk cache is likewise shared, which is the point: a query in one
    request and an ingest slice in another count against the same cache.
    """
    global _DENSE_EMBEDDINGS
    if _DENSE_EMBEDDINGS is None:
        _DENSE_EMBEDDINGS = CachedHuggingFaceEmbeddings(settings)
    return _DENSE_EMBEDDINGS


# ---------------------------------------------------------------------------
# Idle release
# ---------------------------------------------------------------------------


def release_dense_embeddings() -> bool:
    """Drop the loaded weights; ``True`` if there were any to drop.

    Only the model is cleared -- the ``CachedHuggingFaceEmbeddings`` instance
    stays, and so does the module-level singleton. That is deliberate on both
    counts:

    * Clearing ``_model`` frees the ``SentenceTransformer`` and its torch
      tensors even if something else still holds the embedder object (a
      ``QdrantVectorStore`` built during the call that just finished, say).
      Dropping the singleton instead would leave that stale reference holding
      the weights while a new singleton built a second copy.
    * The instance carries the cache hit/miss counters and the cache directory.
      Keeping it means "the model was evicted" and "the cache was thrown away"
      stay separate facts -- re-ingesting after an eviction still hits the disk
      cache rather than re-embedding a corpus.

    Idempotent: calling it twice, or on a process that never loaded a model,
    is a no-op returning ``False``.
    """
    instance = _DENSE_EMBEDDINGS
    if instance is None or instance._model is None:
        return False
    instance._model = None
    return True


def dense_model_loaded() -> bool:
    """Whether the dense model's weights are currently resident."""
    return _DENSE_EMBEDDINGS is not None and _DENSE_EMBEDDINGS._model is not None


# ---------------------------------------------------------------------------
# Sparse embeddings (BM25)
# ---------------------------------------------------------------------------


def get_sparse_embeddings(settings: Settings | None = None) -> "FastEmbedSparse":
    """Sparse BM25 embeddings for the hybrid collection.

    ``FastEmbedSparse`` implements Qdrant's ``SparseEmbeddings`` protocol
    through FastEmbed's local ONNX runtime -- free, offline after the first
    download, and near-instant on CPU because BM25 is lexical, not neural.
    """
    from langchain_qdrant import FastEmbedSparse

    settings = settings or get_settings()
    return FastEmbedSparse(
        model_name=settings.sparse_model,
        cache_dir=str(settings.embedding_cache_dir),
    )
