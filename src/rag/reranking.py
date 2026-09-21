"""Cross-encoder reranking over retrieved candidates.

Reranking is where relevance quality is won on CPU. The hybrid retriever
returns ``fetch_k`` (default 30) candidates; a cross-encoder re-scores every
(query, candidate) pair jointly -- unlike bi-encoder embeddings which score a
query and a document independently and miss lexical/semantic interactions.

The stack stays free and local: :class:`LocalCrossEncoder` implements
``langchain_core.cross_encoders.BaseCrossEncoder`` around
``sentence_transformers.CrossEncoder`` with
``cross-encoder/ms-marco-MiniLM-L-6-v2`` -- 22M parameters, ~80MB, and
comfortably fast on CPU (~10-30ms per pair).

:func:`rerank_documents` is the entry point the query graph calls. It scores
the candidates, attaches each score to its document as ``rerank_score`` (the
CLI prints it and the citation map carries it), truncates to ``top_k``, and
drops anything under ``rerank_score_threshold`` -- never all of it: the best
candidate survives a threshold that would otherwise empty the context.

LangChain's ``CrossEncoderReranker`` /
``ContextualCompressionRetriever`` composition is deliberately not used: the
retriever re-runs the base retrieval that produced the candidates, and the
compressor returns documents with the scores discarded -- losing both the
printed citation scores and the threshold.
"""

from __future__ import annotations

from typing import Any

from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from rag.config import Settings, get_settings
from rag.logging_utils import get_logger, timed
from rag.model_lifecycle import RERANKER_BUILD_LOCK, use
from rag.state import MetadataKeys as MK

logger = get_logger(__name__)


class LocalCrossEncoder(BaseCrossEncoder):
    """``BaseCrossEncoder`` adapter over a local sentence-transformers model.

    Implementing the core interface (rather than subclassing the sunset
    ``langchain_community`` wrapper) keeps the dependency surface to
    langchain-core + sentence-transformers. Scores are raw logits;
    :func:`rerank_documents` compares them against ``rerank_score_threshold``
    directly.
    """

    model_name: str = ""
    device: str = "cpu"
    batch_size: int = 16

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.settings = settings or get_settings()
        self.model_name = self.settings.active_reranker_model
        self.batch_size = self.settings.rerank_batch_size
        self.device = self.settings.resolved_device
        self._model = None

    @property
    def model(self):
        """The cross-encoder, loaded on first use (lazy, like the embedder).

        Serialised and re-checked under
        :data:`~rag.model_lifecycle.RERANKER_BUILD_LOCK`, for the reason the
        embedder's is: a query can land in the middle of the warm-up the UI
        started when the page opened, and the second caller must wait for the
        first rather than build a second copy of the weights.
        """
        if self._model is None:
            with RERANKER_BUILD_LOCK:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    with timed(logger, f"Loading reranker {self.model_name}"):
                        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, candidate) pairs; returns raw logit scores.

        An empty list short-circuits: ``predict`` on an empty batch raises
        rather than returning ``[]``, so a caller that has already filtered
        its candidates down to nothing would get a traceback instead of "no
        passages found". ``rerank_documents`` guards this case before it
        builds any pairs, so the branch is a backstop, not the live path.

        The scoring itself runs under :func:`~rag.model_lifecycle.use`, which
        holds the reranker open for the duration and refreshes the idle clock
        -- reranking is the last model call in a query, so without that the
        models would look idle from the moment the query started.
        """
        if not pairs:
            return []
        with use():
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
            )
        return [float(s) for s in scores]


_CROSS_ENCODER: LocalCrossEncoder | None = None


def get_cross_encoder(settings: Settings | None = None) -> LocalCrossEncoder:
    """Process-wide cross-encoder; the loaded ~90MB model is reused across
    queries instead of being reloaded from disk on every rerank."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = LocalCrossEncoder(settings)
    return _CROSS_ENCODER


def release_cross_encoder() -> bool:
    """Drop the loaded cross-encoder; ``True`` if there was one to drop.

    Mirrors :func:`rag.embedding.release_dense_embeddings`: only ``_model`` is
    cleared, so the singleton survives and a stale reference cannot strand a
    dead model beside a freshly loaded second copy. Idempotent.
    """
    instance = _CROSS_ENCODER
    if instance is None or instance._model is None:
        return False
    instance._model = None
    return True


def cross_encoder_loaded() -> bool:
    """Whether the cross-encoder's weights are currently resident."""
    return _CROSS_ENCODER is not None and _CROSS_ENCODER._model is not None


def rerank_documents(
    question: str,
    candidates: list[Document],
    settings: Settings | None = None,
) -> list[Document]:
    """Rerank a candidate list in place, outside the retriever chain.

    The query graph calls this when it has already retrieved (with neighbour
    expansion applied) and wants scores attached to each document without
    re-running the base retrieval.
    """
    settings = settings or get_settings()
    if not candidates:
        return []

    pairs = [(question, d.page_content) for d in candidates]
    scores = get_cross_encoder(settings).score(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    ranked = ranked[: settings.top_k]

    results: list[Document] = []
    for score, doc in ranked:
        meta = dict(doc.metadata)
        meta["rerank_score"] = score
        results.append(Document(page_content=doc.page_content, metadata=meta))

    threshold = settings.rerank_score_threshold
    if threshold > 0 and results:
        # The best passage survives whatever it scored. A threshold exists to
        # remove noise, and an empty context is not a cleaner answer than a
        # weak one -- it is a different and worse failure, because a model
        # handed nothing can only report that nothing covers the question, and
        # that report is the signal the UI reads before offering a web search.
        # A threshold that empties the context would therefore manufacture the
        # exact symptom the fallback exists to diagnose.
        kept = [d for d in results if d.metadata[MK.RERANK_SCORE] >= threshold]
        kept = kept or results[:1]
        if len(kept) != len(results):
            logger.info(
                "  rerank threshold %.2f dropped %d candidate(s)",
                threshold,
                len(results) - len(kept),
            )
        results = kept

    return results
