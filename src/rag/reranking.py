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
        """The cross-encoder, loaded on first use (lazy, like the embedder)."""
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
        """
        if not pairs:
            return []
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
