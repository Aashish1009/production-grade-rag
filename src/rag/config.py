"""Central configuration.

Every tunable in the pipeline lives here so that nothing downstream hardcodes
a model name, a batch size, or a path. Values are overridable by environment
variable (prefix ``RAG_``) or a local ``.env`` file -- see ``.env.example``.

The settings object is cached, so ``get_settings()`` is cheap to call from
anywhere and always returns the same instance.
"""

from __future__ import annotations

import importlib.util
import logging
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RetrievalMode(StrEnum):
    """How candidates are pulled from the vector store."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class PdfStrategy(StrEnum):
    """Unstructured parsing strategy for PDFs.

    ``hi_res`` runs layout detection and is the only strategy that recovers
    tables; ``fast`` is roughly 5x quicker but finds none. ``ocr_only`` is for
    scanned documents with no text layer.
    """

    FAST = "fast"
    HI_RES = "hi_res"
    OCR_ONLY = "ocr_only"
    AUTO = "auto"


class Settings(BaseSettings):
    """Runtime configuration for the whole pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # The API-key fields declare ``validation_alias`` so that a
        # conventional OPENAI_API_KEY / GROQ_API_KEY in the environment is
        # honoured without a RAG_ prefix. Without ``populate_by_name`` that
        # alias becomes the *only* accepted spelling, and because
        # ``extra="ignore"`` swallows anything unrecognised, calling
        # ``Settings(openai_api_key=...)`` in Python would be silently
        # discarded rather than raising. Tests and embedders do exactly that.
        populate_by_name=True,
    )

    # --- paths -------------------------------------------------------------

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    qdrant_path: Path = Field(default=PROJECT_ROOT / "qdrant_db")
    embedding_cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "embedding_cache")
    uploads_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "uploads",
        description=(
            "Where the API parks uploaded files. Under data/, which "
            "loaders.EXCLUDED_DIRS already skips during discovery, so a "
            "recursive ingest can never pick up its own uploads."
        ),
    )

    # --- server ------------------------------------------------------------

    api_host: str = Field(
        default="127.0.0.1",
        description=(
            "Loopback by default. This server has no authentication and can "
            "read and delete the index, so binding it to a public interface "
            "is an explicit decision, not a default."
        ),
    )
    api_port: int = Field(default=8000, gt=0, lt=65536)

    # --- models ------------------------------------------------------------

    dense_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description=(
            "33M params, 384-dim, 512-token context -- the best "
            "quality-per-byte free embedding model for CPU-only English "
            "corpora. Swap to BAAI/bge-m3 (8192 tokens, multilingual, ~2.3GB) "
            "if either of those constraints bites."
        ),
    )
    sparse_model: str = Field(
        default="Qdrant/bm25",
        description="FastEmbed BM25. Classic lexical scoring, no neural inference.",
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    # --- reranking ---------------------------------------------------------
    # Cross-encoders score every (query, candidate) pair, so cost scales with
    # fetch_k. ms-marco-MiniLM-L-6-v2 is 22M parameters (~90MB) and scores
    # ~10-30ms per pair on CPU, which keeps a 30-candidate rerank under a
    # second. Swap to BAAI/bge-reranker-v2-m3 (568M) on a GPU if quality
    # matters more than latency.

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder via sentence-transformers, loaded locally.",
    )

    # --- chunking ----------------------------------------------------------

    chunk_size: int = Field(
        default=384,
        gt=0,
        description=(
            "In dense-model tokens. Below bge-small's 512-token window so "
            "overlap never pushes a chunk past the truncation point."
        ),
    )
    chunk_overlap: int = Field(default=48, ge=0)
    min_chunk_tokens: int = Field(
        default=24,
        ge=0,
        description="Chunks below this are dropped as fragments.",
    )
    max_model_tokens: int = Field(
        default=512,
        gt=0,
        description="Hard context ceiling of the dense model.",
    )

    # --- parsing -----------------------------------------------------------

    pdf_strategy: PdfStrategy = PdfStrategy.HI_RES
    pdf_batch_size: int = Field(
        default=20,
        gt=0,
        description="Pages per split batch; keeps peak memory bounded.",
    )
    dedupe_window: int = Field(
        default=10,
        ge=0,
        description=(
            "Local sliding window for near-duplicate removal. Deliberately "
            "local, never global: text repeated far apart in a document is "
            "usually intentional, while text repeated a few elements apart is "
            "an extraction artifact."
        ),
    )
    furniture_page_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Text appearing on more than this fraction of pages is treated as "
            "running header/footer furniture and dropped regardless of the "
            "category Unstructured assigned it."
        ),
    )
    ocr_languages: list[str] = Field(default_factory=lambda: ["eng"])

    # --- retrieval ---------------------------------------------------------

    collection_name: str = "documents"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    fetch_k: int = Field(
        default=30,
        gt=0,
        description=(
            "Candidates before rerank. Tuned down from the usual 50 because "
            "reranking cost is linear in this number and this project runs on "
            "CPU. Raise it if you move to a GPU."
        ),
    )
    top_k: int = Field(
        default=8,
        gt=0,
        description=(
            "Chunks returned after rerank -- which is also the whole context "
            "budget the answer stage spends. Raised from 5 once chunk packing "
            "landed: the reranker already scores fetch_k candidates, so "
            "widening top_k costs prompt length and nothing else."
        ),
    )
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rerank_score_threshold: float = Field(
        default=0.0,
        description=(
            "Drop reranked chunks scoring below this logit. 0.0 disables it. "
            "ms-marco cross-encoders put the decision boundary at 0 -- a "
            "relevant pair scores positive, an unrelated one strongly "
            "negative -- so raising this above 0 trims passages the model is "
            "being invited to blend into the answer. Whatever it is set to, "
            "the best chunk always survives: see rerank_documents. An empty "
            "context is a worse failure than a weak one, because the model "
            "can only report that nothing covers the question, and that is "
            "the signal the web-search fallback reads."
        ),
    )
    context_neighbour_radius: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "How many chunks either side of each reranked passage to pull in "
            "as context. 0 disables widening. Bounded at 3 because the cost is "
            "quadratic-ish in this number -- every extra step is another "
            "batched read and another chunk of prompt per winning passage."
        ),
    )
    context_token_budget: int = Field(
        default=4000,
        gt=0,
        description=(
            "Ceiling on the whole rendered context, in dense-model tokens, "
            "after widening. Widening multiplies the prompt by up to "
            "2*radius+1, so this is what keeps ``top_k`` a budget rather than "
            "a multiplier. Context is trimmed from the lowest-ranked passages "
            "inward, and a passage is never dropped in favour of its own "
            "neighbours."
        ),
    )

    # --- batching ----------------------------------------------------------
    # Sized for CPU; raise substantially on GPU.

    rerank_batch_size: int = Field(default=16, gt=0)

    ingest_slice_size: int = Field(
        default=64,
        gt=0,
        description=(
            "Chunks per Qdrant upsert during ingest. The knob trades query "
            "latency against progress granularity: the store lock is taken "
            "per slice, so a query arriving mid-ingest waits for one slice "
            "rather than for the whole document. Lower it if queries feel "
            "blocked during a large ingest; raise it for faster ingest."
        ),
    )

    max_upload_mb: int = Field(
        default=512,
        gt=0,
        description=(
            "Largest file the API will accept. Sized so a scanned 600-page "
            "book PDF fits comfortably; this is a local single-user server, "
            "so the limit guards against a mis-dragged disk image rather "
            "than against an attacker."
        ),
    )

    # --- optional generation ----------------------------------------------
    # Retrieval, embedding, reranking and storage are all local and free.
    # Generation is the only stage that calls out to a hosted API, and it is
    # off by default -- the query graph returns ranked, cited context without
    # it. Provider selection under "auto": OpenAI if an OpenAI key is present,
    # otherwise Groq (which has a free tier), otherwise generation stays off.

    enable_generation: bool = False
    llm_provider: Literal["auto", "openai", "groq"] = "auto"

    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = Field(
        default=None,
        repr=False,
        # Excluded from model_dump()/model_dump_json() as well as from repr,
        # so no future call site can serialise a secret by forgetting to pass
        # an exclude set -- ``GET /api/config`` dumps the model directly, and
        # that is exactly the kind of memory that fails.
        exclude=True,
        # Accept the conventional OPENAI_API_KEY as well as the prefixed form,
        # so an existing shell environment works without being re-exported.
        validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )

    # LiteLLM model strings, provider prefix included ("groq/..."), so they
    # can be passed to the gateway verbatim. Defaults are Groq's free-tier
    # models as of 2026-09-17: one strong general model first, then a lighter
    # one, then a different family -- each free-tier model carries its own
    # daily request budget, so a second family buys a second budget rather
    # than one more slot in the same queue.
    #
    # Both ids this used to default to (llama-3.3-70b-versatile as primary,
    # llama-3.1-8b-instant as the last fallback) were decommissioned for
    # non-Enterprise tiers on 2026-08-16 and now fail with
    # "model_decommissioned"; Groq's own migration table names these gpt-oss
    # models as their replacements. Re-check before trusting these ids:
    # https://console.groq.com/docs/deprecations
    groq_model: str = "groq/openai/gpt-oss-120b"
    groq_fallback_models: list[str] = Field(
        default_factory=lambda: [
            "groq/openai/gpt-oss-20b",
            "groq/qwen/qwen3.6-27b",
        ],
        description=(
            "Tried in order when the primary Groq model errors or is rate "
            "limited. Groq's free tier has aggressive per-model limits, so a "
            "fallback chain is what keeps generation usable."
        ),
    )
    groq_api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
        validation_alias=AliasChoices("RAG_GROQ_API_KEY", "GROQ_API_KEY"),
    )

    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- optional web search ----------------------------------------------
    # The fallback for a question the corpus cannot answer. This is the only
    # path in the pipeline that sends anything off the machine -- every other
    # stage, retrieval included, is local -- and it only ever runs because a
    # user clicked a button, never on its own.
    #
    # Two backends, chosen by what is configured rather than by preference:
    #
    #   tavily      A real search API with a free tier (1,000 searches a
    #               month). It runs the search *and* fetches each result,
    #               returning the page text, so the whole fallback is one
    #               request. Needs TAVILY_API_KEY.
    #   duckduckgo  Keyless, through the ``ddgs`` library, which reads
    #               DuckDuckGo's HTML endpoint. No signup -- but it is
    #               unofficial, so DDG can rate-limit it and a change to their
    #               markup breaks it. Returns snippets only, so the top
    #               results are fetched separately.
    #
    # ``auto`` prefers Tavily when a key is present and falls back to
    # DuckDuckGo, which is what makes the feature work on a fresh checkout
    # with no signup at all.

    enable_web_search: bool = Field(
        default=True,
        description=(
            "On, unlike generation, because it cannot run by itself: the "
            "feature is reachable only by clicking a button on an answer that "
            "failed to find anything."
        ),
    )
    web_search_provider: Literal["auto", "tavily", "duckduckgo"] = "auto"
    tavily_api_key: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
        validation_alias=AliasChoices("RAG_TAVILY_API_KEY", "TAVILY_API_KEY"),
    )
    tavily_api_base_url: str = Field(
        default="https://api.tavily.com",
        description=(
            "The LangChain tool appends '/search' to this. A setting rather "
            "than a constant so tests can point it at a fixture."
        ),
    )
    web_search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = Field(
        default="basic",
        description=(
            "'advanced' costs two Tavily credits per search rather than one, "
            "which halves the free tier's thousand a month."
        ),
    )
    web_search_max_results: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Tavily caps at 20; ddgs has no cap but gets slower.",
    )
    web_fetch_pages: int = Field(
        default=3,
        gt=0,
        description=(
            "Pages fetched on the keyless path, which has snippets only. "
            "Ignored by Tavily, which fetches every result it returns."
        ),
    )
    web_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        description=(
            "Per request. Deliberately shorter than the generation timeout: "
            "this is fetching a page, not writing an answer."
        ),
    )
    web_max_page_chars: int = Field(
        default=6000,
        gt=0,
        description=(
            "Characters kept per page. Three pages at this bound is roughly "
            "4,500 tokens, which leaves room for the answer inside a "
            "free-tier model's context window."
        ),
    )
    web_page_max_bytes: int = Field(
        default=2_000_000,
        gt=0,
        description="A response larger than this is skipped rather than parsed.",
    )

    # --- logging -----------------------------------------------------------

    log_level: str = "INFO"

    # --- validation --------------------------------------------------------

    @field_validator(
        "data_dir", "qdrant_path", "embedding_cache_dir", "uploads_dir", mode="after"
    )
    @classmethod
    def _resolve_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def _check_chunk_budget(self) -> Settings:
        """Guard the failure mode that silently truncates embeddings.

        If ``chunk_size`` exceeds the model's context window, sentence
        transformers truncates without warning -- the stored text is complete
        but only its opening tokens are ever embedded, and retrieval quality
        degrades invisibly. Fail loudly at startup instead.
        """
        if self.chunk_size > self.max_model_tokens:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) exceeds max_model_tokens "
                f"({self.max_model_tokens}). The embedding model would silently "
                f"truncate every chunk. Lower chunk_size or pick a model with a "
                f"larger context window."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise splitting cannot "
                f"make forward progress."
            )
        if self.top_k > self.fetch_k:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed fetch_k ({self.fetch_k}) -- "
                f"the reranker can only narrow the candidate set, not widen it."
            )
        if self.enable_generation and self.resolved_llm_provider is None:
            raise ValueError(
                "enable_generation is true but no API key was found for any "
                "provider. Set OPENAI_API_KEY for gpt-4o-mini, or GROQ_API_KEY "
                "to use Groq's free tier, or leave generation disabled to run "
                "retrieval-only."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("llm_provider is 'openai' but OPENAI_API_KEY is unset.")
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError("llm_provider is 'groq' but GROQ_API_KEY is unset.")
        if self.web_search_provider == "tavily" and not self.tavily_api_key:
            raise ValueError(
                "web_search_provider is 'tavily' but TAVILY_API_KEY is unset. "
                "Set the key, use 'duckduckgo' for the keyless backend, or "
                "'auto' to take whichever is available."
            )
        return self

    # --- derived -----------------------------------------------------------

    @property
    def resolved_device(self) -> str:
        """Concrete torch device, resolving ``auto`` against what is installed."""
        if self.device != "auto":
            return self.device
        return _detect_device()

    @property
    def active_reranker_model(self) -> str:
        """Model id of the configured reranker."""
        return self.reranker_model

    @property
    def resolved_llm_provider(self) -> str | None:
        """Which chat provider to use, or ``None`` if no key is available.

        Under ``auto`` OpenAI wins when its key is set, since an explicitly
        configured paid key signals intent; Groq is the free fallback.
        """
        if self.llm_provider == "openai":
            return "openai" if self.openai_api_key else None
        if self.llm_provider == "groq":
            return "groq" if self.groq_api_key else None

        if self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        return None

    @property
    def resolved_web_provider(self) -> str | None:
        """Which web-search backend to use, or ``None`` if neither can run.

        ``None`` is the load-bearing answer. It is what stops the UI offering
        a "Search the web" button on a machine where the search would only
        fail: the query path reads this to decide whether to suggest one, so a
        checkout with no Tavily key and no ``ddgs`` installed simply never
        shows the offer rather than showing a dead one.
        """
        if not self.enable_web_search:
            return None
        if self.web_search_provider == "tavily":
            return "tavily" if self.tavily_api_key else None
        if self.web_search_provider == "duckduckgo":
            return "duckduckgo" if _ddgs_available() else None

        if self.tavily_api_key:
            return "tavily"
        return "duckduckgo" if _ddgs_available() else None

    def ensure_directories(self) -> None:
        """Create every directory the pipeline writes to."""
        for path in (
            self.data_dir,
            self.qdrant_path,
            self.embedding_cache_dir,
            self.uploads_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _detect_device() -> str:
    """Pick the best available torch backend, degrading to CPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        logger.warning("torch not importable; falling back to CPU")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def _ddgs_available() -> bool:
    """Whether the keyless web-search backend's library is installed.

    ``find_spec`` rather than an import: this runs while settings are being
    resolved, which includes every ``GET /api/config`` and every corpus query
    deciding whether to offer the button -- and importing ``ddgs`` there would
    pay for a package most queries never use. ``find_spec`` only locates it.
    """
    try:
        return importlib.util.find_spec("ddgs") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False
