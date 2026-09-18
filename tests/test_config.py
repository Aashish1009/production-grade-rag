"""Unit tests for configuration and its startup validation.

The validators exist to turn silent quality failures (truncated embeddings, a
reranker that cannot widen, a generation stage with no key) into loud startup
errors, so each one is pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient keys and the developer's .env out of these assertions."""
    for var in ("OPENAI_API_KEY", "GROQ_API_KEY", "RAG_OPENAI_API_KEY", "RAG_GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# startup validation
# ---------------------------------------------------------------------------


def test_chunk_size_beyond_the_model_window_is_rejected() -> None:
    # Exceeding the model's context silently truncates every embedding: the
    # stored text is complete but only its opening tokens are ever compared.
    with pytest.raises(ValueError, match="truncate"):
        _settings(chunk_size=900, max_model_tokens=512)


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _settings(chunk_size=384, chunk_overlap=384)


def test_top_k_cannot_exceed_fetch_k() -> None:
    with pytest.raises(ValueError, match="fetch_k"):
        _settings(fetch_k=5, top_k=10)


def test_generation_without_any_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="no API key"):
        _settings(enable_generation=True)


def test_explicit_provider_without_its_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _settings(llm_provider="openai")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        _settings(llm_provider="groq")


# ---------------------------------------------------------------------------
# provider resolution
# ---------------------------------------------------------------------------


def test_no_keys_resolves_to_no_provider() -> None:
    assert _settings().resolved_llm_provider is None


def test_groq_key_resolves_to_groq() -> None:
    assert _settings(groq_api_key="gsk_test").resolved_llm_provider == "groq"


def test_openai_wins_over_groq_under_auto() -> None:
    settings = _settings(openai_api_key="sk-test", groq_api_key="gsk_test")
    assert settings.resolved_llm_provider == "openai"


def test_forced_provider_without_its_key_is_rejected_not_downgraded() -> None:
    # llm_provider="groq" with only an OpenAI key must fail at startup rather
    # than silently falling back to OpenAI: the operator asked for one
    # provider on purpose, and quietly billing the other one is worse than
    # refusing to start.
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        _settings(llm_provider="groq", openai_api_key="sk-test")


def test_api_key_field_names_are_accepted_outside_the_environment() -> None:
    # These fields carry a validation_alias so that a conventional
    # OPENAI_API_KEY / GROQ_API_KEY in the shell is honoured without a RAG_
    # prefix. With extra="ignore", an alias that is not also populated by
    # field name makes Settings(openai_api_key=...) vanish without a warning.
    settings = _settings(openai_api_key="sk-test")
    assert settings.openai_api_key == "sk-test"
    assert settings.resolved_llm_provider == "openai"


def test_keys_are_hidden_from_the_repr_and_from_dumps() -> None:
    # repr=False alone would only hide the value from repr; `GET /api/config`
    # dumps the model with model_dump_json, so the fields also carry
    # exclude=True.
    settings = _settings(openai_api_key="sk-secret", groq_api_key="gsk-secret")
    rendered = repr(settings) + settings.model_dump_json()
    assert "sk-secret" not in rendered
    assert "gsk-secret" not in rendered


# ---------------------------------------------------------------------------
# derived values and paths
# ---------------------------------------------------------------------------


def test_paths_are_expanded_and_resolved() -> None:
    # A leading ~ in an env var must reach the filesystem as a real path, and
    # resolve() keeps relative RAG_DATA_DIR values anchored to the project.
    settings = _settings(data_dir="~/rag-data-test")
    assert settings.data_dir.is_absolute()
    assert "~" not in str(settings.data_dir)


def test_explicit_device_is_respected() -> None:
    assert _settings(device="cpu").resolved_device == "cpu"


def test_ensure_directories_creates_every_writable_path(tmp_path: Path) -> None:
    settings = _settings(
        data_dir=tmp_path / "data",
        qdrant_path=tmp_path / "qdrant",
        embedding_cache_dir=tmp_path / "cache",
    )
    settings.ensure_directories()

    assert settings.data_dir.is_dir()
    assert settings.qdrant_path.is_dir()
    assert settings.embedding_cache_dir.is_dir()


def test_retrieval_mode_values_match_the_qdrant_client() -> None:
    from langchain_qdrant import RetrievalMode

    for mode in ("dense", "sparse", "hybrid"):
        # The pipeline constructs RetrievalMode from this enum's .value, so a
        # drift between the two would surface as a key error at query time.
        assert RetrievalMode(_settings(retrieval_mode=mode).retrieval_mode.value)
