"""Phase 1 smoke tests: configuration loads and enforces its own invariants."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_defaults_match_readme(env):
    env(ANTHROPIC_API_KEY="sk-test")
    settings = Settings()

    assert settings.llm_provider == "claude"
    assert settings.claude_model == "claude-opus-5"
    assert settings.vector_store == "pgvector"
    assert settings.auth_mode == "local"


def test_claude_provider_requires_api_key(env):
    env(LLM_PROVIDER="claude", ANTHROPIC_API_KEY="")
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings()


def test_ollama_provider_needs_no_api_key(env):
    env(LLM_PROVIDER="ollama", ANTHROPIC_API_KEY="", OLLAMA_MODEL="llama3.1")
    settings = Settings()

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"


def test_jwt_mode_requires_secret(env):
    env(AUTH_MODE="jwt", ANTHROPIC_API_KEY="sk-test", JWT_SECRET="")
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings()


def test_jwt_mode_with_secret_is_valid(env):
    env(AUTH_MODE="jwt", ANTHROPIC_API_KEY="sk-test", JWT_SECRET="dev-secret")
    assert Settings().auth_mode == "jwt"


def test_unknown_vector_store_is_rejected(env):
    env(ANTHROPIC_API_KEY="sk-test", VECTOR_STORE="faiss")
    with pytest.raises(ValidationError):
        Settings()


def test_chunk_overlap_must_be_smaller_than_chunk_size(env):
    env(ANTHROPIC_API_KEY="sk-test", CHUNK_SIZE="500", CHUNK_OVERLAP="500")
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings()


def test_langfuse_disabled_without_keys(env):
    env(ANTHROPIC_API_KEY="sk-test")
    assert Settings().langfuse_enabled is False


def test_get_settings_is_cached(env):
    env(ANTHROPIC_API_KEY="sk-test")
    assert get_settings() is get_settings()
