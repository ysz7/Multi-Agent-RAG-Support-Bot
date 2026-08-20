"""Application configuration.

Every backend choice in this project is environment-driven: no module may hardcode a
vector store, an LLM provider, or an auth mode. Read them from `get_settings()` only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["claude", "ollama"]
VectorStoreName = Literal["pgvector", "qdrant"]
AuthMode = Literal["local", "jwt"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General -----------------------------------------------------------
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # --- LLM provider ------------------------------------------------------
    llm_provider: LLMProviderName = "claude"
    anthropic_api_key: str | None = None
    # Pinned deliberately; adaptive thinking is configured in the provider (Phase 3).
    claude_model: str = "claude-opus-5"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    # Cap on generated tokens. Reasoning models spend part of this budget on
    # `message.thinking`, so a small value can yield an empty answer.
    ollama_num_predict: int = 2048
    ollama_timeout_s: float = 300.0
    # Cap, not a target — Claude stops at end_turn.
    claude_max_tokens: int = 16000
    # Thinking depth / token spend. "medium" keeps support answers responsive.
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Re-route server-side if a safety classifier declines, instead of erroring.
    claude_refusal_fallback: bool = True

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # --- Vector store ------------------------------------------------------
    vector_store: VectorStoreName = "pgvector"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ragbot"
    # Qdrant adapter is written but not bundled; requires the [qdrant] extra.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 6
    db_pool_max_size: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 150
    documents_dir: Path = Path("data/documents")

    # --- Observability -----------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # --- Auth --------------------------------------------------------------
    # "local" resolves a fixed principal from config (no tokens, no login).
    # "jwt" verifies a bearer token and maps its claims onto the same Principal.
    auth_mode: AuthMode = "local"
    local_user_id: str = "local-user"
    local_tenant_id: str = "default"
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    jwt_issuer: str | None = None
    jwt_audience: str | None = None

    @model_validator(mode="after")
    def _check_required_by_mode(self) -> Settings:
        if self.llm_provider == "claude" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude")
        if self.auth_mode == "jwt" and not self.jwt_secret:
            raise ValueError("JWT_SECRET is required when AUTH_MODE=jwt")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Call `get_settings.cache_clear()` in tests."""
    return Settings()
