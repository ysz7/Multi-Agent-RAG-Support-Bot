import pytest

from app.core.config import get_settings
from app.core.observability import reset_observability

# Env vars Settings reads; cleared before each test so a developer's real
# environment can never leak into an assertion.
_MANAGED_VARS = (
    "APP_ENV",
    "LOG_LEVEL",
    "LLM_PROVIDER",
    "ANTHROPIC_API_KEY",
    "CLAUDE_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "VECTOR_STORE",
    "DATABASE_URL",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "RETRIEVAL_TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "DOCUMENTS_DIR",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "AUTH_MODE",
    "LOCAL_USER_ID",
    "LOCAL_TENANT_ID",
    "JWT_SECRET",
    "JWT_ALGORITHM",
)


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    """Isolate configuration: no cached settings, no ambient env, no .env on disk.

    Running from an empty tmp dir keeps `Settings`' relative `env_file=".env"`
    from picking up the repository's own .env.
    """
    get_settings.cache_clear()
    # Tracing state is a module global; a client configured by one test must
    # not follow the suite into the next one.
    reset_observability()
    for name in _MANAGED_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    def _set(**values: str) -> None:
        for key, value in values.items():
            monkeypatch.setenv(key.upper(), value)

    yield _set
    get_settings.cache_clear()
    reset_observability()
