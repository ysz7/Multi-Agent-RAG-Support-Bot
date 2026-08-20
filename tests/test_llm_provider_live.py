"""Live smoke tests against a running Ollama.

Skipped automatically when Ollama is unreachable, so CI and offline runs stay
green. These are the only tests that touch a real model.
"""

from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.core.llm_provider import OllamaEmbeddingProvider, OllamaProvider

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.live


def _ollama_up(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=2.0).status_code == 200
    except Exception:
        return False


def _model_installed(base_url: str, model: str) -> bool:
    """Ollama answers 404 on /api/chat for a model that was never pulled."""
    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=5.0).json()
    except Exception:
        return False
    return any(m.get("name") == model for m in tags.get("models", []))


@pytest.fixture
def live_settings(env) -> Settings:
    """Deliberately reads the repo's real .env — these tests exercise the models
    actually installed locally, not the library defaults."""
    env(ANTHROPIC_API_KEY="sk-test", LLM_PROVIDER="ollama")
    settings = Settings(_env_file=ROOT / ".env")
    if settings.llm_provider != "ollama":
        pytest.skip("LLM_PROVIDER is not ollama")
    if not _ollama_up(settings.ollama_base_url):
        pytest.skip(f"ollama not reachable at {settings.ollama_base_url}")
    if not _model_installed(settings.ollama_base_url, settings.ollama_model):
        pytest.skip(f"model {settings.ollama_model} not pulled")
    return settings


async def test_live_chat_returns_text(live_settings):
    provider = OllamaProvider(live_settings)
    try:
        result = await provider.chat(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            system="Answer with one word.",
        )
    finally:
        await provider.aclose()

    assert result.text.strip(), "answer text was empty — check OLLAMA_NUM_PREDICT"
    assert result.output_tokens and result.output_tokens > 0


async def test_live_stream_yields_text(live_settings):
    provider = OllamaProvider(live_settings)
    try:
        chunks = [c async for c in provider.stream([{"role": "user", "content": "Count: 1 2 3"}])]
    finally:
        await provider.aclose()

    assert "".join(chunks).strip()


async def test_live_embedding_matches_configured_dim(live_settings):
    """Guards the contract between the embedding model and vector(N) in Postgres."""
    provider = OllamaEmbeddingProvider(live_settings)
    try:
        vectors = await provider.embed(["hello world", "second chunk"])
    finally:
        await provider.aclose()

    assert len(vectors) == 2
    assert all(len(v) == live_settings.embedding_dim for v in vectors)
