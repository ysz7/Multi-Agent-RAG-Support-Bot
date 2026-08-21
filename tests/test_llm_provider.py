"""Phase 3: the provider abstraction.

Network-free. The Claude path is exercised against a fake SDK client, so these
run identically whether or not ANTHROPIC_API_KEY is set.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.core.llm_provider import (
    ChatResult,
    ClaudeProvider,
    EmbeddingDimensionMismatch,
    EmbeddingProvider,
    LLMError,
    LLMProvider,
    LLMRefusal,
    OllamaEmbeddingProvider,
    OllamaProvider,
    get_embedding_provider,
    get_llm_provider,
)


def _settings(env, **overrides) -> Settings:
    env(ANTHROPIC_API_KEY="sk-test", **overrides)
    return Settings()


def _mock_ollama(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")


# --- factory ---------------------------------------------------------------


def test_factory_selects_ollama(env):
    provider = get_llm_provider(_settings(env, LLM_PROVIDER="ollama"))
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama"


def test_factory_selects_claude(env):
    provider = get_llm_provider(_settings(env, LLM_PROVIDER="claude"))
    assert isinstance(provider, ClaudeProvider)
    assert provider.model == "claude-opus-5"


def test_both_providers_satisfy_the_protocol(env):
    for name in ("claude", "ollama"):
        assert isinstance(get_llm_provider(_settings(env, LLM_PROVIDER=name)), LLMProvider)


def test_embedding_provider_is_ollama_even_under_claude(env):
    """Anthropic has no embeddings endpoint, so this must not follow LLM_PROVIDER."""
    provider = get_embedding_provider(_settings(env, LLM_PROVIDER="claude"))
    assert isinstance(provider, OllamaEmbeddingProvider)
    assert isinstance(provider, EmbeddingProvider)


# --- ollama chat -----------------------------------------------------------


async def test_ollama_chat_reads_content_not_thinking(env):
    """Reasoning models put the answer in `content` and reasoning in `thinking`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "lfm2.5:8b",
                "message": {"role": "assistant", "content": "OK", "thinking": "hmm..."},
                "prompt_eval_count": 11,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    provider = OllamaProvider(_settings(env), client=_mock_ollama(handler))
    result = await provider.chat([{"role": "user", "content": "hi"}])

    assert isinstance(result, ChatResult)
    assert result.text == "OK"
    assert result.thinking == "hmm..."
    assert (result.input_tokens, result.output_tokens) == (11, 5)


async def test_ollama_chat_strips_leaked_think_tags(env):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "<think>reasoning here</think>\nThe answer."}},
        )

    provider = OllamaProvider(_settings(env), client=_mock_ollama(handler))
    result = await provider.chat([{"role": "user", "content": "hi"}])
    assert result.text == "The answer."


async def test_ollama_system_prompt_is_prepended(env):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = OllamaProvider(_settings(env), client=_mock_ollama(handler))
    await provider.chat([{"role": "user", "content": "hi"}], system="be terse")

    assert seen["messages"][0] == {"role": "system", "content": "be terse"}
    assert seen["stream"] is False


async def test_ollama_think_is_only_sent_when_disabled(env):
    """A model with no thinking mode rejects the key, so it is omitted by default."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = OllamaProvider(_settings(env), client=_mock_ollama(handler))
    await provider.chat([{"role": "user", "content": "hi"}])
    assert "think" not in seen[-1]

    provider = OllamaProvider(_settings(env, OLLAMA_THINK="false"), client=_mock_ollama(handler))
    await provider.chat([{"role": "user", "content": "hi"}])
    assert seen[-1]["think"] is False


async def test_ollama_max_tokens_overrides_the_configured_budget(env):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = OllamaProvider(_settings(env), client=_mock_ollama(handler))
    await provider.chat([{"role": "user", "content": "hi"}], max_tokens=8192)
    assert seen["options"]["num_predict"] == 8192


async def test_ollama_stream_yields_content_deltas_only(env):
    lines = [
        {"message": {"content": "", "thinking": "The"}, "done": False},
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = "\n".join(json.dumps(line) for line in lines)
        return httpx.Response(200, text=body)

    provider = OllamaProvider(_settings(env), client=_mock_ollama(handler))
    chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]
    assert "".join(chunks) == "Hello"


async def test_ollama_http_error_becomes_llm_error(env):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    provider = OllamaProvider(_settings(env), client=_mock_ollama(handler))
    with pytest.raises(LLMError, match="ollama chat failed"):
        await provider.chat([{"role": "user", "content": "hi"}])


# --- embeddings ------------------------------------------------------------


async def test_embed_returns_vectors(env):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1] * 768 for _ in payload["input"]]})

    provider = OllamaEmbeddingProvider(_settings(env), client=_mock_ollama(handler))
    vectors = await provider.embed(["a", "b"])
    assert len(vectors) == 2 and len(vectors[0]) == 768


async def test_embed_rejects_wrong_dimensions(env):
    """A mismatch must fail loudly here, not as an opaque INSERT error later."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * 384]})

    provider = OllamaEmbeddingProvider(_settings(env), client=_mock_ollama(handler))
    with pytest.raises(EmbeddingDimensionMismatch, match="384 dims"):
        await provider.embed(["a"])


async def test_embed_empty_input_short_circuits(env):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not call the API for an empty batch")

    provider = OllamaEmbeddingProvider(_settings(env), client=_mock_ollama(handler))
    assert await provider.embed([]) == []


# --- claude ----------------------------------------------------------------


class _FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._message


def _claude_with(monkeypatch, env, message, capture: dict | None = None):
    provider = ClaudeProvider(_settings(env, LLM_PROVIDER="claude"))

    def fake_stream(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return _FakeStream(message)

    monkeypatch.setattr(provider._client.beta.messages, "stream", fake_stream)
    return provider


async def test_claude_chat_extracts_text_and_usage(monkeypatch, env):
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="reasoning"),
            SimpleNamespace(type="text", text="Paris."),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        stop_reason="end_turn",
    )
    provider = _claude_with(monkeypatch, env, message)
    result = await provider.chat([{"role": "user", "content": "capital of France?"}])

    assert result.text == "Paris."
    assert result.thinking == "reasoning"
    assert result.provider == "claude"
    assert (result.input_tokens, result.output_tokens) == (12, 3)


async def test_claude_request_uses_adaptive_thinking_and_no_budget_tokens(monkeypatch, env):
    message = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    captured: dict = {}
    provider = _claude_with(monkeypatch, env, message, captured)
    await provider.chat([{"role": "user", "content": "hi"}], system="be terse")

    assert captured["model"] == "claude-opus-5"
    assert captured["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in json.dumps(captured)
    assert captured["output_config"] == {"effort": "medium"}
    assert captured["system"] == "be terse"
    assert captured["fallbacks"] == "default"


async def test_claude_refusal_raises(monkeypatch, env):
    message = SimpleNamespace(
        content=[],
        usage=None,
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
    )
    provider = _claude_with(monkeypatch, env, message)
    with pytest.raises(LLMRefusal, match="cyber"):
        await provider.chat([{"role": "user", "content": "hi"}])
