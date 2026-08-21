"""Provider abstraction between the graph and whatever is actually answering.

Nodes import `get_llm_provider()` / `get_embedding_provider()` and nothing else —
they never import `anthropic` or talk to Ollama directly. Swapping backends is a
change to `.env`, not to any calling code.

Two things worth knowing about the split:

* **Chat and embeddings are separate protocols.** Anthropic has no embeddings
  endpoint, so `ClaudeProvider` cannot implement `embed()`. Embeddings always go
  through Ollama, including when `LLM_PROVIDER=claude`.
* **Ollama reasoning models stream into a separate field.** Their answer text
  arrives in `message.content` while reasoning goes to `message.thinking`.
  Passing `think: false` does not help — the model then leaks raw `<think>` tags
  into `content` instead. We keep thinking structured and read `content`.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable

import httpx

from app.core.config import Settings, get_settings

Role = Literal["system", "user", "assistant"]

# Safety net for models that emit reasoning inline despite the structured field.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Provider-neutral result. `text` is always the answer only, never reasoning."""

    text: str
    model: str
    provider: str
    thinking: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class LLMError(RuntimeError):
    """Any provider-level failure, normalised across backends."""


class LLMRefusal(LLMError):
    """Claude declined the request (`stop_reason == "refusal"`)."""


class EmbeddingDimensionMismatch(LLMError):
    """Embedding width does not match the `vector(N)` column in Postgres."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).lstrip()


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudeProvider:
    """Anthropic Claude via the official SDK.

    Always streams: it is the SDK's own guidance for avoiding HTTP timeouts on
    large `max_tokens`, and `get_final_message()` still yields the whole message.
    """

    name = "claude"

    def __init__(self, settings: Settings | None = None) -> None:
        from anthropic import AsyncAnthropic

        self._settings = settings or get_settings()
        self.model = self._settings.claude_model
        self._client = AsyncAnthropic(api_key=self._settings.anthropic_api_key)

    def _request_kwargs(self, messages, system, max_tokens) -> dict:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self._settings.claude_max_tokens,
            "messages": messages,
            # Adaptive is the only supported on-mode for current models;
            # `budget_tokens` is rejected with a 400.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._settings.claude_effort},
        }
        if system:
            kwargs["system"] = system
        if self._settings.claude_refusal_fallback:
            # Server-side fallback: if a safety classifier declines, the request
            # is re-routed by refusal category instead of failing outright.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def chat(self, messages, *, system=None, max_tokens=None) -> ChatResult:
        kwargs = self._request_kwargs(messages, system, max_tokens)
        try:
            async with self._client.beta.messages.stream(**kwargs) as stream:
                message = await stream.get_final_message()
        except Exception as exc:  # normalise SDK errors for callers
            raise LLMError(f"claude chat failed: {exc}") from exc

        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMRefusal(f"claude declined the request (category={category})")

        text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        thinking = "".join(
            getattr(b, "thinking", "") or ""
            for b in message.content
            if getattr(b, "type", None) == "thinking"
        )
        usage = getattr(message, "usage", None)
        return ChatResult(
            text=text,
            model=self.model,
            provider=self.name,
            thinking=thinking or None,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            stop_reason=getattr(message, "stop_reason", None),
        )

    async def stream(self, messages, *, system=None, max_tokens=None) -> AsyncIterator[str]:
        kwargs = self._request_kwargs(messages, system, max_tokens)
        try:
            async with self._client.beta.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise LLMError(f"claude stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaProvider:
    """Local Ollama model over its HTTP API."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self.model = self._settings.ollama_model
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=httpx.Timeout(self._settings.ollama_timeout_s),
        )

    def _payload(self, messages, system, max_tokens, *, stream: bool) -> dict:
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": max_tokens or self._settings.ollama_num_predict},
        }
        # Only sent when disabling: a model with no thinking mode rejects the key.
        # With thinking on, a reasoning model can spend the whole budget on
        # `message.thinking` and return an empty answer — the tags it leaks into
        # `content` instead are stripped on the way out.
        if not self._settings.ollama_think:
            payload["think"] = False
        return payload

    async def chat(self, messages, *, system=None, max_tokens=None) -> ChatResult:
        payload = self._payload(messages, system, max_tokens, stream=False)
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise LLMError(f"ollama chat failed: {exc}") from exc

        message = data.get("message") or {}
        return ChatResult(
            text=_strip_think_tags(message.get("content") or ""),
            model=data.get("model", self.model),
            provider=self.name,
            thinking=message.get("thinking") or None,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            stop_reason=data.get("done_reason"),
        )

    async def stream(self, messages, *, system=None, max_tokens=None) -> AsyncIterator[str]:
        payload = self._payload(messages, system, max_tokens, stream=True)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if error := chunk.get("error"):
                        raise LLMError(f"ollama stream error: {error}")
                    # Reasoning arrives under "thinking"; only surface the answer.
                    if text := (chunk.get("message") or {}).get("content"):
                        yield text
                    if chunk.get("done"):
                        break
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"ollama stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaEmbeddingProvider:
    """Embeddings via Ollama — used regardless of which chat provider is active."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self.model = self._settings.embedding_model
        self.dim = self._settings.embedding_dim
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=httpx.Timeout(self._settings.ollama_timeout_s),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.post(
                "/api/embed", json={"model": self.model, "input": texts}
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings") or []
        except Exception as exc:
            raise LLMError(f"ollama embed failed: {exc}") from exc

        if len(vectors) != len(texts):
            raise LLMError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        # Caught here rather than as an opaque INSERT failure against vector(N).
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingDimensionMismatch(
                    f"{self.model} returned {len(vector)} dims, but EMBEDDING_DIM is "
                    f"{self.dim}. Update EMBEDDING_DIM and re-create the schema "
                    f"(`make clean && make up`), or use a matching model."
                )
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type] = {"claude": ClaudeProvider, "ollama": OllamaProvider}


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    try:
        provider_cls = _PROVIDERS[settings.llm_provider]
    except KeyError:  # pragma: no cover - Settings already constrains the value
        raise LLMError(f"unknown LLM_PROVIDER: {settings.llm_provider}") from None
    return provider_cls(settings)


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Always Ollama: Anthropic exposes no embeddings endpoint.

    This means `LLM_PROVIDER=claude` still needs a reachable Ollama for indexing
    and retrieval.
    """
    return OllamaEmbeddingProvider(settings or get_settings())
