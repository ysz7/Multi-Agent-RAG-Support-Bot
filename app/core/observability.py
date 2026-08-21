"""Langfuse tracing.

One HTTP request becomes one trace. Inside it:

    chat.request                     (root span, carries user/session/tenant)
    └─ LangGraph run                 (from the LangChain callback handler)
       ├─ router / simple_rag / supervisor / ... one span per node
       │  └─ llm.chat | llm.stream   (generation: model, prompt, usage)
       ├─ retrieve                   (LCEL step inside the RAG chain)
       └─ tool.search_documents      (MCP call, with its approval state)

Two mechanisms, deliberately:

* **The LangChain/LangGraph callback handler** covers everything composed as a
  Runnable — the graph, its nodes, the LCEL chain's steps.
* **Explicit wrappers** (`instrument_llm`, `instrument_tools`) cover what is
  *not* a Runnable. Model calls go through `LLMProvider` and MCP calls through
  `MCPToolClient`, neither of which LangChain can see, so without these the
  trace would show a node span with nothing inside it.

Tracing is best-effort by construction: if Langfuse is unconfigured or
unreachable, every helper here degrades to a pass-through. `_client` is `None`
when disabled, and each wrapper checks it before doing anything. Export happens
on a background thread with its own retry/backoff, so a dead Langfuse costs a
log line, never a failed chat.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger("observability")

# Set by `configure_observability()`; `None` means tracing is off.
_client: Any | None = None
_configured = False

# Long values (retrieved documents, drafts) bloat a trace without adding much.
_MAX_VALUE_CHARS = 4000


def configure_observability(settings: Settings | None = None) -> Any | None:
    """Initialise the Langfuse client once. Returns `None` when tracing is off.

    Never raises: a misconfigured or missing Langfuse must not stop the app
    from starting.
    """
    global _client, _configured
    settings = settings or get_settings()

    if _configured:
        return _client
    _configured = True

    if not settings.langfuse_enabled:
        logger.info("langfuse: disabled (no keys configured)")
        _client = None
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=settings.app_env,
            tracing_enabled=True,
        )
        logger.info("langfuse: tracing to %s (env=%s)", settings.langfuse_host, settings.app_env)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("langfuse: disabled, client init failed: %s", exc)
        _client = None
    return _client


def get_observability() -> Any | None:
    """The configured client, or `None`. Does not configure implicitly."""
    return _client


def reset_observability() -> None:
    """Drop the client so the next `configure_observability()` runs again (tests)."""
    global _client, _configured
    _client = None
    _configured = False


def flush() -> None:
    """Force-export buffered spans. Best-effort."""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("langfuse: flush failed: %s", exc)


def shutdown() -> None:
    """Flush and stop the exporter thread. Best-effort."""
    if _client is None:
        return
    try:
        _client.shutdown()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("langfuse: shutdown failed: %s", exc)


def auth_check() -> bool:
    """Whether the configured credentials reach the Langfuse server. Blocking."""
    if _client is None:
        return False
    try:
        return bool(_client.auth_check())
    except Exception as exc:
        logger.debug("langfuse: auth check failed: %s", exc)
        return False


def callback_handler() -> Any | None:
    """LangChain callback handler for graph and chain runs, or `None`."""
    if _client is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("langfuse: callback handler unavailable: %s", exc)
        return None


def graph_callbacks(config: dict | None = None) -> dict:
    """Add the Langfuse handler to a LangGraph/LCEL config, if tracing is on."""
    config = dict(config or {})
    handler = callback_handler()
    if handler is not None:
        config["callbacks"] = [*config.get("callbacks", []), handler]
    return config


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
        return value[:_MAX_VALUE_CHARS] + f"… [{len(value) - _MAX_VALUE_CHARS} more chars]"
    return value


class _TraceHandle:
    """What `request_trace()` yields: a trace id, or nothing when tracing is off."""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id


@contextlib.contextmanager
def request_trace(
    *,
    name: str,
    question: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tenant_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[_TraceHandle]:
    """Open the root span of a request and attach identity to the whole trace.

    `session_id` is the thread id, so a conversation — including the second leg
    of an approval, which arrives as a *separate* HTTP request — groups into one
    session in the dashboard.

    Failures *setting up* tracing are swallowed; failures inside the body are
    recorded on the span and re-raised. Those are different things, and
    conflating them would turn a broken Langfuse into a broken request.
    """
    handle = _TraceHandle()
    if _client is None:
        yield handle
        return

    attributes: dict[str, Any] = {
        "user_id": user_id,
        "session_id": session_id,
        "metadata": {"tenant_id": tenant_id, **(metadata or {})},
        "trace_name": name,
    }
    if tags:
        attributes["tags"] = tags

    try:
        from langfuse import propagate_attributes

        stack = contextlib.ExitStack()
        stack.enter_context(propagate_attributes(**attributes))
        span = stack.enter_context(
            _client.start_as_current_observation(
                name=name, as_type="span", input=_truncate(question)
            )
        )
        handle.trace_id = _client.get_current_trace_id()
    except Exception as exc:  # tracing must never break the request
        logger.debug("langfuse: request trace unavailable: %s", exc)
        yield handle
        return

    with stack:
        try:
            yield handle
        except Exception as exc:
            update_observation(span, level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise


@contextlib.contextmanager
def observation(name: str, *, as_type: str = "span", **kwargs: Any) -> Iterator[Any | None]:
    """A nested observation, or a no-op when tracing is off or unavailable."""
    if _client is None:
        yield None
        return
    try:
        manager = _client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
        span = manager.__enter__()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("langfuse: observation %r unavailable: %s", name, exc)
        yield None
        return

    try:
        yield span
    except BaseException as exc:
        with contextlib.suppress(Exception):
            manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        with contextlib.suppress(Exception):
            manager.__exit__(None, None, None)


def update_observation(span: Any | None, **kwargs: Any) -> None:
    """Update a span if there is one. Safe to call with `None`."""
    if span is None:
        return
    with contextlib.suppress(Exception):
        span.update(**kwargs)


# ---------------------------------------------------------------------------
# Wrappers for the non-Runnable parts of the pipeline
# ---------------------------------------------------------------------------


class TracedLLMProvider:
    """`LLMProvider` decorator that records each call as a generation.

    Transparent: same protocol, same exceptions. It exists because provider
    calls bypass LangChain entirely, so the callback handler cannot see them.
    """

    def __init__(self, inner: Any, settings: Settings | None = None) -> None:
        self._inner = inner
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "")

    def _model_parameters(self) -> dict[str, Any]:
        if self.name == "claude":
            return {
                "max_tokens": self._settings.claude_max_tokens,
                "effort": self._settings.claude_effort,
            }
        return {"num_predict": self._settings.ollama_num_predict}

    @staticmethod
    def _input(messages: Any, system: str | None) -> Any:
        payload: dict[str, Any] = {"messages": messages}
        if system:
            payload["system"] = _truncate(system)
        return payload

    async def chat(self, messages, *, system=None, max_tokens=None):
        if _client is None:
            return await self._inner.chat(messages, system=system, max_tokens=max_tokens)

        with observation(
            f"{self.name}.chat",
            as_type="generation",
            input=self._input(messages, system),
            model=self.model,
            model_parameters=self._model_parameters(),
        ) as span:
            try:
                result = await self._inner.chat(messages, system=system, max_tokens=max_tokens)
            except Exception as exc:
                update_observation(
                    span, level="ERROR", status_message=f"{type(exc).__name__}: {exc}"
                )
                raise
            usage = {
                k: v
                for k, v in {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                }.items()
                if v is not None
            }
            update_observation(
                span,
                output=_truncate(result.text),
                model=result.model or self.model,
                usage_details=usage or None,
                metadata={"stop_reason": result.stop_reason, "provider": result.provider},
            )
            return result

    async def stream(self, messages, *, system=None, max_tokens=None) -> AsyncIterator[str]:
        if _client is None:
            async for piece in self._inner.stream(messages, system=system, max_tokens=max_tokens):
                yield piece
            return

        with observation(
            f"{self.name}.stream",
            as_type="generation",
            input=self._input(messages, system),
            model=self.model,
            model_parameters=self._model_parameters(),
        ) as span:
            parts: list[str] = []
            first_token_at: datetime | None = None
            try:
                async for piece in self._inner.stream(
                    messages, system=system, max_tokens=max_tokens
                ):
                    if first_token_at is None:
                        first_token_at = datetime.now(UTC)
                    parts.append(piece)
                    yield piece
            except Exception as exc:
                update_observation(
                    span, level="ERROR", status_message=f"{type(exc).__name__}: {exc}"
                )
                raise
            # Time to first token is the number that matters for a streamed UI.
            update_observation(
                span,
                output=_truncate("".join(parts)),
                completion_start_time=first_token_at,
            )

    async def aclose(self) -> None:
        await self._inner.aclose()


class TracedToolClient:
    """`MCPToolClient` decorator recording each tool call as a tool observation.

    The approval state is part of the span metadata: a trace shows not just
    that a tool ran, but that it ran *with* a human decision behind it.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def list_tools(self):
        return await self._inner.list_tools()

    async def call(self, name: str, arguments: dict | None = None, *, approved: bool = False):
        if _client is None:
            return await self._inner.call(name, arguments, approved=approved)

        with observation(
            f"tool.{name}",
            as_type="tool",
            input=arguments or {},
            metadata={"approved": approved},
        ) as span:
            try:
                result = await self._inner.call(name, arguments, approved=approved)
            except Exception as exc:
                update_observation(
                    span, level="ERROR", status_message=f"{type(exc).__name__}: {exc}"
                )
                raise
            update_observation(
                span,
                output=_truncate(result.content),
                level="ERROR" if result.is_error else None,
            )
            return result


def instrument_llm(provider: Any, settings: Settings | None = None) -> Any:
    """Wrap a provider for tracing. Idempotent."""
    if isinstance(provider, TracedLLMProvider):
        return provider
    return TracedLLMProvider(provider, settings)


def instrument_tools(client: Any) -> Any:
    """Wrap an MCP tool client for tracing. Idempotent."""
    if isinstance(client, TracedToolClient):
        return client
    return TracedToolClient(client)
