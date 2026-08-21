"""Phase 12: Langfuse tracing.

Two properties matter here, and neither needs a Langfuse server:

1. **Tracing off changes nothing.** Every helper is a pass-through, and the
   wrappers return exactly what the wrapped object returned.
2. **Tracing on records the right thing, and never breaks the call.** A client
   that raises on every span must not turn a working chat into a failed one.
"""

import pytest

from app.core import observability as obs
from app.core.config import Settings
from app.core.llm_provider import ChatResult
from app.mcp_server.client import ApprovalRequired, ToolCallResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSpan:
    def __init__(self, name, as_type, kwargs):
        self.name = name
        self.as_type = as_type
        self.start_kwargs = kwargs
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    @property
    def last(self) -> dict:
        merged: dict = {}
        for update in self.updates:
            merged.update(update)
        return merged


class FakeClient:
    """Records observations the way the real client would create them."""

    def __init__(self, *, explode=False):
        self.spans: list[FakeSpan] = []
        self.explode = explode
        self.flushed = 0

    def start_as_current_observation(self, *, name, as_type="span", **kwargs):
        if self.explode:
            raise RuntimeError("langfuse is down")
        span = FakeSpan(name, as_type, kwargs)
        self.spans.append(span)

        class _CM:
            def __enter__(self_inner):
                return span

            def __exit__(self_inner, *exc):
                span.ended = True
                return False

        return _CM()

    def get_current_trace_id(self):
        return "trace-abc"

    def flush(self):
        self.flushed += 1

    def shutdown(self):
        self.flushed += 1

    def auth_check(self):
        return True

    def by_name(self, name) -> FakeSpan:
        return next(s for s in self.spans if s.name == name)


class FakeLLM:
    name = "ollama"
    model = "test-model"

    def __init__(self, *, error=None):
        self.error = error
        self.calls: list[dict] = []

    async def chat(self, messages, *, system=None, max_tokens=None):
        self.calls.append({"messages": messages, "system": system})
        if self.error:
            raise self.error
        return ChatResult(
            text="hello",
            model=self.model,
            provider=self.name,
            input_tokens=11,
            output_tokens=3,
            stop_reason="end_turn",
        )

    async def stream(self, messages, *, system=None, max_tokens=None):
        self.calls.append({"messages": messages, "system": system})
        if self.error:
            raise self.error
        for piece in ("he", "llo"):
            yield piece

    async def aclose(self):
        self.closed = True


class FakeTools:
    def __init__(self, *, error=None):
        self.error = error
        self.calls: list[tuple] = []

    async def call(self, name, arguments=None, *, approved=False):
        self.calls.append((name, arguments, approved))
        if self.error:
            raise self.error
        return ToolCallResult(name=name, content="result text")

    async def list_tools(self):
        return []


@pytest.fixture
def traced(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(obs, "_client", client)
    return client


@pytest.fixture(autouse=True)
def _reset():
    obs.reset_observability()
    yield
    obs.reset_observability()


def settings(**kwargs) -> Settings:
    return Settings(llm_provider="ollama", **kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_disabled_without_keys():
    assert obs.configure_observability(settings()) is None
    assert obs.get_observability() is None
    assert obs.callback_handler() is None
    assert obs.auth_check() is False


def test_enabled_with_keys():
    client = obs.configure_observability(
        settings(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            langfuse_host="http://localhost:3000",
        )
    )
    assert client is not None
    assert obs.callback_handler() is not None


def test_configuration_failure_is_not_fatal(monkeypatch):
    """A broken Langfuse must degrade to "no tracing", never to a failed boot."""
    import langfuse

    def boom(**kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(langfuse, "Langfuse", boom)
    assert (
        obs.configure_observability(
            settings(langfuse_public_key="pk-lf-x", langfuse_secret_key="sk-lf-x")
        )
        is None
    )


def test_flush_and_shutdown_are_safe_when_disabled():
    obs.flush()
    obs.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# Pass-through when tracing is off
# ---------------------------------------------------------------------------


async def test_helpers_are_noops_when_disabled():
    assert obs.graph_callbacks({"configurable": {}}) == {"configurable": {}}

    with obs.request_trace(name="chat", question="q") as trace:
        assert trace.trace_id is None

    with obs.observation("x") as span:
        assert span is None
    obs.update_observation(None, output="ignored")


async def test_wrapped_llm_returns_the_same_result_when_disabled():
    provider = obs.instrument_llm(FakeLLM(), settings())
    result = await provider.chat([{"role": "user", "content": "hi"}])
    assert result.text == "hello"
    assert [piece async for piece in provider.stream([])] == ["he", "llo"]


async def test_wrapped_tools_pass_through_when_disabled():
    tools = obs.instrument_tools(FakeTools())
    result = await tools.call("search_documents", {"query": "refund"})
    assert result.content == "result text"


def test_instrumentation_is_idempotent():
    provider = obs.instrument_llm(FakeLLM(), settings())
    assert obs.instrument_llm(provider, settings()) is provider
    tools = obs.instrument_tools(FakeTools())
    assert obs.instrument_tools(tools) is tools


# ---------------------------------------------------------------------------
# Recording when tracing is on
# ---------------------------------------------------------------------------


def test_graph_callbacks_appends_the_handler(monkeypatch, traced):
    monkeypatch.setattr(obs, "callback_handler", lambda: "handler")
    config = obs.graph_callbacks({"configurable": {"thread_id": "t"}})
    assert config["callbacks"] == ["handler"]
    assert config["configurable"] == {"thread_id": "t"}


def test_request_trace_exposes_the_trace_id(traced):
    with obs.request_trace(
        name="chat", question="q", user_id="u", session_id="t", tenant_id="acme"
    ) as trace:
        assert trace.trace_id == "trace-abc"
    assert traced.by_name("chat").ended


def test_request_trace_records_and_reraises_errors(traced):
    with pytest.raises(ValueError):  # noqa: SIM117 - the nesting is the point
        with obs.request_trace(name="chat", question="q"):
            raise ValueError("boom")
    span = traced.by_name("chat")
    assert span.last["level"] == "ERROR"
    assert "boom" in span.last["status_message"]


async def test_chat_records_a_generation(traced):
    provider = obs.instrument_llm(FakeLLM(), settings())
    result = await provider.chat([{"role": "user", "content": "hi"}], system="sys")
    assert result.text == "hello"

    span = traced.by_name("ollama.chat")
    assert span.as_type == "generation"
    assert span.start_kwargs["model"] == "test-model"
    assert span.start_kwargs["input"]["system"] == "sys"
    assert span.last["output"] == "hello"
    assert span.last["usage_details"] == {"input": 11, "output": 3}
    assert span.last["metadata"]["stop_reason"] == "end_turn"


async def test_stream_records_output_and_first_token_time(traced):
    provider = obs.instrument_llm(FakeLLM(), settings())
    assert [p async for p in provider.stream([{"role": "user", "content": "hi"}])] == ["he", "llo"]

    span = traced.by_name("ollama.stream")
    assert span.last["output"] == "hello"
    assert span.last["completion_start_time"] is not None


async def test_generation_errors_are_recorded_and_reraised(traced):
    from app.core.llm_provider import LLMError

    provider = obs.instrument_llm(FakeLLM(error=LLMError("upstream 500")), settings())
    with pytest.raises(LLMError):
        await provider.chat([])
    assert traced.by_name("ollama.chat").last["level"] == "ERROR"


async def test_tool_call_records_its_approval_state(traced):
    tools = obs.instrument_tools(FakeTools())
    await tools.call("send_email", {"to": "a@example.com"}, approved=True)

    span = traced.by_name("tool.send_email")
    assert span.as_type == "tool"
    assert span.start_kwargs["input"] == {"to": "a@example.com"}
    assert span.start_kwargs["metadata"] == {"approved": True}
    assert span.last["output"] == "result text"


async def test_refused_tool_call_is_traced_as_an_error(traced):
    """A gate refusal is exactly the kind of thing an audit trail should show."""
    tools = obs.instrument_tools(FakeTools(error=ApprovalRequired("needs approval")))
    with pytest.raises(ApprovalRequired):
        await tools.call("send_email", {"to": "a@example.com"})

    span = traced.by_name("tool.send_email")
    assert span.start_kwargs["metadata"] == {"approved": False}
    assert span.last["level"] == "ERROR"


async def test_long_values_are_truncated(traced):
    class Chatty(FakeLLM):
        async def chat(self, messages, *, system=None, max_tokens=None):
            return ChatResult(text="x" * 9000, model=self.model, provider=self.name)

    provider = obs.instrument_llm(Chatty(), settings())
    await provider.chat([])
    output = traced.by_name("ollama.chat").last["output"]
    assert len(output) < 9000
    assert "more chars" in output


# ---------------------------------------------------------------------------
# Degradation: a broken Langfuse must not break the pipeline
# ---------------------------------------------------------------------------


async def test_a_failing_langfuse_does_not_break_calls(monkeypatch):
    monkeypatch.setattr(obs, "_client", FakeClient(explode=True))

    provider = obs.instrument_llm(FakeLLM(), settings())
    assert (await provider.chat([])).text == "hello"
    assert [p async for p in provider.stream([])] == ["he", "llo"]

    tools = obs.instrument_tools(FakeTools())
    assert (await tools.call("search_documents", {})).content == "result text"

    with obs.request_trace(name="chat", question="q") as trace:
        assert trace.trace_id is None

    with obs.observation("x") as span:
        assert span is None
