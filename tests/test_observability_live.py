"""Phase 12: tracing against the real self-hosted Langfuse.

The offline suite proves the wrappers record the right things. This proves the
spans actually reach the server: credentials work, the exporter flushes, and a
run through the real graph produces one trace with a real id.

Langfuse v4 self-hosted runs its read APIs in "events_only" mode, so a test
cannot fetch the finished trace back over HTTP. What is asserted here is
therefore the ingestion side; the trace *contents* were verified by reading
ClickHouse directly (see PLAN.md Phase 12).
"""

from pathlib import Path

import pytest

from app.core import observability as obs
from app.core.config import Settings

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def live_settings(env) -> Settings:
    settings = Settings(_env_file=ROOT / ".env")
    if not settings.langfuse_enabled:
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not configured")
    obs.reset_observability()
    obs.configure_observability(settings)
    if not obs.auth_check():
        obs.reset_observability()
        pytest.skip("langfuse is not reachable at " + settings.langfuse_host)
    yield settings
    obs.flush()
    obs.reset_observability()


def test_credentials_authenticate(live_settings):
    assert obs.auth_check() is True
    assert obs.callback_handler() is not None


def test_request_trace_produces_a_real_trace_id(live_settings):
    with obs.request_trace(
        name="test.trace",
        question="does tracing work?",
        user_id="test-user",
        session_id="test-session",
        tenant_id="default",
        tags=["test"],
    ) as trace:
        assert trace.trace_id and len(trace.trace_id) == 32
        with obs.observation("test.child", as_type="tool", input={"x": 1}) as span:
            obs.update_observation(span, output="done")

    obs.flush()  # raises nothing if the exporter is healthy


async def test_a_graph_run_is_traced_end_to_end(live_settings):
    """One question through the real graph: one trace, spans for every stage."""
    from app.graph.build import build_graph
    from app.graph.state import initial_state

    graph = build_graph(live_settings)
    with obs.request_trace(
        name="chat",
        question="How long do I have to get a refund?",
        user_id=live_settings.local_user_id,
        session_id="live-test-thread",
        tenant_id=live_settings.local_tenant_id,
    ) as trace:
        assert trace.trace_id
        state = await graph.ainvoke(
            initial_state(
                "How long do I have to get a refund?",
                tenant_id=live_settings.local_tenant_id,
                user_id=live_settings.local_user_id,
                thread_id="live-test-thread",
                trace_id=trace.trace_id,
            ),
            obs.graph_callbacks({"configurable": {"thread_id": "live-test-thread"}}),
        )

    assert state["answer"]
    obs.flush()
