"""Phase 10: durable approval against real Postgres.

The point of these, over the in-memory tests: a run paused by one graph object
is resumed by a **different** one, reading the thread back out of Postgres. That
is what makes the gate survive a process restart.
"""

import json
import uuid
from pathlib import Path

import pytest
from langgraph.types import Command

from app.core.config import Settings
from app.graph.build import GRAPH_RECURSION_LIMIT, compiled_graph
from app.graph.state import initial_state

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent
ACTION_QUESTION = "Email me a summary of the refund policy"


@pytest.fixture
def live_settings(env):
    env(ANTHROPIC_API_KEY="sk-test")
    settings = Settings(_env_file=ROOT / ".env")
    if settings.llm_provider != "ollama":
        pytest.skip("live approval tests target the local ollama model")
    return settings


def _outbox(settings: Settings) -> Path:
    return Path(settings.documents_dir).parent / "outbox.jsonl"


def _outbox_lines(settings: Settings) -> list[str]:
    path = _outbox(settings)
    return path.read_text().splitlines() if path.exists() else []


def _config(thread: str) -> dict:
    return {
        "configurable": {"thread_id": thread},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }


async def test_approval_survives_a_new_graph_instance(live_settings):
    thread = f"approval-{uuid.uuid4().hex[:8]}"
    config = _config(thread)
    before = len(_outbox_lines(live_settings))

    # --- one process pauses ---
    async with compiled_graph(live_settings) as graph_a:
        try:
            result = await graph_a.ainvoke(
                initial_state(ACTION_QUESTION, tenant_id="default", user_id="tester"), config
            )
        except Exception as exc:
            pytest.skip(f"backends unavailable: {exc}")

        assert "__interrupt__" in result, "the run must pause for approval"
        assert result["__interrupt__"][0].value["tool"] == "send_email"

    assert len(_outbox_lines(live_settings)) == before, "nothing may be sent while paused"

    # --- a *different* graph object resumes the same thread ---
    async with compiled_graph(live_settings) as graph_b:
        snapshot = await graph_b.aget_state(config)
        assert snapshot.next, "the paused thread must be reloadable from Postgres"
        assert snapshot.values["pending_action"]["tool"] == "send_email"

        final = await graph_b.ainvoke(Command(resume={"decision": "approve"}), config)

    assert final["executed_actions"][0]["tool"] == "send_email"
    after = _outbox_lines(live_settings)
    assert len(after) == before + 1, "approval must send exactly one message"
    assert json.loads(after[-1])["to"]


async def test_rejection_sends_nothing_and_tells_the_user(live_settings):
    thread = f"reject-{uuid.uuid4().hex[:8]}"
    config = _config(thread)
    before = len(_outbox_lines(live_settings))

    async with compiled_graph(live_settings) as graph:
        try:
            result = await graph.ainvoke(
                initial_state(ACTION_QUESTION, tenant_id="default", user_id="tester"), config
            )
        except Exception as exc:
            pytest.skip(f"backends unavailable: {exc}")
        assert "__interrupt__" in result

        final = await graph.ainvoke(
            Command(resume={"decision": "reject", "note": "recipient not verified"}), config
        )

    assert len(_outbox_lines(live_settings)) == before, "a rejected action must not send"
    assert final["rejected_actions"][0]["tool"] == "send_email"
    assert "not approved" in final["answer"]
    assert "recipient not verified" in final["answer"]


async def test_abandoned_approval_stays_pending(live_settings):
    """A run nobody answers must remain paused, not time out into sending."""
    thread = f"abandoned-{uuid.uuid4().hex[:8]}"
    config = _config(thread)
    before = len(_outbox_lines(live_settings))

    async with compiled_graph(live_settings) as graph:
        try:
            await graph.ainvoke(
                initial_state(ACTION_QUESTION, tenant_id="default", user_id="tester"), config
            )
        except Exception as exc:
            pytest.skip(f"backends unavailable: {exc}")

    async with compiled_graph(live_settings) as graph2:
        snapshot = await graph2.aget_state(config)

    assert snapshot.next == ("approval_gate",)
    assert snapshot.values["approval"] == "pending"
    assert len(_outbox_lines(live_settings)) == before
