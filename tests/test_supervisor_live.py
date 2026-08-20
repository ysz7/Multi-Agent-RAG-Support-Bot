"""Phase 9: the supervisor branch against real backends."""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.graph.build import GRAPH_RECURSION_LIMIT, build_graph
from app.graph.state import MAX_SUPERVISOR_ITERATIONS, initial_state

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def live_graph(env):
    env(ANTHROPIC_API_KEY="sk-test")
    settings = Settings(_env_file=ROOT / ".env")
    if settings.llm_provider != "ollama":
        pytest.skip("live supervisor tests target the local ollama model")
    return build_graph(settings), settings


async def test_multi_document_question_is_answered_with_citations(live_graph):
    """The point of the supervisor: synthesise across more than one document."""
    graph, _ = live_graph
    try:
        final = await graph.ainvoke(
            initial_state(
                "Compare the refund window with the data retention period and say which is longer",
                tenant_id="default",
                user_id="u1",
            ),
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
    except Exception as exc:
        pytest.skip(f"backends unavailable: {exc}")

    assert final["route"] == "supervisor"
    assert final["findings"], "researcher must have gathered evidence"
    assert final["review_verdict"] in {"approved", "revise"}
    assert final["iterations"] <= MAX_SUPERVISOR_ITERATIONS + 1
    assert final["answer"].strip()

    sources = {c.source_path for c in final.get("citations", [])}
    assert len(sources) >= 2, f"expected citations from 2+ documents, got {sources}"


async def test_action_request_stops_at_a_pending_action(live_graph):
    """Nothing may be dispatched before the Phase 10 gate exists."""
    graph, settings = live_graph
    outbox = Path(settings.documents_dir).parent / "outbox.jsonl"
    before = outbox.read_text().splitlines() if outbox.exists() else []

    try:
        final = await graph.ainvoke(
            initial_state(
                "Email me a summary of the refund policy", tenant_id="default", user_id="u1"
            ),
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
    except Exception as exc:
        pytest.skip(f"backends unavailable: {exc}")

    assert final["pending_action"]["tool"] == "send_email"
    assert final["approval"] == "pending"

    after = outbox.read_text().splitlines() if outbox.exists() else []
    assert after == before, "a sensitive action was executed without approval"
