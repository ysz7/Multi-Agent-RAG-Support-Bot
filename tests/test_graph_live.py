"""Phase 8: routing accuracy and durable checkpointing against real backends."""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.llm_provider import get_llm_provider
from app.graph.build import build_graph, compiled_graph
from app.graph.nodes import RouterNode
from app.graph.state import initial_state

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent

# Labelled set. `simple` = one factual lookup; `supervisor` = multi-step,
# ambiguous, or action-requesting.
LABELLED: list[tuple[str, str]] = [
    ("How long do I have to request a refund?", "simple"),
    ("What is the uptime commitment?", "simple"),
    ("When are invoices generated?", "simple"),
    ("How long do password reset links last?", "simple"),
    ("How long is customer data retained?", "simple"),
    ("Compare the refund policy with the retention policy and say which is longer", "supervisor"),
    ("Payment failed and I was suspended, but I have a pending refund - what now?", "supervisor"),
    ("Email me a summary of the refund policy", "supervisor"),
    ("Cancel my subscription and confirm the refund amount", "supervisor"),
    ("It's broken", "supervisor"),
]

# Below this the router is not fit for purpose; above it, occasional drift on
# genuinely borderline questions is tolerated.
MIN_ROUTING_ACCURACY = 0.8


@pytest.fixture
def live_settings(env):
    env(ANTHROPIC_API_KEY="sk-test")
    settings = Settings(_env_file=ROOT / ".env")
    if settings.llm_provider != "ollama":
        pytest.skip("live graph tests target the local ollama model")
    return settings


async def test_router_accuracy_on_labelled_questions(live_settings):
    llm = get_llm_provider(live_settings)
    router = RouterNode(llm, live_settings)
    misses: list[str] = []
    try:
        for question, expected in LABELLED:
            try:
                result = await router({"question": question})
            except Exception as exc:
                pytest.skip(f"model unavailable: {exc}")
            if result["route"] != expected:
                misses.append(f"{question!r}: expected {expected}, got {result['route']}")
    finally:
        await llm.aclose()

    accuracy = 1 - len(misses) / len(LABELLED)
    assert accuracy >= MIN_ROUTING_ACCURACY, (
        f"routing accuracy {accuracy:.0%} below {MIN_ROUTING_ACCURACY:.0%}: " + "; ".join(misses)
    )


async def test_action_requests_are_never_routed_simple(live_settings):
    """Deterministic pre-check: this must hold for every model."""
    llm = get_llm_provider(live_settings)
    router = RouterNode(llm, live_settings)
    try:
        for question, _expected in LABELLED:
            if "Email me" in question or "Cancel my" in question:
                result = await router({"question": question})
                assert result["route"] == "supervisor"
    finally:
        await llm.aclose()


async def test_state_survives_a_checkpoint_round_trip(live_settings):
    """Durability is what makes the Phase 10 approval gate resumable."""
    thread = "phase8-durability-test"
    async with compiled_graph(live_settings) as graph:
        config = {"configurable": {"thread_id": thread}}
        try:
            await graph.ainvoke(
                initial_state(
                    "How long is customer data retained?", tenant_id="default", user_id="u1"
                ),
                config=config,
            )
        except Exception as exc:
            pytest.skip(f"backends unavailable: {exc}")

        snapshot = await graph.aget_state(config)

    assert snapshot.values["tenant_id"] == "default"
    assert snapshot.values["route"] in {"simple", "supervisor"}

    # A *new* graph object reads the same persisted thread — the point of
    # Postgres checkpointing rather than an in-memory saver.
    async with compiled_graph(live_settings) as graph2:
        reloaded = await graph2.aget_state({"configurable": {"thread_id": thread}})

    assert reloaded.values["question"] == "How long is customer data retained?"
    chunks = reloaded.values.get("chunks") or []
    if chunks:
        assert type(chunks[0]).__name__ == "RetrievedChunk", "dataclass must round-trip typed"


async def test_graph_without_checkpointer_still_runs(live_settings):
    graph = build_graph(live_settings)
    try:
        final = await graph.ainvoke(
            initial_state("What is the uptime commitment?", tenant_id="default", user_id="u1")
        )
    except Exception as exc:
        pytest.skip(f"backends unavailable: {exc}")
    assert final.get("answer")
