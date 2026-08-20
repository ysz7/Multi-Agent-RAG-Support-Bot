"""Phase 9: supervisor coordination, specialists, and termination guarantees."""

from app.core.config import Settings
from app.core.llm_provider import ChatResult, LLMError
from app.graph.build import GRAPH_RECURSION_LIMIT, build_graph
from app.graph.state import (
    MAX_RESEARCH_QUERIES,
    MAX_REVISIONS,
    MAX_SUPERVISOR_ITERATIONS,
    initial_state,
)
from app.graph.supervisor import (
    ActionTakerNode,
    ResearcherNode,
    ReviewerNode,
    SupervisorNode,
    citations_from_findings,
    supervisor_branch,
)
from tests.supervisor_fakes import FakeTools


class _ScriptedLLM:
    """Returns queued replies in order, repeating the last one."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *replies, error: Exception | None = None):
        self.replies = list(replies) or [""]
        self.error = error
        self.systems: list[str] = []
        self.calls = 0

    async def chat(self, messages, *, system=None, max_tokens=None):
        self.calls += 1
        self.systems.append(system or "")
        if self.error:
            raise self.error
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return ChatResult(text=reply, model=self.model, provider=self.name)

    async def stream(self, messages, *, system=None, max_tokens=None):
        yield self.replies[0]

    async def aclose(self):
        return None


def _settings(env) -> Settings:
    env(ANTHROPIC_API_KEY="sk-test")
    return Settings()


def _finding(citation="handbook.md (Refunds)", content="Refunds within 30 days."):
    return {
        "query": "refund",
        "citation": citation,
        "content": content,
        "score": 0.9,
        "source_path": "/corpus/handbook.md",
        "document_id": "doc-1",
        "chunk_index": 0,
    }


# --- supervisor decisions --------------------------------------------------


async def test_first_pass_goes_to_research(env):
    node = SupervisorNode(_ScriptedLLM(), _settings(env))
    result = await node({"question": "What is the refund window?"})
    assert result["next_step"] == "research"
    assert result["iterations"] == 1


async def test_action_question_proposes_before_drafting(env):
    node = SupervisorNode(_ScriptedLLM(), _settings(env))
    result = await node({"question": "Email me the refund policy", "findings": [_finding()]})
    assert result["next_step"] == "act"


async def test_draft_is_written_then_reviewed(env):
    llm = _ScriptedLLM("Refunds take 30 days [1].")
    node = SupervisorNode(llm, _settings(env))
    result = await node(
        {"question": "refund window?", "findings": [_finding()], "needs_action": False}
    )
    assert result["next_step"] == "review"
    assert result["draft"] == "Refunds take 30 days [1]."


async def test_approved_draft_finalizes_with_citations(env):
    node = SupervisorNode(_ScriptedLLM(), _settings(env))
    result = await node(
        {
            "question": "refund window?",
            "findings": [_finding()],
            "needs_action": False,
            "draft": "Refunds take 30 days [1].",
            "review_verdict": "approved",
        }
    )
    assert result["next_step"] == "finalize"
    assert result["answer"] == "Refunds take 30 days [1]."
    assert result["citations"][0].citation == "handbook.md (Refunds)"


async def test_revision_request_loops_back_to_research(env):
    node = SupervisorNode(_ScriptedLLM(), _settings(env))
    result = await node(
        {
            "question": "q?",
            "findings": [_finding()],
            "needs_action": False,
            "draft": "unsupported claim",
            "review_verdict": "revise",
            "revisions": 0,
        }
    )
    assert result["next_step"] == "research"
    assert result["revisions"] == 1
    assert result["draft"] == "", "the rejected draft must be cleared"


async def test_revision_budget_is_capped(env):
    """A second rejection must finalise, not loop again."""
    node = SupervisorNode(_ScriptedLLM(), _settings(env))
    result = await node(
        {
            "question": "q?",
            "findings": [_finding()],
            "needs_action": False,
            "draft": "still unsupported",
            "review_verdict": "revise",
            "revisions": MAX_REVISIONS,
        }
    )
    assert result["next_step"] == "finalize"


async def test_iteration_cap_forces_termination(env):
    node = SupervisorNode(_ScriptedLLM(), _settings(env))
    result = await node({"question": "q?", "findings": [], "iterations": MAX_SUPERVISOR_ITERATIONS})
    assert result["next_step"] == "finalize"


async def test_draft_failure_does_not_crash_the_run(env):
    node = SupervisorNode(_ScriptedLLM(error=LLMError("down")), _settings(env))
    result = await node({"question": "q?", "findings": [_finding()], "needs_action": False})
    assert result["draft"] == ""
    assert result["next_step"] == "review"


# --- researcher ------------------------------------------------------------


async def test_researcher_uses_model_written_queries(env):
    tools = FakeTools()
    llm = _ScriptedLLM('["refund window", "partial refunds"]')
    result = await ResearcherNode(llm, tools, _settings(env))({"question": "refunds?"})

    queries = [args["query"] for name, args in tools.calls if name == "search_documents"]
    assert queries == ["refund window", "partial refunds"]
    assert result["findings"][0]["citation"] == "handbook.md (Refunds)"


async def test_researcher_caps_query_count(env):
    tools = FakeTools()
    llm = _ScriptedLLM('["a","b","c","d","e"]')
    await ResearcherNode(llm, tools, _settings(env))({"question": "q?"})
    assert len(tools.calls) == MAX_RESEARCH_QUERIES


async def test_researcher_falls_back_to_the_raw_question(env):
    tools = FakeTools()
    llm = _ScriptedLLM("not json at all")
    await ResearcherNode(llm, tools, _settings(env))({"question": "refund window?"})
    assert tools.calls[0][1]["query"] == "refund window?"


async def test_researcher_deduplicates_against_existing_findings(env):
    tools = FakeTools()
    llm = _ScriptedLLM('["refund"]')
    result = await ResearcherNode(llm, tools, _settings(env))(
        {"question": "q?", "findings": [_finding()]}
    )
    assert result["findings"] == [], "already-seen citation must not be re-added"


async def test_researcher_records_a_sentinel_when_nothing_is_found(env):
    """Otherwise the supervisor would loop back into research forever."""
    tools = FakeTools(hits=[])
    llm = _ScriptedLLM('["nothing"]')
    result = await ResearcherNode(llm, tools, _settings(env))({"question": "q?"})
    assert len(result["findings"]) == 1
    assert result["findings"][0]["citation"] == ""


# --- action taker ----------------------------------------------------------


async def test_action_taker_proposes_without_executing(env):
    tools = FakeTools()
    result = await ActionTakerNode(_ScriptedLLM(), tools, _settings(env))(
        {"question": "Email me the policy", "user_id": "u1", "findings": [_finding()]}
    )

    assert result["pending_action"]["tool"] == "send_email"
    assert result["approval"] == "pending"
    assert tools.calls == [], "no tool may be invoked while proposing"


async def test_proposed_action_is_inspectable(env):
    """A human has to be able to see exactly what would be sent."""
    result = await ActionTakerNode(_ScriptedLLM(), FakeTools(), _settings(env))(
        {"question": "Email me", "user_id": "u1", "draft": "Your refund is approved."}
    )
    arguments = result["pending_action"]["arguments"]
    assert set(arguments) == {"to", "subject", "body"}
    assert arguments["body"] == "Your refund is approved."


# --- reviewer --------------------------------------------------------------


async def test_reviewer_approves_a_supported_draft(env):
    llm = _ScriptedLLM('{"verdict": "approved"}')
    result = await ReviewerNode(llm, _settings(env))(
        {"draft": "Refunds take 30 days [1].", "findings": [_finding()]}
    )
    assert result["review_verdict"] == "approved"


async def test_reviewer_requests_revision_with_a_reason(env):
    llm = _ScriptedLLM('{"verdict": "revise", "note": "no source for the 90 day claim"}')
    result = await ReviewerNode(llm, _settings(env))(
        {"draft": "Refunds take 90 days.", "findings": [_finding()]}
    )
    assert result["review_verdict"] == "revise"
    assert "90 day" in result["review_note"]


async def test_reviewer_handles_fenced_json(env):
    llm = _ScriptedLLM('```json\n{"verdict": "approved"}\n```')
    result = await ReviewerNode(llm, _settings(env))({"draft": "d [1]", "findings": [_finding()]})
    assert result["review_verdict"] == "approved"


async def test_reviewer_failure_approves_rather_than_blocking(env):
    llm = _ScriptedLLM(error=LLMError("down"))
    result = await ReviewerNode(llm, _settings(env))({"draft": "d", "findings": [_finding()]})
    assert result["review_verdict"] == "approved"


async def test_reviewer_does_not_loop_when_nothing_was_retrieved(env):
    result = await ReviewerNode(_ScriptedLLM(), _settings(env))(
        {"draft": "d", "findings": [_finding(citation="", content="")]}
    )
    assert result["review_verdict"] == "approved"


async def test_missing_draft_is_sent_back(env):
    result = await ReviewerNode(_ScriptedLLM(), _settings(env))(
        {"draft": "", "findings": [_finding()]}
    )
    assert result["review_verdict"] == "revise"


# --- citations from findings -----------------------------------------------


def test_citations_map_to_findings_by_position():
    findings = [_finding(citation="a.md (X)"), _finding(citation="b.md (Y)")]
    citations = citations_from_findings("Both [1] and [2].", findings)
    assert [c.citation for c in citations] == ["a.md (X)", "b.md (Y)"]


def test_out_of_range_and_empty_citations_are_dropped():
    assert citations_from_findings("[9]", [_finding()]) == []
    assert citations_from_findings("[1]", [_finding(citation="")]) == []


# --- branch routing --------------------------------------------------------


def test_supervisor_branch_mapping():
    assert supervisor_branch({"next_step": "research"}) == "researcher"
    assert supervisor_branch({"next_step": "act"}) == "action_taker"
    assert supervisor_branch({"next_step": "review"}) == "reviewer"
    assert supervisor_branch({"next_step": "finalize"}) == "__end__"
    assert supervisor_branch({}) == "__end__"


# --- full branch through the compiled graph --------------------------------


async def test_complex_question_runs_research_then_review(env):
    llm = _ScriptedLLM(
        "COMPLEX",  # router
        '["refund window"]',  # researcher queries
        "Refunds take 30 days [1].",  # draft
        '{"verdict": "approved"}',  # review
    )
    graph = build_graph(_settings(env), llm=llm, tools=FakeTools(), chain=None)
    final = await graph.ainvoke(
        initial_state("Compare the refund and retention policies", tenant_id="t1", user_id="u1"),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )

    assert final["route"] == "supervisor"
    assert final["findings"], "researcher must have run"
    assert final["review_verdict"] == "approved", "reviewer must have run"
    assert final["answer"] == "Refunds take 30 days [1]."
    assert final["citations"][0].citation == "handbook.md (Refunds)"


async def test_graph_terminates_even_if_the_reviewer_never_approves(env):
    """The revision cap must break the research → draft → review cycle."""
    llm = _ScriptedLLM(
        "COMPLEX",
        '["q"]',
        "draft",
        '{"verdict": "revise", "note": "never happy"}',
    )
    graph = build_graph(_settings(env), llm=llm, tools=FakeTools(), chain=None)
    final = await graph.ainvoke(
        initial_state("something ambiguous", tenant_id="t1", user_id="u1"),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )

    assert final["revisions"] <= MAX_REVISIONS
    assert final["iterations"] <= MAX_SUPERVISOR_ITERATIONS + 1
    assert "answer" in final


async def test_action_branch_ends_with_a_pending_action(env):
    llm = _ScriptedLLM("COMPLEX", '["refund"]', "Draft body [1].", '{"verdict": "approved"}')
    tools = FakeTools()
    graph = build_graph(_settings(env), llm=llm, tools=tools, chain=None)
    final = await graph.ainvoke(
        initial_state("Email me the refund policy", tenant_id="t1", user_id="u1"),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )

    assert final["pending_action"]["tool"] == "send_email"
    assert final["approval"] == "pending"
    assert all(name != "send_email" for name, _ in tools.calls), "must not execute"
