"""Human-in-the-loop gate.

    action_taker → approval_gate ─┬→ dispatch_action → supervisor
                                  └→ supervisor            (rejected)

`approval_gate` calls LangGraph's `interrupt()`, which raises out of the node and
returns control to the caller with the proposal attached. Nothing downstream
runs. Because the graph is checkpointed in Postgres, the paused run is durable:
a human can approve it minutes later, from a different process, after a restart.

Three resume outcomes:

* **approve** — dispatch exactly as proposed.
* **edit** — dispatch with the human's corrected arguments (they may only replace
  argument values; the tool name is fixed by the proposal, so an edit cannot
  escalate to a different tool).
* **reject** — never dispatch; record the reason and report it in the answer.

Execute-exactly-once is enforced by clearing `pending_action` when the dispatch
succeeds and recording it in `executed_actions`. A resumed or replayed run finds
nothing pending and does nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from app.core.config import Settings, get_settings
from app.graph.state import GraphState
from app.mcp_server.client import MCPToolClient
from app.mcp_server.tools import is_sensitive

logger = logging.getLogger("graph.approval")

REJECTION_TEMPLATE = (
    "I prepared the action `{tool}` but it was not approved, so I have not run it.{reason}"
)


def _normalise(decision: Any, proposal: dict) -> tuple[str, dict, str | None]:
    """Interpret whatever the caller passed to `Command(resume=...)`.

    Accepts a bare bool, a bare string, or a dict with `decision` / `arguments`
    / `note`. Anything unrecognised is treated as a rejection — the gate fails
    closed, so a malformed resume can never dispatch.
    """
    arguments = dict(proposal.get("arguments") or {})
    note: str | None = None

    if decision is True:
        return "approved", arguments, None
    if decision is False or decision is None:
        return "rejected", arguments, None

    if isinstance(decision, str):
        verdict = decision.strip().lower()
        approved = verdict in {"approve", "approved", "yes"}
        return ("approved" if approved else "rejected"), arguments, None

    if isinstance(decision, dict):
        verdict = str(decision.get("decision", "")).strip().lower()
        note = decision.get("note")
        edited = decision.get("arguments")
        if edited and isinstance(edited, dict):
            # Only argument values may be edited; the tool name is fixed.
            arguments.update(edited)
        if verdict in {"approve", "approved", "yes", "edit", "edited"}:
            return "approved", arguments, note
        return "rejected", arguments, note

    return "rejected", arguments, None


class ApprovalGateNode:
    """Pauses the graph until a human decides on the pending action."""

    async def __call__(self, state: GraphState) -> GraphState:
        proposal = state.get("pending_action")
        if not proposal:
            return {"approval": None}

        tool = proposal.get("tool", "")
        if not is_sensitive(tool):
            # Read-only tools do not need a gate; nothing to ask about.
            return {"approval": "approved"}

        # Raises GraphInterrupt: nothing below this line runs until resumed.
        decision = interrupt(
            {
                "type": "approval_request",
                "tool": tool,
                "arguments": proposal.get("arguments", {}),
                "reason": proposal.get("reason", ""),
                "proposed_by": proposal.get("proposed_by", ""),
                "question": state.get("question", ""),
            }
        )

        verdict, arguments, note = _normalise(decision, proposal)
        logger.info("approval decision for %r: %s", tool, verdict)

        update: GraphState = {"approval": verdict, "approval_note": note}
        if arguments != (proposal.get("arguments") or {}):
            update["pending_action"] = {**proposal, "arguments": arguments}
        return update


class DispatchActionNode:
    """Executes the approved action, exactly once."""

    def __init__(self, tools: MCPToolClient, settings: Settings | None = None) -> None:
        self._tools = tools
        self._settings = settings or get_settings()

    async def __call__(self, state: GraphState) -> GraphState:
        proposal = state.get("pending_action")
        if not proposal:
            return {}
        if state.get("approval") != "approved":
            # Belt and braces: this node is only reachable when approved.
            logger.error("dispatch reached without approval; refusing")
            return {"error": "dispatch attempted without approval"}

        tool = proposal["tool"]
        try:
            result = await self._tools.call(tool, proposal.get("arguments") or {}, approved=True)
        except Exception as exc:
            logger.exception("approved action %r failed", tool)
            return {"error": f"action {tool} failed: {exc}", "pending_action": None}

        logger.info("dispatched approved action %r", tool)
        return {
            # Clearing this is what makes a replay a no-op.
            "pending_action": None,
            "executed_actions": [{"tool": tool, "result": result.content}],
        }


class RejectionNode:
    """Records a refused action so the final answer can mention it."""

    async def __call__(self, state: GraphState) -> GraphState:
        proposal = state.get("pending_action") or {}
        tool = proposal.get("tool", "the action")
        note = state.get("approval_note")
        reason = f" Reason given: {note}" if note else ""

        return {
            "pending_action": None,
            "rejected_actions": [{"tool": tool, "note": note}],
            "draft": REJECTION_TEMPLATE.format(tool=tool, reason=reason),
            # A rejected action is a final outcome, not something to re-review.
            "review_verdict": "approved",
        }


def approval_branch(state: GraphState) -> str:
    """Conditional edge out of the gate."""
    return "dispatch_action" if state.get("approval") == "approved" else "rejected_action"
