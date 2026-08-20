"""Adapter the graph uses to call MCP tools.

Phase 9's action-taker proposes a tool call; Phase 10's gate approves or rejects
it; this is what finally dispatches it.

The tools run in-process against the same `MCPServer` object the standalone
process exposes, so there is one definition of a tool and one implementation —
no drift between "what Claude Desktop can call" and "what the graph can call".
A subprocess/stdio client would be the alternative, but it buys isolation we do
not need and costs a process boundary on every call.

The gate is enforced here rather than trusted to callers: `call()` refuses a
sensitive tool unless `approved=True` is passed explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.mcp_server.server import build_server
from app.mcp_server.tools import ToolSpec, is_sensitive


class ToolError(RuntimeError):
    """Tool call failed."""


class ApprovalRequired(ToolError):
    """A sensitive tool was dispatched without an approval."""


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    name: str
    content: Any
    is_error: bool = False


class MCPToolClient:
    """In-process MCP client over the shared server definition."""

    def __init__(self, settings: Settings | None = None, *, tenant_id: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._server = build_server(self._settings, tenant_id=tenant_id)

    async def list_tools(self) -> list[ToolSpec]:
        """Advertised tools, with sensitivity read from the server's own metadata."""
        specs: list[ToolSpec] = []
        for tool in await self._server.list_tools():
            meta = tool.meta or {}
            annotations = tool.annotations
            specs.append(
                ToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    sensitive=bool(meta.get("sensitive", is_sensitive(tool.name))),
                    read_only=bool(getattr(annotations, "read_only_hint", False)),
                )
            )
        return specs

    async def call(
        self, name: str, arguments: dict | None = None, *, approved: bool = False
    ) -> ToolCallResult:
        if is_sensitive(name) and not approved:
            raise ApprovalRequired(
                f"{name!r} is a sensitive tool and needs human approval before it runs"
            )

        try:
            result = await self._server.call_tool(name, arguments or {})
        except Exception as exc:
            raise ToolError(f"tool {name!r} failed: {exc}") from exc

        return ToolCallResult(
            name=name,
            content=_unwrap(result),
            is_error=bool(getattr(result, "is_error", False)),
        )

    async def aclose(self) -> None:
        toolbox = getattr(self._server, "toolbox", None)
        if toolbox is not None:
            await toolbox.aclose()


def _unwrap(result: Any) -> Any:
    """Normalise an MCP result back into plain Python.

    A tool returning a list arrives as one `TextContent` block *per element*, so
    the blocks are parsed individually — joining them first would produce
    `{...}\n{...}`, which is not valid JSON.
    """
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        # MCP wraps a bare list/scalar return under "result".
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured

    blocks = [b.text for b in getattr(result, "content", []) or [] if hasattr(b, "text")]
    if not blocks:
        return None

    parsed = []
    for block in blocks:
        try:
            parsed.append(json.loads(block))
        except (ValueError, TypeError):
            parsed.append(block)

    return parsed[0] if len(parsed) == 1 else parsed
