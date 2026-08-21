"""Request and response models.

Note what `ChatRequest` does **not** have: a `tenant_id`. It is not "ignored if
supplied" by convention — there is no field for it, so a client cannot express
one, and `model_config` forbids extras so sending it is a 422 rather than a
silent no-op.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None, description="Resume or continue a conversation thread."
    )


class CitationOut(BaseModel):
    index: int
    citation: str
    source_path: str
    score: float


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    citations: list[CitationOut] = []
    route: str | None = None
    approval_required: bool = False
    pending_action: dict[str, Any] | None = None
    # Present only when Langfuse tracing is on; lets a caller correlate an
    # answer with its trace in the dashboard.
    trace_id: str | None = None


class ApprovalOut(BaseModel):
    thread_id: str
    question: str
    tool: str
    arguments: dict[str, Any]
    reason: str | None = None
    status: str
    created_at: datetime


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject", "edit"]
    arguments: dict[str, Any] | None = Field(
        default=None, description="Only for 'edit': replacement argument values."
    )
    note: str | None = Field(default=None, max_length=1000)


class ApprovalResult(BaseModel):
    thread_id: str
    status: str
    answer: str | None = None
    executed: list[dict[str, Any]] = []
    trace_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    auth_mode: str
    llm_provider: str
    vector_store: str
    database: str
    checks: dict[str, str] = {}
