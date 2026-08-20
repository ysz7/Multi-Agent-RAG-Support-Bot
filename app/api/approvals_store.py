"""Index of runs paused for approval.

LangGraph checkpoints keep the paused *state*, but they are keyed by thread id
and offer no "list every thread awaiting approval for tenant X" query. This
small table is that index: one row per pending action, written when a run
interrupts and closed when it is decided.

The table is created on startup rather than in the Compose init script, because
that script only runs on an empty volume — an existing database would never get
the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS pending_approvals (
    thread_id   text PRIMARY KEY,
    tenant_id   text        NOT NULL,
    user_id     text        NOT NULL,
    question    text        NOT NULL,
    tool        text        NOT NULL,
    arguments   jsonb       NOT NULL DEFAULT '{}'::jsonb,
    reason      text,
    status      text        NOT NULL DEFAULT 'pending',
    decided_by  text,
    note        text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    decided_at  timestamptz
);
CREATE INDEX IF NOT EXISTS pending_approvals_tenant_idx
    ON pending_approvals (tenant_id, status);
"""


@dataclass(frozen=True, slots=True)
class PendingApproval:
    thread_id: str
    tenant_id: str
    user_id: str
    question: str
    tool: str
    arguments: dict[str, Any]
    reason: str | None
    status: str
    created_at: datetime


class ApprovalStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    async def _connect(self) -> AsyncConnection:
        return await AsyncConnection.connect(self._database_url, row_factory=dict_row)

    async def setup(self) -> None:
        connection = await self._connect()
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(CREATE_TABLE)
            await connection.commit()
        finally:
            await connection.close()

    async def record(
        self,
        *,
        thread_id: str,
        tenant_id: str,
        user_id: str,
        question: str,
        tool: str,
        arguments: dict,
        reason: str | None,
    ) -> None:
        connection = await self._connect()
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO pending_approvals
                        (thread_id, tenant_id, user_id, question, tool, arguments, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE
                        SET arguments = EXCLUDED.arguments, status = 'pending'
                    """,
                    (thread_id, tenant_id, user_id, question, tool, Jsonb(arguments), reason),
                )
            await connection.commit()
        finally:
            await connection.close()

    async def list_pending(self, *, tenant_id: str) -> list[PendingApproval]:
        connection = await self._connect()
        try:
            async with connection.cursor() as cursor:
                # Tenant is always a bound parameter, never interpolated.
                await cursor.execute(
                    """
                    SELECT * FROM pending_approvals
                    WHERE tenant_id = %s AND status = 'pending'
                    ORDER BY created_at DESC
                    """,
                    (tenant_id,),
                )
                rows = await cursor.fetchall()
        finally:
            await connection.close()

        return [
            PendingApproval(
                thread_id=row["thread_id"],
                tenant_id=row["tenant_id"],
                user_id=row["user_id"],
                question=row["question"],
                tool=row["tool"],
                arguments=row["arguments"] or {},
                reason=row["reason"],
                status=row["status"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def get(self, *, thread_id: str, tenant_id: str) -> PendingApproval | None:
        connection = await self._connect()
        try:
            async with connection.cursor() as cursor:
                # tenant_id in the WHERE clause is the isolation boundary: a
                # caller cannot decide another tenant's approval by guessing an id.
                await cursor.execute(
                    "SELECT * FROM pending_approvals WHERE thread_id = %s AND tenant_id = %s",
                    (thread_id, tenant_id),
                )
                row = await cursor.fetchone()
        finally:
            await connection.close()

        if row is None:
            return None
        return PendingApproval(
            thread_id=row["thread_id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            question=row["question"],
            tool=row["tool"],
            arguments=row["arguments"] or {},
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
        )

    async def close(self, *, thread_id: str, status: str, decided_by: str, note: str | None):
        connection = await self._connect()
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE pending_approvals
                    SET status = %s, decided_by = %s, note = %s, decided_at = now()
                    WHERE thread_id = %s
                    """,
                    (status, decided_by, note, thread_id),
                )
            await connection.commit()
        finally:
            await connection.close()
