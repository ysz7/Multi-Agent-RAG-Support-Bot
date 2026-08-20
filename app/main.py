"""FastAPI application.

    uvicorn app.main:app --reload

Startup wires the durable pieces once: the Postgres checkpointer (so approval
interrupts survive a restart) and the pending-approvals table.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from app.api import approvals as approvals_routes
from app.api import chat as chat_routes
from app.api import health as health_routes
from app.api.approvals_store import ApprovalStore
from app.core.config import get_settings
from app.graph.build import build_graph, postgres_checkpointer

logger = logging.getLogger("api")

DESCRIPTION = """Document-grounded support agent.

Simple questions get a single-shot RAG answer; complex or action-bearing ones go
through a supervisor with specialist sub-agents. Any sensitive action pauses the
run until a human approves it via `/approvals`.

Tenant identity always comes from the caller's credentials — never from a
request body.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(postgres_checkpointer(settings))

        approvals = ApprovalStore(settings.database_url)
        await approvals.setup()

        app.state.settings = settings
        app.state.graph = build_graph(settings, checkpointer=checkpointer)
        app.state.approvals = approvals

        logger.info(
            "ready: llm=%s vector_store=%s auth_mode=%s",
            settings.llm_provider,
            settings.vector_store,
            settings.auth_mode,
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Agent RAG Support Bot",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(approvals_routes.router)
    return app


app = create_app()
