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
from app.core.observability import configure_observability
from app.core.observability import shutdown as shutdown_observability
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
    # Tracing is configured before anything is built, so the wrappers created
    # below see a live client. It never raises: no Langfuse just means no spans.
    configure_observability(settings)

    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(postgres_checkpointer(settings))

        approvals = ApprovalStore(settings.database_url)
        await approvals.setup()

        app.state.settings = settings
        app.state.graph = build_graph(settings, checkpointer=checkpointer)
        app.state.approvals = approvals

        logger.info(
            "ready: llm=%s vector_store=%s auth_mode=%s tracing=%s",
            settings.llm_provider,
            settings.vector_store,
            settings.auth_mode,
            "langfuse" if settings.langfuse_enabled else "off",
        )
        try:
            yield
        finally:
            # Flush buffered spans; a dying process would otherwise drop them.
            shutdown_observability()


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
