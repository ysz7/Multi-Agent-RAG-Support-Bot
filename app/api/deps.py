"""Shared API dependencies."""

from __future__ import annotations

from fastapi import Request

from app.api.approvals_store import ApprovalStore
from app.core.config import Settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_graph(request: Request):
    return request.app.state.graph


def get_approvals(request: Request) -> ApprovalStore:
    return request.app.state.approvals
