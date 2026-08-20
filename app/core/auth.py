"""Caller identity.

Everything downstream sees a `Principal` and nothing else. Two resolvers sit
behind it:

* `AUTH_MODE=local` (default) — a fixed principal from configuration. No tokens,
  no login endpoint, no user table. Right for a self-hosted single-user bot.
* `AUTH_MODE=jwt` — verifies a bearer token and maps its claims onto the same
  `Principal`. This app only ever *verifies* tokens; issuing them is someone
  else's job (an identity provider, or `python -m scripts.make_token` for dev).

The security property the README claims lives here: `tenant_id` comes from the
principal, never from a request body. A handler that wants a tenant has to take
it from `Depends(get_principal)`, and there is no code path that reads a tenant
from client input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

# auto_error=False so we can raise our own 401 with a useful message, and so
# local mode does not demand a header that will never exist.
_bearer = HTTPBearer(auto_error=False)

SCOPE_CHAT = "chat"
SCOPE_APPROVE = "approvals:write"


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is asking. The only source of tenant identity in the system."""

    user_id: str
    tenant_id: str
    scopes: frozenset[str] = field(default_factory=frozenset)

    def has(self, scope: str) -> bool:
        return scope in self.scopes


def _local_principal(settings: Settings) -> Principal:
    return Principal(
        user_id=settings.local_user_id,
        tenant_id=settings.local_tenant_id,
        # Local mode is a trusted single user: grant everything.
        scopes=frozenset({SCOPE_CHAT, SCOPE_APPROVE}),
    )


def _decode(token: str, settings: Settings) -> dict:
    from jose import JWTError, jwt

    options = {"verify_aud": settings.jwt_audience is not None}
    try:
        return jwt.decode(
            token,
            settings.jwt_secret or "",
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options=options,
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _jwt_principal(credentials: HTTPAuthorizationCredentials | None, settings: Settings):
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = _decode(credentials.credentials, settings)
    tenant_id = claims.get("tenant_id") or claims.get("tid")
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="token has no tenant_id claim"
        )

    raw_scopes = claims.get("scope") or claims.get("scopes") or ""
    scopes = set(raw_scopes.split()) if isinstance(raw_scopes, str) else set(raw_scopes)

    return Principal(
        user_id=str(claims.get("sub") or "unknown"),
        tenant_id=str(tenant_id),
        scopes=frozenset(scopes or {SCOPE_CHAT}),
    )


def get_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """FastAPI dependency resolving the caller's identity."""
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    if settings.auth_mode == "jwt":
        return _jwt_principal(credentials, settings)
    return _local_principal(settings)


def require_scope(scope: str):
    """Dependency factory gating an endpoint on a scope."""

    def _dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has(scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"missing scope: {scope}"
            )
        return principal

    return _dependency


# Module-level dependency singletons: built once, not per request.
RequireChat = Depends(require_scope(SCOPE_CHAT))
RequireApprove = Depends(require_scope(SCOPE_APPROVE))
CurrentPrincipal = Depends(get_principal)
