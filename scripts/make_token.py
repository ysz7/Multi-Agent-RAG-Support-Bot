"""Mint a development JWT.

    python -m scripts.make_token --tenant acme --user alice

Exists so `AUTH_MODE=jwt` is usable without standing up an identity provider.
The app only verifies tokens; there is no login endpoint and no user store.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

from app.core.auth import SCOPE_APPROVE, SCOPE_CHAT
from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=settings.local_tenant_id)
    parser.add_argument("--user", default=settings.local_user_id)
    parser.add_argument("--scopes", default=f"{SCOPE_CHAT} {SCOPE_APPROVE}")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    if not settings.jwt_secret:
        print("JWT_SECRET is not set in .env - cannot sign a token.", file=sys.stderr)
        raise SystemExit(1)

    from jose import jwt

    now = datetime.now(UTC)
    claims = {
        "sub": args.user,
        "tenant_id": args.tenant,
        "scope": args.scopes,
        "iat": now,
        "exp": now + timedelta(days=args.days),
    }
    if settings.jwt_issuer:
        claims["iss"] = settings.jwt_issuer
    if settings.jwt_audience:
        claims["aud"] = settings.jwt_audience

    print(jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm))


if __name__ == "__main__":
    main()
