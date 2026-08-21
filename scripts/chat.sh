#!/usr/bin/env bash
#
# Interactive terminal chat against the local bot.
#
#   ./scripts/chat.sh                     # start everything, then chat
#   ./scripts/chat.sh -q "your question"  # ask once and exit
#   ./scripts/chat.sh --no-stream         # POST /chat instead of SSE
#
# Starts whatever is not already up: Docker Desktop, postgres, ollama, uvicorn.

set -Eeuo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
trace_errors

case "${1:-}" in
  -h|--help|help)
    sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
esac

ensure_env_file
resolve_python
ensure_postgres
ensure_ollama
ensure_api

cd "$ROOT"
# `"$@"` with no positional parameters trips `set -u` on bash 3.2, so branch on it.
if [ "$#" -gt 0 ]; then
  exec "$PY" -m scripts.chat_cli --url "$(api_url)" "$@"
fi
exec "$PY" -m scripts.chat_cli --url "$(api_url)"
