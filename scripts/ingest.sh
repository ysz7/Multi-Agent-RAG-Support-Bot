#!/usr/bin/env bash
#
# Index everything under DOCUMENTS_DIR into the vector store.
#
#   ./scripts/ingest.sh                    # index data/documents
#   ./scripts/ingest.sh ~/Downloads/docs   # index another folder
#   ./scripts/ingest.sh --prune            # drop docs whose source file is gone
#   ./scripts/ingest.sh --dry-run          # report, write nothing
#
# Starts Docker Desktop, postgres and ollama first if they are not running.
# Re-running is cheap: unchanged files are skipped by content hash before the
# embedding call.

set -Eeuo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
trace_errors

# A first bare argument that is an existing directory is shorthand for --path.
PATH_ARG=""
if [ "${1:-}" ] && [ -d "$1" ]; then
  PATH_ARG="$1"
  shift
fi

usage() {
  sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}
case "${1:-}" in -h|--help|help) usage ;; esac

ensure_env_file
resolve_python

DOCS_DIR="$(env_get DOCUMENTS_DIR data/documents)"
case "$DOCS_DIR" in /*) : ;; *) DOCS_DIR="$ROOT/$DOCS_DIR" ;; esac
TARGET_DIR="$DOCS_DIR"
if [ -n "$PATH_ARG" ]; then
  TARGET_DIR="$PATH_ARG"
fi

mkdir -p "$DOCS_DIR"
COUNT="$(find "$TARGET_DIR" -type f \( -iname '*.pdf' -o -iname '*.md' -o -iname '*.markdown' \
  -o -iname '*.txt' -o -iname '*.text' \) 2>/dev/null | wc -l | tr -d ' ')"
if [ "$COUNT" = "0" ]; then
  warn "no supported files in $TARGET_DIR (.pdf / .md / .markdown / .txt / .text)"
  warn "drop documents there and run again - or: python -m scripts.seed_documents"
  exit 1
fi
info "found $COUNT file(s) in $TARGET_DIR"

ensure_postgres
ensure_ollama

TENANT="$(env_get LOCAL_TENANT_ID default)"
info "indexing under tenant=$TENANT"

cd "$ROOT"
# Rebuild the argument list positionally: bash 3.2 trips over `"$@"` and empty
# arrays under `set -u`, and $@ here may legitimately be empty.
if [ -n "$PATH_ARG" ]; then
  set -- --path "$PATH_ARG" ${1:+"$@"}
fi
if [ "$#" -gt 0 ]; then
  exec "$PY" -m scripts.index_documents "$@"
fi
exec "$PY" -m scripts.index_documents
