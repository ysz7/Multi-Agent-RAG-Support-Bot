#!/usr/bin/env bash
# Shared preflight for scripts/ingest.sh and scripts/chat.sh.
#
# Nothing here is imported by the Python app - it only brings the local stack up
# (postgres, ollama, uvicorn) and reads a few keys out of .env without
# `source`-ing it, because .env holds unquoted values with spaces.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/.run"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_DIM=$'\033[2m'; C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YLW=$'\033[33m'; C_OFF=$'\033[0m'
else
  C_DIM=''; C_RED=''; C_GRN=''; C_YLW=''; C_OFF=''
fi

info() { printf '%s> %s%s\n' "$C_DIM" "$*" "$C_OFF" >&2; }
ok()   { printf '%sOK %s%s\n' "$C_GRN" "$*" "$C_OFF" >&2; }
warn() { printf '%s! %s%s\n' "$C_YLW" "$*" "$C_OFF" >&2; }
die()  { printf '%sERROR %s%s\n' "$C_RED" "$*" "$C_OFF" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "$1 not found in PATH${2:+ - $2}"
}

# Turn a silent `set -e` death into a line number. Needs `set -E` in the caller,
# or the trap is not inherited by functions - which is how the first version of
# these scripts managed to die without printing anything at all.
trace_errors() {
  set -E
  mkdir -p "$RUN_DIR"
  # A file, not a variable: a failure inside `$(...)` happens in a subshell, so a
  # variable set there would be lost and the same error reported twice.
  __GUARD="$RUN_DIR/.err-reported"
  rm -f "$__GUARD"
  trap '__st=$?; if [ ! -e "$__GUARD" ]; then : >"$__GUARD"; \
    printf "%sERROR %s: failed at line %s (exit %s)%s\n" \
      "$C_RED" "${BASH_SOURCE[0]##*/}" "$LINENO" "$__st" "$C_OFF" >&2; fi' ERR
}

# env_get KEY [DEFAULT] - the process environment wins, then .env, then the default.
# `eval` rather than `${!key}` so this also works under bash 3.2, which macOS ships.
env_get() {
  local key="$1" default="${2:-}" val=""
  eval "val=\${$key:-}"
  if [ -n "$val" ]; then printf '%s' "$val"; return 0; fi
  if [ -f "$ROOT/.env" ]; then
    val="$(sed -n "s/^[[:space:]]*${key}=//p" "$ROOT/.env" | tail -n 1)"
    val="${val%$'\r'}"
    case "$val" in
      \"*\") val="${val#\"}"; val="${val%\"}" ;;
      \'*\') val="${val#\'}"; val="${val%\'}" ;;
    esac
  fi
  [ -n "$val" ] || val="$default"
  printf '%s' "$val"
}

compose() {
  if [ -z "${COMPOSE_CMD:-}" ]; then
    if docker compose version >/dev/null 2>&1; then
      COMPOSE_CMD="docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
      COMPOSE_CMD="docker-compose"
    else
      die "docker compose not found - install Docker Desktop"
    fi
  fi
  ( cd "$ROOT" && $COMPOSE_CMD "$@" )
}

# wait_http URL TRIES - poll until it answers, one second apart.
wait_http() {
  local url="$1" tries="${2:-30}" i=0
  until curl -fsS -o /dev/null "$url" 2>/dev/null; do
    i=$((i + 1))
    [ "$i" -ge "$tries" ] && return 1
    sleep 1
  done
  return 0
}

resolve_python() {
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
    warn "no .venv - falling back to $PY"
  else
    die "python3 not found"
  fi
  export PY
}

ensure_env_file() {
  [ -f "$ROOT/.env" ] || die ".env is missing - copy it: cp .env.example .env"
}

# The `docker` CLI existing says nothing about the daemon being up. On macOS the
# daemon is Docker Desktop, an app - so launch it and wait, rather than failing.
ensure_docker_daemon() {
  docker info >/dev/null 2>&1 && return 0

  if [ "$(uname -s)" = "Darwin" ] && [ -d "/Applications/Docker.app" ]; then
    info "Docker daemon is down - launching Docker Desktop (first start takes ~30s)..."
    open -a Docker || die "could not launch Docker Desktop"
    local i=0
    until docker info >/dev/null 2>&1; do
      i=$((i + 1))
      [ "$i" -ge 120 ] && die "Docker Desktop did not become ready in 120s - start it by hand and retry"
      sleep 1
    done
    ok "Docker daemon ready"
    return 0
  fi

  die "Docker daemon is not running - start Docker Desktop and retry"
}

ensure_postgres() {
  need docker "required for postgres/pgvector - https://docker.com"
  ensure_docker_daemon

  local health=""
  # `|| true`: a failing pipeline inside an assignment kills the script under
  # `set -e`, and its stderr is hidden here, so the death would be silent.
  health="$(compose ps --format '{{.Health}}' postgres 2>/dev/null | head -n 1 || true)"
  if [ "$health" != "healthy" ]; then
    info "starting postgres..."
    compose up -d postgres >/dev/null || die "could not start postgres - try: docker compose up postgres"
  fi

  local i=0
  while [ "$(compose ps --format '{{.Health}}' postgres 2>/dev/null | head -n 1 || true)" != "healthy" ]; do
    i=$((i + 1))
    [ "$i" -gt 60 ] && die "postgres never became healthy (120s) - check: docker compose logs postgres"
    sleep 2
  done
  ok "postgres ready"
}

# Embeddings always go through Ollama, whatever LLM_PROVIDER says.
ensure_ollama() {
  local base model tags
  base="$(env_get OLLAMA_BASE_URL http://localhost:11434)"
  model="$(env_get EMBEDDING_MODEL nomic-embed-text)"

  if ! curl -fsS -o /dev/null "$base/api/tags" 2>/dev/null; then
    need ollama "required for embeddings - https://ollama.com"
    info "starting ollama serve..."
    mkdir -p "$RUN_DIR"
    nohup ollama serve >"$RUN_DIR/ollama.log" 2>&1 &
    echo $! >"$RUN_DIR/ollama.pid"
    wait_http "$base/api/tags" 30 || die "ollama is not answering on $base - log: $RUN_DIR/ollama.log"
  fi

  tags="$(curl -fsS "$base/api/tags" 2>/dev/null || true)"
  # An empty body here means the server answered but told us nothing; treat a
  # missing model as "pull it" rather than assuming it is there.
  if ! printf '%s' "$tags" | grep -q "\"$model"; then
    need ollama
    info "pulling embedding model $model (one time)..."
    OLLAMA_HOST="$base" ollama pull "$model" || die "could not pull $model"
  fi
  ok "ollama ready ($model)"
}

api_url() {
  env_get RAGBOT_API_URL "http://127.0.0.1:$(env_get RAGBOT_PORT 8000)"
}

ensure_api() {
  local url port
  url="$(api_url)"
  port="${url##*:}"
  port="${port%%/*}"

  if curl -fsS -o /dev/null "$url/health" 2>/dev/null; then
    ok "API already listening on $url"
    return 0
  fi

  info "starting uvicorn on $url..."
  mkdir -p "$RUN_DIR"
  (
    cd "$ROOT" || exit 1
    nohup "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$port" \
      >"$RUN_DIR/api.log" 2>&1 &
    echo $! >"$RUN_DIR/api.pid"
  )
  if ! wait_http "$url/health" 90; then
    printf '%s\n' "--- last lines of $RUN_DIR/api.log ---" >&2
    tail -n 30 "$RUN_DIR/api.log" >&2 || true
    die "API did not come up"
  fi
  ok "API ready ($url)"
  warn "server left running in the background: kill \$(cat .run/api.pid)"
}
