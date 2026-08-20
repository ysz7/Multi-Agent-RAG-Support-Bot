"""Phase 2: static checks on the Compose stack.

Docker is not required to run these — they guard the invariants that are easy
to break by editing YAML: port collisions, leaked env vars, config drift
between .env.example and Settings defaults.
"""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from app.core.config import Settings

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"


def _env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def test_default_stack_has_no_qdrant(compose):
    """Qdrant is opt-in via the overlay; it must never be in the default stack."""
    assert "qdrant" not in compose["services"]
    assert (ROOT / "docker-compose.qdrant.yml").exists()


def test_app_postgres_has_pgvector(compose):
    assert "pgvector" in compose["services"]["postgres"]["image"]


def test_no_duplicate_host_ports(compose):
    published = [
        str(port).rsplit(":", 1)[0]
        for service in compose["services"].values()
        for port in service.get("ports", [])
    ]
    duplicates = [port for port, count in Counter(published).items() if count > 1]
    assert not duplicates, f"host port collision: {duplicates}"


def test_langfuse_reachable_on_3000(compose):
    """The README points users at http://localhost:3000 for the dashboard."""
    ports = compose["services"]["langfuse-web"]["ports"]
    assert any(str(p).startswith("3000:") for p in ports)


def test_worker_does_not_inherit_web_only_env(compose):
    worker = compose["services"]["langfuse-worker"]["environment"]
    assert "NEXTAUTH_SECRET" not in worker
    assert not any(key.startswith("LANGFUSE_INIT") for key in worker)


def test_worker_and_web_share_required_env(compose):
    worker = compose["services"]["langfuse-worker"]["environment"]
    web = compose["services"]["langfuse-web"]["environment"]
    for key in ("DATABASE_URL", "CLICKHOUSE_URL", "REDIS_AUTH", "ENCRYPTION_KEY", "SALT"):
        assert key in worker and key in web, key


def test_services_have_healthchecks(compose):
    """depends_on: service_healthy only works if the dependency defines one."""
    for name in ("postgres", "clickhouse", "redis", "minio"):
        assert "healthcheck" in compose["services"][name], name


def test_postgres_init_script_is_executable():
    script = ROOT / "docker" / "postgres" / "init" / "01-init.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111, "init script must be executable"


def test_embedding_dim_matches_settings_default(env):
    """The init script sizes vector(N) from EMBEDDING_DIM; a mismatch with the
    app's default would only surface as a runtime insert error."""
    env(ANTHROPIC_API_KEY="sk-test")
    assert _env_example()["EMBEDDING_DIM"] == str(Settings().embedding_dim)


def test_database_url_points_at_the_compose_database():
    values = _env_example()
    assert values["DATABASE_URL"].endswith("/" + values["POSTGRES_DB"])
    assert f":{values['POSTGRES_PORT']}/" in values["DATABASE_URL"]
