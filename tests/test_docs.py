"""Phase 15: the README must not claim properties the code lacks.

Documentation drift is invisible until someone follows the Quickstart on a clean
machine. These pin the claims that were actually wrong at some point: an auth
scheme the project does not implement, a Compose stack that never had Qdrant in
it, a dataset size, a project tree listing files that moved.
"""

import re
from pathlib import Path

import pytest
import yaml

from app.core.config import AuthMode, Settings, VectorStoreName

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()


def test_license_file_exists_and_matches_the_readme():
    license_text = (ROOT / "LICENSE").read_text()
    assert "MIT License" in license_text
    assert "MIT" in README.split("## License")[1]


def test_contributing_exists_and_is_linked():
    assert (ROOT / "CONTRIBUTING.md").exists()
    assert "CONTRIBUTING.md" in README


# --- the two deliberate deviations ----------------------------------------


def test_readme_does_not_promise_jwt_in_the_architecture():
    """AUTH_MODE=local is the default: no tokens, no login, no user table."""
    diagram = README.split("```")[1]
    assert "JWT auth" not in diagram
    assert "auth-gated" in diagram


def test_readme_documents_both_auth_modes():
    for mode in AuthMode.__args__:
        assert f"AUTH_MODE={mode}" in README


def test_quickstart_does_not_start_qdrant():
    """Qdrant is an opt-in overlay; the default stack never had it."""
    quickstart = README.split("## Quickstart")[1].split("###")[0]
    assert "qdrant" not in quickstart.lower()
    assert "docker-compose.qdrant.yml" in README, "the overlay should still be documented"


def test_readme_lists_every_vector_store_the_config_allows():
    for name in VectorStoreName.__args__:
        assert f"VECTOR_STORE={name}" in README or f"or: {name}" in README


def test_readme_says_embeddings_are_always_local():
    """The Phase 3 deviation: Anthropic has no embeddings endpoint."""
    assert "## Embeddings" in README
    settings = Settings(anthropic_api_key="x")
    assert settings.embedding_model in README


# --- claims that drift as the code moves ----------------------------------


def test_readme_dataset_size_matches_the_dataset():
    import json

    questions = json.loads((ROOT / "evals" / "golden_dataset.json").read_text())["questions"]
    answerable = sum(1 for q in questions if q["kind"] == "answerable")
    out_of_scope = len(questions) - answerable

    assert f"{len(questions)} questions" in README
    assert f"{answerable} answerable, {out_of_scope} out-of-scope" in README


def _tree_entries() -> list[str]:
    """File and directory names listed in the README's project tree."""
    tree = README.split("## Project structure")[1].split("```")[1]
    names: list[str] = []
    for line in tree.splitlines():
        match = re.match(r"^[\s│├└─]*([A-Za-z0-9_.-]+/?)", line)
        if match:
            names.append(match.group(1))
    return names


@pytest.mark.parametrize("name", _tree_entries())
def test_every_path_in_the_project_tree_exists(name):
    """The tree is a map; a wrong entry sends a reader to a file that is not there."""
    assert list(ROOT.rglob(name.rstrip("/"))), f"{name!r} in the project tree does not exist"


def test_the_project_tree_is_not_empty():
    """Guards the parser above: a tree that stops parsing would test nothing."""
    assert len(_tree_entries()) > 25


def test_documented_make_targets_exist():
    makefile = (ROOT / "Makefile").read_text()
    documented = re.search(r"`make help` lists the shortcuts: (.+?)\.", README, re.DOTALL)
    assert documented, "the Makefile shortcuts paragraph is gone"
    for target in re.findall(r"`(\w[\w-]*)`", documented.group(1)):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE), f"no make target {target!r}"


def test_documented_scripts_exist():
    for script in re.findall(r"python -m (scripts\.\w+)", README):
        assert (ROOT / script.replace(".", "/")).with_suffix(".py").exists(), script


def test_evaluation_command_matches_the_ci_workflow():
    """One documented threshold pair, used in both places."""
    workflow = (ROOT / ".github/workflows/evals.yml").read_text()
    for flag, value in (("--min-faithfulness", "0.85"), ("--min-context-recall", "0.60")):
        assert f"{flag} {value}" in README
        assert f"{flag} {value}" in workflow


def test_env_example_covers_every_key_the_readme_shows():
    """A key in the Quickstart that .env.example lacks is a broken first run."""
    example = (ROOT / ".env.example").read_text()
    for key in re.findall(r"^([A-Z][A-Z0-9_]{3,})=", README, re.MULTILINE):
        assert f"{key}=" in example, f"{key} is documented but missing from .env.example"


def test_compose_services_named_in_the_readme_exist():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    quickstart = README.split("## Quickstart")[1].split("###")[0]
    named = re.search(r"docker compose up -d\s+#\s*(.+)", quickstart)
    assert named, "the compose line lost its comment"
    for word in re.findall(r"[a-z][a-z-]+", named.group(1)):
        if word in {"postgres", "langfuse", "web", "worker", "clickhouse", "redis", "minio"}:
            assert any(word in service for service in compose["services"]), word
