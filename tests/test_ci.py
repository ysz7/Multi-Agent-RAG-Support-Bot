"""Phase 14: static checks on the CI workflows.

GitHub Actions cannot be run here, so these guard the things that silently rot:
a threshold that drifts away from the documented one, a job that installs the
wrong extras, an image that no longer matches Compose, or a step that runs
without the secret it needs.
"""

from pathlib import Path

import pytest
import yaml

from app.core.config import Settings

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"


@pytest.fixture(scope="module")
def ci() -> dict:
    return yaml.safe_load((WORKFLOWS / "ci.yml").read_text())


@pytest.fixture(scope="module")
def evals() -> dict:
    return yaml.safe_load((WORKFLOWS / "evals.yml").read_text())


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job]["steps"]


def _run_script(workflow: dict, job: str) -> str:
    return "\n".join(step.get("run", "") for step in _steps(workflow, job))


# --- lint and unit tests ---------------------------------------------------


def test_both_workflows_run_on_pull_requests(ci, evals):
    # PyYAML parses a bare `on:` key as the boolean True.
    assert "pull_request" in ci[True]
    assert "pull_request" in evals[True]


def test_every_job_has_a_timeout(ci, evals):
    """A hung model call must not hold a runner for six hours."""
    for workflow in (ci, evals):
        for name, job in workflow["jobs"].items():
            assert job.get("timeout-minutes"), f"{name} has no timeout"


def test_lint_job_checks_formatting_too(ci):
    script = _run_script(ci, "lint")
    assert "ruff check ." in script
    assert "ruff format --check ." in script


def test_unit_tests_exclude_live_tests(ci):
    """`live` needs a model and a database; that is the evals workflow's job."""
    assert 'pytest -q -m "not live"' in _run_script(ci, "tests")


def test_unit_tests_install_the_extras_they_import(ci):
    """The offline suite imports `evals.judge` (ragas) and the jwt auth path."""
    script = _run_script(ci, "tests")
    assert "dev" in script and "jwt" in script and "evals" in script


def test_ci_python_version_satisfies_the_project(ci):
    """CI must not run on a version the package refuses to install on."""
    import re

    pyproject = (ROOT / "pyproject.toml").read_text()
    minimum = re.search(r'requires-python = ">=([\d.]+)"', pyproject).group(1)
    version = str(ci["env"]["PYTHON_VERSION"])

    as_tuple = lambda text: tuple(int(part) for part in text.split("."))  # noqa: E731
    assert as_tuple(version) >= as_tuple(minimum)


# --- the evals workflow ----------------------------------------------------


def test_evals_uses_the_compose_postgres_image(evals):
    """One schema path: CI and local must not drift onto different images."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    ci_image = evals["jobs"]["ragas"]["services"]["postgres"]["image"]
    assert ci_image == compose["services"]["postgres"]["image"]


def test_evals_creates_the_schema_with_the_compose_init_script(evals):
    script = _run_script(evals, "ragas")
    assert "docker/postgres/init/01-init.sh" in script
    assert (ROOT / "docker/postgres/init/01-init.sh").exists()


def test_evals_embedding_dim_matches_the_settings_default(evals):
    """A mismatch here fails as an opaque INSERT error, deep in indexing."""
    env = evals["jobs"]["ragas"]["env"]
    assert int(env["EMBEDDING_DIM"]) == Settings(anthropic_api_key="x").embedding_dim


def test_evals_enforces_the_documented_thresholds(evals):
    """The README promises these numbers; CI must actually apply them."""
    script = _run_script(evals, "ragas")
    readme = (ROOT / "README.md").read_text()
    for flag, value in (
        ("--min-faithfulness", "0.85"),
        ("--min-context-recall", "0.60"),
    ):
        assert f"{flag} {value}" in script
        assert f"{flag} {value}" in readme
    # Beyond the README: retrieval and refusals are checked too (Phase 13).
    assert "--min-source-accuracy" in script
    assert "--min-refusal-rate" in script


def test_evals_indexes_the_corpus_itself(evals):
    """`--skip-index` is a local convenience; CI must prove indexing works."""
    assert "--skip-index" not in _run_script(evals, "ragas")


def test_evals_steps_are_gated_on_the_api_key(evals):
    """A fork PR has no secrets: it should skip cleanly, not fail confusingly."""
    steps = _steps(evals, "ragas")
    guard = next(step for step in steps if step.get("id") == "guard")
    assert "ANTHROPIC_API_KEY" in guard["run"]

    after_guard = steps[steps.index(guard) + 1 :]
    for step in after_guard:
        assert "steps.guard.outputs.run == 'true'" in step.get("if", ""), step.get("name")


def test_evals_uploads_the_report_even_on_failure(evals):
    upload = next(
        step for step in _steps(evals, "ragas") if "upload-artifact" in step.get("uses", "")
    )
    assert upload["if"].startswith("always()")


def test_evals_runs_on_a_schedule(evals):
    """Model drift is not tied to someone opening a pull request."""
    assert evals[True]["schedule"]
