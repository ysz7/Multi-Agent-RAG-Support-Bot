"""Every third-party module the app imports must be a declared dependency.

`mcp` was imported at module level by `app/mcp_server` for six phases without
ever appearing in `pyproject.toml`. It was installed in the dev environment, so
nothing local ever noticed; the first clean install — CI — failed to collect
nine test modules.

This walks the first-party packages, resolves each imported top-level module to
the distribution that provides it, and checks that distribution is named in
`pyproject.toml`. Imports inside functions are included: a lazy import of an
optional backend still has to be declared under *some* extra.
"""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("app", "evals", "scripts")

# First-party packages, not distributions we could declare.
IGNORED = {"app", "evals", "scripts", "tests"}


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _declared() -> set[str]:
    """Distribution names from `dependencies` and every extra, normalised.

    An extra on a requirement (`psycopg[binary,pool]`) is expanded to the
    distributions it pulls in (`psycopg-binary`, `psycopg-pool`), which is what
    an import of `psycopg_pool` actually resolves to.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)

    names: set[str] = set()
    for requirement in requirements:
        match = re.match(r"^([A-Za-z0-9._-]+)(?:\[([^\]]*)\])?", requirement.strip())
        if not match:
            continue
        name = _normalise(match.group(1))
        names.add(name)
        for feature in (match.group(2) or "").split(","):
            if feature.strip():
                names.add(f"{name}-{_normalise(feature)}")
    return names


def _imported_modules() -> dict[str, set[Path]]:
    """Top-level module name -> the files that import it."""
    found: dict[str, set[Path]] = {}
    for package in PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module] if node.module and node.level == 0 else []
                else:
                    continue
                for module in modules:
                    found.setdefault(module.split(".")[0], set()).add(path)
    return found


def test_all_imports_are_declared_dependencies():
    declared = _declared()
    distributions = packages_distributions()
    undeclared: list[str] = []

    for module, files in sorted(_imported_modules().items()):
        if module in IGNORED or module in sys.stdlib_module_names:
            continue
        # An optional backend may not be installed at all (qdrant ships as an
        # extra and is never installed by default). Fall back to matching the
        # module name itself, so "not installed" is not a free pass.
        providers = distributions.get(module) or [module]
        if not any(_normalise(name) in declared for name in providers):
            where = ", ".join(sorted(str(f.relative_to(ROOT)) for f in files))
            undeclared.append(f"{module} (provided by {providers}) imported in {where}")

    assert not undeclared, "undeclared dependencies:\n  " + "\n  ".join(undeclared)


def test_the_mcp_dependency_is_declared():
    """The specific regression: the tool server is runtime, not a dev extra."""
    text = (ROOT / "pyproject.toml").read_text()
    runtime = text.split("[project.optional-dependencies]")[0]
    assert '"mcp>=' in runtime
