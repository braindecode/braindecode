"""The core dependencies in pyproject.toml match what ``braindecode`` imports."""

import ast
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, packages_distributions, requires
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _name(requirement):
    return (
        re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
    )


def _imports(lazy):
    """Third-party top-level modules imported by ``braindecode`` (module level
    only, or anywhere with ``lazy``)."""
    found = set()
    for path in (ROOT / "braindecode").rglob("*.py"):
        tree = ast.parse(path.read_text())
        nodes = ast.walk(tree) if lazy else tree.body
        for node in nodes:
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                found.add(node.module.split(".")[0])
    return {m for m in found if m != "braindecode" and m not in sys.stdlib_module_names}


@pytest.fixture(scope="module")
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_module_level_imports_are_declared(project):
    """A module-level import comes from a core dependency or one of its own
    requirements (e.g. numpy via mne), never from an unrelated package."""
    core = {_name(r) for r in project["dependencies"]}
    allowed = set(core)
    for dist in core:
        try:
            allowed |= {_name(r) for r in requires(dist) or () if "extra ==" not in r}
        except PackageNotFoundError:
            pass
    dists = packages_distributions()
    undeclared = {
        module
        for module in _imports(lazy=False)
        if not {_name(d) for d in dists.get(module, [module])} & allowed
    }
    assert not undeclared, f"imported at module level but not declared: {undeclared}"


def test_core_dependencies_are_imported(project):
    """Every core dependency is imported somewhere in ``braindecode``."""
    imported = set()
    for module, dists in packages_distributions().items():
        if module in _imports(lazy=True):
            imported |= {_name(d) for d in dists}
    unused = {_name(r) for r in project["dependencies"]} - imported
    assert not unused, f"core dependencies never imported: {unused}"
