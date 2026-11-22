"""Enforce no error catching outside optional imports in models, modules, and functional."""

import ast
from pathlib import Path

import braindecode

_ALLOWED_EXCEPT = {
    ("braindecode/models/config.py", "ImportError"),
    ("braindecode/functional/_lfilter.py", "ImportError"),
    ("braindecode/models/base.py", "ImportError"),
}


def test_no_error_catching_outside_optional_imports():
    repo = Path(braindecode.__file__).parent.parent
    found = set()
    for package in ("models", "modules", "functional"):
        for path in (repo / "braindecode" / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ExceptHandler):
                    kind = ast.unparse(node.type) if node.type else "bare"
                    found.add((path.relative_to(repo).as_posix(), kind))
    assert found <= _ALLOWED_EXCEPT, sorted(found - _ALLOWED_EXCEPT)
