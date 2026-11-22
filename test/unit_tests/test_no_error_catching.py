# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

"""Enforce that models, modules and functional check conditions instead of catching errors."""

import ast
from pathlib import Path

import braindecode

_ALLOWED_EXCEPT = {
    ("braindecode/models/config.py", "ImportError"),
    ("braindecode/functional/_lfilter.py", "ImportError"),
    ("braindecode/models/base.py", "ImportError"),
    ("braindecode/models/base.py", "NotImplementedError"),  # removed by Task 9
}


def _collect_except_handlers(source: str, rel_path: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ExceptHandler):
            kind = ast.unparse(node.type) if node.type else "bare"
            found.add((rel_path, kind))
    return found


def test_no_error_catching_outside_optional_imports():
    repo = Path(braindecode.__file__).parent.parent
    found = set()
    for package in ("models", "modules", "functional"):
        for path in (repo / "braindecode" / package).rglob("*.py"):
            rel_path = path.relative_to(repo).as_posix()
            found |= _collect_except_handlers(
                path.read_text(encoding="utf-8"), rel_path
            )
    assert found <= _ALLOWED_EXCEPT, sorted(found - _ALLOWED_EXCEPT)
    assert not (_ALLOWED_EXCEPT - found), (
        f"Stale _ALLOWED_EXCEPT entries: {sorted(_ALLOWED_EXCEPT - found)}"
    )


def test_except_scanner_flags_unallowlisted_and_bare_handlers():
    sample = (
        "try:\n"
        "    x = 1\n"
        "except ValueError:\n"
        "    x = 2\n"
        "try:\n"
        "    y = 1\n"
        "except:\n"
        "    y = 2\n"
    )
    assert _collect_except_handlers(sample, "braindecode/models/dummy.py") == {
        ("braindecode/models/dummy.py", "ValueError"),
        ("braindecode/models/dummy.py", "bare"),
    }
