"""Plan documentation CI execution scope from git diff and AST public-API inspection.

Outputs GitHub Actions key=value pairs deciding:
- ``build``: ``"true"`` or ``"false"`` (skips PRs that only touch tests, non-doc CI,
  internal function bodies, or ``docs/whats_new.rst``).
- ``run_examples``: ``"all"``, pipe-delimited ``plot_*`` stems, or ``""`` (no-plot).
- ``gallery_arg``: Sphinx CLI override for ``sphinx-gallery``.
- ``extras``: minimal ``pyproject.toml`` extras string required for the chosen mode.
- ``use_gallery_cache``: ``"true"`` on ``master`` pushes, ``"false"`` otherwise.
"""

from __future__ import annotations

import ast
import os
import subprocess
import tomllib
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

BASE_EXTRAS = "docs,moabb,hub,eegprep,typing,viz"
GALLERY_EXTRAS = f"docs,docs-gallery,moabb,hub,eegprep,typing,viz"
FULL_EXTRAS = f"{GALLERY_EXTRAS},sae"

DOC_CI_PATHS = frozenset(
    {
        ".github/workflows/docs.yml",
        ".github/workflows/push-artifact-to-circleci.yml",
        ".github/scripts/plan_docs.py",
        ".github/scripts/sync_gallery_cache.py",
        ".circleci/config.yml",
        "README.rst",
    }
)

DUNDER_DOC_METHODS = frozenset({"__init__", "__call__", "__getitem__", "__len__"})
DOCSTRING_HELPER_FUNCTIONS = frozenset(
    {
        "update_estimator_docstring",
        "_clean_docstring_sections",
        "_update_moabb_docstring",
    }
)


def _format_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ast.unparse(fn.args)
    returns = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None else ""
    decorators = [ast.unparse(d) for d in fn.decorator_list]
    return f"{decorators}:{fn.name}({args}){returns}"


def public_api_fingerprint(source: str) -> tuple[object, ...] | None:
    """Return a deterministic tuple capturing public API signatures and docstrings.

    Returns ``None`` if ``source`` cannot be parsed as Python AST.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    items: list[object] = [("module_doc", ast.get_docstring(tree, clean=False))]

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    items.append(("__all__", ast.unparse(node.value)))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                val = ast.unparse(node.value) if node.value is not None else None
                items.append(("var", node.target.id, ast.unparse(node.annotation), val))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in DOCSTRING_HELPER_FUNCTIONS:
                items.append(("doc_helper", node.name, ast.unparse(node)))
            elif not node.name.startswith("_"):
                items.append(
                    (
                        "func",
                        _format_signature(node),
                        ast.get_docstring(node, clean=False),
                    )
                )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            bases = tuple(ast.unparse(b) for b in node.bases)
            decorators = tuple(ast.unparse(d) for d in node.decorator_list)
            members: list[object] = []
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not stmt.name.startswith("_") or stmt.name in DUNDER_DOC_METHODS:
                        members.append(
                            (
                                "method",
                                _format_signature(stmt),
                                ast.get_docstring(stmt, clean=False),
                            )
                        )
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    if not stmt.target.id.startswith("_"):
                        val = (
                            ast.unparse(stmt.value) if stmt.value is not None else None
                        )
                        members.append(
                            (
                                "field",
                                stmt.target.id,
                                ast.unparse(stmt.annotation),
                                val,
                            )
                        )
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id in {
                            "doc",
                            "__doc__",
                        }:
                            members.append(("doc_attr", target.id, ast.unparse(stmt)))
            items.append(
                (
                    "class",
                    node.name,
                    bases,
                    decorators,
                    ast.get_docstring(node, clean=False),
                    tuple(members),
                )
            )

    return tuple(items)


def _pyproject_docs_changed(old_toml: str | None, new_toml: str | None) -> bool:
    if old_toml is None or new_toml is None:
        return True
    try:
        old_proj = tomllib.loads(old_toml).get("project", {})
        new_proj = tomllib.loads(new_toml).get("project", {})
    except tomllib.TOMLDecodeError:
        return True

    if old_proj.get("dependencies") != new_proj.get("dependencies"):
        return True
    old_opt = old_proj.get("optional-dependencies", {})
    new_opt = new_proj.get("optional-dependencies", {})
    doc_keys = {
        "docs",
        "docs-gallery",
        "moabb",
        "hub",
        "eegprep",
        "typing",
        "viz",
        "sae",
    }
    return any(old_opt.get(k) != new_opt.get(k) for k in doc_keys)


def compute_plan(
    *,
    event_name: str,
    ref: str,
    head_ref: str,
    changed_entries: list[tuple[str, str]],
    read_old_file: Callable[[str], str | None],
    read_new_file: Callable[[str], str | None],
) -> dict[str, str]:
    """Compute documentation plan dictionary from git status-path entries.

    ``changed_entries`` is a list of ``(status, posix_path)`` pairs from
    ``git diff --name-status`` (e.g. ``("M", "braindecode/models/eegnet.py")``,
    ``("D", "examples/plot_old.py")``).
    """
    is_partial_pr = event_name == "pull_request" and not head_ref.startswith("release/")
    if not is_partial_pr:
        use_cache = event_name == "push" and ref == "refs/heads/master"
        return {
            "build": "true",
            "run_examples": "all",
            "gallery_arg": "",
            "extras": FULL_EXTRAS,
            "use_gallery_cache": "true" if use_cache else "false",
        }

    changed_plots = sorted(
        {
            Path(path).stem
            for status, path in changed_entries
            if not status.startswith("D")
            and path.startswith("examples/")
            and Path(path).name.startswith("plot_")
            and path.endswith(".py")
        }
    )
    if changed_plots:
        run_examples = "|".join(changed_plots)
        needs_sae = any(stem.startswith("plot_sae_") for stem in changed_plots)
        return {
            "build": "true",
            "run_examples": run_examples,
            "gallery_arg": (
                f"-D sphinx_gallery_conf.filename_pattern=/({run_examples})\\.py"
            ),
            "extras": f"{GALLERY_EXTRAS},sae" if needs_sae else GALLERY_EXTRAS,
            "use_gallery_cache": "false",
        }

    docs_affected = False
    for status, path in changed_entries:
        if path == "docs/whats_new.rst":
            continue
        if (
            path.startswith("docs/")
            or path.startswith("examples/")
            or path in DOC_CI_PATHS
            or path == "braindecode/models/summary.csv"
        ):
            docs_affected = True
            break
        if path == "pyproject.toml":
            if _pyproject_docs_changed(read_old_file(path), read_new_file(path)):
                docs_affected = True
                break
        if path.startswith("braindecode/") and path.endswith(".py"):
            if status[:1] in {"A", "D", "R"}:
                docs_affected = True
                break
            old_src = read_old_file(path)
            new_src = read_new_file(path)
            if old_src is None or new_src is None:
                docs_affected = True
                break
            old_fp = public_api_fingerprint(old_src)
            new_fp = public_api_fingerprint(new_src)
            if old_fp is None or new_fp is None or old_fp != new_fp:
                docs_affected = True
                break

    if not docs_affected:
        return {
            "build": "false",
            "run_examples": "",
            "gallery_arg": "-D plot_gallery=0",
            "extras": "",
            "use_gallery_cache": "false",
        }

    return {
        "build": "true",
        "run_examples": "",
        "gallery_arg": "-D plot_gallery=0",
        "extras": BASE_EXTRAS,
        "use_gallery_cache": "false",
    }


def _git_changed_entries(
    repo_root: Path, base_rev: str, head_rev: str
) -> list[tuple[str, str]]:
    proc = subprocess.run(
        ["git", "diff", "--name-status", base_rev, head_rev],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # Fallback if shallow clone lacks base_rev: trigger safe no-plot doc build.
        return [("M", "docs/index.rst")]
    entries: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            entries.append((parts[0], parts[-1]))
    return entries


def _git_show_file(repo_root: Path, rev: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--base-rev", default="HEAD^1")
    parser.add_argument("--head-rev", default="HEAD")
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    event_name = os.environ.get("GITHUB_EVENT_NAME", "push")
    ref = os.environ.get("GITHUB_REF", "")
    head_ref = os.environ.get("GITHUB_HEAD_REF", "")

    entries = (
        _git_changed_entries(repo_root, args.base_rev, args.head_rev)
        if event_name == "pull_request"
        else []
    )
    plan = compute_plan(
        event_name=event_name,
        ref=ref,
        head_ref=head_ref,
        changed_entries=entries,
        read_old_file=lambda p: _git_show_file(repo_root, args.base_rev, p),
        read_new_file=lambda p: _git_show_file(repo_root, args.head_rev, p),
    )

    lines = "".join(f"{k}={v}\n" for k, v in plan.items())
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as fh:
            fh.write(lines)
    print(lines, end="")


if __name__ == "__main__":
    main()
