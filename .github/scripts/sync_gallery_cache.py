"""Manage persistent Sphinx-Gallery build caches across ``master`` CI runs.

Subcommands:
- ``prepare``: compare the cached ``.cached_sha`` against ``--head-sha``, delete
  ``.py.md5`` checksums for any gallery script whose imported ``braindecode``
  modules changed (or all checksums if ``docs/conf.py`` / ``pyproject.toml``
  changed or the base commit is unreachable), and rewrite cell-0 commit provenance
  in surviving cached notebooks and zip bundles.
- ``stamp``: record ``--head-sha`` in ``docs/auto_examples/.cached_sha`` after a
  successful Sphinx build.
"""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import zipfile
from argparse import ArgumentParser
from pathlib import Path

_EXACT_PIP_RE = re.compile(
    r"(%pip install git\+https://github\.com/braindecode/braindecode\.git@)[0-9a-zA-Z._/-]+"
)
_SHARED_CORE_MODULES = frozenset(
    {
        "braindecode.classifier",
        "braindecode.regressor",
        "braindecode.util",
        "braindecode.models.base",
        "braindecode.models.util",
        "braindecode.datasets.base",
        "braindecode.preprocessing.preprocess",
        "braindecode.preprocessing.windowers",
    }
)


def _path_to_module(path: str) -> str | None:
    if not (path.startswith("braindecode/") and path.endswith(".py")):
        return None
    parts = list(Path(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _extract_referenced_modules(
    codeobj_path: Path, source_py_path: Path | None
) -> set[str] | None:
    if not codeobj_path.is_file():
        return None
    modules: set[str] = set()
    try:
        data = json.loads(codeobj_path.read_text(encoding="utf-8"))
        for entries in data.values():
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, dict):
                        val = item.get("module")
                        if isinstance(val, str) and val.startswith("braindecode"):
                            modules.add(val)
    except (OSError, json.JSONDecodeError):
        return None

    if source_py_path is not None and source_py_path.is_file():
        try:
            tree = ast.parse(source_py_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("braindecode"):
                            modules.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("braindecode"):
                        for alias in node.names:
                            modules.add(f"{node.module}.{alias.name}")
        except (OSError, SyntaxError):
            pass

    return modules


def _module_matches(changed_mod: str, referenced: set[str]) -> bool:
    for ref in referenced:
        if ref == changed_mod or ref.startswith(f"{changed_mod}."):
            return True
    return False


def _rewrite_notebook_bytes(raw_bytes: bytes, head_sha: str) -> bytes | None:
    try:
        nb = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    first = cells[0]
    if first.get("cell_type") != "code":
        return None
    src = first.get("source")
    changed = False
    if isinstance(src, list):
        new_src = []
        for line in src:
            updated = _EXACT_PIP_RE.sub(rf"\g<1>{head_sha}", line)
            if updated != line:
                changed = True
            new_src.append(updated)
        first["source"] = new_src
    elif isinstance(src, str):
        updated = _EXACT_PIP_RE.sub(rf"\g<1>{head_sha}", src)
        if updated != src:
            changed = True
            first["source"] = updated
    if not changed:
        return None
    return (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _update_cached_provenance(auto_examples_dir: Path, head_sha: str) -> int:
    updated_count = 0
    for md5_file in auto_examples_dir.rglob("*.py.md5"):
        stem_path = md5_file.with_suffix("")  # .../plot_foo.py
        ipynb_path = stem_path.with_suffix(".ipynb")
        if ipynb_path.is_file():
            new_bytes = _rewrite_notebook_bytes(ipynb_path.read_bytes(), head_sha)
            if new_bytes is not None:
                ipynb_path.write_bytes(new_bytes)
                updated_count += 1

        for zip_path in ipynb_path.parent.glob(f"{ipynb_path.stem}*.zip"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zin:
                    entries = {name: zin.read(name) for name in zin.namelist()}
            except (OSError, zipfile.BadZipFile):
                continue
            zip_changed = False
            for name, content in list(entries.items()):
                if name.endswith(".ipynb"):
                    new_nb = _rewrite_notebook_bytes(content, head_sha)
                    if new_nb is not None:
                        entries[name] = new_nb
                        zip_changed = True
            if zip_changed:
                buf = io.BytesIO()
                with zipfile.ZipFile(
                    buf, "w", compression=zipfile.ZIP_DEFLATED
                ) as zout:
                    for name, content in entries.items():
                        zout.writestr(name, content)
                zip_path.write_bytes(buf.getvalue())
    return updated_count


def prepare_cache(
    *,
    docs_dir: Path,
    examples_dir: Path,
    head_sha: str,
    changed_files: list[str] | None,
) -> dict[str, int]:
    """Invalidate stale ``.py.md5`` files and refresh notebook commit provenance.

    If ``changed_files`` is ``None`` (e.g. cached SHA unreachable), all ``.py.md5``
    files are invalidated conservatively.
    """
    auto_examples = docs_dir / "auto_examples"
    if not auto_examples.is_dir():
        return {"invalidated": 0, "kept": 0, "updated_notebooks": 0}

    md5_files = sorted(auto_examples.rglob("*.py.md5"))
    if not md5_files:
        return {"invalidated": 0, "kept": 0, "updated_notebooks": 0}

    full_invalidate = changed_files is None or any(
        p in {"docs/conf.py", "pyproject.toml"} for p in changed_files
    )
    changed_mods = {
        mod for p in (changed_files or []) if (mod := _path_to_module(p)) is not None
    }
    if changed_mods & _SHARED_CORE_MODULES:
        full_invalidate = True

    invalidated = 0
    if full_invalidate:
        for md5_file in md5_files:
            md5_file.unlink(missing_ok=True)
            invalidated += 1
        return {"invalidated": invalidated, "kept": 0, "updated_notebooks": 0}

    if changed_mods:
        for md5_file in md5_files:
            rel = md5_file.relative_to(auto_examples)
            py_rel = Path(str(rel).removesuffix(".md5"))  # subsec/plot_foo.py
            codeobj_path = auto_examples / py_rel.with_suffix(".codeobj.json")
            source_py = examples_dir / py_rel
            referenced = _extract_referenced_modules(codeobj_path, source_py)
            if referenced is None or any(
                _module_matches(m, referenced) for m in changed_mods
            ):
                md5_file.unlink(missing_ok=True)
                invalidated += 1

    remaining = len(list(auto_examples.rglob("*.py.md5")))
    updated = _update_cached_provenance(auto_examples, head_sha)
    return {
        "invalidated": invalidated,
        "kept": remaining,
        "updated_notebooks": updated,
    }


def stamp_cache(docs_dir: Path, head_sha: str) -> Path:
    marker = docs_dir / "auto_examples" / ".cached_sha"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{head_sha}\n", encoding="utf-8")
    return marker


def _git_diff_names(
    repo_root: Path, cached_sha: str, head_sha: str
) -> list[str] | None:
    check = subprocess.run(
        ["git", "cat-file", "-e", f"{cached_sha}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        return None
    diff = subprocess.run(
        ["git", "diff", "--name-only", cached_sha, head_sha],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if diff.returncode != 0:
        return None
    return [line.strip() for line in diff.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--repo-root", type=Path, default=Path("."))
    prep.add_argument("--docs-dir", type=Path, default=Path("docs"))
    prep.add_argument("--examples-dir", type=Path, default=Path("examples"))
    prep.add_argument("--head-sha", required=True)

    st = sub.add_parser("stamp")
    st.add_argument("--docs-dir", type=Path, default=Path("docs"))
    st.add_argument("--head-sha", required=True)

    args = parser.parse_args()
    if args.cmd == "stamp":
        marker = stamp_cache(args.docs_dir.resolve(), args.head_sha)
        print(f"Stamped gallery cache SHA {args.head_sha} -> {marker}")
        return

    docs_dir = args.docs_dir.resolve()
    marker = docs_dir / "auto_examples" / ".cached_sha"
    if not marker.is_file():
        print("No prior Sphinx-Gallery cache marker found; building cold.")
        return

    cached_sha = marker.read_text(encoding="utf-8").strip()
    changed = _git_diff_names(args.repo_root.resolve(), cached_sha, args.head_sha)
    stats = prepare_cache(
        docs_dir=docs_dir,
        examples_dir=args.examples_dir.resolve(),
        head_sha=args.head_sha,
        changed_files=changed,
    )
    print(f"Sphinx-Gallery cache sync ({cached_sha[:8]}..{args.head_sha[:8]}): {stats}")


if __name__ == "__main__":
    main()
