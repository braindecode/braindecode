#!/usr/bin/env python3
"""Stable and development release versions for braindecode.

Commands used by the publishing workflows:

``compute``
    Print ``current_version``, ``release_version``, ``next_dev_version``,
    ``release_tag`` and ``next_dev_base`` as ``key=value`` lines that the
    workflow can redirect into ``$GITHUB_OUTPUT``.

``finalize``
    Rewrite ``braindecode/version.py`` to the release version (drop the
    ``.devN`` / ``devN`` suffix) and flip the matching
    ``Current X.Y.Z (GitHub)`` heading in ``docs/whats_new.rst`` to
    ``Current X.Y.Z (YYYY-MM-DD)``.

``next-dev``
    Rewrite ``braindecode/version.py`` to the next dev seed (e.g.
    ``1.7.0dev0`` for a minor bump after ``1.6.0``) and insert a fresh
    ``Current X.Y.Z (GitHub)`` section at the top of ``docs/whats_new.rst``.

``check-changes``
    Report whether the current source has changes to release since the last stable tag.

``dev-build``
    Set the development version from the first-parent history of the checked-out commit.

Bumping rules (``--bump``)
    ``minor`` (default): ``1.6.0`` -> next dev ``1.7.0dev0``.
    ``patch``:           ``1.6.0`` -> next dev ``1.6.1dev0``.
    ``major``:           ``1.6.0`` -> next dev ``2.0.0dev0``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess  # nosec B404 - invokes Git with a fixed argument list
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover - exercised on CI bootstrap
    print(
        "ERROR: 'packaging' is required. Install with: pip install packaging",
        file=sys.stderr,
    )
    sys.exit(2)


# Capture ``prefix`` (``__version__ = ``), the actual opening quote, the
# version literal and require the closing quote to match the opening one so
# corrupt input like ``__version__ = "1.6.0'`` is rejected upfront instead of
# silently round-tripped.
VERSION_RE = re.compile(
    r"(?P<prefix>__version__\s*=\s*)(?P<q>['\"])(?P<ver>[^'\"]+)(?P=q)"
)

# Matches: ``Current X.Y[.Z] (GitHub)`` exactly. Captures the version string.
GITHUB_HEADING_RE_TEMPLATE = r"^Current {version} \(GitHub\)[ \t]*$"

# Matches any heading that opens a section for the given version, regardless
# of the suffix (``(GitHub)``, ``(in development)``, dated stable, etc.).
# Used by ``insert_next_dev_section`` to stay idempotent even when a
# maintainer pre-edited the heading.
ANY_HEADING_RE_TEMPLATE = r"^Current {version}\b"

# The skeleton of an empty "what's new" section we drop in for the next dev
# cycle. It mirrors the existing braindecode style in ``docs/whats_new.rst``:
# section title underlined by ``===``, blank line, list with ``- None yet``.
EMPTY_SECTION_TEMPLATE = """\
Current {base} (GitHub)
===============================

Enhancements
============

- None yet

API and behavior changes
========================

- None yet

Requirements
============

- None yet

Bug fixes
==========

- None yet

Code health
============

- None yet


"""


def read_version(path: Path) -> str:
    """Return the ``__version__`` string declared in ``path``."""
    text = path.read_text(encoding="utf-8")
    m = VERSION_RE.search(text)
    if not m:
        raise SystemExit(f"__version__ assignment not found in {path}")
    return m.group("ver")


def write_version(path: Path, new_version: str) -> None:
    """Replace the ``__version__`` literal in ``path`` with ``new_version``."""
    text = path.read_text(encoding="utf-8")
    new_text = replace_version(text, new_version)
    path.write_text(new_text, encoding="utf-8")


def replace_version(text: str, new_version: str) -> str:
    """Replace only the version literal, preserving its quotes and surrounding code."""
    match = VERSION_RE.search(text)
    if match is None:
        raise SystemExit("__version__ assignment not found")
    return text[: match.start("ver")] + new_version + text[match.end("ver") :]


def release_from_current(current: str) -> str:
    """Return the PEP 440 release segment of ``current`` as ``X.Y.Z``.

    ``1.6.0dev0`` -> ``1.6.0``; ``1.6`` -> ``1.6.0``; ``1.6.0.dev17`` -> ``1.6.0``.
    """
    release_parts = list(Version(current).release)
    while len(release_parts) < 3:
        release_parts.append(0)
    major, minor, patch = release_parts[:3]
    return f"{major}.{minor}.{patch}"


def compute_versions(current: str, bump: str) -> dict[str, str]:
    """Return the planned ``release`` and ``next_dev`` versions."""
    release_version = release_from_current(current)
    major, minor, patch = map(int, release_version.split("."))

    if bump == "major":
        next_base = f"{major + 1}.0.0"
    elif bump == "minor":
        next_base = f"{major}.{minor + 1}.0"
    elif bump == "patch":
        next_base = f"{major}.{minor}.{patch + 1}"
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"unknown bump: {bump}")

    next_dev = f"{next_base}dev0"

    return {
        "current_version": current,
        "release_version": release_version,
        "release_tag": f"v{release_version}",
        "next_dev_version": next_dev,
        "next_dev_base": next_base,
    }


def _find_github_heading(text: str, version: str) -> re.Match[str] | None:
    pat = re.compile(
        GITHUB_HEADING_RE_TEMPLATE.format(version=re.escape(version)),
        re.M,
    )
    return pat.search(text)


def finalize_changelog(changelog: Path, release_version: str) -> None:
    """Rename ``Current X.Y.Z (GitHub)`` heading to a dated stable heading.

    Uses UTC date so the same line is produced regardless of runner timezone.
    Hard-fails when the expected heading is absent: shipping a stable release
    with a stale ``(GitHub)`` (or missing) heading would be a silent docs bug
    that the workflow cannot recover from after PyPI upload.
    """
    text = changelog.read_text(encoding="utf-8")
    today = datetime.now(timezone.utc).date().isoformat()

    m = _find_github_heading(text, release_version)
    if not m:
        raise SystemExit(
            f"ERROR: no 'Current {release_version} (GitHub)' heading found in "
            f"{changelog}. Refusing to finalize a release with an unfinalized "
            "changelog. Add the heading (or correct the version) and re-run."
        )

    new_heading = f"Current {release_version} ({today})"
    new_text = text[: m.start()] + new_heading + text[m.end() :]
    changelog.write_text(new_text, encoding="utf-8")


def insert_next_dev_section(changelog: Path, next_dev_base: str) -> None:
    """Insert a fresh ``Current X.Y.Z (GitHub)`` section after ``.. _current:``."""
    text = changelog.read_text(encoding="utf-8")
    updated = next_dev_changelog(text, next_dev_base)
    if updated != text:
        changelog.write_text(updated, encoding="utf-8")


def next_dev_changelog(text: str, next_dev_base: str) -> str:
    """Return the changelog with an empty section for the next cycle."""
    # Idempotent: any pre-existing ``Current <next> ...`` heading (regardless
    # of suffix — ``(GitHub)``, ``(in development)``, dated stable, etc.) is
    # treated as "already there" so a re-run does not append a duplicate
    # section above a maintainer-edited variant.
    any_existing = re.compile(
        ANY_HEADING_RE_TEMPLATE.format(version=re.escape(next_dev_base)),
        re.M,
    )
    if any_existing.search(text):
        return text

    anchor_re = re.compile(r"^\.\. _current:[ \t]*\n", re.M)
    m = anchor_re.search(text)
    if m:
        # Skip any blank lines after the anchor so the new section sits flush
        # against the existing structure.
        insert_at = m.end()
        while insert_at < len(text) and text[insert_at] == "\n":
            insert_at += 1
    else:
        # Fall back to inserting before the first existing "Current X.Y.Z"
        # heading.
        m2 = re.search(r"^Current \d", text, re.M)
        if not m2:
            raise SystemExit(
                "Could not find an insertion point in the changelog: "
                "no '.. _current:' anchor and no 'Current X.Y.Z' heading."
            )
        insert_at = m2.start()

    section = EMPTY_SECTION_TEMPLATE.format(base=next_dev_base)
    return text[:insert_at] + section + text[insert_at:]


def git_output(*args: str) -> str:
    """Run Git on the checked-out release history and preserve its output verbatim."""
    git = shutil.which("git")
    if git is None:
        raise SystemExit("Git is required to inspect the release history.")
    # Callers supply fixed options and tag names matched against vX.Y.Z.
    return subprocess.check_output([git, *args], text=True)  # nosec B603


def latest_stable_tag() -> str:
    """Return the highest version-sorted tag whose entire name is vX.Y.Z."""
    tags = git_output("tag", "--list", "--sort=-version:refname").splitlines()
    return next(
        (tag for tag in tags if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag)), ""
    )


def has_release_changes() -> bool:
    """Check master against the stable tag, including its off-master release commit."""
    tag = latest_stable_tag()
    if not tag:
        return True
    count = int(git_output("rev-list", "--first-parent", "--count", f"{tag}..HEAD"))
    if count != 1:
        return count > 0

    # A version-bump PR mirrors the release commit's changelog on master,
    # changes the version to the next dev seed and opens an empty section.
    version_path = Path("braindecode/version.py")
    changelog = Path("docs/whats_new.rst")
    changes = git_output("diff", "--name-status", tag, "HEAD")
    if set(changes.splitlines()) != {
        f"M\t{version_path.as_posix()}",
        f"M\t{changelog.as_posix()}",
    }:
        return True
    current = read_version(version_path)
    seeds = {
        compute_versions(tag[1:], bump)["next_dev_version"]
        for bump in ("major", "minor", "patch")
    }
    if current not in seeds:
        return True
    old_version = git_output("show", f"{tag}:{version_path.as_posix()}")
    old_notes = git_output("show", f"{tag}:{changelog.as_posix()}")
    expected_version = replace_version(old_version, current)
    expected_notes = next_dev_changelog(old_notes, release_from_current(current))
    return (expected_version, expected_notes) != (
        version_path.read_text(encoding="utf-8"),
        changelog.read_text(encoding="utf-8"),
    )


def dev_build(init_file: Path) -> str:
    """Number master commits in order, above the former 32-bit hash suffixes."""
    if git_output("rev-parse", "--is-shallow-repository").strip() == "true":
        raise SystemExit("Development versions require a full Git history.")
    version = Version(read_version(init_file))
    if not version.is_devrelease:
        raise SystemExit(
            f"Expected a development version in {init_file}, got {version}."
        )
    count = int(git_output("rev-list", "--first-parent", "--count", "HEAD"))
    build = f"{version.base_version}.dev{2**32 + count}"
    write_version(init_file, build)
    return build


def cmd_compute(args: argparse.Namespace) -> int:
    current = read_version(args.init_file)
    info = compute_versions(current, args.bump)
    for key, value in info.items():
        print(f"{key}={value}")
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    # ``--bump`` is intentionally not consulted here: the release version is
    # always the PEP 440 release segment of the current dev seed, independent
    # of the bump kind (which only governs the *next* dev cycle).
    current = read_version(args.init_file)
    release_version = release_from_current(current)
    if current == release_version:
        raise SystemExit(
            f"ERROR: __version__ is already '{release_version}' (no .dev suffix). "
            "Refusing to re-finalize. If a previous run partially succeeded, "
            "drop the orphan release commit/tag or bump version.py back to "
            f"'{release_version}dev0' before re-running."
        )
    write_version(args.init_file, release_version)
    finalize_changelog(args.changelog, release_version)
    print(f"{args.init_file} -> {release_version}")
    return 0


def cmd_next_dev(args: argparse.Namespace) -> int:
    current = read_version(args.init_file)
    info = compute_versions(current, args.bump)
    write_version(args.init_file, info["next_dev_version"])
    insert_next_dev_section(args.changelog, info["next_dev_base"])
    print(f"{args.init_file} -> {info['next_dev_version']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument(
        "--init-file",
        default=Path("braindecode/version.py"),
        type=Path,
        help="Path to the file holding the __version__ literal.",
    )
    paths.add_argument(
        "--changelog",
        default=Path("docs/whats_new.rst"),
        type=Path,
        help="Path to the RST changelog.",
    )

    # ``--bump`` governs the next dev cycle, so it is wired only into the
    # ``compute`` and ``next-dev`` subcommands. ``finalize`` is independent
    # of bump kind and accepting the flag there would be a silent footgun.
    bump = argparse.ArgumentParser(add_help=False)
    bump.add_argument(
        "--bump",
        default="minor",
        choices=("major", "minor", "patch"),
        help="How to bump the version for the next development cycle.",
    )

    # Guard against ``python -OO`` (or any caller that strips docstrings):
    # ``__doc__`` is ``None`` in that case and ``None.split(...)`` crashes.
    description = (__doc__ or "").split("\n", 1)[0]
    ap = argparse.ArgumentParser(description=description)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "compute",
        parents=[paths, bump],
        help="Print the planned versions as key=value lines.",
    )
    sub.add_parser(
        "finalize",
        parents=[paths],
        help="Write the release version to version.py and date the changelog heading.",
    )
    sub.add_parser(
        "next-dev",
        parents=[paths, bump],
        help="Write the next .dev0 version and add a blank changelog section.",
    )
    sub.add_parser(
        "check-changes", help="Check for changes since the last stable release."
    )
    dev = sub.add_parser("dev-build", help="Write this commit's development version.")
    dev.add_argument("--init-file", type=Path, default=Path("braindecode/version.py"))
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "check-changes":
        print(f"skip={str(not has_release_changes()).lower()}")
        return 0
    if args.cmd == "dev-build":
        print(dev_build(args.init_file))
        return 0
    if args.cmd == "compute":
        return cmd_compute(args)
    if args.cmd == "finalize":
        return cmd_finalize(args)
    if args.cmd == "next-dev":
        return cmd_next_dev(args)
    ap.error(f"unknown subcommand: {args.cmd}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
