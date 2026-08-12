#!/usr/bin/env python3
"""Robust release version bumper.
- Reads __version__ from --init-file (default: braindecode/version.py)
- Normalizes to base release (drops any .dev/a/b/rc/post/local)
- Appends a PEP 440 compliant .devN
  * If --pr is provided and numeric -> N = PR number
  * If --pr is provided but NOT numeric -> treat its value as a suffix seed
  * Else if --suffix provided -> N derived from suffix
  * Else -> error
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

try:
    from packaging.version import Version
except Exception:
    print(
        "ERROR: 'packaging' is required. Install with: pip install packaging",
        file=sys.stderr,
    )
    sys.exit(2)


def numeric_from_suffix(s: str) -> int:
    """Turn an arbitrary string into a stable positive integer for .devN.

    Strategy:
    - If purely digits -> int(s)
    - If it contains a hex-like run (e.g., git sha) of length>=5 -> use first 7 chars as hex
    - Otherwise -> sha1(s) and use first 8 hex chars as int
    """
    s = (s or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    m = re.search(r"([0-9a-fA-F]{5,})", s)
    if m:
        return int(m.group(1)[:7], 16)
    return int(hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16)


def read_current_version(init_path: Path) -> str:
    text = init_path.read_text(encoding="utf-8")
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        raise RuntimeError(f"__version__ not found in {init_path}")
    return m.group(1), text, m.start(1), m.end(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--init-file",
        default="braindecode/version.py",
        help="Path to version.py holding __version__",
    )
    # Make --pr a *string*, not int, so we can gracefully handle accidental non-numeric values.
    ap.add_argument(
        "--pr",
        required=False,
        help="PR number (or accidental string; non-numeric will be interpreted as suffix)",
    )
    ap.add_argument(
        "--suffix",
        required=False,
        help="Arbitrary suffix used to derive a numeric .devN when no numeric --pr is given",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print new version without writing file"
    )
    args = ap.parse_args()

    init_path = Path(args.init_file)
    if not init_path.exists():
        print(f"ERROR: Cannot find {init_path}", file=sys.stderr)
        return 1

    try:
        current, text, vstart, vend = read_current_version(init_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        base_release = ".".join(map(str, Version(current).release))
    except Exception as e:
        print(
            f"ERROR: Could not parse current version '{current}': {e}", file=sys.stderr
        )
        return 1

    # Determine N
    N = None
    if args.pr:
        if args.pr.isdigit():
            N = int(args.pr)
        else:
            # Be forgiving: treat as suffix
            N = numeric_from_suffix(args.pr)
            print(
                f"WARN: --pr value '{args.pr}' is not numeric; treating it as suffix → dev{N}",
                file=sys.stderr,
            )
    elif args.suffix:
        N = numeric_from_suffix(args.suffix)

    if N is None:
        print("ERROR: Either --pr or --suffix must be provided", file=sys.stderr)
        return 2

    new_version = f"{base_release}.dev{N}"

    new_text = text[:vstart] + new_version + text[vend:]

    if args.dry_run:
        print(new_version)
        return 0

    init_path.write_text(new_text, encoding="utf-8")
    print(new_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
