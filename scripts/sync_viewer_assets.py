#!/usr/bin/env python
# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Vendor the eegdash-viewer static assets into the braindecode package.

The viewer (https://github.com/eegdash/eegdash-viewer) is plain ES
scripts with no build step, so "vendoring" is a filtered copy into
``braindecode/datasets/viewer_static/`` where ``_viewer_server`` looks
for it. Re-run whenever the upstream viewer changes.

Usage:
    python scripts/sync_viewer_assets.py --src ../eegdash/eegdash-viewer
"""

import argparse
import shutil
from pathlib import Path

# Top-level entries the running viewer needs. Everything else (tests,
# benches, mutation sandboxes, node_modules) stays out of the wheel.
RUNTIME_ENTRIES = (
    "index.html",
    "styles.css",
    "viewer.js",
    "worker.js",
    "traces.js",
    "filters.js",
    "perf.js",
    "bids-loader.js",
    "bids-recording.js",
    "pose-panel.js",
    "eegdash-logo.svg",
    "viewer",
    "traces",
    "formats",
    "bids-recording",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="Path to an eegdash-viewer checkout.",
    )
    args = parser.parse_args()

    src = Path(args.src).resolve()
    if not (src / "index.html").is_file():
        raise SystemExit(f"{src} does not look like an eegdash-viewer checkout")

    dst = Path(__file__).resolve().parents[1] / "braindecode" / "datasets" / "viewer_static"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    n_files = 0
    for name in RUNTIME_ENTRIES:
        item = src / name
        if item.is_dir():
            shutil.copytree(
                item,
                dst / name,
                ignore=shutil.ignore_patterns("__pycache__", "node_modules"),
            )
        elif item.is_file():
            shutil.copy2(item, dst / name)
        else:
            print(f"warning: {name} missing upstream, skipped")
    n_files = sum(1 for f in dst.rglob("*") if f.is_file())
    print(f"vendored {n_files} files -> {dst}")


if __name__ == "__main__":
    main()
