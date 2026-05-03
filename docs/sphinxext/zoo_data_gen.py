"""Generate `_static/zoo-data.js` from `braindecode/models/summary.csv`.

The landing page's interactive model-zoo browser used to ship a hand-curated
JS array of ~33 models. Single source of truth is the CSV that already feeds
the docs' Models Summary table, so we read it at build time and emit a JS
file that defines ``window.BD_MODELS`` / ``window.BD_CATEGORIES`` for the
landing's vanilla-JS browser to consume.
"""

from __future__ import annotations

import csv
import json
import os.path as op

from sphinx.util import logging

logger = logging.getLogger(__name__)

# Maps the CSV's "Categorization" tags (comma-separated, multi-tag per model)
# to the canonical category IDs used by the landing's chip filter.
# The first matched tag wins; a model with "Convolution,Attention/Transformer"
# is filed under "convolution" because that's its main building block.
CATEGORY_PRIORITY: list[tuple[str, str, str]] = [
    # (csv-tag-substring, id, label) — order = priority. A model tagged
    # "Convolution,Recurrent,Attention/Transformer" should land under
    # Attention because that is its most distinctive structural feature.
    ("Foundation Model", "foundation", "Foundation"),
    ("Filterbank", "filterbank", "Filterbank"),
    ("Graph Neural Network", "graph", "Graph"),
    ("GNN", "graph", "Graph"),
    ("Symmetric Positive", "spd", "SPD"),
    ("SPD", "spd", "SPD"),
    ("Attention", "attention", "Attention"),
    ("Transformer", "attention", "Attention"),
    ("Interpretab", "interpretable", "Interpretable"),
    ("Channel", "channel", "Channel"),
    ("Recurrent", "recurrent", "Recurrent"),
    ("Convolution", "convolution", "Convolution"),
]


def _classify(cat_field: str) -> tuple[str, str]:
    """Pick a primary (id, label) for a CSV "Categorization" cell."""
    if not cat_field:
        return ("other", "Other")
    for needle, cid, label in CATEGORY_PRIORITY:
        if needle.lower() in cat_field.lower():
            return (cid, label)
    return ("other", "Other")


def _format_params(raw: str) -> str:
    """Format a raw integer param-count as compact "1.2 M" / "47 K"."""
    try:
        n = int(raw.strip().replace(",", ""))
    except (TypeError, ValueError):
        return raw or ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M".replace(".0 ", " ")
    if n >= 1_000:
        return f"{n / 1_000:.1f} K".replace(".0 ", " ")
    return str(n)


def generate_zoo_data(app=None, *_args) -> None:
    """Sphinx ``config-inited`` hook. Writes ``docs/_static/zoo-data.js``."""
    here = op.dirname(op.abspath(__file__))
    csv_path = op.normpath(
        op.join(here, "..", "..", "braindecode", "models", "summary.csv")
    )
    out_path = op.normpath(op.join(here, "..", "_static", "zoo-data.js"))

    if not op.exists(csv_path):
        # Don't fail the build if the CSV is missing — landing.js falls back
        # to its inline list. The build still works for partial trees.
        logger.warning("zoo_data_gen: %s not found, skipping", csv_path)
        return

    # Skip the rebuild if zoo-data.js is already up to date with the CSV
    # *and* this generator script. Editing classifier rules in this file
    # without touching the CSV would otherwise leave stale JS in place.
    if op.exists(out_path):
        out_mtime = op.getmtime(out_path)
        if out_mtime >= op.getmtime(csv_path) and out_mtime >= op.getmtime(__file__):
            return

    models: list[dict] = []
    cat_counts: dict[str, dict] = {
        "all": {"id": "all", "label": "All", "color": "gray", "n": 0},
    }

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            name = (row.get("Model") or "").strip()
            if not name:
                continue
            # Skip the auto-channel-interpolation wrappers; they shadow the
            # base foundation models (BENDR / BIOT / LaBraM / SignalJEPA)
            # and aren't useful as separate cards on the landing page.
            if name.startswith("Interpolated"):
                continue
            cat_field = (row.get("Categorization") or "").strip()
            cid, clabel = _classify(cat_field)
            params = _format_params(row.get("#Parameters") or "")
            application = (row.get("Application") or "").strip()

            # `desc` falls back to Application + Type so the card always says
            # something useful even without a docstring summary.
            type_field = (row.get("Type") or "").strip()
            if application and type_field:
                desc = f"{type_field} for {application}"
            elif application:
                desc = application
            elif type_field:
                desc = type_field
            else:
                desc = "Published architecture"

            models.append(
                {
                    "name": name,
                    "cat": cid,
                    "year": None,
                    "params": params,
                    "paper": "",
                    "desc": desc,
                }
            )
            if cid not in cat_counts:
                cat_counts[cid] = {
                    "id": cid,
                    "label": clabel,
                    "color": _color_for(cid),
                    "n": 0,
                }
            cat_counts[cid]["n"] += 1
            cat_counts["all"]["n"] += 1

    # Order categories: All first, then by descending count.
    ordered = [cat_counts["all"]] + sorted(
        (v for k, v in cat_counts.items() if k != "all"),
        key=lambda c: -c["n"],
    )
    categories = [
        {"id": c["id"], "label": c["label"], "color": c["color"]} for c in ordered
    ]

    js = (
        "// Auto-generated by docs/sphinxext/zoo_data_gen.py from\n"
        "// braindecode/models/summary.csv. Edit the CSV, not this file.\n"
        f"window.BD_MODELS = {json.dumps(models, indent=2)};\n"
        f"window.BD_CATEGORIES = {json.dumps(categories, indent=2)};\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(js)


def _color_for(cid: str) -> str:
    """CSS modifier class (matches landing.css `.zoo-card-cat.<color>`)."""
    return {
        "convolution": "",
        "attention": "warm",
        "filterbank": "green",
        "foundation": "purple",
        "recurrent": "",
        "graph": "green",
        "channel": "warm",
        "interpretable": "",
        "spd": "purple",
        "other": "gray",
    }.get(cid, "")


def setup(app):
    app.connect("config-inited", generate_zoo_data)
    return {"version": "1.0", "parallel_read_safe": True}
