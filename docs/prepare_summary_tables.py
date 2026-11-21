"""Generate ``models_summary_table.html`` and ``models_analysis.html`` from CSV.

Uses only the Python standard library (``csv``, ``json``, ``html``) with an
mtime fast-path so incremental Sphinx builds skip regeneration when inputs are
unchanged and do not require ``pandas`` or ``plotly`` at build time.
"""

from __future__ import annotations

import csv
import html
import json
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_DOCS_DIR = Path(__file__).resolve().parent
_DEFAULT_OUT_HTML = "_static/model/models_analysis.html"
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_FONT_FAMILY = (
    "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
)

_TABLE_COLUMNS = (
    "Model",
    "Application",
    "Modality",
    "Type",
    "Categorization",
    "Sampling Frequency (Hz)",
    "#Parameters",
    "Hyperparameters",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN check without math/pandas
        return True
    return str(value).strip() == ""


def _coerce_rows(rows_or_df: Any) -> list[dict[str, Any]]:
    if hasattr(rows_or_df, "to_dict"):
        return list(rows_or_df.to_dict("records"))
    return [dict(row) for row in rows_or_df]


def gen_models_visualization(
    rows_or_df: Iterable[dict[str, Any]],
    out_html: str = _DEFAULT_OUT_HTML,
    bar_color: str = "#3B82F6",
) -> str:
    rows = _coerce_rows(rows_or_df)

    replace_pipeline = {
        "EEGNet": "EEGNet",
        "TSception": "TSception",
        "ShallowFBCSPNet": "ShallowNet",
        "AttentionBaseNet": "AttnBaseNet",
        "SleepStagerChambon2018": "SleepChambon2018",
        "SleepStagerBlanco2020": "SleepBlanco2020",
    }
    label_map = {
        "SignalJEPA_PostLocal": "sJEPA<sub>pos</sub>",
        "SignalJEPA_PreLocal": "sJEPA<sub>pre</sub>",
        "SignalJEPA_Contextual": "sJEPA<sub>con</sub>",
    }

    param_pairs: list[tuple[int | float, str]] = []
    app_counter: Counter[str] = Counter()

    for row in rows:
        raw_model = str(row.get("Model") or "").strip()
        model_label = replace_pipeline.get(raw_model, raw_model)

        raw_params = row.get("#Parameters")
        if not _is_missing(raw_params):
            try:
                num_val = float(str(raw_params).strip().replace(",", ""))
                param_val: int | float = (
                    int(num_val) if num_val.is_integer() else num_val
                )
                param_pairs.append((param_val, model_label))
            except ValueError:
                pass

        raw_app = row.get("Application")
        if not _is_missing(raw_app):
            for token in str(raw_app).split(","):
                cleaned = token.strip()
                if cleaned:
                    app_counter[cleaned] += 1

    param_pairs.sort(key=lambda item: item[0])
    param_x = [item[0] for item in param_pairs]
    param_y = [item[1] for item in param_pairs]

    sorted_apps = sorted(app_counter.items(), key=lambda kv: kv[1])
    app_y = [item[0] for item in sorted_apps]
    app_x = [item[1] for item in sorted_apps]
    par_alias = {p: p.replace(" ", "<br>") for p in app_y}

    row_px_left = 26
    row_px_right = 28
    dynamic_h = max(
        520, 120 + max(len(param_pairs) * row_px_left, len(sorted_apps) * row_px_right)
    )

    traces = [
        {
            "type": "bar",
            "x": param_x,
            "y": param_y,
            "orientation": "h",
            "marker": {"color": bar_color, "line": {"color": "#2b4ea1", "width": 0.8}},
            "hovertemplate": "<b>%{y}</b><br>#Params: %{x:,}<extra></extra>",
            "xaxis": "x",
            "yaxis": "y",
        },
        {
            "type": "bar",
            "x": app_x,
            "y": app_y,
            "orientation": "h",
            "text": app_x,
            "textposition": "auto",
            "marker": {"color": bar_color, "line": {"color": "#2b4ea1", "width": 0.8}},
            "hovertemplate": "%{y}: %{x}<extra></extra>",
            "xaxis": "x2",
            "yaxis": "y2",
        },
    ]

    # Match make_subplots(rows=1, cols=2, column_widths=[0.76, 0.24], horizontal_spacing=0.10):
    # usable width = 0.90 -> col 1 = [0.0, 0.684], col 2 = [0.784, 1.0].
    layout = {
        "height": dynamic_h,
        "bargap": 0.32,
        "margin": {"l": 140, "r": 30, "t": 60, "b": 50},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "showlegend": False,
        "font": {"family": _FONT_FAMILY, "size": 14},
        "uniformtext": {"minsize": 10, "mode": "hide"},
        "xaxis": {
            "domain": [0.0, 0.684],
            "anchor": "y",
            "type": "log",
            "title": {"text": "#Parameters (log scale)"},
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.18)",
            "gridwidth": 1,
            "minor": {
                "showgrid": True,
                "dtick": "D1",
                "gridcolor": "rgba(0,0,0,0.10)",
                "griddash": "dot",
                "gridwidth": 0.5,
            },
            "tickformat": "~s",
            "ticks": "outside",
            "ticklen": 6,
        },
        "yaxis": {
            "domain": [0.0, 1.0],
            "anchor": "x",
            "title": {"text": "Model"},
            "categoryorder": "array",
            "categoryarray": param_y,
            "labelalias": label_map,
            "automargin": True,
        },
        "xaxis2": {
            "domain": [0.784, 1.0],
            "anchor": "y2",
            "title": {"text": "Count"},
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.12)",
            "ticks": "outside",
            "ticklen": 6,
        },
        "yaxis2": {
            "domain": [0.0, 1.0],
            "anchor": "x2",
            "labelalias": par_alias,
            "automargin": True,
        },
        "annotations": [
            {
                "text": "Model Complexity (log₁₀ #Parameters)",
                "x": 0.342,
                "y": 1.0,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 16, "color": "#111", "family": _FONT_FAMILY},
            },
            {
                "text": "Models per Application",
                "x": 0.892,
                "y": 1.0,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 16, "color": "#111", "family": _FONT_FAMILY},
            },
        ],
    }

    config = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }

    if out_html == _DEFAULT_OUT_HTML:
        out_path = _DOCS_DIR / out_html
    else:
        out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snippet = (
        "<div>"
        '<script type="text/javascript">'
        "window.PlotlyConfig = {MathJaxConfig: 'local'};"
        "</script>"
        f'<script charset="utf-8" src="{_PLOTLY_CDN}"></script>'
        f'<div id="models-analysis" class="plotly-graph-div" '
        f'style="height:{dynamic_h}px; width:100%;"></div>'
        '<script type="text/javascript">'
        "window.PLOTLYENV=window.PLOTLYENV || {};"
        'if (document.getElementById("models-analysis")) {'
        "Plotly.newPlot("
        '"models-analysis", '
        f"{json.dumps(traces, ensure_ascii=False)}, "
        f"{json.dumps(layout, ensure_ascii=False)}, "
        f"{json.dumps(config)}"
        ");}"
        "</script>"
        "</div>\n"
    )
    out_path.write_text(snippet, encoding="utf-8")
    return str(out_path)


def wrap_tags(paradigm_cell: Any) -> str:
    if _is_missing(paradigm_cell):
        return ""
    paradigms = [
        f'<span class="tag">{html.escape(p.strip())}</span>'
        for p in str(paradigm_cell).split(",")
        if p.strip()
    ]
    return " ".join(paradigms)


def wrap_model_name(name: str) -> str:
    clean = str(name).strip()
    escaped = html.escape(clean)
    url = f"../generated/braindecode.models.{escaped}.html"
    return f'<a href="{url}">{escaped}</a>'


def wrap_hyperparameters(spec: str) -> str:
    """Turn 'n_chans, n_outputs, n_times,sfreq' into HTML with Font Awesome icons."""
    if _is_missing(spec):
        return ""

    icons = {
        "n_chans": ("wave-square", "n_chans"),
        "n_times": ("clock", "n_times"),
        "n_outputs": ("shapes", "n_outputs"),
        "sfreq": ("signal", "freq&nbsp;(Hz)"),
        "chs_info": ("circle-info", "chs_info"),
    }

    def fa(icon: str) -> str:
        return f'<span class="fa-solid fa-{icon}" aria-hidden="true"></span>'

    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]

    seen: set[str] = set()
    lines: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        if t in icons:
            icon, label = icons[t]
            lines.append(f"{fa(icon)}&nbsp;<strong>{label}</strong>")
        else:
            lines.append(f"<strong>{html.escape(t)}</strong>")

    return "<br/>".join(lines)


def wrap_categorization(spec: str, sep: str = "<br/>") -> str:
    """Render 'Convolution, Recurrent, Attention/Transformer' as colored pills."""
    if _is_missing(spec):
        return ""
    palette = {
        "Convolution": "success",
        "Recurrent": "secondary",
        "Attention/Transformer": "info",
        "Filterbank": "primary",
        "Interpretability": "warning",
        "SPD": "dark",
        "Foundation Model": "danger",
        "Graph Neural Network": "light",
        "Channel": "dark-line",
    }

    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        tone = palette.get(t, "secondary")
        out.append(f'<span class="tag tag-{tone}">{html.escape(t)}</span>')
    return sep.join(out)


def _render_summary_table(rows: list[dict[str, str]]) -> str:
    header_cells = "".join(
        f"      <th>{html.escape(col)}</th>\n" for col in _TABLE_COLUMNS
    )
    body_rows: list[str] = []
    for row in rows:
        cells = [
            wrap_model_name(row.get("Model", "")),
            wrap_tags(row.get("Application", "")),
            wrap_tags(row.get("Modality", "")),
            wrap_tags(row.get("Type", "")),
            wrap_tags(row.get("Categorization", "")),
            html.escape(str(row.get("Sampling Frequency (Hz)") or "").strip()),
            html.escape(str(row.get("#Parameters") or "").strip()),
            wrap_hyperparameters(row.get("Hyperparameters", "")),
        ]
        rendered_cells = "".join(f"      <td>{cell}</td>\n" for cell in cells)
        body_rows.append(f"    <tr>\n{rendered_cells}    </tr>\n")

    return (
        '<table border="1" class="dataframe sd-table sortable">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        f"{header_cells}"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        f"{''.join(body_rows)}"
        "  </tbody>\n"
        "</table>"
    )


def main(
    source_dir: str,
    target_dir: str,
    out_html: str = _DEFAULT_OUT_HTML,
    force: bool = False,
) -> None:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(Path(source_dir).glob("*.csv"))
    if not csv_files:
        return

    table_path = target_path / "models_summary_table.html"
    analysis_path = (
        _DOCS_DIR / out_html if out_html == _DEFAULT_OUT_HTML else Path(out_html)
    )

    if not force and table_path.is_file() and analysis_path.is_file():
        newest_input = max(
            *(f.stat().st_mtime for f in csv_files),
            Path(__file__).stat().st_mtime,
        )
        oldest_output = min(table_path.stat().st_mtime, analysis_path.stat().st_mtime)
        if oldest_output >= newest_input:
            return

    for csv_path in csv_files:
        print(f"Processing {csv_path} -> {table_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh, skipinitialspace=True)
            rows = [
                {
                    k: (v.strip() if isinstance(v, str) else "")
                    for k, v in row.items()
                    if k
                }
                for row in reader
            ]
        gen_models_visualization(rows, out_html=str(analysis_path))
        table_path.write_text(_render_summary_table(rows), encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.source_dir, args.target_dir, force=args.force)
    print(args.target_dir)
