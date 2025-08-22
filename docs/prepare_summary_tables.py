import glob
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def gen_models_visualization(
    df: pd.DataFrame,
    out_html: str = "_static/model/models_analysis.html",
    bar_color: str = "#3B82F6",
):
    d = df.copy()

    replace_pipeline = {
        "EEGNetv4": "EEGNet",
        "TSceptionV1": "TSception",
        "ShallowFBCSPNet": "ShallowNet",
        "AttentionBaseNet": "AttnBaseNet",
        "SleepStagerChambon2018": "SleepChambon2018",
        "SleepStagerBlanco2020": "SleepBlanco2020",
        "SleepStagerEldele2021": "SleepEldele2021",
    }
    label_map = {
        "SignalJEPA_PostLocal": "sJEPA<sub>pos</sub>",
        "SignalJEPA_PreLocal": "sJEPA<sub>pre</sub>",
        "SignalJEPA_Contextual": "sJEPA<sub>con</sub>",
    }
    d["Model"] = d["Model"].replace(replace_pipeline)

    # Left: #Parameters per model (log x)
    d_params = (
        d[["Model", "#Parameters"]]
        .assign(**{"#Parameters": pd.to_numeric(d["#Parameters"], errors="coerce")})
        .dropna(subset=["#Parameters"])
        .sort_values("#Parameters", ascending=True)
    )

    # Right: models per paradigm
    def _split(s):
        if pd.isna(s):
            return []
        return [t.strip() for t in str(s).split(",") if t.strip()]

    d_par = d.assign(Paradigms=d["Paradigm"].apply(_split)).explode("Paradigms")
    counts = d_par["Paradigms"].value_counts().sort_values(ascending=True)

    # ---- dynamic height based on the tallest panel ----
    row_px_left = 26  # pixels per bar on left
    row_px_right = 28  # pixels per bar on right
    n_left = len(d_params)
    n_right = len(counts)
    dynamic_h = max(520, 120 + max(n_left * row_px_left, n_right * row_px_right))

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.10,
        subplot_titles=("Model Complexity (log₁₀ #Parameters)", "Models per Paradigm"),
    )

    # bars: left
    fig.add_trace(
        go.Bar(
            x=d_params["#Parameters"],
            y=d_params["Model"],
            orientation="h",
            marker=dict(color=bar_color, line=dict(color="#2b4ea1", width=0.8)),
            hovertemplate="<b>%{y}</b><br>#Params: %{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(labelalias=label_map, row=1, col=1)

    # bars: right
    fig.add_trace(
        go.Bar(
            x=counts.values,
            y=counts.index.tolist(),
            orientation="h",
            text=counts.values,
            textposition="auto",
            marker=dict(color=bar_color, line=dict(color="#2b4ea1", width=0.8)),
            hovertemplate="%{y}: %{x}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    # make "Motor Imagery" -> "Motor<br>Imagery", etc.
    par_alias = {p: p.replace(" ", "<br>") for p in counts.index}
    fig.update_yaxes(labelalias=par_alias, row=1, col=2)

    # Axes & grids
    fig.update_xaxes(
        type="log",
        title_text="#Parameters (log scale)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        griddash="solid",
        minor=dict(showgrid=True, dtick="D2"),
        minor_gridcolor="rgba(0,0,0,0.10)",
        minor_griddash="dot",
        tickformat="~s",
        ticks="outside",
        ticklen=6,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Model",
        categoryorder="array",
        categoryarray=d_params["Model"].tolist(),
        automargin=True,  # let labels push margins
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Count",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        ticks="outside",
        ticklen=6,
        row=1,
        col=2,
    )

    # Layout: bigger left margin for long model names + dynamic height
    fig.update_layout(
        height=dynamic_h,
        bargap=0.32,
        margin=dict(l=140, r=30, t=60, b=50),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
        ),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    # tidy subplot titles
    for ann in fig.layout.annotations or []:
        ann.font = dict(
            size=16,
            color="#111",
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        )

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        full_html=False,
        include_plotlyjs="cdn",
        div_id="models-analysis",
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    return str(out_path)


def wrap_tags(paradigm_cell: str):
    if pd.isna(paradigm_cell):
        return ""
    paradigms = [
        f'<span class="tag">{p.strip()}</span>' for p in paradigm_cell.split(", ")
    ]
    return " ".join(paradigms)


def wrap_model_name(name: str):
    # Remove any surrounding whitespace
    name = name.strip()
    # Construct the URL based on the model name
    url = f"generated/braindecode.models.{name}.html#braindecode.models.{name}"
    return f'<a href="{url}">{name}</a>'


def wrap_hyperparameters(spec: str) -> str:
    """Turn 'n_chans, n_outputs, n_times,sfreq' into HTML with Font Awesome icons,
    one per line. Use with DataFrame.to_html(escape=False).
    """
    if not spec:
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

    tokens = [t.strip() for t in spec.split(",") if t.strip()]

    seen, lines = set(), []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        if t in icons:
            icon, label = icons[t]
            lines.append(f"{fa(icon)}&nbsp;<strong>{label}</strong>")
        else:
            lines.append(f"<strong>{t}</strong>")  # fallback

    return "<br/>".join(lines)


def wrap_categorization(spec: str, sep: str = "<br/>") -> str:
    """Render 'Convolution, Recurrent, Small Attention' as colored <span class="tag ..."> pills.
    Use with DataFrame.to_html(escape=False).
    """
    if not spec:
        return ""
    # label -> tone (choose any names you like)
    palette = {
        "Convolution": "success",
        "Recurrent": "secondary",
        "Small Attention": "info",
        "Filterbank": "primary",
        "Interpretability": "warning",
        "SPD": "dark",
        "Large Language Model": "danger",
        "Graph Neural Network": "light",
        "Channel": "dark-line",  # outline
    }

    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    seen, out = set(), []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        tone = palette.get(t, "secondary")
        out.append(f'<span class="tag tag-{tone}">{t}</span>')
    return sep.join(out)


def main(source_dir: str, target_dir: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(Path(source_dir) / "*.csv"))
    for f in files:
        target_file = target_dir / Path(f).name
        print(f"Processing {f} -> {target_file}")
        df = pd.read_csv(f, index_col=False, header=0, skipinitialspace=True)
        df.drop(columns="get_#Parameters", inplace=True)
        gen_models_visualization(df)
        df["Model"] = df["Model"].apply(wrap_model_name)
        df["Paradigm"] = df["Paradigm"].apply(wrap_tags)
        df["Hyperparameters"] = df["Hyperparameters"].apply(wrap_hyperparameters)
        df["Categorization"] = df["Categorization"].str.replace(",", ", ")
        df["Categorization"] = df["Categorization"].apply(wrap_tags)
        df["Type"] = df["Type"].apply(wrap_tags)

        df = df[
            [
                "Model",
                "Paradigm",
                "Type",
                "Categorization",
                "Freq(Hz)",
                "#Parameters",
                "Hyperparameters",
            ]
        ]

        html_table = df.to_html(
            classes=["sd-table", "sortable"], index=False, escape=False
        )
        with open(
            f"{target_dir}/models_summary_table.html", "w", encoding="utf-8"
        ) as f:
            f.write(html_table)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()
    main(args.source_dir, args.target_dir)
    print(args.target_dir)
