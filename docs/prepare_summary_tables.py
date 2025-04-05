import glob
import math
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def gen_models_visualization(df: pd.DataFrame):
    sns.set_style("whitegrid")
    df["log_#Parameters"] = df["#Parameters"].map(math.log10)

    plt.figure(figsize=(15, 6))

    # Plot 1: Number of Parameters Comparison
    plt.subplot(1, 2, 1)
    sns.barplot(
        data=df.sort_values("#Parameters", ascending=False),
        x="#Parameters",
        y="Model",
        palette="viridis",
    )
    plt.xscale("log")
    plt.xlabel("Number of Parameters (log scale)", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.title("Model Complexity Comparison", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Plot 2: Model Distribution by Paradigm
    plt.subplot(1, 2, 2)

    df_paradigms = df.assign(Paradigms=df["Paradigm"].str.split(", ")).explode(
        "Paradigms"
    )
    paradigm_counts = df_paradigms["Paradigms"].value_counts()
    color_mapping = dict(
        zip(paradigm_counts, sns.color_palette("rocket", len(paradigm_counts)))
    )

    sns.barplot(
        x=paradigm_counts.values,
        y=paradigm_counts.index,
        palette=[color_mapping[v] for v in paradigm_counts.values],
    )
    plt.title("Model Distribution by Paradigm", fontsize=14)
    plt.xlabel("Number of Models", fontsize=12)
    plt.ylabel("Paradigm", fontsize=12)
    plt.grid(True, which="major", linestyle="--", alpha=0.5)
    plt.xticks(range(0, paradigm_counts.max() + 1, 1))
    plt.tight_layout()
    plt.savefig("_static/model/models_analysis.png", bbox_inches="tight")

    df.drop(columns="log_#Parameters", inplace=True)


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
        html_table = df.to_html(classes="sortable", index=False, escape=False)
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
