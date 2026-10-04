"""Publish version-scoped Markdown siblings from a completed Sphinx build.

No imports of Braindecode, execution of examples, or network requests here.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlsplit

# Deliberately curated: these are entry points, not a claim of full-corpus coverage.
CRITICAL_PAGES = {
    "install/install_pip": "Installation from PyPI",
    "api": "API and model input conventions",
    "models/models": "The decoding problem",
    "models/models_table": "Model selection (interactive table in HTML)",
    "generated/braindecode.models.EEGNet": "EEGNet model reference",
    "generated/braindecode.models.ShallowFBCSPNet": "ShallowFBCSPNet model reference",
    "auto_examples/index": "Tutorials",
    "auto_examples/model_building/plot_bcic_iv_2a_moabb_trial": "Motor imagery tutorial",
    "cite": "Citing Braindecode",
}


def publish(markdown_root, html_root, *, site_url, version, revision):
    """Validate every advertised page before writing any public assets."""
    markdown_root, html_root = Path(markdown_root), Path(html_root)
    url = urlsplit(site_url)
    if (
        url.scheme != "https"
        or not url.netloc
        or url.query
        or url.fragment
        or not url.path.strip("/")
        or ".." in url.path.split("/")
    ):
        raise ValueError("site_url must be an HTTPS URL with an explicit version path")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be the full source commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9.+_-]+", version):
        raise ValueError("version must be a package version, not free text")
    site_url = site_url.rstrip("/") + "/"
    pages = {}
    for name in CRITICAL_PAGES:
        md = markdown_root / f"{name}.md"
        html = html_root / f"{name}.html"
        if not md.is_file() or not html.is_file():
            raise ValueError(f"Missing critical Markdown/HTML page: {name}")
        text = md.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty critical Markdown page: {name}")
        pages[name] = text
    # Sphinx-Gallery's thumbnail-only index does not provide ordinary Markdown
    # navigation. Keep the curated tutorial reachable without scraping raw HTML.
    pages["auto_examples/index"] += (
        "\n\n## Selected tutorial\n\n"
        "- [Motor imagery tutorial]"
        "(model_building/plot_bcic_iv_2a_moabb_trial.html)\n"
    )

    source = f"https://github.com/braindecode/braindecode/tree/{revision}"
    attribution = (
        f"Braindecode package version: {version}\n\n"
        f"Documentation scope: {site_url}\n\nSource commit: [{revision}]({source})\n\n"
    )
    index = (
        "# Braindecode\n\n"
        "> PyTorch-based deep learning for EEG, ECoG and MEG.\n\n"
        + attribution
        + "This index describes only this documentation build. The dev path is a "
        "moving development snapshot, even when the package version has no dev suffix. "
        "Match your installed version before using API examples; do not infer stable "
        "release behavior from dev documentation.\n\n"
        "Selected Markdown pages are rendered from canonical Sphinx documentation. "
        "HTML remains authoritative for interactive tables, figures and custom "
        "directives. Examples may download data or weights and run training; reading "
        "these pages does not execute them.\n\n## Documentation\n\n"
    )
    for name, title in CRITICAL_PAGES.items():
        target = html_root / f"{name}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            attribution
            + f"Canonical HTML: [{title}]({site_url}{name}.html)\n\n---\n\n"
            + pages[name]
            + "\n",
            encoding="utf-8",
        )
        index += f"- [{title}]({site_url}{name}.md)\n"
    (html_root / "llms.txt").write_text(index, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-root", type=Path, required=True)
    parser.add_argument("--html-root", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    publish(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
