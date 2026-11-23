"""Offline unit tests for pre-rendered model zoo HTML generation."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

DOCS_DIR = Path(__file__).parents[1]
REPO_ROOT = DOCS_DIR.parent
CSV_PATH = REPO_ROOT / "braindecode" / "models" / "summary.csv"

SPEC = importlib.util.spec_from_file_location(
    "zoo_data_gen", DOCS_DIR / "sphinxext" / "zoo_data_gen.py"
)
zoo_data_gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(zoo_data_gen)


def test_generate_zoo_data_writes_static_html(tmp_path):
    app = SimpleNamespace(srcdir=str(tmp_path))
    zoo_data_gen.generate_zoo_data(app)

    out_file = tmp_path / "generated" / "zoo_browser.html"
    assert out_file.is_file()
    html = out_file.read_text(encoding="utf-8")

    cards = re.findall(r'<a class="zoo-card"[^>]+>', html)
    assert len(cards) >= 100

    # Zero vague 'General' copy and zero purple badge classes
    assert "Prediction for General" not in html
    assert "for General" not in html
    assert "purple" not in html.lower()

    # Dual-axis controls, native <search> landmark, and progressive disclosure
    assert '<search class="zoo-search' in html
    assert 'id="bd-zoo-q"' in html
    assert 'id="bd-zoo-more"' in html
    assert "Show all architectures ↓" in html

    for mod_val in ("all", "eeg", "ieeg", "fnirs", "meg", "semg"):
        assert f'name="bd-zoo-mod" value="{mod_val}"' in html
    for cat_val in (
        "all",
        "foundation",
        "convolution",
        "attention",
        "channel",
        "recurrent",
        "filterbank",
        "interpretable",
        "graph",
    ):
        assert f'name="bd-zoo-cat" value="{cat_val}"' in html

    featured = re.findall(r'data-featured="1"', html)
    extra = re.findall(r'data-fold="extra"', html)
    assert len(featured) == 12
    assert len(featured) + len(extra) == len(cards)
    assert 'id="bd-zoo-pop"' in html
    assert "Channel &amp; 3D Montage-Aware" in html
    assert len(re.findall(r'class="zoo-card"[^>]*data-api-code="', html)) == len(cards)


def test_cards_follow_citation_count():
    models = zoo_data_gen.parse_summary_csv(str(CSV_PATH))
    assert set(zoo_data_gen.CITATIONS) == {m["name"] for m in models}
    featured = [m["name"] for m in models if m["featured"]]
    extra = [m["name"] for m in models if not m["featured"]]
    for names in (featured, extra):
        keys = [(-zoo_data_gen.CITATIONS[n], n) for n in names]
        assert keys == sorted(keys)


def test_cross_modal_and_links_match_api_reference():
    models = zoo_data_gen.parse_summary_csv(str(CSV_PATH))
    assert len(models) >= 100

    api_text = (DOCS_DIR / "api.rst").read_text(encoding="utf-8")
    api_models = set(re.findall(r"^\s+([A-Za-z0-9_]+)\s*$", api_text, re.M))

    html = zoo_data_gen.render_zoo_html(models)
    hrefs = re.findall(r'class="zoo-card" href="([^"]+)"', html)
    assert len(hrefs) == len(models)

    for href in hrefs:
        assert href.startswith("generated/braindecode.models.")
        assert href.endswith(".html")
        cls_name = href.removeprefix("generated/braindecode.models.").removesuffix(
            ".html"
        )
        assert cls_name in api_models, f"{cls_name} missing from docs/api.rst"

    # Cross-modal models expose space-separated modality tokens (e.g. 'eeg meg')
    by_name = {m["name"]: m for m in models}
    assert by_name["BrainOmni"]["mods"] == ["eeg", "meg"]
    assert 'data-mod="eeg meg"' in html
    assert "Cross-modal EEG + MEG" in by_name["BrainOmni"]["desc"]
