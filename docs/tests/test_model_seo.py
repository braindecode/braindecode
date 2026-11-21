"""Unit and Sphinx integration tests for ``docs/sphinxext/model_seo.py`` and root SEO files."""

from __future__ import annotations

import importlib.util
import json
import re
from io import StringIO
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DOCS_DIR.parent

_SEO_SPEC = importlib.util.spec_from_file_location(
    "model_seo", DOCS_DIR / "sphinxext" / "model_seo.py"
)
model_seo = importlib.util.module_from_spec(_SEO_SPEC)
_SEO_SPEC.loader.exec_module(model_seo)

_AGENT_SPEC = importlib.util.spec_from_file_location(
    "build_agent_docs", DOCS_DIR / "build_agent_docs.py"
)
agent_docs = importlib.util.module_from_spec(_AGENT_SPEC)
_AGENT_SPEC.loader.exec_module(agent_docs)


def _extract_jsonld_blocks(html_text: str) -> list[dict]:
    blocks = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        html_text,
        flags=re.DOTALL,
    )
    return [json.loads(b) for b in blocks]


def test_all_summary_csv_models_produce_150_to_190_char_descriptions():
    csv_path = REPO_ROOT / "braindecode" / "models" / "summary.csv"
    models = model_seo.load_summary_csv(csv_path)
    assert len(models) >= 100

    for name, row in models.items():
        desc = model_seo.build_model_description(row)
        assert 150 <= len(desc) <= 190, f"{name} length {len(desc)}: {desc!r}"
        assert desc.startswith(f"{name} is an open-source PyTorch ")
        assert "Braindecode" in desc
        assert "Examples using" not in desc
        assert f"{row['Sampling Frequency (Hz)']} Hz" in desc
        assert model_seo.format_params(row["#Parameters"]) in desc


def test_upsert_meta_tag_replaces_both_attribute_orders_without_duplicates():
    initial = (
        '<meta content="Examples using braindecode.models.EEGNet" name="description">\n'
        '<meta property="og:title" content="braindecode.models.EEGNet" />\n'
        '<meta content="Examples using braindecode.models.EEGNet" property="og:description" />\n'
    )
    updated = model_seo.upsert_meta_tag(
        initial, "name", "description", "Clean description."
    )
    updated = model_seo.upsert_meta_tag(
        updated, "property", "og:description", "Clean description."
    )

    assert len(re.findall(r'name="description"', updated)) == 1
    assert len(re.findall(r'property="og:description"', updated)) == 1
    assert "Examples using" not in updated
    assert '<meta name="description" content="Clean description." />' in updated
    assert '<meta property="og:description" content="Clean description." />' in updated


@pytest.mark.parametrize(
    "is_stable,deploy_url,expected_robots",
    [
        (
            False,
            "https://braindecode.org/dev/",
            "noindex, follow",
        ),
        (
            True,
            "https://braindecode.org/stable/",
            "index, follow, max-image-preview:large, max-snippet:-1",
        ),
    ],
)
def test_sphinx_html_integration_seo_tags_and_jsonld(
    tmp_path, is_stable, deploy_url, expected_robots
):
    from sphinx.application import Sphinx

    source = tmp_path / "source"
    out = tmp_path / "html"
    doctrees = tmp_path / "doctrees"
    (source / "generated").mkdir(parents=True)
    (source / "auto_examples" / "model_building").mkdir(parents=True)

    ext_dir = str(DOCS_DIR / "sphinxext")
    csv_path = str(REPO_ROOT / "braindecode" / "models" / "summary.csv")
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {ext_dir!r})\n"
        'project = "Braindecode"\n'
        'release = "2.1.0"\n'
        'extensions = ["sphinxext.opengraph", "model_seo"]\n'
        'html_baseurl = "https://braindecode.org/stable/"\n'
        f"ogp_site_url = {deploy_url!r}\n"
        'ogp_canonical_url = "https://braindecode.org/stable/"\n'
        'ogp_site_name = "Braindecode"\n'
        f'ogp_image = "{deploy_url}_static/braindecode.png"\n'
        "ogp_enable_meta_description = True\n"
        f"model_seo_summary_csv = {csv_path!r}\n"
        "html_context = {\n"
        f"    'docs_url': {deploy_url!r},\n"
        "    'canonical_base_url': 'https://braindecode.org/stable/',\n"
        f"    'is_stable_build': {is_stable!r},\n"
        "}\n",
        encoding="utf-8",
    )

    (source / "index.rst").write_text(
        ".. meta::\n"
        "   :description: Homepage custom meta description for Braindecode.\n\n"
        "Braindecode\n"
        "===========\n\n"
        ".. toctree::\n"
        "   :hidden:\n\n"
        "   generated/braindecode.models.EEGNet\n"
        "   generated/braindecode.models.FNIRSNet\n"
        "   auto_examples/index\n",
        encoding="utf-8",
    )

    # Model with gallery backreference fallback text AND in PAGE_SPECS
    (source / "generated" / "braindecode.models.EEGNet.rst").write_text(
        "braindecode.models.EEGNet\n"
        "=========================\n\n"
        "Examples using ``braindecode.models.EEGNet``: Motor imagery decoding.\n",
        encoding="utf-8",
    )

    # Model with no prose paragraph outside <dl> (sphinxext.opengraph misses it) AND not in PAGE_SPECS
    (source / "generated" / "braindecode.models.FNIRSNet.rst").write_text(
        "braindecode.models.FNIRSNet\n===========================\n",
        encoding="utf-8",
    )

    (source / "auto_examples" / "index.rst").write_text(
        "Examples\n========\n\n.. toctree::\n\n"
        "   model_building/index\n"
        "   sg_execution_times\n",
        encoding="utf-8",
    )
    (source / "auto_examples" / "sg_execution_times.rst").write_text(
        "Computation times\n=================\n\n00:01.0\n",
        encoding="utf-8",
    )
    (source / "auto_examples" / "model_building" / "index.rst").write_text(
        "Model building\n==============\n\n.. toctree::\n\n"
        "   plot_bcic_iv_2a_moabb_trial\n",
        encoding="utf-8",
    )
    (
        source / "auto_examples" / "model_building" / "plot_bcic_iv_2a_moabb_trial.rst"
    ).write_text(
        "Basic trialwise decoding on BCIC IV 2a\n"
        "======================================\n\n"
        "End-to-end trialwise motor imagery decoding tutorial with Braindecode.\n",
        encoding="utf-8",
    )

    warnings = StringIO()
    app = Sphinx(
        str(source),
        str(source),
        str(out),
        str(doctrees),
        "html",
        status=StringIO(),
        warning=warnings,
        freshenv=True,
    )
    app.build()
    assert app.statuscode == 0, warnings.getvalue()

    # 1. Index page: has robots tag, canonical to /stable/, backfilled og:description, NO model_seo JSON-LD
    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert (
        '<link rel="canonical" href="https://braindecode.org/stable/index.html"'
        in index_html
    )
    assert f'<meta name="robots" content="{expected_robots}" />' in index_html
    assert _extract_jsonld_blocks(index_html) == []
    assert (
        '<meta property="og:description" content="Homepage custom meta description for Braindecode." />'
        in index_html
    )

    # 2. EEGNet page: updated <title>, single replaced description, .md link, model JSON-LD
    eegnet_html = (out / "generated" / "braindecode.models.EEGNet.html").read_text(
        encoding="utf-8"
    )
    assert "<title>EEGNet — PyTorch EEG Model" in eegnet_html
    assert (
        '<meta property="og:title" content="EEGNet — PyTorch EEG Model" />'
        in eegnet_html
    )
    assert (
        '<link rel="canonical" href="https://braindecode.org/stable/generated/braindecode.models.EEGNet.html"'
        in eegnet_html
    )
    assert (
        f'<link rel="alternate" type="text/markdown" href="{deploy_url}generated/braindecode.models.EEGNet.md" />'
        in eegnet_html
    )
    desc_matches = re.findall(
        r'<meta\s+name="description"\s+content="([^"]+)"\s*/>', eegnet_html
    )
    og_desc_matches = re.findall(
        r'<meta\s+property="og:description"\s+content="([^"]+)"\s*/>', eegnet_html
    )
    assert len(desc_matches) == 1
    assert len(og_desc_matches) == 1
    assert desc_matches[0] == og_desc_matches[0]
    assert 150 <= len(desc_matches[0]) <= 190
    assert "Examples using" not in desc_matches[0]

    eegnet_ld = _extract_jsonld_blocks(eegnet_html)
    assert len(eegnet_ld) == 1
    types = {item["@type"] for item in eegnet_ld[0]["@graph"]}
    assert types == {"BreadcrumbList", "SoftwareSourceCode"}
    code_node = next(
        item for item in eegnet_ld[0]["@graph"] if item["@type"] == "SoftwareSourceCode"
    )
    assert code_node["name"] == "braindecode.models.EEGNet"
    assert code_node["runtimePlatform"] == "PyTorch"
    assert (
        code_node["url"]
        == "https://braindecode.org/stable/generated/braindecode.models.EEGNet.html"
    )

    # 3. FNIRSNet page: inserted description where opengraph found none, NO .md link
    fnirs_html = (out / "generated" / "braindecode.models.FNIRSNet.html").read_text(
        encoding="utf-8"
    )
    assert "<title>FNIRSNet — PyTorch fNIRS Model" in fnirs_html
    assert 'type="text/markdown"' not in fnirs_html
    fnirs_desc = re.findall(
        r'<meta\s+name="description"\s+content="([^"]+)"\s*/>', fnirs_html
    )
    assert len(fnirs_desc) == 1
    assert 150 <= len(fnirs_desc[0]) <= 190
    assert "fNIRS" in fnirs_desc[0]

    # 4. Tutorial page: TechArticle + BreadcrumbList JSON-LD and .md link
    tut_html = (
        out / "auto_examples" / "model_building" / "plot_bcic_iv_2a_moabb_trial.html"
    ).read_text(encoding="utf-8")
    assert (
        f'<link rel="alternate" type="text/markdown" href="{deploy_url}auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.md" />'
        in tut_html
    )
    tut_ld = _extract_jsonld_blocks(tut_html)
    assert len(tut_ld) == 1
    assert {item["@type"] for item in tut_ld[0]["@graph"]} == {
        "BreadcrumbList",
        "TechArticle",
    }

    # 5. Gallery indexes & sg_execution_times do NOT emit TechArticle JSON-LD
    for non_tutorial in (
        out / "auto_examples" / "index.html",
        out / "auto_examples" / "model_building" / "index.html",
        out / "auto_examples" / "sg_execution_times.html",
    ):
        assert _extract_jsonld_blocks(non_tutorial.read_text(encoding="utf-8")) == []


def test_origin_root_seo_assets_and_agent_specs():
    assert not (DOCS_DIR / "_extra" / "robots.txt").exists()
    assert hasattr(agent_docs, "PAGE_SPECS")
    assert agent_docs.PAGE_SPECS == agent_docs.CRITICAL_PAGES

    robots = (DOCS_DIR / "_root" / "robots.txt").read_text(encoding="utf-8")
    assert "Content-Signal: ai-train=yes, search=yes, ai-input=yes" in robots
    assert "Allow: /stable/" in robots
    assert "Sitemap: https://braindecode.org/sitemap.xml" in robots
    for legacy in ("0.6", "0.7", "0.8", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"):
        assert f"Disallow: /{legacy}/" in robots
    assert "search.html" in robots
    for bot in (
        "OAI-SearchBot",
        "GPTBot",
        "ChatGPT-User",
        "ClaudeBot",
        "PerplexityBot",
        "Bingbot",
    ):
        assert f"User-agent: {bot}" in robots

    sitemap_xml = (DOCS_DIR / "_root" / "sitemap.xml").read_text(encoding="utf-8")
    assert sitemap_xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert (
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        in sitemap_xml
    )
    assert "</sitemapindex>" in sitemap_xml
    locs = re.findall(r"<loc>([^<]+)</loc>", sitemap_xml)
    assert locs == ["https://braindecode.org/stable/sitemap.xml"]

    root_index = (DOCS_DIR / "_root" / "index.html").read_text(encoding="utf-8")
    assert 'content="0; url=https://braindecode.org/stable/index.html"' in root_index
    assert (
        '<link rel="canonical" href="https://braindecode.org/stable/index.html" />'
        in root_index
    )
    assert (
        '<link rel="alternate" type="text/markdown" href="https://braindecode.org/llms.txt" />'
        in root_index
    )
    assert ".png" in root_index
    assert ".svg" not in root_index
