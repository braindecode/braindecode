"""Offline generator contract tests; no Braindecode imports or datasets."""

import importlib.util
import re
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_agent_docs", Path(__file__).parents[1] / "build_agent_docs.py"
)
agent_docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent_docs)


@pytest.fixture
def corpus(tmp_path):
    markdown, html = tmp_path / "markdown", tmp_path / "html"
    for name in agent_docs.CRITICAL_PAGES:
        for root, suffix in ((markdown, ".md"), (html, ".html")):
            path = root / (name + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Canonical content\n\nDo not invent model requirements.\n"
            )
    return dict(
        markdown_root=markdown,
        html_root=html,
        site_url="https://example.org/1.9/",
        version="1.9.0",
        revision="a" * 40,
    )


def test_version_scope_provenance_determinism_and_links(corpus):
    agent_docs.publish(**corpus)
    html = corpus["html_root"]
    before = {
        p.relative_to(html): p.read_bytes() for p in html.rglob("*") if p.is_file()
    }
    index = (html / "llms.txt").read_text()
    links = re.findall(r"\]\(https://example.org/1.9/([^)]*)\)", index)
    assert len(links) == len(agent_docs.CRITICAL_PAGES)
    for link in links:
        text = (html / link).read_text()
        assert "1.9.0" in text and "a" * 40 in text
        assert "# Canonical content" in text
        assert "https://example.org/1.9/" + link.replace(".md", ".html") in text
        assert "braindecode.org/dev" not in text
    agent_docs.publish(**corpus)
    assert before == {
        p.relative_to(html): p.read_bytes() for p in html.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("name", agent_docs.CRITICAL_PAGES)
@pytest.mark.parametrize("kind", ["markdown_root", "html_root"])
def test_every_missing_critical_page_blocks_publication(corpus, name, kind):
    suffix = ".md" if kind == "markdown_root" else ".html"
    (corpus[kind] / (name + suffix)).unlink()
    with pytest.raises(ValueError, match="Missing critical"):
        agent_docs.publish(**corpus)
    assert not (corpus["html_root"] / "llms.txt").exists()
    assert not list(corpus["html_root"].rglob("*.md"))


def test_empty_page_blocks_publication(corpus):
    (corpus["markdown_root"] / "cite.md").write_text(" ")
    with pytest.raises(ValueError, match="Empty critical"):
        agent_docs.publish(**corpus)


@pytest.mark.parametrize(
    "key,value",
    [
        ("site_url", "https://braindecode.org/"),
        ("site_url", "https://braindecode.org/dev/?version=stable"),
        ("site_url", "http://braindecode.org/dev/"),
        ("site_url", "https://braindecode.org/dev/../stable/"),
        ("revision", "master"),
        ("version", "1.9\nwrong"),
    ],
)
def test_invalid_provenance(corpus, key, value):
    corpus[key] = value
    with pytest.raises(ValueError):
        agent_docs.publish(**corpus)


def test_real_sphinx_builder_on_canonical_install_page(tmp_path):
    """Exercise the dependency on real RST, without the heavy project config."""
    from sphinx.application import Sphinx

    source = tmp_path / "source"
    source.mkdir()
    canonical = Path(__file__).parents[1] / "install" / "install_pip.rst"
    (source / "index.rst").write_text(canonical.read_text())
    (source / "links.inc").write_text(
        (Path(__file__).parents[1] / "links.inc").read_text()
    )
    (source / "help.rst").write_text("Help\n====\n")
    (source / "conf.py").write_text(
        'extensions = ["sphinx_markdown_builder"]\nmarkdown_uri_doc_suffix = ".html"\n'
    )
    app = Sphinx(
        str(source),
        str(source),
        str(tmp_path / "out"),
        str(tmp_path / "doctrees"),
        "markdown",
        freshenv=True,
    )
    app.build()
    text = (tmp_path / "out" / "index.md").read_text()
    assert "pip install" in text
    assert "braindecode" in text


def test_critical_pages_have_canonical_sources():
    """Catch drift in curated paths before the full gallery build."""
    docs = Path(__file__).parents[1]
    for name in agent_docs.CRITICAL_PAGES:
        if name.startswith("generated/"):
            model = name.removeprefix("generated/braindecode.models.")
            assert re.search(
                r"^ +" + model + r"$", (docs / "api.rst").read_text(), re.M
            )
        elif name == "auto_examples/index":
            assert (docs.parent / "examples" / "README.rst").is_file()
        elif name.startswith("auto_examples/"):
            assert (
                docs.parent / "examples" / (name.removeprefix("auto_examples/") + ".py")
            ).is_file()
        else:
            assert (docs / (name + ".rst")).is_file()


def test_gallery_index_keeps_curated_navigation(corpus):
    agent_docs.publish(**corpus)
    text = (corpus["html_root"] / "auto_examples/index.md").read_text()
    assert "(model_building/plot_bcic_iv_2a_moabb_trial.html)" in text


def test_html_then_markdown_gallery_integration(tmp_path):
    """Exercise real extension nodes and shared doctrees without any datasets."""
    from io import StringIO

    from sphinx.application import Sphinx

    source, examples = tmp_path / "source", tmp_path / "examples"
    source.mkdir()
    examples.mkdir()
    extension_dir = Path(__file__).parents[1] / "sphinxext"
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(extension_dir)!r})\n"
        'extensions = ["sphinx_markdown_builder", "agent_markdown", '
        '"sphinx_gallery.gen_gallery", "sphinx_design", "sphinxcontrib.bibtex"]\n'
        'markdown_uri_doc_suffix = ".html"\nmarkdown_anchor_sections = True\n'
        'bibtex_bibfiles = ["refs.bib"]\n'
        'sphinx_gallery_conf = {"examples_dirs": "../examples", '
        '"gallery_dirs": "auto_examples", "image_scrapers": ("matplotlib",), '
        '"reset_modules": (), "download_all_examples": False}\n'
    )
    (source / "refs.bib").write_text(
        "@article{one, author={A. Author}, title={Example reference}, "
        "journal={Test}, year={2026}}\n"
    )
    (source / "index.rst").write_text(
        "Test\n====\n\n.. toctree::\n\n   auto_examples/index\n   nested/page\n\n"
        ".. button-ref:: nested/page\n    :ref-type: doc\n\n    Nested page\n\n"
        ":cite:p:`one`\n\n.. bibliography::\n"
    )
    (source / "nested").mkdir()
    (source / "nested/page.rst").write_text(
        "Page\n====\n\n.. image:: ../sample.svg\n\n"
        ".. py:class:: Example(n_chans)\n\n    :param int n_chans: Channels.\n\n"
        ".. raw:: html\n\n    <script>unwanted_widget()</script>\n"
    )
    (source / "sample.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (examples / "README.txt").write_text("Examples\n========\n")
    (examples / "plot_test.py").write_text(
        '"""\nExample\n=======\n\nA no-data fixture.\n"""\n'
        'from pathlib import Path\np = Path("executions.txt")\n'
        'p.write_text(p.read_text() + "x" if p.exists() else "x")\nprint("ran")\n'
        "import matplotlib.pyplot as plt\nplt.plot([0, 1], [0, 1])\n"
    )
    for builder in ("html", "markdown"):
        warnings = StringIO()
        app = Sphinx(
            str(source),
            str(source),
            str(tmp_path / builder),
            str(tmp_path / "doctrees"),
            builder,
            confoverrides={"plot_gallery": 0} if builder == "markdown" else {},
            status=StringIO(),
            warning=warnings,
        )
        app.build()
        assert app.statuscode == 0
        assert "unknown node type" not in warnings.getvalue()
    assert (examples / "executions.txt").read_text() == "x"
    markdown = tmp_path / "markdown"
    page = (markdown / "nested/page.md").read_text()
    assert "![image](../_images/sample.svg)" in page
    assert (tmp_path / "html/_images/sample.svg").is_file()
    assert "unwanted_widget" not in page
    assert "n_chans" in page
    index = (markdown / "index.md").read_text()
    assert "nested/page.html" in index
    assert "Example reference" in index
    assert '<a id="id' in index
    assert "auto_examples/plot_test.html" in index
    tutorial = (markdown / "auto_examples/plot_test.md").read_text()
    assert "ran" in tutorial
    assert "../_images/sphx_glr_plot_test_001.png" in tutorial
    assert (tmp_path / "html/_images/sphx_glr_plot_test_001.png").is_file()
    assert '<a id="example"></a>' in tutorial
