import ast
import doctest
import importlib.util
import posixpath
import re
from importlib.metadata import metadata
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import urldefrag

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_landing_model_cards_open_filterable_models_table():
    zoo_gen = (REPO_ROOT / "docs/sphinxext/zoo_data_gen.py").read_text(encoding="utf-8")
    models_table = (REPO_ROOT / "docs/models/models_table.rst").read_text(
        encoding="utf-8"
    )

    assert 'href = f"generated/braindecode.models.{quote(name)}.html"' in zoo_gen
    assert 'href="models/models_table.html"' in zoo_gen
    assert "new URLSearchParams(window.location.search).get('model')" in models_table
    assert "table.search(model).draw()" in models_table


def test_landing_visible_model_links_go_to_models_table():
    index = (REPO_ROOT / "docs/index.rst").read_text(encoding="utf-8")

    assert "models/models_table.html" in index
    assert "Browse the full zoo on the Models table" in index
    assert 'href="models/models.html"' not in index


def test_documented_model_extras_are_published():
    from braindecode.models.util import models_dict

    published = set(metadata("braindecode").get_all("Provides-Extra"))
    docs = [(REPO_ROOT / "docs/api.rst").read_text(encoding="utf-8")]
    docs.extend(model.__doc__ or "" for model in models_dict.values())
    for extras in re.findall(r"braindecode\[([\w., -]+)\]", "\n".join(docs)):
        assert {extra.strip() for extra in extras.split(",")} <= published


def test_bad_channel_guide():
    guide = REPO_ROOT / "docs/models/channel_strategies.rst"
    result = doctest.testfile(str(guide), module_relative=False, raise_on_error=True)
    assert result.attempted > 0


def test_html_redirects_stub_generation_and_guards(tmp_path):
    html_redirects = _load_module(
        "html_redirects", REPO_ROOT / "docs/sphinxext/html_redirects.py"
    )

    registered_values = []
    connected = []
    mock_setup_app = SimpleNamespace(
        add_config_value=lambda *args: registered_values.append(args),
        connect=lambda *args: connected.append(args),
    )
    meta = html_redirects.setup(mock_setup_app)
    assert meta["parallel_read_safe"] is True
    assert meta["parallel_write_safe"] is True
    assert ("html_redirects", {}, "html") in registered_values
    assert ("build-finished", html_redirects.write_html_redirects) in connected

    outdir = tmp_path / "html"
    outdir.mkdir()
    existing_file = outdir / "existing.html"
    existing_file.write_text("real page", encoding="utf-8")

    redirects = {
        "user_guide/channel_strategies.html": "../models/channel_strategies.html",
        "user_guide/cropped_inference.html": (
            "../auto_examples/model_building/"
            "plot_bcic_iv_2a_moabb_cropped.html#cropped-inference"
        ),
        "existing.html": "../models/channel_strategies.html",
    }
    config = SimpleNamespace(
        html_redirects=redirects,
        html_baseurl="https://braindecode.org/stable/",
    )
    env = SimpleNamespace(found_docs={"existing"})

    html_app = SimpleNamespace(
        builder=SimpleNamespace(format="html", name="html"),
        config=config,
        env=env,
        outdir=str(outdir),
    )
    html_redirects.write_html_redirects(html_app, None)

    ch_stub = (outdir / "user_guide/channel_strategies.html").read_text(
        encoding="utf-8"
    )
    assert '<meta charset="utf-8" />' in ch_stub
    assert '<meta name="robots" content="noindex, follow" />' in ch_stub
    assert (
        '<link rel="canonical" href="https://braindecode.org/stable/models/channel_strategies.html" />'
        in ch_stub
    )
    assert (
        '<meta http-equiv="refresh" content="0; url=../models/channel_strategies.html" />'
        in ch_stub
    )
    assert 'window.location.replace("../models/channel_strategies.html");' in ch_stub
    assert '<a href="../models/channel_strategies.html">' in ch_stub

    crop_stub = (outdir / "user_guide/cropped_inference.html").read_text(
        encoding="utf-8"
    )
    assert (
        '<link rel="canonical" href="https://braindecode.org/stable/'
        'auto_examples/model_building/plot_bcic_iv_2a_moabb_cropped.html" />'
        in crop_stub
    )
    assert "#cropped-inference" not in crop_stub.split("<link rel=\"canonical\"", 1)[1].split(">", 1)[0]
    assert (
        '<meta http-equiv="refresh" content="0; url=../auto_examples/model_building/'
        'plot_bcic_iv_2a_moabb_cropped.html#cropped-inference" />' in crop_stub
    )
    assert (
        'window.location.replace("../auto_examples/model_building/'
        'plot_bcic_iv_2a_moabb_cropped.html" + (window.location.hash || "#cropped-inference"));'
        in crop_stub
    )
    assert existing_file.read_text(encoding="utf-8") == "real page"

    # Non-HTML, epub, and failed-build runs must not write redirect stubs.
    for out_name, builder, exc in (
        ("markdown", SimpleNamespace(format="markdown", name="markdown"), None),
        ("epub", SimpleNamespace(format="html", name="epub"), None),
        ("failed", SimpleNamespace(format="html", name="html"), RuntimeError("boom")),
    ):
        other_out = tmp_path / out_name
        other_out.mkdir()
        app = SimpleNamespace(
            builder=builder,
            config=config,
            env=SimpleNamespace(found_docs=set()),
            outdir=str(other_out),
        )
        html_redirects.write_html_redirects(app, exc)
        assert not (other_out / "user_guide").exists()


@pytest.mark.parametrize(
    "bad_source",
    [
        "../escape.html",
        "/abs.html",
        "a/../../escape.html",
        "dir/",
        "",
        "sub\\escape.html",
    ],
)
def test_html_redirects_rejects_path_traversal(tmp_path, bad_source):
    html_redirects = _load_module(
        "html_redirects", REPO_ROOT / "docs/sphinxext/html_redirects.py"
    )
    outdir = tmp_path / "html"
    outdir.mkdir()
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", name="html"),
        config=SimpleNamespace(
            html_redirects={bad_source: "target.html"},
            html_baseurl="https://braindecode.org/stable/",
        ),
        env=SimpleNamespace(found_docs=set()),
        outdir=str(outdir),
    )
    with pytest.raises(ValueError):
        html_redirects.write_html_redirects(app, None)
    assert not (tmp_path / "escape.html").exists()


def test_html_redirects_sphinx_integration(tmp_path):
    sphinx_app = pytest.importorskip("sphinx.application")

    source = tmp_path / "source"
    source.mkdir()
    ext_dir = str(REPO_ROOT / "docs/sphinxext")
    (source / "index.rst").write_text(
        "Home\n====\n\n.. toctree::\n\n   existing\n", encoding="utf-8"
    )
    (source / "existing.rst").write_text(
        "Existing\n========\n\nKept page.\n", encoding="utf-8"
    )
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {ext_dir!r})\n"
        'project = "Braindecode"\n'
        'extensions = ["html_redirects"]\n'
        'html_baseurl = "https://braindecode.org/stable/"\n'
        "html_redirects = {\n"
        '    "user_guide/channel_strategies.html": "../models/channel_strategies.html",\n'
        '    "user_guide/cropped_inference.html": (\n'
        '        "../auto_examples/model_building/"\n'
        '        "plot_bcic_iv_2a_moabb_cropped.html#cropped-inference"\n'
        "    ),\n"
        '    "existing.html": "../models/channel_strategies.html",\n'
        "}\n",
        encoding="utf-8",
    )

    html_out = tmp_path / "html"
    app = sphinx_app.Sphinx(
        str(source),
        str(source),
        str(html_out),
        str(tmp_path / "doctrees_html"),
        "html",
        freshenv=True,
    )
    app.build()

    assert (html_out / "user_guide/channel_strategies.html").is_file()
    crop_html = (html_out / "user_guide/cropped_inference.html").read_text(
        encoding="utf-8"
    )
    assert "#cropped-inference" in crop_html
    existing_html = (html_out / "existing.html").read_text(encoding="utf-8")
    assert "Kept page." in existing_html
    assert 'http-equiv="refresh"' not in existing_html

    if importlib.util.find_spec("sphinx_markdown_builder") is not None:
        md_out = tmp_path / "markdown"
        md_app = sphinx_app.Sphinx(
            str(source),
            str(source),
            str(md_out),
            str(tmp_path / "doctrees_md"),
            "markdown",
            freshenv=True,
        )
        md_app.build()
        assert not (md_out / "user_guide").exists()


def test_conf_html_redirects_resolve_to_existing_docs():
    conf_tree = ast.parse((REPO_ROOT / "docs/conf.py").read_text(encoding="utf-8"))
    conf_values = {}
    for node in conf_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (
                "extensions",
                "html_redirects",
            ):
                conf_values[target.id] = ast.literal_eval(node.value)

    assert "html_redirects" in conf_values["extensions"]
    redirects = conf_values["html_redirects"]
    assert set(redirects) == {
        "user_guide/channel_strategies.html",
        "user_guide/cropped_inference.html",
    }

    for src, target in redirects.items():
        target_no_frag, frag = urldefrag(target)
        resolved = posixpath.normpath(
            str(PurePosixPath(src).parent / target_no_frag)
        )
        assert not resolved.startswith("..")
        assert resolved.endswith(".html")
        stem = resolved.removesuffix(".html")
        if stem.startswith("auto_examples/"):
            example_py = (
                REPO_ROOT / "examples" / (stem.removeprefix("auto_examples/") + ".py")
            )
            assert example_py.is_file()
            if frag:
                assert f".. _{frag}:" in example_py.read_text(encoding="utf-8")
        else:
            assert (REPO_ROOT / "docs" / f"{stem}.rst").is_file()


def test_ci_plan_docs_contract_targets():
    ci_plan = _load_module("ci_plan", REPO_ROOT / "scripts/ci_plan.py")
    targets = ci_plan.DOCS_CONTRACT_TARGETS
    assert "docs/models/channel_strategies.rst" in targets
    assert "docs/sphinxext/html_redirects.py" in targets
    assert "docs/user_guide/channel_strategies.rst" not in targets
    for rel_path in targets:
        assert (REPO_ROOT / rel_path).is_file(), f"Missing CI contract target: {rel_path}"
