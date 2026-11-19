import doctest
import re
from importlib.metadata import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_landing_model_cards_open_filterable_models_table():
    landing_js = (REPO_ROOT / "docs/_static/landing.js").read_text(encoding="utf-8")
    models_table = (REPO_ROOT / "docs/models/models_table.rst").read_text(
        encoding="utf-8"
    )

    assert "models/models_table.html?model=" in landing_js
    assert 'Open " + m.name + " in the models table' in landing_js
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
    guide = REPO_ROOT / "docs/user_guide/channel_strategies.rst"
    result = doctest.testfile(str(guide), module_relative=False, raise_on_error=True)
    assert result.attempted > 0
