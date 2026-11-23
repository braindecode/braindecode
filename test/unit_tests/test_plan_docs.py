"""Unit tests for documentation CI planning, gallery cache invalidation, and summary tables."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan_docs = _load_script("plan_docs", ".github/scripts/plan_docs.py")
sync_cache = _load_script("sync_gallery_cache", ".github/scripts/sync_gallery_cache.py")
summary_tables = _load_script(
    "prepare_summary_tables", "docs/prepare_summary_tables.py"
)


def test_plan_skips_internal_body_test_and_whats_new_only_pr():
    old_py = (
        "class EEGNet:\n"
        '    """EEGNet docstring."""\n'
        "    def __init__(self, n_chans: int):\n"
        '        """Init docstring."""\n'
        "        self.x = n_chans\n"
        "    def forward(self, x):\n"
        "        return x + 1\n"
    )
    new_py = old_py.replace("return x + 1", "y = x * 2\n        return y - x + 1")

    plan = plan_docs.compute_plan(
        event_name="pull_request",
        ref="refs/pull/123/merge",
        head_ref="fix/internal-math",
        changed_entries=[
            ("M", "braindecode/models/eegnet.py"),
            ("M", "test/unit_tests/models/test_eegnet.py"),
            ("M", ".github/workflows/tests.yml"),
            ("M", "docs/whats_new.rst"),
        ],
        read_old_file=lambda p: old_py if p.endswith("eegnet.py") else "",
        read_new_file=lambda p: new_py if p.endswith("eegnet.py") else "",
    )
    assert plan["build"] == "false"
    assert plan["run_examples"] == ""


def test_plan_builds_noplot_when_public_docstring_or_signature_changes():
    old_py = 'def foo(x: int) -> int:\n    """Old doc."""\n    return x\n'
    new_py = 'def foo(x: int) -> int:\n    """Updated doc."""\n    return x\n'

    plan = plan_docs.compute_plan(
        event_name="pull_request",
        ref="refs/pull/124/merge",
        head_ref="doc/foo",
        changed_entries=[("M", "braindecode/functional/functions.py")],
        read_old_file=lambda _: old_py,
        read_new_file=lambda _: new_py,
    )
    assert plan["build"] == "true"
    assert plan["run_examples"] == ""
    assert plan["gallery_arg"] == "-D plot_gallery=0"
    assert "docs-gallery" not in plan["extras"]
    assert "sae" not in plan["extras"]


@pytest.mark.parametrize(
    "example_path,expect_sae",
    [
        ("examples/applied_examples/plot_dance_event_detection.py", False),
        ("examples/advanced_training/plot_sae_reve.py", True),
    ],
)
def test_plan_runs_only_changed_gallery_plots_and_gates_sae_extra(
    example_path, expect_sae
):
    stem = Path(example_path).stem
    plan = plan_docs.compute_plan(
        event_name="pull_request",
        ref="refs/pull/125/merge",
        head_ref="doc/example",
        changed_entries=[("M", example_path)],
        read_old_file=lambda _: "",
        read_new_file=lambda _: "",
    )
    assert plan["build"] == "true"
    assert plan["run_examples"] == stem
    assert stem in plan["gallery_arg"]
    assert "docs-gallery" in plan["extras"]
    assert (",sae" in plan["extras"]) is expect_sae
    assert plan["use_gallery_cache"] == "false"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs a POSIX shell")
def test_gallery_arg_survives_the_shell():
    # The docs Makefile hands SPHINXOPTS, with this argument, to /bin/sh unquoted.
    plan = plan_docs.compute_plan(
        event_name="pull_request",
        ref="refs/pull/125/merge",
        head_ref="doc/example",
        changed_entries=[
            ("M", "examples/model_building/plot_a.py"),
            ("M", "examples/advanced_training/plot_b.py"),
        ],
        read_old_file=lambda _: "",
        read_new_file=lambda _: "",
    )
    words = subprocess.run(
        ["sh", "-c", f"printf '%s\\n' {plan['gallery_arg']}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert words == ["-D", "sphinx_gallery_conf.filename_pattern=/(plot_a|plot_b)\\.py"]


@pytest.mark.parametrize(
    "event_name,ref,head_ref,expect_cache",
    [
        ("push", "refs/heads/master", "", "true"),
        ("push", "refs/tags/v2.1.0", "", "false"),
        ("schedule", "refs/heads/master", "", "false"),
        ("pull_request", "refs/pull/10/merge", "release/2.1.0", "false"),
    ],
)
def test_plan_full_gallery_modes_and_cache_policy(
    event_name, ref, head_ref, expect_cache
):
    plan = plan_docs.compute_plan(
        event_name=event_name,
        ref=ref,
        head_ref=head_ref,
        changed_entries=[],
        read_old_file=lambda _: None,
        read_new_file=lambda _: None,
    )
    assert plan["build"] == "true"
    assert plan["run_examples"] == "all"
    assert plan["gallery_arg"] == ""
    assert "sae" in plan["extras"]
    assert plan["use_gallery_cache"] == expect_cache


def test_sync_gallery_cache_selective_invalidation_and_notebook_provenance(tmp_path):
    docs_dir = tmp_path / "docs"
    examples_dir = tmp_path / "examples" / "model_building"
    subsec = docs_dir / "auto_examples" / "model_building"
    examples_dir.mkdir(parents=True)
    subsec.mkdir(parents=True)

    # Example A references EEGNet; Example B references ShallowFBCSPNet
    (examples_dir / "plot_a.py").write_text("from braindecode.models import EEGNet\n")
    (examples_dir / "plot_b.py").write_text(
        "from braindecode.models import ShallowFBCSPNet\n"
    )
    (subsec / "plot_a.py.md5").write_text("md5a")
    (subsec / "plot_b.py.md5").write_text("md5b")
    (subsec / "plot_a.codeobj.json").write_text(
        json.dumps({"EEGNet": [{"module": "braindecode.models.eegnet"}]})
    )
    (subsec / "plot_b.codeobj.json").write_text(
        json.dumps(
            {"ShallowFBCSPNet": [{"module": "braindecode.models.shallow_fbcsp"}]}
        )
    )

    nb_payload = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "# Install the development version of braindecode\n",
                    "%pip install git+https://github.com/braindecode/braindecode.git@1111111111111111111111111111111111111111",
                ],
            }
        ]
    }
    nb_bytes = json.dumps(nb_payload).encode("utf-8")
    (subsec / "plot_b.ipynb").write_bytes(nb_bytes)
    with zipfile.ZipFile(subsec / "plot_b_jupyter.zip", "w") as zf:
        zf.writestr("plot_b.ipynb", nb_bytes)

    new_sha = "2" * 40
    stats = sync_cache.prepare_cache(
        docs_dir=docs_dir,
        examples_dir=tmp_path / "examples",
        head_sha=new_sha,
        changed_files=["braindecode/models/eegnet.py"],
    )
    assert stats == {"invalidated": 1, "kept": 1, "updated_notebooks": 1}
    assert not (subsec / "plot_a.py.md5").exists()
    assert (subsec / "plot_b.py.md5").exists()
    assert new_sha in (subsec / "plot_b.ipynb").read_text()
    with zipfile.ZipFile(subsec / "plot_b_jupyter.zip", "r") as zf:
        assert new_sha in zf.read("plot_b.ipynb").decode("utf-8")


def test_prepare_summary_tables_stdlib_output_and_mtime_skip(tmp_path):
    target_dir = tmp_path / "generated"
    out_html = tmp_path / "models_analysis.html"
    summary_tables.main(
        str(REPO_ROOT / "braindecode" / "models"),
        str(target_dir),
        out_html=str(out_html),
    )

    table_file = target_dir / "models_summary_table.html"
    assert table_file.is_file()
    assert out_html.is_file()

    table_html = table_file.read_text(encoding="utf-8")
    assert '<table border="1" class="dataframe sd-table sortable">' in table_html
    assert "../generated/braindecode.models.EEGNet.html" in table_html
    assert "Plotly.newPlot(" in out_html.read_text(encoding="utf-8")

    # Set future mtime and sentinel comment to verify mtime fast-path skips rewrite
    sentinel = table_html + "\n<!-- cached -->"
    table_file.write_text(sentinel, encoding="utf-8")
    future = (
        max(
            (REPO_ROOT / "braindecode" / "models" / "summary.csv").stat().st_mtime,
            (REPO_ROOT / "docs" / "prepare_summary_tables.py").stat().st_mtime,
        )
        + 60
    )
    os.utime(table_file, (future, future))
    os.utime(out_html, (future, future))

    summary_tables.main(
        str(REPO_ROOT / "braindecode" / "models"),
        str(target_dir),
        out_html=str(out_html),
    )
    assert table_file.read_text(encoding="utf-8").endswith("<!-- cached -->")
