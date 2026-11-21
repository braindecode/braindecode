"""Unit tests for the 2D deterministic CI diff planner (``scripts/ci_plan.py``)."""

from __future__ import annotations

import importlib.util
import json
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci_plan.py"


def _load_ci_plan() -> Any:
    spec = importlib.util.spec_from_file_location("ci_plan", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_plan"] = module
    spec.loader.exec_module(module)
    return module


ci_plan = _load_ci_plan()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(  # nosec B603, B607
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write(root: Path, rel: str, content: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Create a minimal synthetic git repository mimicking Braindecode's barrels."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@braindecode.org")
    _git(repo, "config", "user.name", "Test Bot")

    _write(
        repo,
        "braindecode/__init__.py",
        """from typing import TYPE_CHECKING
_SUBMODULES = frozenset({"models", "modules", "training"})
__all__ = ["EEGClassifier"]
if TYPE_CHECKING:
    from .classifier import EEGClassifier
""",
    )
    _write(repo, "braindecode/classifier.py", "class EEGClassifier:\n    pass\n")
    _write(
        repo,
        "braindecode/models/__init__.py",
        """from typing import TYPE_CHECKING
_LAZY: dict[str, str]
__all__ = ["EEGModuleMixin", "EEGNet", "SAMBA"]
if TYPE_CHECKING:
    from .base import EEGModuleMixin
    from .eegnet import EEGNet
    from .samba import SAMBA
else:
    _LAZY = {"EEGModuleMixin": "base", "EEGNet": "eegnet", "SAMBA": "samba"}
""",
    )
    _write(repo, "braindecode/models/base.py", "class EEGModuleMixin:\n    pass\n")
    _write(
        repo,
        "braindecode/models/eegnet.py",
        "from braindecode.modules import Conv2dWithConstraint\nclass EEGNet:\n    pass\n",
    )
    _write(repo, "braindecode/models/samba.py", "class SAMBA:\n    pass\n")
    _write(
        repo,
        "braindecode/models/util.py",
        """def build_model_config(cls):
    return {}

def _samba_helper():
    return {"n_times": 128}

models_mandatory_parameters = [
    ("EEGNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SAMBA", ["chs_info", "n_outputs"], _samba_helper),
]
""",
    )
    _write(
        repo,
        "braindecode/models/summary.csv",
        "Model,Chans,Freq\nEEGNet,22,250\nSAMBA,19,128\n",
    )
    _write(repo, "braindecode/models/model.html.jinja", "<div>{{ name }}</div>\n")
    _write(
        repo,
        "braindecode/modules/__init__.py",
        """from .convolution import Conv2dWithConstraint
from .activation import SafeLog
__all__ = ["Conv2dWithConstraint", "SafeLog"]
""",
    )
    _write(
        repo,
        "braindecode/modules/convolution.py",
        "class Conv2dWithConstraint:\n    pass\n",
    )
    _write(repo, "braindecode/modules/activation.py", "class SafeLog:\n    pass\n")
    _write(
        repo,
        "braindecode/training/__init__.py",
        """from .scoring import trial_preds_from_window_preds
__all__ = ["trial_preds_from_window_preds"]
""",
    )
    _write(
        repo,
        "braindecode/training/scoring.py",
        "def trial_preds_from_window_preds(preds):\n    return preds\n",
    )

    for rel, body in [
        (
            "test/unit_tests/models/test_integration.py",
            "from braindecode.models import EEGNet\n"
            "def test_act(model_class):\n"
            "    if model_class in [EEGNet]:\n"
            "        return\n",
        ),
        ("test/unit_tests/models/test_model_contract.py", "def test_c():\n    pass\n"),
        (
            "test/unit_tests/models/test_pretrained_compat.py",
            "def test_p():\n    pass\n",
        ),
        ("test/unit_tests/models/test_config.py", "def test_cfg():\n    pass\n"),
        ("test/unit_tests/models/test_models.py", "def test_m():\n    pass\n"),
        ("test/unit_tests/models/test_huggingface.py", "def test_hf():\n    pass\n"),
        ("test/unit_tests/models/test_base.py", "def test_b():\n    pass\n"),
        ("test/unit_tests/models/test_util.py", "def test_u():\n    pass\n"),
        (
            "test/unit_tests/models/test_samba.py",
            "from braindecode.models import SAMBA\ndef test_s():\n    assert SAMBA\n",
        ),
        (
            "test/unit_tests/models/test_eegnet_only.py",
            "from braindecode.models import EEGNet\ndef test_e():\n    assert EEGNet\n",
        ),
        (
            "test/unit_tests/modules/test_act.py",
            "from braindecode.modules import SafeLog\ndef test_sl():\n    assert SafeLog\n",
        ),
        (
            "test/unit_tests/training/test_scoring.py",
            "from braindecode.training import trial_preds_from_window_preds\n"
            "def test_sc():\n    assert trial_preds_from_window_preds(1) == 1\n",
        ),
        ("test/unit_tests/test_lazy_imports.py", "def test_li():\n    pass\n"),
        ("test/unit_tests/test_dependencies.py", "def test_dep():\n    pass\n"),
        (
            "test/unit_tests/test_compile_estimator.py",
            "import pytest\ndef test_fast():\n    pass\n"
            "@pytest.mark.slow\ndef test_inductor():\n    pass\n",
        ),
        (
            "test/integration_tests/test_distributed_windows.py",
            "import pytest\npytestmark = [pytest.mark.slow]\ndef test_dist():\n    pass\n",
        ),
    ]:
        _write(repo, rel, body)

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_full_suite_4_shard_partition_completeness_on_real_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Union of default + slow files across Shards 0..3 equals 100% of test files."""
    monkeypatch.delenv("BRAINDECODE_CI_SELECTION", raising=False)
    plan = ci_plan.compute_plan(
        checkout_root=REPO_ROOT,
        git_dir=REPO_ROOT,
        base_rev=None,
        head_rev="HEAD",
        force_full=True,
    )
    assert plan["full_suite"] is True
    assert plan["run_tests"] is True
    assert plan["shard_count"] == 4
    assert plan["shards"] == [0, 1, 2, 3]
    assert plan["select_models"] == ""
    assert plan["run_acceptance"] is True
    assert plan["run_docs"] is True

    shard_files = plan["shard_files"]
    slow_by_shard = plan["slow_files_by_shard"]

    # 1. Default files across shards 0..3 have 0 duplicates
    default_lists = [f for s in ("0", "1", "2", "3") for f in shard_files[s]]
    assert len(default_lists) == len(set(default_lists)), "Duplicate default test file!"

    # 2. Slow files across shards 0..3 have 0 duplicates
    slow_lists = [f for s in ("0", "1", "2", "3") for f in slow_by_shard[s]]
    assert len(slow_lists) == len(set(slow_lists)), "Duplicate slow test file!"
    assert set(slow_lists) == {
        "test/unit_tests/test_compile_estimator.py",
        "test/integration_tests/test_distributed_windows.py",
    }

    # 3. Per-shard combined sets (default | slow) are 100% pairwise disjoint across shards
    per_shard_unions = [
        set(shard_files[s]) | set(slow_by_shard[s]) for s in ("0", "1", "2", "3")
    ]
    for i in range(4):
        for j in range(i + 1, 4):
            overlap = per_shard_unions[i] & per_shard_unions[j]
            assert not overlap, f"Shards {i} and {j} overlap on {overlap}"

    # 4. Union across Shards 0..3 equals 100% of test/**/test_*.py (0 missing, 0 extra)
    expected_all = {
        p.relative_to(REPO_ROOT).as_posix()
        for pat in ("test_*.py", "*_test.py")
        for p in (REPO_ROOT / "test").rglob(pat)
        if p.is_file()
    }
    actual_union = set().union(*per_shard_unions)
    assert actual_union == expected_all

    # 5. Mixed-tier test_compile_estimator.py runs both its non-slow eager tests
    # and its @pytest.mark.slow inductor tests, while 100%-slow distributed_windows
    # runs only in -m slow.
    assert "test/unit_tests/test_compile_estimator.py" in set(default_lists)
    assert "test/integration_tests/test_distributed_windows.py" not in set(
        default_lists
    )


def test_barrel_symbol_resolution_isolates_unrelated_models_and_modules(
    mini_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAINDECODE_CI_SELECTION", raising=False)
    base = _git(mini_repo, "rev-parse", "HEAD")

    # 1. Edit braindecode/modules/activation.py (SafeLog) -> hits test_act.py, NOT EEGNet!
    _write(
        mini_repo,
        "braindecode/modules/activation.py",
        "class SafeLog:\n    eps = 1e-6\n",
    )
    _git(mini_repo, "commit", "-am", "edit SafeLog")
    head1 = _git(mini_repo, "rev-parse", "HEAD")
    plan1 = ci_plan.compute_plan(mini_repo, mini_repo, base, head1)
    files1 = set().union(*plan1["shard_files"].values())
    assert "test/unit_tests/modules/test_act.py" in files1
    assert "test/unit_tests/models/test_eegnet_only.py" not in files1
    assert plan1["select_models"] == ""

    # 2. Edit braindecode/modules/convolution.py (used by EEGNet via barrel) -> selects EEGNet!
    _write(
        mini_repo,
        "braindecode/modules/convolution.py",
        "class Conv2dWithConstraint:\n    max_norm = 1.0\n",
    )
    _git(mini_repo, "commit", "-am", "edit Conv2dWithConstraint")
    head2 = _git(mini_repo, "rev-parse", "HEAD")
    plan2 = ci_plan.compute_plan(mini_repo, mini_repo, head1, head2)
    files2 = set().union(*plan2["shard_files"].values())
    assert plan2["select_models"] == "EEGNet"
    assert "test/unit_tests/models/test_eegnet_only.py" in files2
    assert "test/unit_tests/models/test_samba.py" not in files2

    # 3. Add a new symbol to braindecode/training/__init__.py + scoring.py
    _write(
        mini_repo,
        "braindecode/training/scoring.py",
        "def trial_preds_from_window_preds(preds):\n    return preds\n"
        "def szcore_event_scores():\n    return {}\n",
    )
    _write(
        mini_repo,
        "braindecode/training/__init__.py",
        'from .scoring import szcore_event_scores, trial_preds_from_window_preds\n'
        '__all__ = ["szcore_event_scores", "trial_preds_from_window_preds"]\n',
    )
    _git(mini_repo, "commit", "-am", "add szcore_event_scores")
    head3 = _git(mini_repo, "rev-parse", "HEAD")
    plan3 = ci_plan.compute_plan(mini_repo, mini_repo, head2, head3)
    files3 = set().union(*plan3["shard_files"].values())
    assert plan3["full_suite"] is False
    assert "test/unit_tests/training/test_scoring.py" in files3
    assert "test/unit_tests/test_lazy_imports.py" in files3


def test_models_util_mandatory_params_vs_core_helper(
    mini_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAINDECODE_CI_SELECTION", raising=False)
    base = _git(mini_repo, "rev-parse", "HEAD")

    # 1. Edit only SAMBA's entry in models_mandatory_parameters -> surgical SAMBA!
    _write(
        mini_repo,
        "braindecode/models/util.py",
        """def build_model_config(cls):
    return {}

def _samba_helper():
    return {"n_times": 128}

models_mandatory_parameters = [
    ("EEGNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SAMBA", ["chs_info", "n_outputs", "n_times"], _samba_helper),
]
""",
    )
    _git(mini_repo, "commit", "-am", "update SAMBA mandatory params")
    head_mmp = _git(mini_repo, "rev-parse", "HEAD")
    plan_mmp = ci_plan.compute_plan(mini_repo, mini_repo, base, head_mmp)
    assert plan_mmp["full_suite"] is False
    assert plan_mmp["select_models"] == "SAMBA"

    # 2. Edit non-core helper _samba_helper referenced by SAMBA -> surgical SAMBA!
    _write(
        mini_repo,
        "braindecode/models/util.py",
        """def build_model_config(cls):
    return {}

def _samba_helper():
    return {"n_times": 256}

models_mandatory_parameters = [
    ("EEGNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SAMBA", ["chs_info", "n_outputs", "n_times"], _samba_helper),
]
""",
    )
    _git(mini_repo, "commit", "-am", "update _samba_helper")
    head_helper = _git(mini_repo, "rev-parse", "HEAD")
    plan_helper = ci_plan.compute_plan(mini_repo, mini_repo, head_mmp, head_helper)
    assert plan_helper["full_suite"] is False
    assert plan_helper["select_models"] == "SAMBA"

    # 3. Edit core helper build_model_config -> global 4-shard full-suite fallback!
    _write(
        mini_repo,
        "braindecode/models/util.py",
        """def build_model_config(cls):
    return {"changed": True}

def _samba_helper():
    return {"n_times": 256}

models_mandatory_parameters = [
    ("EEGNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SAMBA", ["chs_info", "n_outputs", "n_times"], _samba_helper),
]
""",
    )
    _git(mini_repo, "commit", "-am", "edit build_model_config")
    head_core = _git(mini_repo, "rev-parse", "HEAD")
    plan_core = ci_plan.compute_plan(mini_repo, mini_repo, head_helper, head_core)
    assert plan_core["full_suite"] is True
    assert plan_core["shard_count"] == 4
    assert plan_core["select_models"] == ""


def test_summary_csv_and_jinja_data_assets(
    mini_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAINDECODE_CI_SELECTION", raising=False)
    base = _git(mini_repo, "rev-parse", "HEAD")

    # 1. Single row diff in summary.csv
    _write(
        mini_repo,
        "braindecode/models/summary.csv",
        "Model,Chans,Freq\nEEGNet,22,250\nSAMBA,32,128\n",
    )
    _git(mini_repo, "commit", "-am", "update SAMBA row in summary.csv")
    head_csv = _git(mini_repo, "rev-parse", "HEAD")
    plan_csv = ci_plan.compute_plan(mini_repo, mini_repo, base, head_csv)
    assert plan_csv["full_suite"] is False
    assert plan_csv["select_models"] == "SAMBA"
    csv_files = set().union(*plan_csv["shard_files"].values())
    assert "test/unit_tests/models/test_integration.py" in csv_files

    # 2. Header/schema change in summary.csv disables model filter
    _write(
        mini_repo,
        "braindecode/models/summary.csv",
        "Model,Chans,Freq,Modality\nEEGNet,22,250,EEG\nSAMBA,32,128,EEG\n",
    )
    _git(mini_repo, "commit", "-am", "add Modality column")
    head_schema = _git(mini_repo, "rev-parse", "HEAD")
    plan_schema = ci_plan.compute_plan(mini_repo, mini_repo, head_csv, head_schema)
    assert plan_schema["select_models"] == ""

    # 3. model.html.jinja edit selects test_huggingface.py + test_base.py
    _write(
        mini_repo,
        "braindecode/models/model.html.jinja",
        "<section>{{ name }}</section>\n",
    )
    _git(mini_repo, "commit", "-am", "edit jinja template")
    head_jinja = _git(mini_repo, "rev-parse", "HEAD")
    plan_jinja = ci_plan.compute_plan(mini_repo, mini_repo, head_schema, head_jinja)
    jinja_files = set().union(*plan_jinja["shard_files"].values())
    assert jinja_files == {
        "test/unit_tests/models/test_huggingface.py",
        "test/unit_tests/models/test_base.py",
    }
    assert plan_jinja["select_models"] == "BIOT,EEGNet"


def test_deleted_and_renamed_files_and_inline_skip_lists(
    mini_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAINDECODE_CI_SELECTION", raising=False)
    base = _git(mini_repo, "rev-parse", "HEAD")

    # 1. Inline skip-list edit in test_integration.py alongside model edit keeps --select-models
    _write(mini_repo, "braindecode/models/samba.py", "class SAMBA:\n    v = 2\n")
    _write(
        mini_repo,
        "test/unit_tests/models/test_integration.py",
        "from braindecode.models import EEGNet, SAMBA\n"
        "def test_act(model_class):\n"
        "    if model_class in [EEGNet, SAMBA]:\n"
        "        return\n",
    )
    _git(mini_repo, "commit", "-am", "edit SAMBA and skip-list")
    head_skip = _git(mini_repo, "rev-parse", "HEAD")
    plan_skip = ci_plan.compute_plan(mini_repo, mini_repo, base, head_skip)
    assert plan_skip["select_models"] == "SAMBA"

    # 2. Real assertion change in test_integration.py disables --select-models
    _write(
        mini_repo,
        "test/unit_tests/models/test_integration.py",
        "from braindecode.models import EEGNet, SAMBA\n"
        "def test_act(model_class):\n"
        "    assert model_class is not None\n",
    )
    _git(mini_repo, "commit", "-am", "edit test_act body")
    head_logic = _git(mini_repo, "rev-parse", "HEAD")
    plan_logic = ci_plan.compute_plan(mini_repo, mini_repo, head_skip, head_logic)
    assert plan_logic["select_models"] == ""

    # 3. Rename a test file and delete another test file -> old/deleted paths never emitted
    _git(
        mini_repo,
        "mv",
        "test/unit_tests/modules/test_act.py",
        "test/unit_tests/modules/test_activation_renamed.py",
    )
    _git(mini_repo, "rm", "test/unit_tests/models/test_eegnet_only.py")
    _git(mini_repo, "commit", "-m", "rename and delete test files")
    head_mv = _git(mini_repo, "rev-parse", "HEAD")
    plan_mv = ci_plan.compute_plan(mini_repo, mini_repo, head_logic, head_mv)
    mv_files = set().union(*plan_mv["shard_files"].values())
    assert "test/unit_tests/modules/test_activation_renamed.py" in mv_files
    assert "test/unit_tests/modules/test_act.py" not in mv_files
    assert "test/unit_tests/models/test_eegnet_only.py" not in mv_files

    # 4. Delete a model file -> triggers cross-model completeness tests with no filter
    _git(mini_repo, "rm", "braindecode/models/samba.py")
    _git(mini_repo, "commit", "-m", "delete samba.py")
    head_del_model = _git(mini_repo, "rev-parse", "HEAD")
    plan_del_model = ci_plan.compute_plan(
        mini_repo, mini_repo, head_mv, head_del_model
    )
    del_files = set().union(*plan_del_model["shard_files"].values())
    assert plan_del_model["select_models"] == ""
    assert "test/unit_tests/models/test_integration.py" in del_files
    assert "test/unit_tests/test_lazy_imports.py" in del_files


def test_kill_switch_and_cli_github_output(
    mini_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    base = _git(mini_repo, "rev-parse", "HEAD")
    _write(mini_repo, "README.md", "# docs tweak\n")
    _git(mini_repo, "add", "README.md")
    _git(mini_repo, "commit", "-m", "readme only")
    head = _git(mini_repo, "rev-parse", "HEAD")

    # Without kill-switch: README-only diff -> 0 shards!
    monkeypatch.delenv("BRAINDECODE_CI_SELECTION", raising=False)
    gh_out_0 = tmp_path / "gh_output_0.txt"
    rc = ci_plan.main(
        [
            "--root",
            str(mini_repo),
            "--base",
            base,
            "--head",
            head,
            "--pretty",
            "--github-output",
            str(gh_out_0),
        ]
    )
    assert rc == 0
    stdout_0 = json.loads(capsys.readouterr().out)
    assert stdout_0["shard_count"] == 0
    assert stdout_0["run_tests"] is False
    out_kv_0 = dict(
        line.split("=", 1) for line in gh_out_0.read_text().strip().splitlines()
    )
    assert out_kv_0["shard_count"] == "0"
    assert out_kv_0["run_tests"] == "false"

    # With BRAINDECODE_CI_SELECTION=full kill-switch: forces 4-shard full suite!
    monkeypatch.setenv("BRAINDECODE_CI_SELECTION", "full")
    gh_out_full = tmp_path / "gh_output_full.txt"
    rc2 = ci_plan.main(
        [
            "--root",
            str(mini_repo),
            "--base",
            base,
            "--head",
            head,
            "--github-output",
            str(gh_out_full),
        ]
    )
    assert rc2 == 0
    stdout_full = json.loads(capsys.readouterr().out)
    assert stdout_full["full_suite"] is True
    assert stdout_full["shard_count"] == 4
    assert "BRAINDECODE_CI_SELECTION=full" in stdout_full["reason"]
    out_kv_full = dict(
        line.split("=", 1) for line in gh_out_full.read_text().strip().splitlines()
    )
    assert out_kv_full["full_suite"] == "true"
    assert out_kv_full["shard_count"] == "4"
