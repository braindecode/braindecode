"""Rules of the shared ``test/conftest.py``: slow and network tiers, caller state."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

pytest_plugins = ["pytester"]

CONFTEST = (Path(__file__).resolve().parents[1] / "conftest.py").read_text()

MARKED_TESTS = """
import pytest

def test_plain():
    pass

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.network
def test_network():
    pass

@pytest.mark.slow
@pytest.mark.network
def test_slow_network():
    pass
"""

NETWORK = {"test_network", "test_slow_network"}
ALL = {"test_plain", "test_slow"} | NETWORK


@pytest.mark.parametrize(
    "args, passed, skipped",
    [
        ([], {"test_plain"}, NETWORK),
        # Network tests that are also slow (the Hub module) need no --run-slow.
        (["--run-network"], {"test_plain"} | NETWORK, set()),
        (["--run-slow"], {"test_plain", "test_slow"}, NETWORK),
        (["-m", "slow"], {"test_slow"}, {"test_slow_network"}),
        (["--run-slow", "--run-network"], ALL, set()),
    ],
    ids=["default", "run-network", "run-slow", "m-slow", "run-slow-and-network"],
)
def test_slow_and_network_selection(pytester, args, passed, skipped):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_marked=MARKED_TESTS)
    reprec = pytester.inline_run("-p", "no:cacheprovider", "-p", "no:xdist", *args)
    outcomes = [{r.nodeid.split("::")[-1] for r in rs} for rs in reprec.listoutcomes()]
    assert outcomes == [passed, skipped, set()]
    # Whatever neither runs nor skips is deselected, never silently dropped.
    calls = reprec.getcalls("pytest_deselected")
    deselected = {item.name for call in calls for item in call.items}
    assert deselected == ALL - passed - skipped


def _caller_state():
    return (
        random.getstate(),
        np.random.get_state()[1].tobytes(),
        np.random.get_state()[2],
        torch.get_rng_state().tolist(),
        torch.get_num_threads(),
        torch.is_grad_enabled(),
    )


def test_acceleration_caller_state_restores_after_pass_and_failure(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        """
        import random

        import numpy as np
        import pytest
        import torch

        # Fixed targets: toggling would undo a missing restore in the next test.
        THREADS, GRAD = torch.get_num_threads() + 1, not torch.is_grad_enabled()

        @pytest.mark.parametrize("fail", [False, True])
        def test_recipe(acceleration_caller_state, fail):
            random.seed(72)
            np.random.seed(72)
            torch.manual_seed(72)
            torch.set_num_threads(THREADS)
            torch.set_grad_enabled(GRAD)
            assert not fail
        """
    )
    before = _caller_state()
    reprec = pytester.inline_run("-p", "no:cacheprovider", "-p", "no:xdist")
    reprec.assertoutcome(passed=1, failed=1)
    assert _caller_state() == before


MODEL_SELECTION_TESTS = """
import os
import pytest

class EEGNet:
    pass

class SAMBA:
    pass

class Deep4Net:
    pass

def test_unparametrized_contract():
    assert "EEGNet" in os.environ.get("BRAINDECODE_SELECT_MODELS", "")

@pytest.mark.parametrize("model_name", ["EEGNet", "SAMBA", "Deep4Net", "PopT"])
def test_by_string(model_name):
    pass

@pytest.fixture(
    params=[
        ("EEGNet", ["n_chans"], None),
        ("SAMBA", ["chs_info"], None),
        ("Deep4Net", ["n_chans"], None),
    ],
    ids=lambda p: p[0],
)
def model_tuple(request):
    return request.param

def test_by_fixture_tuple(model_tuple):
    pass

@pytest.mark.parametrize("model_cls", [EEGNet, SAMBA, Deep4Net])
def test_by_class(model_cls):
    pass
"""


def test_select_models_filters_all_parametrization_shapes_with_eegnet_canary(
    pytester, monkeypatch
):
    monkeypatch.delenv("BRAINDECODE_SELECT_MODELS", raising=False)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_models_sel=MODEL_SELECTION_TESTS)
    reprec = pytester.inline_run(
        "-p",
        "no:cacheprovider",
        "-p",
        "no:xdist",
        "--select-models=SAMBA,PopulationTransformer",
    )
    passed = {r.nodeid.split("::")[-1] for r in reprec.listoutcomes()[0]}
    expected_passed = {
        "test_unparametrized_contract",
        "test_by_string[EEGNet]",
        "test_by_string[SAMBA]",
        "test_by_string[PopT]",
        "test_by_fixture_tuple[EEGNet]",
        "test_by_fixture_tuple[SAMBA]",
        "test_by_class[EEGNet]",
        "test_by_class[SAMBA]",
    }
    assert passed == expected_passed
    calls = reprec.getcalls("pytest_deselected")
    deselected = {item.name for call in calls for item in call.items}
    assert deselected == {
        "test_by_string[Deep4Net]",
        "test_by_fixture_tuple[Deep4Net]",
        "test_by_class[Deep4Net]",
    }


def test_select_models_reads_env_var_fallback(pytester, monkeypatch):
    monkeypatch.setenv("BRAINDECODE_SELECT_MODELS", "SAMBA")
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_models_env=MODEL_SELECTION_TESTS)
    reprec = pytester.inline_run("-p", "no:cacheprovider", "-p", "no:xdist")
    passed = {r.nodeid.split("::")[-1] for r in reprec.listoutcomes()[0]}
    assert "test_by_string[SAMBA]" in passed
    assert "test_by_string[EEGNet]" in passed
    assert "test_by_string[Deep4Net]" not in passed
