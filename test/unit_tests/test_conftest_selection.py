"""Rules of the shared ``test/conftest.py``: slow and network tiers, caller state."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

pytest_plugins = ["pytester"]

CONFTEST = (Path(__file__).resolve().parents[1] / "conftest.py").read_text()


@pytest.fixture
def conftest_project(pytester, monkeypatch):
    """Run the copied conftest in a project with only pytest's built-in plugins."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    pytester.makeconftest(CONFTEST)
    return pytester


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

# Catalog cases with a model name work without network access.
@pytest.mark.parametrize("model", ["REVE"])
def test_catalog(model):
    pass
"""

NETWORK = {"test_network", "test_slow_network"}
OFFLINE = {"test_plain", "test_catalog[REVE]"}
ALL = OFFLINE | {"test_slow"} | NETWORK


@pytest.mark.parametrize(
    "args, passed, skipped",
    [
        ([], OFFLINE, NETWORK),
        # Network tests that are also slow (the Hub module) need no --run-slow.
        (["--run-network"], OFFLINE | NETWORK, set()),
        (["--run-slow"], OFFLINE | {"test_slow"}, NETWORK),
        (["-m", "slow"], {"test_slow"}, NETWORK - {"test_network"}),
        (["--run-slow", "--run-network"], ALL, set()),
    ],
    ids=["default", "run-network", "run-slow", "m-slow", "run-slow-and-network"],
)
def test_slow_and_network_selection(conftest_project, args, passed, skipped):
    conftest_project.makepyfile(test_marked=MARKED_TESTS)
    reprec = conftest_project.inline_run("-p", "no:cacheprovider", *args)
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


def test_acceleration_caller_state_restores_after_pass_and_failure(conftest_project):
    conftest_project.makepyfile(
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
    reprec = conftest_project.inline_run("-p", "no:cacheprovider")
    reprec.assertoutcome(passed=1, failed=1)
    assert _caller_state() == before
