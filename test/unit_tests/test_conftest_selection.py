"""Rules of the shared ``test/conftest.py``: slow and network tiers, caller state."""

import pickle
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

# conftest marks REVE-parametrized tests as network.
@pytest.mark.slow
@pytest.mark.parametrize("model", ["REVE"])
def test_slow_param(model):
    pass
"""

NETWORK = {"test_network", "test_slow_network", "test_slow_param[REVE]"}
ALL = {"test_plain", "test_slow"} | NETWORK


@pytest.mark.parametrize(
    "args, passed, skipped",
    [
        ([], {"test_plain"}, NETWORK),
        # Network tests that are also slow (the Hub module) need no --run-slow.
        (["--run-network"], {"test_plain"} | NETWORK, set()),
        (["--run-slow"], {"test_plain", "test_slow"}, NETWORK),
        (["-m", "slow"], {"test_slow"}, NETWORK - {"test_network"}),
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
        pickle.dumps(np.random.get_state()),
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

        @pytest.mark.parametrize("fail", [False, True])
        def test_recipe(acceleration_caller_state, fail):
            random.seed(72)
            np.random.seed(72)
            torch.manual_seed(72)
            torch.set_num_threads(2 if torch.get_num_threads() == 1 else 1)
            torch.set_grad_enabled(not torch.is_grad_enabled())
            assert not fail
        """
    )
    before = _caller_state()
    reprec = pytester.inline_run("-p", "no:cacheprovider", "-p", "no:xdist")
    reprec.assertoutcome(passed=1, failed=1)
    assert _caller_state() == before
