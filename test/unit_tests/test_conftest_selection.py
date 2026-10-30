"""Selection rules of the shared ``test/conftest.py`` (slow and network tiers)."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"

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

INI = """
[pytest]
markers =
    network: tests that require network access
    slow: slow tests
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
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makeini(INI)
    pytester.makepyfile(test_marked=MARKED_TESTS)
    reprec = pytester.inline_run("-p", "no:cacheprovider", "-p", "no:xdist", *args)
    outcomes = {"passed": set(), "skipped": set(), "failed": set()}
    for report in reprec.getreports("pytest_runtest_logreport"):
        name = report.nodeid.split("::")[-1]
        if report.when == "call" and report.passed:
            outcomes["passed"].add(name)
        elif report.skipped:
            outcomes["skipped"].add(name)
        elif report.failed:
            outcomes["failed"].add(name)
    assert outcomes == {"passed": passed, "skipped": skipped, "failed": set()}
    # Whatever neither runs nor skips is deselected, never silently dropped.
    deselected = {
        item.name
        for call in reprec.getcalls("pytest_deselected")
        for item in call.items
    }
    assert deselected == ALL - passed - skipped
