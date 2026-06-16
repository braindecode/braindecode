import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that require network access",
    )


def pytest_collection_modifyitems(config, items):
    skip_network = pytest.mark.skip(
        reason="need --run-network option to run network-dependent tests"
    )
    run_network = config.getoption("--run-network")

    for item in items:
        callspec = getattr(item, "callspec", None)
        params = getattr(callspec, "params", {}).values()
        if any(_is_reve_param(param) for param in params):
            item.add_marker(pytest.mark.network)
            item.add_marker(pytest.mark.huggingface)

        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)


def _is_reve_param(param):
    if isinstance(param, str):
        return param == "REVE"

    name = getattr(param, "__name__", None)
    if name == "REVE":
        return True

    return isinstance(param, (tuple, list)) and len(param) > 0 and param[0] == "REVE"
