import pytest


@pytest.fixture
def acceleration_caller_state():
    """Restore the RNG, thread and grad-mode state a test changes, even if it fails."""
    import random

    import numpy as np
    import torch

    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng, threads = torch.get_rng_state(), torch.get_num_threads()
    grad_enabled = torch.is_grad_enabled()
    # Do not initialize CUDA just to snapshot it.
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    yield
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    torch.set_num_threads(threads)
    torch.set_grad_enabled(grad_enabled)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="include slow compiler and study qualification tests (or use -m slow)",
    )
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
    # Mark REVE-parametrized items as network before the slow deselection reads
    # their keywords, so a slow REVE test follows the --run-network rule too.
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = getattr(callspec, "params", {}).values()
        if any(_is_reve_param(param) for param in params):
            item.add_marker(pytest.mark.network)
            item.add_marker(pytest.mark.huggingface)

    # Explicit marker selection remains a standard way to run just this tier.
    run_slow = config.getoption("--run-slow") or "slow" in config.option.markexpr
    if not run_slow:
        # Network tests (e.g. the Hub integration module, also marked slow)
        # stay governed by --run-network alone, as before the slow tier.
        def is_slow_only(item):
            return "slow" in item.keywords and "network" not in item.keywords

        selected = [item for item in items if not is_slow_only(item)]
        deselected = [item for item in items if is_slow_only(item)]
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    for item in items:
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)


def _is_reve_param(param):
    """Return whether a parametrized value identifies the REVE model."""
    if isinstance(param, str):
        return param == "REVE"

    name = getattr(param, "__name__", None)
    if name == "REVE":
        return True

    return isinstance(param, (tuple, list)) and len(param) > 0 and param[0] == "REVE"
