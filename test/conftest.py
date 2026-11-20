import gc

import pytest


def pytest_collection_finish(session):
    """Freeze what the imports and the collection allocated, so that the
    collection after each test only walks the objects tests create."""
    if gc.get_freeze_count() == 0:  # once: not again in a pytester session
        gc.collect()
        gc.freeze()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Free what a test leaves in reference cycles before the next test runs.

    A model in a cycle (e.g. a bound method of itself kept as a hook or as
    ``forward``) outlives its test until a full collection, which Python runs
    rarely: 1.2 GB LUNA-Huge copies piled up in test_foundation_models.py.
    Collected after the protocol, once pytest has dropped the fixture values.
    """
    yield
    gc.collect()


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
        help="run tests marked slow (or select only them with -m slow)",
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
    run_slow = config.getoption("--run-slow") or "slow" in config.option.markexpr
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = getattr(callspec, "params", {}).values()
        # 69.5M-parameter REVE and 92.9M-parameter Pratibha2 builds exhaust the parallel job's memory.
        if any(_is_full_size_model(param) for param in params):
            item.add_marker(pytest.mark.slow)

        # Network tests, even slow ones (the Hub module), follow --run-network alone.
        if "network" in item.keywords:
            if not run_network:
                item.add_marker(skip_network)
        elif "slow" in item.keywords and not run_slow:
            deselected.append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item not in deselected]


_FULL_SIZE_MODELS = {"REVE", "Pratibha2"}


def _is_full_size_model(param):
    """Return whether a parametrized value identifies a model of
    ``_FULL_SIZE_MODELS``."""
    if isinstance(param, str):
        return param in _FULL_SIZE_MODELS

    name = getattr(param, "__name__", None)
    if name in _FULL_SIZE_MODELS:
        return True

    return (
        isinstance(param, (tuple, list))
        and len(param) > 0
        and isinstance(param[0], str)
        and param[0] in _FULL_SIZE_MODELS
    )
