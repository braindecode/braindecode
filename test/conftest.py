import gc
import inspect
import os

import pytest

# Alias map for test registries whose parameter ID/key differs from cls.__name__
_MODEL_PARAM_ALIASES = {
    "PopT": "PopulationTransformer",
}


def _known_model_names() -> frozenset[str]:
    try:
        from braindecode import models

        return frozenset(models._LAZY) | frozenset(_MODEL_PARAM_ALIASES)
    except Exception:
        return frozenset(_MODEL_PARAM_ALIASES)


def _extract_item_model_name(item, known_models: frozenset[str]) -> str | None:
    """Return canonical model class name if ``item`` is parametrized over a model."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    for val in callspec.params.values():
        cand = None
        if isinstance(val, str) and val in known_models:
            cand = val
        elif isinstance(val, tuple) and len(val) >= 1:
            first = val[0]
            if isinstance(first, str) and first in known_models:
                cand = first
            elif inspect.isclass(first) and first.__name__ in known_models:
                cand = first.__name__
        elif inspect.isclass(val) and val.__name__ in known_models:
            cand = val.__name__
        if cand is not None:
            return _MODEL_PARAM_ALIASES.get(cand, cand)
    return None


def selected_models_set(config) -> frozenset[str] | None:
    """Return selected model names (+ ``EEGNet`` canary) when ``--select-models`` is set."""
    raw = config.getoption("--select-models", default="") or os.environ.get(
        "BRAINDECODE_SELECT_MODELS", ""
    )
    names = {m.strip() for m in raw.split(",") if m.strip()}
    if not names:
        return None
    return frozenset(names | {"EEGNet"})


def pytest_collection_finish(session):
    """Freeze what the imports and the collection allocated, so that the
    collection after each test only walks the objects tests create.

    Only the outermost session freezes: a pytester session runs inside a test,
    which pytest marks in ``PYTEST_CURRENT_TEST``. ``gc.get_freeze_count() == 0``
    cannot tell them apart, as CPython 3.12 starts with 375 objects frozen.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
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
    parser.addoption(
        "--select-models",
        action="store",
        default="",
        help=(
            "comma-separated model class names to select in parametrized model tests "
            "(always includes EEGNet as a baseline canary and enables "
            "test_model_compiled for the selected models)"
        ),
    )


def pytest_configure(config):
    raw_opt = config.getoption("--select-models", default="")
    if raw_opt:
        selected = selected_models_set(config)
        if selected is not None:
            os.environ["BRAINDECODE_SELECT_MODELS"] = ",".join(sorted(selected))


def pytest_collection_modifyitems(config, items):
    skip_network = pytest.mark.skip(
        reason="need --run-network option to run network-dependent tests"
    )
    run_network = config.getoption("--run-network")
    run_slow = config.getoption("--run-slow") or "slow" in config.option.markexpr
    selected_models = selected_models_set(config)
    known_models = _known_model_names() if selected_models is not None else frozenset()

    deselected = []
    for item in items:
        if selected_models is not None:
            item_model = _extract_item_model_name(item, known_models)
            if item_model is not None and item_model not in selected_models:
                deselected.append(item)
                continue
        # Network tests, even slow ones (the Hub module), follow --run-network alone.
        if "network" in item.keywords:
            if not run_network:
                item.add_marker(skip_network)
        elif "slow" in item.keywords and not run_slow:
            deselected.append(item)
    config.hook.pytest_deselected(items=deselected)
    deselected_set = set(deselected)
    items[:] = [item for item in items if item not in deselected_set]
