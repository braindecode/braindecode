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
        # Network tests, even slow ones (the Hub module), follow --run-network alone.
        if "network" in item.keywords:
            if not run_network:
                item.add_marker(skip_network)
        elif "slow" in item.keywords and not run_slow:
            deselected.append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item not in deselected]


@pytest.fixture(scope="session", autouse=True)
def offline_reve_positions(request, tmp_path_factory):
    """Use one position bank for construction and reload, including module fixtures."""
    if request.config.getoption("--run-network"):
        yield
        return

    import json

    import mne

    from braindecode.util import resolve_montage_name

    positions = {}
    for name in (resolve_montage_name("standard_1005"), "GSN-HydroCel-128"):
        montage = mne.channels.make_standard_montage(name)
        positions.update(montage.get_positions()["ch_pos"])
    folder = tmp_path_factory.mktemp("reve_positions")
    (folder / "reve_positions.json").write_text(
        json.dumps({name: xyz.tolist() for name, xyz in positions.items()})
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("REVE_POSITIONS_PATH", str(folder))
        yield


@pytest.fixture
def eeg_recordings():
    """Two independent recordings with EEG, EOG and two annotated trials."""
    import mne
    import numpy as np

    from braindecode.datasets import BaseConcatDataset, RawDataset

    rng = np.random.default_rng(42)
    info = mne.create_info(["C3", "Cz", "C4", "EOG"], 250, ["eeg"] * 3 + ["eog"])
    recordings = []
    for subject in (1, 2):
        raw = mne.io.RawArray(
            rng.normal(scale=1e-6, size=(4, 3000)), info, verbose=False
        )
        raw.set_annotations(mne.Annotations([1.0, 7.0], [2.0, 2.0], ["left", "right"]))
        recordings.append(
            RawDataset(raw, {"subject": subject, "session": "0", "run": "0"})
        )
    return BaseConcatDataset(recordings)
