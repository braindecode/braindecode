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
def offline_model_geometry(request, tmp_path_factory):
    """Supply REVE positions and the released ST-EEGFormer slot order offline."""
    if request.config.getoption("--run-network"):
        yield
        return

    import json
    from types import SimpleNamespace

    import mne

    from braindecode.models import steegformer
    from braindecode.util import resolve_montage_name

    positions = {}
    for name in (resolve_montage_name("standard_1005"), "GSN-HydroCel-128"):
        montage = mne.channels.make_standard_montage(name)
        positions.update(montage.get_positions()["ch_pos"])
    folder = tmp_path_factory.mktemp("model_geometry")
    (folder / "reve_positions.json").write_text(
        json.dumps({name: xyz.tolist() for name, xyz in positions.items()})
    )
    # The released small/base/large checkpoints and CORTEG share these slots.
    names = """
        C1 Pz C4 F6 FTT8h Oz Fp1 FCC5h TPP8h CPP6h C2 F4 OI2h AF4 FCz CCP6h TP8 POO10h FC1 FC6
        C5 P8 FT8 P6 P9 Fz AFF1 TPP10h AFF2 P10 CPP2h M1 FCC6h FTT7h FC2 PPO2 AFp3h AF7 PO10 AF8
        CPP1h P7 F1 AFp4h PO9 FT9 CP2 Iz FCC1h FC5 T5 CP5 CP6 FFC5h F2 M2 POO9h AFF5h PO4 POO3h
        Fp2 T3 CP4 POz TTP7h T7 A2 CCP4h T8 PPO10h FC3 F3 F5 A1 P3 FC4 FCC2h FFC6h FFT8h CCP2h
        CPP4h T6 FTT9h PPO6h CP3 CP1 AF3 FT10 OI1h TPP9h P5 I2 CCP1h T4 CCP3h O1 PO5 PPO9h PPO5h
        P1 AFz PO6 PO3 O2 CPP5h FFC1h FCC4h FFT7h FFC2h FFC4h Cz TP7 Fpz FTT10h PO7 CPP3h P4 P2
        F8 CPz FCC3h FFC3h FT7 I1 TTP8h AFF6h CCP5h C6 PPO1 PO8 C3 POO4h TPP7h F7 T9 TP9 T10
        TP10 POO1 POO2 PPO1h PPO2h
    """.split()
    vocab = folder / "steegformer_channels.json"
    vocab.write_text(json.dumps(names))

    def local_vocabulary(repo, filename, *, revision):
        assert (repo, filename, revision) == (
            "braindecode/STEEGFormer-small",
            vocab.name,
            "633c9c0d76c6e26459e1e0d1337c91858578f13d",
        )
        return vocab

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("REVE_POSITIONS_PATH", str(folder))
        patch.setattr(
            steegformer,
            "huggingface_hub",
            SimpleNamespace(hf_hub_download=local_vocabulary),
        )
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
