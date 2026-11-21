# Authors: Sarthak Tayal <sarthaktayal2@gmail.com>
import importlib
import sys

import mne
import numpy as np
import pytest


@pytest.fixture(params=["numpydantic", "core"])
def configs(request, monkeypatch):
    """The model configs module, with and without numpydantic installed."""
    import braindecode.models.config as module

    if request.param == "numpydantic":
        pytest.importorskip("numpydantic")
        yield module
        return
    # A core install has no numpydantic: reload the configs without it.
    monkeypatch.setitem(sys.modules, "numpydantic", None)
    try:
        yield importlib.reload(module)
    finally:
        monkeypatch.undo()
        importlib.reload(module)  # later tests get the typed configs back


def test_mne_channel_metadata_roundtrip(configs):
    info = mne.create_info(["F3", "C3"], sfreq=128, ch_types="eeg")
    info.set_montage("colin27_1020")
    config = configs.EEGNetConfig(chs_info=info["chs"], n_times=256, n_outputs=2)
    restored = configs.EEGNetConfig.model_validate_json(
        config.model_dump_json()
    ).create_instance()
    assert restored.n_chans == 2
    for result, original in zip(restored.chs_info, info["chs"], strict=True):
        assert result["ch_name"] == original["ch_name"]
        assert result["kind"] == original["kind"]
        assert result["coord_frame"] == original["coord_frame"]
        np.testing.assert_allclose(result["loc"], original["loc"], equal_nan=True)


def test_ch_type_descriptor_roundtrip():
    from braindecode.models.config import EEGNetConfig

    chs = [{"ch_name": "C3", "ch_type": "eeg"}]
    config = EEGNetConfig(chs_info=chs, n_times=256, n_outputs=2)
    assert EEGNetConfig.model_validate_json(config.model_dump_json()).chs_info == chs
