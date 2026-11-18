# Authors: Sarthak Tayal <sarthaktayal2@gmail.com>
import subprocess
import sys

import mne
import numpy as np


def test_mne_channel_metadata_roundtrip():
    from braindecode.models.config import EEGNetConfig

    info = mne.create_info(["F3", "C3"], sfreq=128, ch_types="eeg")
    info.set_montage("standard_1020")
    config = EEGNetConfig(chs_info=info["chs"], n_times=256, n_outputs=2)
    restored = EEGNetConfig.model_validate_json(
        config.model_dump_json()
    ).create_instance()
    assert restored.n_chans == 2
    for result, original in zip(restored.chs_info, info["chs"], strict=True):
        assert result["ch_name"] == original["ch_name"]
        assert result["kind"] == original["kind"]
        assert result["coord_frame"] == original["coord_frame"]
        np.testing.assert_allclose(result["loc"], original["loc"], equal_nan=True)
    # Native model descriptors can also give the modality under ch_type.
    descriptors = [dict(ch, ch_type="eeg") for ch in info["chs"]]
    for ch in descriptors:
        del ch["kind"]
    alias = EEGNetConfig(chs_info=descriptors, n_times=256, n_outputs=2)
    assert (
        EEGNetConfig.model_validate_json(alias.model_dump_json()).chs_info[0]["ch_type"]
        == "eeg"
    )


def test_config_without_numpydantic():
    # Exercise the core install in isolation from the typed configs already imported.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['numpydantic'] = None\n"
            "from test.unit_tests.models.test_config_numpydantic_fallback import "
            "test_mne_channel_metadata_roundtrip\n"
            "test_mne_channel_metadata_roundtrip()",
        ],
        check=True,
    )
