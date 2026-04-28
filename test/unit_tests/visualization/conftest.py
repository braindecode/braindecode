import mne
import pytest

SEED = 0


@pytest.fixture(scope="module")
def chs_info_factory():
    """Build a `chs_info` list for an N-channel standard_1020 montage."""
    montage = mne.channels.make_standard_montage("standard_1020")

    def _factory(n_chans):
        ch_names = montage.ch_names[:n_chans]
        info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
        info.set_montage(montage)
        return info["chs"]

    return _factory
