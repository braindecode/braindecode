# Authors: Pierre Guetschel
#
# License: BSD (3-clause)
import numpy as np
import pytest
import torch

from braindecode.models.eegnet import EEGNet
from braindecode.models.interpolated import InterpolatedModel


def _ch(name, loc=(0.0, 0.0, 0.0)):
    return {"ch_name": name, "kind": "eeg", "loc": np.array(loc, dtype=float)}


def _target_5ch():
    """5 real 10-20 channel positions (enough for MNE spline fitting)."""
    import mne

    mtg = mne.channels.make_standard_montage("standard_1005")
    pos = mtg.get_positions()["ch_pos"]
    return [
        {"ch_name": n, "kind": "eeg", "loc": np.asarray(pos[n], dtype=float)}
        for n in ["Fz", "Cz", "Pz", "C3", "C4"]
    ]


def test_factory_returns_subclass_of_backbone():
    target = _target_5ch()
    Cls = InterpolatedModel(EEGNet, target_chs_info=target)
    assert issubclass(Cls, EEGNet)
    assert Cls.__name__ == "InterpolatedEEGNet"
    assert Cls._TARGET_CHS_INFO == target


def test_factory_target_is_required():
    with pytest.raises(TypeError):
        InterpolatedModel(EEGNet)  # missing target_chs_info


def test_factory_forward_shape_matches_user_channels():
    target = _target_5ch()
    user_4 = _target_5ch()[:4]  # 4 valid EEG channels with real positions
    Cls = InterpolatedModel(EEGNet, target_chs_info=target)
    model = Cls(
        chs_info=user_4,
        n_outputs=2,
        n_times=200,
        interpolation_mode="always",
    )
    y = model(torch.zeros(1, 4, 200))
    assert y.shape[0] == 1  # batch dim preserved
