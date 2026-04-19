# Authors: Pierre Guetschel
#
# License: BSD (3-clause)
import numpy as np
import pytest
import torch

from braindecode.models.eegnet import EEGNet
from braindecode.models.interpolated import (
    InterpolatedModel,
    _build_chs_info_from_montage,
)


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


def test_user_facing_chs_info_and_n_chans_reflect_user_after_init():
    target = _target_5ch()
    # 4 user channels — all names present in target, so name_match avoids
    # MNE for matched channels; only the 1 unmatched target channel needs
    # MNE interpolation (4 src points satisfies MNE's minimum of 4).
    user = _target_5ch()[:4]
    Cls = InterpolatedModel(EEGNet, target_chs_info=target)
    model = Cls(
        chs_info=user,
        n_outputs=2,
        n_times=200,
        interpolation_mode="name_match",
    )
    assert model.n_chans == 4
    assert len(model.chs_info) == 4
    assert model._TARGET_CHS_INFO == target
    assert model.input_shape[1] == 4  # user's channel count, not target's


def test_get_output_shape_does_not_crash():
    target = _target_5ch()
    user = _target_5ch()[:4]
    Cls = InterpolatedModel(EEGNet, target_chs_info=target)
    model = Cls(
        chs_info=user,
        n_outputs=2,
        n_times=200,
        interpolation_mode="name_match",
    )
    shape = model.get_output_shape()
    assert shape[0] == 1  # batch


def test_get_config_preserves_user_chs_info():
    target = _target_5ch()
    user = _target_5ch()[:4]
    Cls = InterpolatedModel(EEGNet, target_chs_info=target)
    model = Cls(
        chs_info=user,
        n_outputs=2,
        n_times=200,
        interpolation_mode="name_match",
    )
    config = model.get_config()
    # chs_info stored is user's (4 channels), not target's (5)
    assert len(config["chs_info"]) == 4
    # extra kwargs are captured
    assert config["interpolation_mode"] == "name_match"
    assert config["n_outputs"] == 2


def test_from_config_round_trip():
    target = _target_5ch()
    user = _target_5ch()[:4]
    Cls = InterpolatedModel(EEGNet, target_chs_info=target)
    model1 = Cls(
        chs_info=user,
        n_outputs=2,
        n_times=200,
        interpolation_mode="name_match",
    )
    config = model1.get_config()
    model2 = Cls.from_config(config)
    assert model2.n_chans == 4
    assert len(model2.chs_info) == 4
    assert model2._TARGET_CHS_INFO == target


def test_build_chs_info_from_montage_returns_list_of_dicts():
    info = _build_chs_info_from_montage(["Fz", "Cz", "Pz"], montage="standard_1005")
    assert isinstance(info, list) and len(info) == 3
    for ch in info:
        assert set(ch.keys()) >= {"ch_name", "loc", "kind"}
        assert ch["kind"] == "eeg"
        assert isinstance(ch["loc"], np.ndarray)
        assert ch["loc"].shape == (3,)


def test_build_chs_info_from_montage_raises_on_unknown_name():
    with pytest.raises(ValueError, match="not found"):
        _build_chs_info_from_montage(["NotAChannel"], montage="standard_1005")
