# Authors: Mahir Jain (mahir@mannas.ai)
#
# License: Apache-2.0
"""Tests specific to :class:`braindecode.models.AXON`."""

import mne
import pytest
import torch

from braindecode.models import AXON
from braindecode.util import resolve_montage_name

SMALL = dict(embed_dim=64, depth=2, num_heads=4)
NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "P4", "O1", "O2"]


def _chs(names=NAMES):
    info = mne.create_info(names, sfreq=200.0, ch_types="eeg")
    info.set_montage(resolve_montage_name("standard_1005"), match_case=False)
    return info["chs"]


def _model(chs_info, **kw):
    torch.manual_seed(0)
    return AXON(chs_info=chs_info, n_outputs=3, sfreq=200.0, **{**SMALL, **kw}).eval()


def test_shapes_and_features():
    model = _model(_chs())
    x = torch.randn(2, len(NAMES), 800)  # 4 patches of 1 s with 0.9 s stride
    with torch.no_grad():
        logits = model(x)
        out = model(x, return_features=True)
    assert logits.shape == (2, 3)
    assert out["features"].shape == (2, 64)
    assert out["tokens"].shape == (2, len(NAMES), 4, 64)
    assert out["cls_token"] is None
    torch.testing.assert_close(model.encode(x), out["tokens"])


def test_channel_order_does_not_matter():
    """Electrodes are identified by position, so permuting channels together
    with chs_info must leave the pooled embedding unchanged."""
    chs = _chs()
    model = _model(chs)
    perm = torch.randperm(len(NAMES))
    permuted = _model([chs[i] for i in perm])
    permuted.load_state_dict(model.state_dict())
    x = torch.randn(2, len(NAMES), 1000)
    with torch.no_grad():
        a = model(x, return_features=True)["features"]
        b = permuted(x[:, perm], return_features=True)["features"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_positions_from_channel_names():
    """Without 'loc', standard channel names resolve to the same positions."""
    with_loc = _model(_chs())
    without_loc = _model([{"ch_name": n, "kind": "eeg"} for n in NAMES])
    torch.testing.assert_close(
        with_loc.encoder.channel_positions, without_loc.encoder.channel_positions
    )


def test_unknown_channel_without_position_raises():
    chs = [{"ch_name": "Fp1", "kind": "eeg"}, {"ch_name": "NOT_A_CHANNEL", "kind": "eeg"}]
    with pytest.raises(ValueError, match="NOT_A_CHANNEL"):
        _model(chs)


def test_weights_load_onto_another_montage():
    """Channel positions are not stored in the weights."""
    source = _model(_chs())
    target = _model(_chs(["C3", "Cz", "C4", "FC3", "CP4"]))
    target.load_state_dict(source.state_dict(), strict=True)
    assert "encoder.channel_positions" not in source.state_dict()


def test_input_unit_does_not_matter():
    """Microvolts and volts (MNE's default) give the same output."""
    model = _model(_chs())
    x_uv = 20.0 * torch.randn(2, len(NAMES), 600) + 5.0
    with torch.no_grad():
        torch.testing.assert_close(model(x_uv), model(x_uv * 1e-6), atol=1e-4, rtol=1e-4)


def test_too_short_window_raises():
    with pytest.raises(ValueError, match="patch_size"):
        AXON(chs_info=_chs(), n_outputs=2, n_times=100, **SMALL)


def test_warns_on_non_200_hz():
    with pytest.warns(UserWarning, match="200 Hz"):
        AXON(chs_info=_chs(), n_outputs=2, sfreq=250.0, **SMALL)
