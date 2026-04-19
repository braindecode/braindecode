# Authors: Pierre Guetschel
#
# License: BSD (3-clause)
import numpy as np
import torch

from braindecode.modules.interpolation import (
    ChannelInterpolationLayer,
    _compute_interpolation_matrix_mne,
)


def _ch(name, loc=(0.0, 0.0, 0.0)):
    return {"ch_name": name, "kind": "eeg", "loc": np.array(loc, dtype=float)}


def test_name_match_all_matches_is_pure_permutation():
    # src has same names as tgt but in a different order.
    src = [_ch("Cz"), _ch("Fz"), _ch("Oz")]
    tgt = [_ch("Fz"), _ch("Oz"), _ch("Cz")]
    layer = ChannelInterpolationLayer(
        src_chs_info=src, tgt_chs_info=tgt, mode="name_match"
    )
    # shape
    assert layer.matrix.shape == (3, 3)
    # exact permutation: row i selects src index of the matching name
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # Fz from src index 1
            [0.0, 0.0, 1.0],  # Oz from src index 2
            [1.0, 0.0, 0.0],  # Cz from src index 0
        ]
    )
    torch.testing.assert_close(layer.matrix, expected)


def test_name_match_is_case_insensitive():
    src = [_ch("FZ"), _ch("cz")]
    tgt = [_ch("Fz"), _ch("Cz")]
    layer = ChannelInterpolationLayer(
        src_chs_info=src, tgt_chs_info=tgt, mode="name_match"
    )
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(layer.matrix, expected)


def test_forward_applies_matrix_over_channel_axis():
    src = [_ch("A"), _ch("B")]
    tgt = [_ch("B"), _ch("A")]  # swap
    layer = ChannelInterpolationLayer(
        src_chs_info=src, tgt_chs_info=tgt, mode="name_match"
    )
    x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])  # (1, 2, 3)
    y = layer(x)
    assert y.shape == (1, 2, 3)
    torch.testing.assert_close(y[0, 0], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(y[0, 1], torch.tensor([1.0, 2.0, 3.0]))


def _montage_ch(name):
    # Positions from a real montage: we pick a handful of 10-20 names.
    import mne

    mtg = mne.channels.make_standard_montage("standard_1005")
    pos = mtg.get_positions()["ch_pos"]
    return {"ch_name": name, "kind": "eeg", "loc": np.asarray(pos[name], dtype=float)}


def test_compute_mne_matrix_returns_correct_shape():
    src = [_montage_ch(n) for n in ["Fz", "Cz", "Pz", "C3", "C4"]]
    tgt = [_montage_ch(n) for n in ["F3", "F4", "P3", "P4"]]
    W = _compute_interpolation_matrix_mne(src, tgt, method="spline")
    assert isinstance(W, torch.Tensor)
    assert W.dtype == torch.float32
    assert W.shape == (4, 5)
    # Should not be all-zero
    assert torch.any(W.abs() > 1e-6)
