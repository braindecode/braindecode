# Authors: Pierre Guetschel
#
# License: BSD (3-clause)
import numpy as np
import torch

from braindecode.modules.interpolation import ChannelInterpolationLayer


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
