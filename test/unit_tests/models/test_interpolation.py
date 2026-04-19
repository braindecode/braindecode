# Authors: Pierre Guetschel
#
# License: BSD (3-clause)
import numpy as np
import pytest
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


def test_always_mode_uses_mne_even_when_names_match():
    # Identical src and tgt by name — in name_match this would be identity,
    # in always mode it uses the MNE matrix (NOT identity).
    # Use at least 4 channels to satisfy MNE's minimum digitization requirement.
    names = ["Fz", "Cz", "Pz", "C3", "C4"]
    src = [_montage_ch(n) for n in names]
    tgt = [_montage_ch(n) for n in names]
    layer = ChannelInterpolationLayer(src, tgt, mode="always")
    assert layer.matrix.shape == (5, 5)
    # MNE on identical positions will approximate identity but likely not
    # be exactly identity. Check non-trivial off-diagonal structure.
    off_diag = layer.matrix - torch.diag(torch.diagonal(layer.matrix))
    assert torch.any(off_diag.abs() > 1e-6), (
        "expected non-trivial MNE-computed matrix, got pure diagonal"
    )


def test_name_match_partial_overwrites_matched_rows_with_one_hots():
    # src: Fz, Cz, Pz, C3, C4 (5 — MNE needs ≥4)
    # tgt: Fz, F3, Cz, P3 — Fz and Cz match; F3 and P3 don't.
    src = [_montage_ch(n) for n in ["Fz", "Cz", "Pz", "C3", "C4"]]
    tgt = [_montage_ch(n) for n in ["Fz", "F3", "Cz", "P3"]]
    layer = ChannelInterpolationLayer(src, tgt, mode="name_match")
    W = layer.matrix
    assert W.shape == (4, 5)

    # Matched rows are one-hots on the matching src index.
    # Fz → tgt row 0, src col 0
    torch.testing.assert_close(
        W[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    )
    # Cz → tgt row 2, src col 1
    torch.testing.assert_close(
        W[2], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    )
    # Unmatched row (F3, row 1) came from MNE: NOT a pure one-hot.
    nonzero_count_row1 = (W[1].abs() > 1e-6).sum().item()
    assert nonzero_count_row1 > 1, (
        f"expected MNE-computed row (>1 nonzeros), got {nonzero_count_row1}"
    )
    # Unmatched row (P3, row 3) came from MNE.
    nonzero_count_row3 = (W[3].abs() > 1e-6).sum().item()
    assert nonzero_count_row3 > 1


def test_non_eeg_channel_in_src_raises():
    src = [_ch("Fz"), {"ch_name": "EMG1", "kind": "emg", "loc": np.zeros(3)}]
    tgt = [_ch("Fz")]
    with pytest.raises(ValueError, match="non-EEG channel"):
        ChannelInterpolationLayer(src, tgt, mode="name_match")


def test_non_eeg_channel_in_tgt_raises():
    src = [_ch("Fz")]
    tgt = [_ch("Fz"), {"ch_name": "EOG1", "kind": "eog", "loc": np.zeros(3)}]
    with pytest.raises(ValueError, match="non-EEG channel"):
        ChannelInterpolationLayer(src, tgt, mode="name_match")


def test_missing_loc_raises_when_mne_needed():
    # mode="always" always calls MNE, so missing loc must raise.
    src = [{"ch_name": "Fz", "kind": "eeg"}]
    tgt = [{"ch_name": "Cz", "kind": "eeg", "loc": np.zeros(3)}]
    with pytest.raises(ValueError, match="'loc'"):
        ChannelInterpolationLayer(src, tgt, mode="always")


def test_missing_loc_is_ok_for_full_name_match():
    # Full coverage → no MNE call → loc not required.
    src = [{"ch_name": "Fz", "kind": "eeg"}, {"ch_name": "Cz", "kind": "eeg"}]
    tgt = [{"ch_name": "Cz", "kind": "eeg"}, {"ch_name": "Fz", "kind": "eeg"}]
    layer = ChannelInterpolationLayer(src, tgt, mode="name_match")
    assert layer.matrix.shape == (2, 2)
