# Authors: Pierre Guetschel
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Matrix checks of the ``spline`` channel strategy (MNE ``interpolate_to``).

The matrix code lives in :mod:`braindecode.modules.channels.strategies.interp`
since the channel layer (:mod:`braindecode.modules.channels`) replaced the
former public interpolation layer.
"""

import mne
import numpy as np
import pytest
import torch

from braindecode.modules.channels import (
    ChannelTarget,
    ChannelTokenizer,
    get_channel_strategy,
    resolve_montage,
)
from braindecode.modules.channels.strategies import interp
from braindecode.modules.channels.strategies.interp import _mne_interp_matrix

_POS = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]


def _ch(name, loc=(0.0, 0.0, 0.0)):
    return {"ch_name": name, "kind": "eeg", "loc": np.array(loc, dtype=float)}


def _montage_ch(name, pos_name=None):
    return {
        "ch_name": name,
        "kind": "eeg",
        "loc": np.asarray(_POS[pos_name or name], dtype=float),
    }


def _spline_weights(src, tgt, **kwargs):
    strategy = get_channel_strategy("spline", **kwargs)
    m = strategy.build(resolve_montage(src), ChannelTarget("montage", chs_info=tgt))
    return m


def test_name_match_all_matches_is_pure_permutation():
    src = [_ch("Cz"), _ch("Fz"), _ch("Oz")]
    tgt = [_ch("Fz"), _ch("Oz"), _ch("Cz")]
    m = _spline_weights(src, tgt)
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # Fz from src index 1
            [0.0, 0.0, 1.0],  # Oz from src index 2
            [1.0, 0.0, 0.0],  # Cz from src index 0
        ]
    )
    torch.testing.assert_close(m.weights, expected)
    assert m.observed.all()


def test_name_match_is_case_insensitive():
    m = _spline_weights([_ch("FZ"), _ch("cz")], [_ch("Fz"), _ch("Cz")])
    torch.testing.assert_close(m.weights, torch.eye(2))


def test_forward_applies_matrix_over_channel_axis():
    src = [_ch("A"), _ch("B")]
    tgt = [_ch("B"), _ch("A")]  # swap
    layer = ChannelTokenizer(
        ChannelTarget("montage", chs_info=tgt), "spline", src_chs_info=src
    )
    x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])  # (1, 2, 3)
    y = layer(x).x
    assert y.shape == (1, 2, 3)
    torch.testing.assert_close(y[0, 0], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(y[0, 1], torch.tensor([1.0, 2.0, 3.0]))


def test_compute_mne_matrix_returns_correct_shape():
    src = np.array([_POS[n] for n in ["Fz", "Cz", "Pz", "C3", "C4"]])
    tgt = np.array([_POS[n] for n in ["F3", "F4", "P3", "P4"]])
    W = _mne_interp_matrix(src, tgt, method="spline", reg=1e-3)
    assert W.shape == (4, 5)
    assert np.any(np.abs(W) > 1e-6)


@pytest.mark.parametrize("reg", [0.0, 1e-3])
def test_spline_onto_other_positions_mixes_sources(reg):
    # Same names, different positions: the positional fill mixes several
    # sources per target and preserves constant signals (rows sum to one).
    src = np.array([_POS[n] for n in ["Fz", "Cz", "Pz", "C3", "C4"]])
    tgt = np.array([_POS[n] for n in ["F3", "F4", "P3", "P4", "Oz"]])
    W = _mne_interp_matrix(src, tgt, method="spline", reg=reg)
    for i, row in enumerate(W):
        n_large = int((np.abs(row) > 0.1).sum())
        assert n_large > 1, f"row {i} looks one-hot ({n_large} large entries)"
    np.testing.assert_allclose(W.sum(axis=1), np.ones(5), atol=1e-2 if reg else 1e-3)


def test_partial_name_match_copies_matched_rows_and_fills_the_rest():
    src = [_montage_ch(n) for n in ["Fz", "Cz", "Pz", "C3", "C4"]]
    tgt = [_montage_ch(n) for n in ["Fz", "F3", "Cz", "P3"]]
    m = _spline_weights(src, tgt)
    W = m.weights
    assert W.shape == (4, 5)
    torch.testing.assert_close(W[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(W[2], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
    assert (W[1].abs() > 1e-6).sum() > 1
    assert (W[3].abs() > 1e-6).sum() > 1
    assert m.observed.tolist() == [True, False, True, False]


def test_non_eeg_channel_in_src_raises():
    src = [_ch("Fz"), {"ch_name": "EMG1", "kind": "emg", "loc": np.zeros(3)}]
    with pytest.raises(ValueError, match="not EEG"):
        resolve_montage(src)


def test_too_few_positions_raises_when_spline_needed():
    # One unknown, unpositioned input cannot feed a spline.
    src = [{"ch_name": "X1", "kind": "eeg"}]
    tgt = [_montage_ch("Cz")]
    with pytest.raises(ValueError, match="position"):
        _spline_weights(src, tgt)


def test_missing_loc_is_ok_for_full_name_match():
    src = [{"ch_name": "Fz", "kind": "eeg"}, {"ch_name": "Cz", "kind": "eeg"}]
    tgt = [{"ch_name": "Cz", "kind": "eeg"}, {"ch_name": "Fz", "kind": "eeg"}]
    m = _spline_weights(src, tgt)
    torch.testing.assert_close(m.weights, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_spline_layer_adds_no_state():
    layer = ChannelTokenizer(
        ChannelTarget("montage", chs_info=[_montage_ch("Cz")]),
        "spline",
        src_chs_info=[_montage_ch(n) for n in ["Fz", "Pz", "C3", "C4"]],
    )
    assert len(layer.state_dict()) == 0
    assert len(list(layer.parameters())) == 0


def test_full_name_coverage_does_not_call_mne(monkeypatch):
    def fake_mne(*args, **kwargs):
        raise AssertionError("MNE should not be called on full name coverage")

    monkeypatch.setattr(interp, "_mne_interp_matrix", fake_mne)
    src = [_ch("Fz"), _ch("Cz"), _ch("Pz")]
    tgt = [_ch("Pz"), _ch("Fz"), _ch("Cz")]
    m = _spline_weights(src, tgt)
    assert m.weights.shape == (3, 3)
