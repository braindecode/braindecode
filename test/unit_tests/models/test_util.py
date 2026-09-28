# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import inspect

import mne
import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from braindecode import models
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    interpolated_models_dict,
    models_dict,
    resolve_channel_indices,
)
from braindecode.modules.util import (
    _pad_shift_array,
    aggregate_probas,
)
from braindecode.util import resolve_montage_name


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    "n_sequences,n_classes,n_windows,stride",
    [[10, 3, 2, 1], [3, 3, 2, 5], [3, 3, 1, 2]],
)
def test_pad_shift_array(n_sequences, n_classes, n_windows, stride, dtype):
    dense_y = (
        np.random.RandomState(33).rand(n_sequences, n_classes, n_windows).astype(dtype)
    )
    n_outputs = (n_sequences - 1) * stride + n_windows

    # Align sequences with _pad_shift_array
    shifted_y = _pad_shift_array(dense_y, stride=stride)

    # Align sequences explicitly (to reproduce output of _pad_shift_array)
    shifted_y2 = np.concatenate(
        [
            np.concatenate(
                (
                    np.zeros((1, n_classes, i * stride)),
                    dense_y[[i]],
                    np.zeros((1, n_classes, n_outputs - n_windows - i * stride)),
                ),
                axis=2,
            )
            for i in range(n_sequences)
        ],
        axis=0,
    )

    assert (shifted_y == shifted_y2).all()


def test_pad_shift_array_not_3d():
    with pytest.raises(NotImplementedError):
        _pad_shift_array(np.zeros((2, 2)))


@pytest.mark.parametrize(
    "n_sequences,n_classes,n_windows,stride",
    [[3, 3, 2, 2], [3, 3, 1, 1], [10, 3, 2, 1]],
)
def test_aggregate_probas(n_sequences, n_classes, n_windows, stride):
    # Create fake matrix of logits where each example has a logit of 1 for the
    # given class and zeros elsewhere
    n_outputs = (n_sequences - 1) * stride + n_windows
    y_true = np.arange(n_outputs) % n_classes  # fake target for each window
    logits = OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape(-1, 1))
    logits = np.lib.stride_tricks.sliding_window_view(  # extract sequences
        logits, n_windows, axis=0
    )[::stride]

    y_pred_probas = aggregate_probas(logits, n_windows_stride=stride)

    # Make sure shape is right
    assert y_pred_probas.ndim == 2
    assert y_pred_probas.shape == (n_outputs, n_classes)

    # Make sure results of aggregation match the original targets
    assert (y_pred_probas.argmax(axis=1) == y_true).all()


def test_models_dict():
    all_models = [
        (name, m)
        for name, m in models.__dict__.items()
        if (
            inspect.isclass(m)
            and issubclass(m, models.base.EEGModuleMixin)
            and m != models.base.EEGModuleMixin
        )
    ]
    # ``models_dict`` and ``interpolated_models_dict`` together must cover all
    # EEGModuleMixin subclasses, and must be disjoint.
    combined = {**models_dict, **interpolated_models_dict}
    assert len(all_models) == len(combined)
    assert set(all_models) == set(combined.items())
    assert set(models_dict).isdisjoint(interpolated_models_dict)


def test_interpolated_models_dict():
    # Interpolated models are separated out of ``models_dict`` and are
    # identified by the ``_TARGET_CHS_INFO`` attribute set by
    # ``InterpolatedModel``.
    assert len(interpolated_models_dict) > 0
    for name, model_cls in interpolated_models_dict.items():
        assert getattr(model_cls, "_TARGET_CHS_INFO", None) is not None
        assert name not in models_dict
    # No interpolated models leaked into ``models_dict``.
    for model_cls in models_dict.values():
        assert getattr(model_cls, "_TARGET_CHS_INFO", None) is None


def test_resolve_channel_indices_transformed_head_geometry():
    """Independent MNE head coordinates expose native/head reference mismatches."""
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    # Drop aliases sharing coordinates, so every expected slot is unambiguous.
    sites = {
        tuple(xyz): name for name, xyz in montage.get_positions()["ch_pos"].items()
    }
    names = list(sites.values())
    info = mne.create_info(names, 250, "eeg")
    info.set_montage(montage)
    head = np.array([ch["loc"][:3] for ch in info["chs"]])
    assert not np.allclose(list(sites), head)
    chs = [dict(ch, ch_name=f"sensor-{i}") for i, ch in enumerate(info["chs"])]
    assert resolve_channel_indices(chs, names, montage="standard_1005") == list(
        range(len(names))
    )


@pytest.mark.parametrize(
    "loc, frame",
    [
        ("missing", "head"),
        (None, "head"),
        (1.0, "head"),
        ("invalid", "head"),
        ([[0.1], [0.2, 0.3]], "head"),
        ([0.1, 0.2], "head"),
        ([[0.1, 0.2, 0.3]], "head"),
        ([0, 0, 0], "head"),
        ([np.nan, 0, 1], "head"),
        ([0.01, 0.02, 0.1], "mri"),
        ([0.01, 0.02, 0.1], 0),
    ],
    ids=[
        "missing", "none", "scalar", "string", "ragged", "short", "matrix",
        "zero", "nonfinite", "mri", "unknown-frame",
    ],
)
def test_resolve_channel_indices_invalid_geometry(loc, frame):
    chs = [{"ch_name": "unknown", "loc": loc, "coord_frame": frame}]
    if isinstance(loc, str) and loc == "missing":
        chs[0].pop("loc")
    # Malformed locations stop extraction, preserving any valid prefix.
    malformed = not (
        isinstance(loc, list) and len(loc) == 3 and np.isscalar(loc[0])
    )
    if frame == "head" and malformed:
        assert extract_channel_locations_from_chs_info(chs) is None
        prefix = {"ch_name": "other", "loc": [0.01, 0.02, 0.1]}
        assert extract_channel_locations_from_chs_info([prefix, *chs]).shape == (1, 3)
        assert resolve_channel_indices(
            [prefix, *chs], ["Cz"], montage="standard_1005"
        ) is None
    assert resolve_channel_indices(chs, ["Cz"], montage="standard_1005") is None
