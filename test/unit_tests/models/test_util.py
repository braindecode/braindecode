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
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.models.signal_jepa import _PRETRAIN_CHS_INFO
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


@pytest.mark.parametrize(
    "names", [LABRAM_CHANNEL_ORDER, [ch["ch_name"] for ch in _PRETRAIN_CHS_INFO]]
)
def test_resolve_channel_indices_pretrained_vocabularies(names):
    """Name semantics agree with LaBraM/SignalJEPA without changing their geometry."""
    selected = [names[-1].lower(), names[0].upper(), names[-1]]
    chs = [{"ch_name": name, "coord_frame": "mri"} for name in selected]
    assert resolve_channel_indices(chs, names) == [len(names) - 1, 0, len(names) - 1]
    assert resolve_channel_indices([{"ch_name": "not-an-electrode"}], names) is None


def _head_info(names):
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    info = mne.create_info(names, 250, "eeg")
    info.set_montage(montage)
    return info


def test_resolve_channel_indices_transformed_head_geometry():
    """Every real head-frame 10-05 location must recover its own vocabulary slot.

    Using native montage coordinates on both sides would mask a frame mismatch.
    A dense vocabulary ensures the old native-vs-head bug changes actual indices.
    """
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    # MNE includes aliases with identical coordinates; retain the first per site.
    names = []
    seen = set()
    for name, xyz in montage.get_positions()["ch_pos"].items():
        if tuple(xyz) not in seen:
            names.append(name)
            seen.add(tuple(xyz))
    info = _head_info(names)
    native = np.array([montage.get_positions()["ch_pos"][name] for name in names])
    head = np.array([ch["loc"][:3] for ch in info["chs"]])
    assert not np.allclose(native, head)
    chs = [dict(ch, ch_name=f"sensor-{i}") for i, ch in enumerate(info["chs"])]
    assert resolve_channel_indices(chs, names, montage="standard_1005") == list(
        range(len(names))
    )


def test_resolve_channel_indices_uses_luna_location_extraction():
    """Shared extraction accepts both compact and full MNE channel dictionaries."""
    names = ["Oz", "Cz", "Fp1"]
    chs = [
        dict(ch, ch_name=f"sensor-{i}") for i, ch in enumerate(_head_info(names)["chs"])
    ]
    locations = extract_channel_locations_from_chs_info(chs)
    compact = [
        {"ch_name": ch["ch_name"], "loc": loc} for ch, loc in zip(chs, locations)
    ]
    assert resolve_channel_indices(compact, names, montage="standard_1005") == [0, 1, 2]
    np.testing.assert_array_equal(
        extract_channel_locations_from_chs_info(compact), locations
    )


@pytest.mark.parametrize(
    "loc", [None, [], [1, 2], [0, 0, 0], [np.nan, 0, 1], [np.inf, 0, 1], "bad"]
)
def test_resolve_channel_indices_invalid_locations(loc):
    assert (
        resolve_channel_indices(
            [{"ch_name": "unknown", "loc": loc}], ["Cz"], montage="standard_1005"
        )
        is None
    )


def test_resolve_channel_indices_degenerate_direction():
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    info = _head_info(montage.ch_names)
    points = np.array([ch["loc"][:3] for ch in info["chs"]])
    centre = np.linalg.lstsq(
        np.c_[2 * points, np.ones(len(points))], (points**2).sum(axis=1), rcond=None
    )[0][:3]
    assert (
        resolve_channel_indices(
            [{"ch_name": "unknown", "loc": centre}], ["Cz"], montage="standard_1005"
        )
        is None
    )


def test_resolve_channel_indices_no_reference_sites():
    assert (
        resolve_channel_indices(
            [{"ch_name": "unknown", "loc": [0.01, 0.02, 0.1]}],
            ["not-a-standard-site"],
            montage="standard_1005",
        )
        is None
    )


def test_resolve_channel_indices_radius_and_input_subset_invariance():
    names = ["Fp1", "Cz", "Oz", "T8"]
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    points = np.array([ch["loc"][:3] for ch in _head_info(montage.ch_names)["chs"]])
    centre = np.linalg.lstsq(
        np.c_[2 * points, np.ones(len(points))], (points**2).sum(axis=1), rcond=None
    )[0][:3]
    chs = [
        {"ch_name": f"sensor-{i}", "loc": centre + 1.3 * (ch["loc"][:3] - centre)}
        for i, ch in enumerate(_head_info(names)["chs"])
    ]
    assert resolve_channel_indices(chs, names, montage="standard_1005") == [0, 1, 2, 3]
    assert resolve_channel_indices(
        [chs[3], chs[0], chs[3]], names, montage="standard_1005"
    ) == [3, 0, 3]
