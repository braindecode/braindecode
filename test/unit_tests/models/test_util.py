# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import inspect
import subprocess  # nosec B404: runs a constant snippet with sys.executable
import sys
import warnings

import mne
import numpy as np
import pytest
import torch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import OneHotEncoder

from braindecode import models
from braindecode.models.util import (
    _chs_info_3ch,
    _draw_chs_info,
    _fixture_montage_chs,
    _get_signal_params,
    extract_channel_locations_from_chs_info,
    has_valid_locations,
    models_dict,
    models_mandatory_parameters,
    positions_from_chs_info,
    resolve_channel_indices,
    valid_location_mask,
    warn_if_sfreq_differs,
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
    # ``models_dict`` covers every EEGModuleMixin subclass.
    assert len(all_models) == len(models_dict)
    assert set(all_models) == set(models_dict.items())


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
def test_resolve_channel_indices_transformed_head_geometry(metric):
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
    assert resolve_channel_indices(
        chs, names, montage="standard_1005", metric=metric
    ) == list(range(len(names)))


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_coordinate_helpers_tensor_contracts(dtype):
    xyz = torch.tensor(
        [[0, 0, 0], [1e-9, 0, 0], [1, 2, 3], [float("nan"), 1, 1],
         [float("inf"), 1, 1]], dtype=dtype
    )
    mask = valid_location_mask(xyz)
    expected = torch.tensor([[False], [bool(xyz[1, 0] != 0)], [True], [False], [False]])
    torch.testing.assert_close(mask, expected)
    assert mask.device == xyz.device
    torch.testing.assert_close(torch.jit.script(valid_location_mask)(xyz), mask)
    assert not has_valid_locations(xyz)
    assert has_valid_locations(xyz[:3])
    assert not has_valid_locations(xyz[:2])
    assert not has_valid_locations(xyz[:0])
    boundary = torch.tensor([[1e-8, 0, 0]], dtype=dtype)
    assert has_valid_locations(boundary) == has_valid_locations(
        [{"loc": boundary[0].tolist()}]
    )

    positions = torch.tensor([[1, 2, 3], [3, 2, 4], [2, 2, 0]], dtype=dtype,
                             requires_grad=True)
    normalized = positions_from_chs_info(positions)
    torch.testing.assert_close(normalized, torch.tensor([[0, 0], [1, 0], [.5, 0]], dtype=dtype))
    assert normalized.device == positions.device
    assert normalized.dtype == dtype
    torch.testing.assert_close(torch.jit.script(positions_from_chs_info)(positions), normalized)
    normalized.sum().backward()
    assert torch.isfinite(positions.grad).all()
    for normalize in (positions_from_chs_info, torch.jit.script(positions_from_chs_info)):
        small = torch.tensor([[0., 0., 0.], [1e-5, 0., 0.]], dtype=dtype,
                             requires_grad=True)
        normalize(small).sum().backward()
        torch.testing.assert_close(small.grad, torch.zeros_like(small))


def test_coordinate_helpers_preserve_dictionary_defaults():
    for locations, expected in [([], False), ([[0, 0, 0]], False),
                                ([[1e-9, 0, 0]], False),
                                ([[0, 0, 0], [1, 2, 3]], True),
                                ([[float("nan"), 0, 1]], False)]:
        chs = [{"loc": xyz} for xyz in locations]
        assert has_valid_locations(chs) is expected
    for chs in [None, [{}], [{"loc": None}], [{"loc": 1}]]:
        assert has_valid_locations(chs) is False
    chs = [{"loc": [1, 2, 3]}, {"loc": [3, 2, 4]}, {"loc": [2, 2, 0]}]
    result = positions_from_chs_info(chs)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[0, 0], [1, 0], [.5, 0]])


@pytest.mark.parametrize("fill_missing", [False, True])
@pytest.mark.parametrize(
    "entry, valid",
    [(None, False), ({}, False), ({"loc": None}, False),
     ({"loc": "invalid"}, False), ({"loc": 1}, False),
     ({"loc": [[1, 2, 3]]}, False), ({"loc": [[1], [2, 3]]}, False),
     ({"loc": [1, 2]}, False), ({"loc": {"x": 1}}, False),
     ({"loc": [1, 2, 3]}, True), ({"loc": np.arange(12)}, True),
     ({"loc": [np.nan, 1, 2]}, True), ({"loc": [np.inf, 1, 2]}, True),
     ({"loc": [0, 0, 0]}, True), (np.array([1, 2, 3]), False)],
)
def test_extract_channel_locations_fill_policy(entry, valid, fill_missing):
    prefix = {"loc": [1, 2, 3]}
    actual = extract_channel_locations_from_chs_info([prefix, entry, prefix], fill_missing=fill_missing)
    if valid:
        expected = [[1, 2, 3], np.asarray(entry["loc"])[:3], [1, 2, 3]]
    elif fill_missing:
        expected = [[1, 2, 3], [np.nan] * 3, [1, 2, 3]]
    else:
        expected = [[1, 2, 3]]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))


@pytest.mark.parametrize("fill_missing", [False, True])
def test_extract_channel_locations_empty_zero_and_requested_count(fill_missing):
    for chs in [None, []]:
        assert extract_channel_locations_from_chs_info(chs, fill_missing=fill_missing) is None
    chs = [{"loc": [0, 0, 0]}]
    assert extract_channel_locations_from_chs_info(chs, num_channels=0, fill_missing=fill_missing) is None
    actual = extract_channel_locations_from_chs_info(chs, fill_missing=fill_missing)
    if fill_missing:
        np.testing.assert_array_equal(actual, np.full((1, 3), np.nan, dtype=np.float32))
    else:
        assert actual is None
    actual = extract_channel_locations_from_chs_info([{"loc": [1, 2, 3]}], num_channels=3, fill_missing=fill_missing)
    expected = [[1, 2, 3], [np.nan] * 3, [np.nan] * 3] if fill_missing else [[1, 2, 3]]
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))


def _montage_head_locs(names):
    """Head-frame ``loc`` of ``names``, as a user gets them from ``set_montage``."""
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    info = mne.create_info(list(montage.ch_names), 250.0, "eeg")
    info.set_montage(montage)
    by_name = {ch["ch_name"]: ch["loc"][:3] for ch in info["chs"]}
    return np.array([by_name[name] for name in names])


def _assert_real_chs_info(chs_info, distinct=True):
    """Names are real 10-05 names, positions are the head-frame ones, in metres."""
    names = [ch["ch_name"] for ch in chs_info]
    if distinct:
        assert len(set(names)) == len(names)
        # Two names for one electrode (legacy aliases) do not exist in a recording.
        locs_head = np.array([ch["loc"][:3] for ch in chs_info])
        if len(chs_info) > 1:
            assert pdist(locs_head).min() > 1e-6
    assert set(names) <= set(mne.channels.make_standard_montage(
        resolve_montage_name("standard_1005")
    ).ch_names)
    locs = np.array([ch["loc"] for ch in chs_info])
    assert locs.shape == (len(chs_info), 12)
    assert all(ch["kind"] == "eeg" for ch in chs_info)
    np.testing.assert_allclose(locs[:, :3], _montage_head_locs(names), atol=1e-8)
    np.testing.assert_array_equal(locs[:, 3:], 0.0)
    # Head frame, in metres: scalp sensors 5 to 20 cm away from the origin.
    radius = np.linalg.norm(locs[:, :3], axis=1)
    assert ((radius > 0.05) & (radius < 0.20)).all()


def test_chs_info_3ch_matches_the_montage():
    assert [ch["ch_name"] for ch in _chs_info_3ch] == ["C1", "C2", "C3"]
    _assert_real_chs_info(_chs_info_3ch)


def test_fixture_montage_is_loaded_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chs = _fixture_montage_chs.__wrapped__()
    assert len(chs) > 300
    assert len({name for name, _ in chs}) == len(chs)


def test_fixture_montage_has_no_colocated_legacy_aliases():
    chs = _fixture_montage_chs()
    assert len(chs) > 300
    positions = np.array([loc[:3] for _, loc in chs])
    assert pdist(positions).min() > 1e-6
    names = {name for name, _ in chs}
    # The modern name is kept, its legacy 10-20 alias is dropped.
    assert {"T7", "T8", "P7", "P8"} <= names
    assert not names & {"T3", "T4", "T5", "T6"}


def test_import_does_not_load_the_fixture_montage():
    # No warning filter here: that the montage loads without a warning is
    # checked in-process above, and an unrelated import-time warning of a
    # dependency must not fail this test.
    code = (
        "import braindecode.models.util as u;"
        "assert u._fixture_montage_chs.cache_info().currsize == 0"
    )
    result = subprocess.run(  # nosec B603: constant code, sys.executable
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("n_chans", [1, 3, 19, 32])
def test_draw_chs_info_real_distinct_channels(n_chans):
    chs_info = _draw_chs_info(n_chans, np.random.default_rng(0))
    assert len(chs_info) == n_chans
    _assert_real_chs_info(chs_info)


def test_draw_chs_info_is_determined_by_the_rng():
    def names(seed):
        return [ch["ch_name"] for ch in _draw_chs_info(19, np.random.default_rng(seed))]

    assert names(0) == names(0)
    assert names(0) != names(1)
    # The draw moves the stream on, so successive draws differ.
    rng = np.random.default_rng(0)
    first, second = (_draw_chs_info(19, rng) for _ in range(2))
    assert [ch["ch_name"] for ch in first] != [ch["ch_name"] for ch in second]


def test_draw_chs_info_returns_independent_arrays():
    first = _draw_chs_info(3, np.random.default_rng(0))
    first[0]["loc"][:] = 99.0
    second = _draw_chs_info(3, np.random.default_rng(0))
    assert second[0]["loc"][0] != 99.0


def test_draw_chs_info_with_more_channels_than_the_montage():
    n_montage = len(_fixture_montage_chs())
    assert len(_draw_chs_info(n_montage, np.random.default_rng(0))) == n_montage
    with pytest.raises(ValueError, match="Cannot draw .* distinct channels"):
        _draw_chs_info(n_montage + 1, np.random.default_rng(0))


def test_get_signal_params_draws_real_channels_for_n_chans():
    chs_info = _get_signal_params({"n_chans": 16})["chs_info"]
    assert len(chs_info) == 16
    _assert_real_chs_info(chs_info)


def test_get_signal_params_channels_do_not_depend_on_the_test_order():
    def names():
        return [ch["ch_name"] for ch in _get_signal_params({"n_chans": 16})["chs_info"]]

    first = names()
    _get_signal_params({"n_chans": 5})  # other draws in between
    assert names() == first


def test_dance_signal_params_are_real_channels_and_stable():
    signal_params = {name: sp for name, _, sp in models_mandatory_parameters}["DANCE"]
    first, second = signal_params(), signal_params()
    assert first["n_chans"] == len(first["chs_info"]) == 19
    _assert_real_chs_info(first["chs_info"])
    assert [ch["ch_name"] for ch in first["chs_info"]] == [
        ch["ch_name"] for ch in second["chs_info"]
    ]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("sfreq", [None, 200, 200.0, 200 + 1e-9])
def test_warn_if_sfreq_differs_silent(sfreq):
    warn_if_sfreq_differs("M", sfreq, 200)


def test_warn_if_sfreq_differs_warns():
    with pytest.warns(UserWarning, match="M was pretrained at 200 Hz"):
        warn_if_sfreq_differs("M", 250, 200)
