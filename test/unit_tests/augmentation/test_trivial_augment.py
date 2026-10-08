# Authors: Li Qing <325196192+qinxwew@users.noreply.github.com>
#
# License: BSD (3-clause)

import pickle

import numpy as np
import pytest
import torch

from braindecode.augmentation import TrivialAugment
from braindecode.augmentation.transforms import _get_standard_10_20_positions
from test.unit_tests.augmentation.test_base import common_transform_assertions


@pytest.fixture
def random_batch(batch_size=200):
    """Deterministic batch of 200 examples, 8 channels, 100 samples."""
    rng = np.random.RandomState(31)
    X = torch.from_numpy(rng.randn(batch_size, 8, 100)).float()
    y = torch.arange(batch_size) % 4
    return X, y


@pytest.fixture
def standard_positions():
    names = "Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2".split()
    return _get_standard_10_20_positions(ordered_ch_names=names)


def _scaling_op(s):
    """Custom op multiplying X by its strength (None -> by 2), y unchanged."""

    def apply(X, y):
        return X * (2.0 if s is None else s), y

    return apply


def test_shape_and_labels_preserved(random_batch):
    transform = TrivialAugment(random_state=0)
    X, y = random_batch
    tr_X, tr_y = transform(X, y)
    common_transform_assertions((X, y), (tr_X, tr_y))


def test_probability_zero_leaves_input_unchanged(random_batch):
    transform = TrivialAugment(probability=0.0, random_state=0)
    X, y = random_batch
    tr_X, tr_y = transform(X, y)
    assert torch.equal(tr_X, X)
    assert torch.equal(tr_y, y)


def test_y_none_returns_x_only(random_batch):
    transform = TrivialAugment(random_state=0)
    X, _ = random_batch
    out = transform(X)
    assert not isinstance(out, tuple)
    assert out.shape == X.shape


def test_single_example_batch():
    transform = TrivialAugment(random_state=0)
    X = torch.randn(1, 8, 100)
    y = torch.tensor([1])
    tr_X, tr_y = transform(X, y)
    assert tr_X.shape == X.shape
    assert tr_y.shape == y.shape


def test_default_pool_composition():
    transform = TrivialAugment(random_state=0)
    assert transform.op_names == [
        "TimeReverse",
        "SignFlip",
        "FTSurrogate",
        "ChannelsDropout",
        "ChannelsShuffle",
        "SmoothTimeMask",
        "GaussianNoise",
        "AmplitudeScale",
    ]


def test_sfreq_adds_frequency_ops():
    transform = TrivialAugment(sfreq=250, random_state=0)
    assert "BandstopFilter" in transform.op_names
    assert "FrequencyShift" in transform.op_names
    assert "SensorsRotation" not in transform.op_names


def test_sensors_positions_adds_rotation(standard_positions):
    transform = TrivialAugment(
        sensors_positions_matrix=standard_positions, random_state=0
    )
    assert "SensorsRotation" in transform.op_names
    assert "BandstopFilter" not in transform.op_names


def test_too_small_sfreq_raises():
    with pytest.raises(ValueError, match="too small"):
        TrivialAugment(sfreq=6)


def test_num_bins_validation():
    with pytest.raises(ValueError, match="num_bins"):
        TrivialAugment(num_bins=0)


def test_per_example_strengths():
    """Per-example (not per-batch) strength sampling, on an isolated op."""
    num_bins = 4
    transform = TrivialAugment(num_bins=num_bins, random_state=2)
    # white-box: restrict the pool to a single deterministic custom op so
    # that per-row scaling factors can be checked against the strength grid
    transform._pool = [transform._check_custom_op(("Scale", _scaling_op, (1.0, 4.0)))]
    transform.op_names = [entry.name for entry in transform._pool]
    rng = np.random.RandomState(5)
    X = torch.from_numpy(rng.randn(100, 8, 50)).float()
    y = torch.zeros(100)
    tr_X, tr_y = transform(X, y)
    strengths = torch.linspace(1.0, 4.0, num_bins)
    row_scale = (tr_X / X).mean(dim=(1, 2))
    for scale in row_scale:
        assert any(torch.isclose(scale, b) for b in strengths)
    # several strengths must have been hit, which is only possible when the
    # bin is sampled per example rather than once per batch
    assert row_scale.unique().numel() > 1
    assert torch.equal(tr_y, y)


def test_custom_ops_validation():
    with pytest.raises(ValueError, match="custom_ops entries must be"):
        TrivialAugment(custom_ops=[("bad",)])
    with pytest.raises(ValueError, match="str name and a callable"):
        TrivialAugment(custom_ops=[("Scale", "not-callable", (1.0, 2.0))])
    with pytest.raises(ValueError, match="lo < hi"):
        TrivialAugment(custom_ops=[("Scale", _scaling_op, (2.0, 1.0))])


def test_binary_custom_op():
    transform = TrivialAugment(custom_ops=[("Double", _scaling_op, None)])
    assert "Double" in transform.op_names
    X = torch.randn(16, 8, 100)
    y = torch.zeros(16)
    tr_X, tr_y = transform(X, y)
    assert tr_X.shape == X.shape
    assert torch.equal(tr_y, y)


def test_reproducibility(random_batch):
    X, y = random_batch
    out_a = TrivialAugment(random_state=7)(X, y)
    out_b = TrivialAugment(random_state=7)(X, y)
    assert torch.equal(out_a[0], out_b[0])
    out_c = TrivialAugment(random_state=8)(X, y)
    assert not torch.equal(out_a[0], out_c[0])


def test_all_default_ops_eventually_hit():
    """Statistical coverage: every default op is sampled on a large batch."""
    transform = TrivialAugment(random_state=42)
    called = set()
    for entry in transform._pool:
        if entry.variants is None:
            continue
        for i, variant in enumerate(entry.variants):
            entry.variants[i] = _spy(variant, entry.name, called)
    orig_dynamic = transform._apply_dynamic

    def spy_dynamic(name, strength, X_sub, y_sub):
        called.add(name)
        return orig_dynamic(name, strength, X_sub, y_sub)

    transform._apply_dynamic = spy_dynamic
    X = torch.randn(2000, 8, 100)
    y = torch.zeros(2000)
    transform(X, y)
    assert called == set(transform.op_names)


def _spy(variant, name, called):
    def wrapped(X, y):
        called.add(name)
        return variant(X, y)

    return wrapped


def test_picklable_for_dataloader_workers():
    transform = TrivialAugment(sfreq=250, random_state=0)
    restored = pickle.loads(pickle.dumps(transform))
    # windows long enough for BandstopFilter's default filter length
    rng = np.random.RandomState(3)
    X = torch.from_numpy(rng.randn(32, 8, 2000)).float()
    y = torch.zeros(32)
    tr_a, y_a = transform(X.clone(), y.clone())
    tr_b, y_b = restored(X.clone(), y.clone())
    assert torch.equal(tr_a, tr_b)
    assert torch.equal(y_a, y_b)


def test_works_inside_augmented_dataloader(random_batch):
    from torch.utils.data import TensorDataset

    from braindecode.augmentation import AugmentedDataLoader

    ds = TensorDataset(random_batch[0], random_batch[1].long())
    loader = AugmentedDataLoader(ds, batch_size=64, transforms=TrivialAugment(random_state=0))
    for Xb, yb in loader:
        assert Xb.shape[0] == yb.shape[0]
        assert Xb.shape[1:] == (8, 100)
