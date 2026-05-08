import pytest
import torch

from braindecode.augmentation.functional import (
    _analytic_transform,
    amplitude_scale,
    band_rotation,
    channels_rereference,
    channels_shuffle,
    segmentation_reconstruction,
)


def test_channels_shuffle():
    X = torch.rand((5, 64, 100))
    # Random EEG data for 5 examples, 64 channels, and 100 time points
    y = torch.randint(0, 2, (5,))
    # Random labels for 5 examples
    p_shuffle = 0
    random_state = 42

    transformed_X, transformed_y = channels_shuffle(X, y, p_shuffle,
                                                    random_state)

    # Check if the output is the same as the input
    assert torch.equal(transformed_X, X)
    assert torch.equal(transformed_y, y)


def test_analytic_transform():
    # Create a real tensor
    x = torch.rand((5, 64,
                    100))
    # Random data for 5 examples, 64 channels, and 100 time points

    # Call the _analytic_transform function
    transformed_x = _analytic_transform(x)

    # Check if the output is complex
    assert transformed_x.is_complex()

    # Check if the output has the same shape as the input
    assert transformed_x.shape == x.shape

    # Check if the function raises a ValueError when the input is complex
    with pytest.raises(ValueError):
        _analytic_transform(torch.complex(torch.rand(5), torch.rand(5)))


def test_analytic_transform_even():
    # Create a real tensor with an even length in the last dimension
    x = torch.rand((5, 64, 100))
    # Random data for 5 examples, 64 channels, and 100 time points

    # Call the _analytic_transform function
    transformed_x = _analytic_transform(x)

    # Check if the output is complex
    assert transformed_x.is_complex()

    # Check if the output has the same shape as the input
    assert transformed_x.shape == x.shape


def test_segmentation_reconstruction():
    X = torch.zeros((20, 64, 100))
    # Random EEG data for 20 examples, 64 channels, and 100 time points
    y = torch.randint(0, 4, (20,))
    # Random labels for 5 examples

    n_segments = 5

    from sklearn.utils import check_random_state
    rng = check_random_state(42)

    classes = torch.unique(y)
    data_classes = [(i, X[y == i]) for i in classes]

    rand_idxs = dict()
    for label, X_class in data_classes:
        n_trials = X_class.shape[0]
        rand_idxs[label] = rng.randint(0, n_trials, (n_trials, n_segments))

    idx_shuffle = rng.permutation(X.shape[0])

    transformed_X, transformed_y = segmentation_reconstruction(X, y, n_segments,
                                                               data_classes, rand_idxs,
                                                               idx_shuffle)

    # Check the output
    assert torch.equal(transformed_X, X)
    # preserve time sequence
    assert torch.equal(torch.bincount(transformed_y), torch.bincount(y))
    # preserve number of occurrences of each label


def test_amplitude_scale():
    X = torch.rand((5, 64, 100))
    # Random EEG data for 5 examples, 64 channels, and 100 time points
    y = torch.randint(0, 2, (5,))
    # Random labels for 5 examples
    random_state = 42

    scale = (1,1) # sanity check
    transformed_X1, _ = amplitude_scale(X, y, scale, random_state)
    scale = (0,0) # sanity check
    transformed_X0, _ = amplitude_scale(X, y, scale, random_state)

    # Check if the output is the same as the input
    assert torch.equal(transformed_X1, X)
    assert torch.equal(transformed_X0, torch.zeros_like(X))


def test_band_rotation_shape_and_seed_reproducibility():
    # 2 bands × 8 electrodes/band; small T for fast assertions.  Force
    # ``band_offsets=(2,)`` so we get a deterministic non-zero rotation
    # regardless of the RNG draw — the previous version asserted the
    # output differs from the input under ``random_state=42``, which
    # silently couples the test to the sampler's exact draw sequence.
    X = torch.randn(4, 16, 32)
    y = torch.zeros(4)
    kwargs = dict(
        num_bands=2, electrodes_per_band=8,
        band_offsets=(2,), max_temporal_jitter=0,
        random_state=42,
    )
    out1, _ = band_rotation(X, y, **kwargs)
    out2, _ = band_rotation(X, y, **kwargs)
    assert out1.shape == X.shape
    assert torch.equal(out1, out2)  # same seed → identical
    assert not torch.equal(out1, X)  # offsets=(2,) guarantees a roll


def test_band_rotation_no_op_when_offsets_zero_and_no_jitter():
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    out, _ = band_rotation(
        X, y, num_bands=2, electrodes_per_band=16,
        band_offsets=(0,), max_temporal_jitter=0,
        random_state=0,
    )
    assert torch.equal(out, X)


def test_band_rotation_rejects_mismatched_channel_count():
    X = torch.randn(2, 30, 64)  # 30 != 2 * 16
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="num_bands \\* electrodes_per_band="):
        band_rotation(
            X, y, num_bands=2, electrodes_per_band=16,
            band_offsets=(-1, 1), max_temporal_jitter=0,
            random_state=0,
        )


def test_band_rotation_rejects_empty_band_offsets():
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="band_offsets must be non-empty"):
        band_rotation(
            X, y, num_bands=2, electrodes_per_band=16,
            band_offsets=(), max_temporal_jitter=0,
            random_state=0,
        )


def test_band_rotation_rejects_negative_jitter():
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="max_temporal_jitter must be >= 0"):
        band_rotation(
            X, y, num_bands=2, electrodes_per_band=16,
            band_offsets=(-1, 1), max_temporal_jitter=-3,
            random_state=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"num_bands": 0}, "num_bands must be >= 1"),
        ({"electrodes_per_band": 0}, "electrodes_per_band must be >= 1"),
        ({"band_offsets": (1.5, 2.0)}, "band_offsets must contain integers"),
    ],
)
def test_band_rotation_rejects_invalid_params(kwargs, msg):
    X = torch.randn(2, 32, 64)
    y = torch.zeros(2)
    base = dict(num_bands=2, electrodes_per_band=16, band_offsets=(-1, 1),
                max_temporal_jitter=0, random_state=0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        band_rotation(X, y, **base)


def test_band_rotation_circular_jitter_wraps_vs_zero_pads():
    """``circular_jitter=False`` zero-pads the gap; ``True`` (default)
    wraps end-of-window samples to the start.  We use a deterministic
    band_offsets=(0,) so the rotation step is a no-op and the only
    visible change is the temporal shift on band 1."""
    # Band 0 = constant 1.0, band 1 = ramp 1..T (avoid 0 so we can check
    # "is 0 present?" as a marker of zero-padding).  Channel layout
    # ``(B=1, num_bands*E=4, T=8)`` keeps assertions readable.
    T = 8
    band0 = torch.ones(1, 2, T)
    band1 = torch.arange(1, T + 1, dtype=torch.float).expand(1, 2, T).clone()
    X = torch.cat([band0, band1], dim=1)  # (1, 4, T)
    y = torch.zeros(1)

    common = dict(
        num_bands=2, electrodes_per_band=2, band_offsets=(0,),
        max_temporal_jitter=3,
    )
    # With a fixed seed, both branches sample the same shift; only the
    # boundary handling differs.
    out_circ, _ = band_rotation(X, y, **common, circular_jitter=True, random_state=11)
    out_pad, _ = band_rotation(X, y, **common, circular_jitter=False, random_state=11)

    # Same shift sampled in both → outputs disagree only inside the
    # |shift|-wide boundary region of band 1.
    diff = (out_circ != out_pad).any(dim=1).squeeze(0)  # (T,)
    n_diff = int(diff.sum())
    assert 0 < n_diff <= common["max_temporal_jitter"]
    # Padded variant has zeros somewhere on band 1's edge; circular has
    # the original ramp values wrapped around.
    band1_pad = out_pad[0, 2:, :]
    band1_circ = out_circ[0, 2:, :]
    assert (band1_pad == 0).any()
    assert not (band1_circ == 0).any()


def test_band_rotation_circular_roll_preserves_values():
    """Per-band roll is circular: every input value should appear in the
    output (no zero-padding) when offsets are non-trivial."""
    # Small, deterministic input so the sorted-list comparison stays
    # ``O(n log n)`` on a few-hundred-element tensor.  Don't scale this up
    # to large shapes — for production-size inputs use a ``set``-based
    # equality check instead.
    X = torch.arange(2 * 16 * 8).float().view(2, 16, 8)
    y = torch.zeros(2)
    out, _ = band_rotation(
        X, y, num_bands=2, electrodes_per_band=8,
        band_offsets=(2,),  # force a +2 roll on every band
        max_temporal_jitter=0,
        random_state=0,
    )
    assert sorted(out.flatten().tolist()) == sorted(X.flatten().tolist())


def test_channel_reref():
    X0 = torch.zeros((5, 64, 100))  # sanity
    X1 = torch.ones((5, 64, 100))  # sanity

    # Random EEG data for 5 examples, 64 channels, and 100 time points
    y = torch.randint(0, 2, (5,))
    # Random labels for 5 examples
    random_state = 42

    transformed_X0, _ = channels_rereference(X0, y, random_state)
    transformed_X1, _ = channels_rereference(X1, y, random_state)
    # Check if the output is the same as the input
    assert torch.equal(transformed_X0, X0)
    assert torch.equal(transformed_X1.sum(dim=(1,2)), -100*torch.ones(5))
