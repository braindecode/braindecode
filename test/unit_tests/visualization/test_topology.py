import numpy as np
import pytest

from braindecode.visualization.topology import project_to_topomap

from .conftest import SEED

N_CHANS = 16


@pytest.fixture(scope="module")
def chs_info(chs_info_factory):
    return chs_info_factory(N_CHANS)


@pytest.fixture
def random_data():
    return np.random.default_rng(SEED).standard_normal(N_CHANS).astype(np.float32)


def test_default_resolution(chs_info, random_data):
    Z = project_to_topomap(random_data, chs_info)
    assert Z.shape == (64, 64)


def test_custom_resolution(chs_info, random_data):
    Z = project_to_topomap(random_data, chs_info, res=32)
    assert Z.shape == (32, 32)


def test_uniform_input_low_spatial_variance(chs_info):
    """All channels with same value → interpolated map must be near-constant."""
    data = np.ones(N_CHANS, dtype=np.float32)
    Z = project_to_topomap(data, chs_info)
    finite = Z[np.isfinite(Z)]
    assert finite.std() < 0.05, f"Std too high for uniform input: {finite.std():.4f}"


def test_nonuniform_input_has_spatial_variation(chs_info):
    """Non-uniform channel values must produce a map with meaningful variance."""
    data = np.random.default_rng(SEED + 1).standard_normal(N_CHANS).astype(np.float32)
    Z = project_to_topomap(data, chs_info)
    finite = Z[np.isfinite(Z)]
    assert finite.std() > 0.01, f"Std too low for non-uniform input: {finite.std():.4f}"


def test_different_inputs_produce_different_maps(chs_info):
    rng = np.random.default_rng(SEED + 2)
    data1 = rng.standard_normal(N_CHANS).astype(np.float32)
    data2 = rng.standard_normal(N_CHANS).astype(np.float32)
    Z1 = project_to_topomap(data1, chs_info)
    Z2 = project_to_topomap(data2, chs_info)
    assert not np.allclose(Z1, Z2, equal_nan=True)


def test_peak_near_active_electrode(chs_info):
    """A single active channel must produce a peak close to that electrode's 2-D position."""
    import mne
    from mne.channels.layout import _find_topomap_coords
    from mne.utils import _check_sphere

    data = np.zeros(N_CHANS, dtype=np.float32)
    data[0] = 1.0

    Z = project_to_topomap(data, chs_info, res=64)

    info = mne.create_info(
        ch_names=[ch["ch_name"] for ch in chs_info],
        sfreq=256.0,
        ch_types="eeg",
    )
    with info._unlock():
        for i, ch in enumerate(chs_info):
            info["chs"][i]["loc"] = ch["loc"]
    sphere = _check_sphere(None)
    pos2d = _find_topomap_coords(info, picks=list(range(N_CHANS)), sphere=sphere)

    # Match the head-extent grid that _make_head_outlines builds (mask_scale=1.25).
    x0, y0 = pos2d[0]
    sx, sy, _, radius = sphere
    grid_extent = radius * 1.25
    xmin, xmax = sx - grid_extent, sx + grid_extent
    ymin, ymax = sy - grid_extent, sy + grid_extent
    col = int(np.clip((x0 - xmin) / (xmax - xmin) * 63, 0, 63))
    row = int(np.clip((y0 - ymin) / (ymax - ymin) * 63, 0, 63))

    peak_value = Z[row, col]
    finite_values = Z[np.isfinite(Z)]
    assert peak_value > np.percentile(finite_values, 75), (
        f"Active electrode pixel value {peak_value:.3f} is not in top 25% of map"
    )


def test_scaled_input_scales_output(chs_info):
    data = np.abs(np.random.default_rng(SEED + 3).standard_normal(N_CHANS)).astype(np.float32) + 0.5
    scale = 3.0
    Z1 = project_to_topomap(data, chs_info)
    Z2 = project_to_topomap(data * scale, chs_info)
    finite1 = Z1[np.isfinite(Z1)]
    finite2 = Z2[np.isfinite(Z2)]
    np.testing.assert_allclose(finite2, finite1 * scale, rtol=1e-3)
