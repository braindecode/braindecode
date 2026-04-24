import mne
import numpy as np
import pytest

from braindecode.visualization.topology import project_to_topomap

N_CHANS = 16
_SEED = 0


@pytest.fixture
def chs_info():
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:N_CHANS]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


@pytest.fixture
def random_data():
    return np.random.default_rng(_SEED).standard_normal(N_CHANS).astype(np.float32)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_default_resolution(chs_info, random_data):
    Z = project_to_topomap(random_data, chs_info)
    assert Z.shape == (64, 64)


def test_custom_resolution(chs_info, random_data):
    Z = project_to_topomap(random_data, chs_info, res=32)
    assert Z.shape == (32, 32)


# ---------------------------------------------------------------------------
# Interpolation correctness
# ---------------------------------------------------------------------------

def test_uniform_input_low_spatial_variance(chs_info):
    """All channels with same value → interpolated map must be near-constant."""
    data = np.ones(N_CHANS, dtype=np.float32)
    Z = project_to_topomap(data, chs_info)
    finite = Z[np.isfinite(Z)]
    assert finite.std() < 0.05, f"Std too high for uniform input: {finite.std():.4f}"


def test_nonuniform_input_has_spatial_variation(chs_info):
    """Non-uniform channel values must produce a map with meaningful variance."""
    data = np.random.default_rng(_SEED + 1).standard_normal(N_CHANS).astype(np.float32)
    Z = project_to_topomap(data, chs_info)
    finite = Z[np.isfinite(Z)]
    assert finite.std() > 0.01, f"Std too low for non-uniform input: {finite.std():.4f}"


def test_different_inputs_produce_different_maps(chs_info):
    """Two different channel vectors must produce different scalp maps."""
    rng = np.random.default_rng(_SEED + 2)
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

    # Activate only the first channel
    data = np.zeros(N_CHANS, dtype=np.float32)
    data[0] = 1.0

    Z = project_to_topomap(data, chs_info, res=64)

    # Get the 2-D projected position of the first electrode
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

    # Map the electrode's 2-D coordinate to a pixel index in Z.
    # The grid spans the head circle boundary — use the sphere radius to get
    # the correct extent rather than the electrode position bounds.
    x0, y0 = pos2d[0]
    sphere = _check_sphere(None)
    sx, sy, _, radius = sphere
    grid_extent = radius * 1.25  # matches mask_scale in _make_head_outlines
    xmin, xmax = sx - grid_extent, sx + grid_extent
    ymin, ymax = sy - grid_extent, sy + grid_extent
    col = int((x0 - xmin) / (xmax - xmin) * (64 - 1))
    row = int((y0 - ymin) / (ymax - ymin) * (64 - 1))
    col = np.clip(col, 0, 63)
    row = np.clip(row, 0, 63)

    # The pixel at the active electrode must be among the top values
    peak_value = Z[row, col]
    finite_values = Z[np.isfinite(Z)]
    assert peak_value > np.percentile(finite_values, 75), (
        f"Active electrode pixel value {peak_value:.3f} is not in top 25% of map"
    )


def test_scaled_input_scales_output(chs_info):
    """Scaling channel values by a constant must scale the map by the same factor."""
    data = np.abs(np.random.default_rng(_SEED + 3).standard_normal(N_CHANS)).astype(np.float32) + 0.5
    scale = 3.0
    Z1 = project_to_topomap(data, chs_info)
    Z2 = project_to_topomap(data * scale, chs_info)
    finite1 = Z1[np.isfinite(Z1)]
    finite2 = Z2[np.isfinite(Z2)]
    np.testing.assert_allclose(finite2, finite1 * scale, rtol=1e-3)
