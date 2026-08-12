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
    Z = project_to_topomap(np.ones(N_CHANS, dtype=np.float32), chs_info)
    finite = Z[np.isfinite(Z)]
    assert finite.std() < 0.05


def test_nonuniform_input_has_spatial_variation(chs_info):
    data = np.random.default_rng(SEED + 1).standard_normal(N_CHANS).astype(np.float32)
    Z = project_to_topomap(data, chs_info)
    finite = Z[np.isfinite(Z)]
    assert finite.std() > 0.01


def test_different_inputs_produce_different_maps(chs_info):
    rng = np.random.default_rng(SEED + 2)
    Z1 = project_to_topomap(rng.standard_normal(N_CHANS).astype(np.float32), chs_info)
    Z2 = project_to_topomap(rng.standard_normal(N_CHANS).astype(np.float32), chs_info)
    assert not np.allclose(Z1, Z2, equal_nan=True)


def test_single_active_channel_produces_peak(chs_info):
    """A single positive channel must yield a positive peak near 1.0."""
    data = np.zeros(N_CHANS, dtype=np.float32)
    data[0] = 1.0
    Z = project_to_topomap(data, chs_info)
    finite = Z[np.isfinite(Z)]
    assert finite.max() > 0.5, f"Peak {finite.max():.3f} is too low for active=1.0 channel"


def test_scaled_input_scales_output(chs_info):
    data = np.abs(np.random.default_rng(SEED + 3).standard_normal(N_CHANS)).astype(np.float32) + 0.5
    scale = 3.0
    Z1 = project_to_topomap(data, chs_info)
    Z2 = project_to_topomap(data * scale, chs_info)
    finite1 = Z1[np.isfinite(Z1)]
    finite2 = Z2[np.isfinite(Z2)]
    np.testing.assert_allclose(finite2, finite1 * scale, rtol=1e-3)
