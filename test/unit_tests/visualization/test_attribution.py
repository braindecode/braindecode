import copy

import numpy as np
import pytest
import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.visualization.attribution import (
    attribute_image_features,
    get_attributions,
)

from .conftest import SEED

N_CHANS = 18
N_TIMES = 600
N_OUTPUTS = 2
BATCH = 6


@pytest.fixture(scope="module")
def model():
    m = ShallowFBCSPNet(
        n_chans=N_CHANS,
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
        final_conv_length="auto",
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch(model):
    torch.manual_seed(SEED)
    X = torch.randn(BATCH, N_CHANS, N_TIMES, requires_grad=True)
    with torch.no_grad():
        y = model(X).argmax(dim=1)
    return X, y


@pytest.mark.parametrize("method", [
    "Saliency",
    "InputXGradient",
    "IntegratedGradients",
    "GuidedBackprop",
    "DeepLift",
])
def test_attribute_image_features_shape(model, batch, method):
    X, y = batch
    attr = attribute_image_features(model, X, y, method)
    assert attr.shape == (BATCH, N_CHANS, N_TIMES), (
        f"{method}: expected {(BATCH, N_CHANS, N_TIMES)}, got {attr.shape}"
    )


def test_layer_grad_cam_shape(model, batch):
    """LayerGradCam output is interpolated back to the input shape."""
    X, y = batch
    layer = model.final_layer.conv_classifier
    attr = attribute_image_features(model, X, y, "LayerGradCam", layer=layer)
    assert attr.shape == (BATCH, N_CHANS, N_TIMES)


def test_attributions_meaningfully_nonzero(model, batch):
    X, y = batch
    attr = attribute_image_features(model, X, y, "Saliency")
    assert np.mean(np.abs(attr)) > 1e-8, (
        f"Saliency mean abs attribution {np.mean(np.abs(attr)):.2e} is too small"
    )


def test_different_inputs_different_attributions(model, batch):
    X, y = batch
    attr1 = attribute_image_features(model, X[:3], y[:3], "Saliency")
    attr2 = attribute_image_features(model, X[3:], y[3:], "Saliency")
    assert not np.allclose(attr1, attr2)


@pytest.mark.parametrize("method", ["LayerGradCam", "GuidedGradCam"])
def test_layer_required_methods_raise_without_layer(model, batch, method):
    X, y = batch
    with pytest.raises(ValueError, match="requires a `layer` argument"):
        attribute_image_features(model, X, y, method)


def test_unsupported_method_raises(model, batch):
    X, y = batch
    with pytest.raises(ValueError, match="Unsupported attribution method"):
        attribute_image_features(model, X, y, "NotAMethod")


def test_get_attributions_output_shapes(model):
    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((BATCH, N_CHANS, N_TIMES)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    attributions, labels = get_attributions(model, X, y, method="Saliency")

    assert attributions.ndim == 3
    assert attributions.shape[1] == N_CHANS
    assert attributions.shape[2] == N_TIMES
    assert labels.shape[0] == attributions.shape[0]


def test_get_attributions_only_correct_samples(model):
    """Module-scoped model is deepcopied because this test mutates classifier weights."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.final_layer.conv_classifier.weight[1, :] = 0.0
        m.final_layer.conv_classifier.bias[1] = -1e6

    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((BATCH, N_CHANS, N_TIMES)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    attributions, labels = get_attributions(m, X, y, method="Saliency")

    assert np.all(labels == 0)
    assert attributions.shape[0] == (y == 0).sum()
