import numpy as np
import pytest
import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.visualization.attribution import (
    attribute_image_features,
    get_attributions,
)

N_CHANS = 18
N_TIMES = 600
N_OUTPUTS = 2
BATCH = 6
_SEED = 0


@pytest.fixture
def model():
    m = ShallowFBCSPNet(
        n_chans=N_CHANS,
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
        final_conv_length="auto",
    )
    m.eval()
    return m


@pytest.fixture
def batch(model):
    torch.manual_seed(_SEED)
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


def test_attribute_image_features_layer_method_shape(model, batch):
    """LayerGradCam output at layer resolution must be interpolated to input shape."""
    X, y = batch
    layer = model.final_layer.conv_classifier
    attr = attribute_image_features(model, X, y, "LayerGradCam", layer=layer)
    assert attr.shape == (BATCH, N_CHANS, N_TIMES), (
        f"LayerGradCam: expected {(BATCH, N_CHANS, N_TIMES)}, got {attr.shape}"
    )


@pytest.mark.parametrize("method", [
    "GradCAM",
    "GradCAMPlusPlus",
    "LayerCAM",
])
def test_cam_methods_shape(model, batch, method):
    """CAM methods return (batch, n_times) — spatial dims of the 3-D input."""
    X, y = batch
    layer = model.final_layer.conv_classifier
    attr = attribute_image_features(model, X, y, method, layer=layer)
    assert attr.shape == (BATCH, N_TIMES), (
        f"{method}: expected {(BATCH, N_TIMES)}, got {attr.shape}"
    )


def test_attributions_nonzero(model, batch):
    """Attributions must not be all-zero — gradient must actually flow."""
    X, y = batch
    attr = attribute_image_features(model, X, y, "Saliency")
    assert np.any(attr != 0), "Saliency returned all-zero attributions"


def test_different_inputs_different_attributions(model, batch):
    """Two different inputs must produce different attribution maps."""
    X, y = batch
    attr1 = attribute_image_features(model, X[:3], y[:3], "Saliency")
    attr2 = attribute_image_features(model, X[3:], y[3:], "Saliency")
    assert not np.allclose(attr1, attr2), "Different inputs produced identical attributions"


def test_get_attributions_output_shapes(model):
    """Returned attribution shape must match (n_correct, n_chans, n_times)."""
    rng = np.random.default_rng(_SEED + 1)
    X = rng.standard_normal((BATCH, N_CHANS, N_TIMES)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    attributions, labels = get_attributions(model, X, y, method="Saliency")

    assert attributions.ndim == 3
    assert attributions.shape[1] == N_CHANS
    assert attributions.shape[2] == N_TIMES
    assert labels.shape[0] == attributions.shape[0]


def test_get_attributions_only_correct_samples(model):
    """Only correctly classified samples must be returned."""
    with torch.no_grad():
        model.final_layer.conv_classifier.weight[1, :] = 0.0
        model.final_layer.conv_classifier.bias[1] = -1e6

    rng = np.random.default_rng(_SEED + 2)
    X = rng.standard_normal((BATCH, N_CHANS, N_TIMES)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    attributions, labels = get_attributions(model, X, y, method="Saliency")

    assert np.all(labels == 0), "Only class-0 samples should be returned"
    assert attributions.shape[0] == (y == 0).sum()
