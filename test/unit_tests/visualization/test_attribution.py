import numpy as np
import pytest
import torch
from torch import nn

# All attribution methods are captum-backed wrappers; skip cleanly if
# captum (the optional `viz` dep) isn't installed.
pytest.importorskip("captum")

from braindecode.models import EEGNet, EEGSimpleConv, ShallowFBCSPNet  # noqa: E402
from braindecode.visualization.attribution import (  # noqa: E402
    input_x_gradient,
    integrated_gradients,
    layer_grad_cam,
    saliency,
)

from .conftest import SEED  # noqa: E402

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
    X = torch.randn(BATCH, N_CHANS, N_TIMES)
    with torch.no_grad():
        y = model(X).argmax(dim=1)
    return X, y


@pytest.mark.parametrize("fn", [saliency, input_x_gradient, integrated_gradients])
def test_input_attribution_shape(model, batch, fn):
    X, y = batch
    attr = fn(model, X, y).detach().cpu().numpy()
    assert attr.shape == (BATCH, N_CHANS, N_TIMES)


@pytest.mark.parametrize("layer_name", ["pool", "final_layer.conv_classifier"])
def test_layer_grad_cam_rejects_layers_that_are_not_channels_by_time(
    model, batch, layer_name
):
    """ShallowFBCSPNet holds time on the height axis after its dimshuffle."""
    X, y = batch
    with pytest.raises(ValueError, match=layer_name):
        layer_grad_cam(model, X, y, model.get_submodule(layer_name))


@pytest.mark.parametrize(
    "build, layer_name, shape",
    [
        (
            lambda: EEGSimpleConv(
                n_chans=3, n_outputs=2, sfreq=80, n_times=128, feature_maps=4
            ),
            "conv",
            (2, 128),
        ),
        (
            lambda: EEGNet(n_chans=3, n_outputs=2, n_times=128),
            "conv_temporal",
            (2, 3, 128),
        ),
    ],
    ids=["one-dimensional", "channels-by-time"],
)
def test_layer_grad_cam_shape(build, layer_name, shape):
    """A one-dimensional layer gives a map over time, a channels x time layer
    a map over both."""
    model = build().eval()
    layer = model.get_submodule(layer_name)
    cam = layer_grad_cam(model, torch.randn(2, 3, 128), torch.tensor([0, 1]), layer)
    assert cam.shape == shape
    assert not cam.requires_grad


@pytest.mark.parametrize(
    "layers, conv_index, peak",
    [
        ((nn.Conv1d(3, 1, 1, bias=False), nn.AdaptiveAvgPool1d(1)), 0, (40,)),
        (
            (
                nn.Unflatten(1, (1, 3)),
                nn.Conv2d(1, 1, 1, bias=False),
                nn.AdaptiveAvgPool2d(1),
            ),
            1,
            (1, 40),
        ),
    ],
    ids=["time", "channels-by-time"],
)
def test_layer_grad_cam_peaks_at_a_burst(layers, conv_index, peak):
    """With every weight 1 and a mean over positions, the map is the signal
    times a positive constant: it peaks where the burst is."""
    model = nn.Sequential(*layers, nn.Flatten(), nn.Linear(1, 2, bias=False))
    for param in model.parameters():
        nn.init.ones_(param)
    x = torch.zeros(1, 3, 64)
    x[0, 1, 40] = 1.0
    cam = layer_grad_cam(model, x, torch.tensor([0]), model[conv_index])
    assert np.unravel_index(cam[0].argmax().item(), cam[0].shape) == peak


def test_layer_grad_cam_requires_spatial_axis():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    with pytest.raises(ValueError, match="one or two spatial dimensions"):
        layer_grad_cam(model, torch.randn(1, 3, 4), torch.tensor([0]), model[1])


def test_saliency_meaningfully_nonzero(model, batch):
    X, y = batch
    attr = saliency(model, X, y).detach().cpu().numpy()
    assert np.mean(np.abs(attr)) > 1e-8


def test_different_inputs_different_attributions(model, batch):
    X, y = batch
    a1 = saliency(model, X[:3], y[:3]).detach().cpu().numpy()
    a2 = saliency(model, X[3:], y[3:]).detach().cpu().numpy()
    assert not np.allclose(a1, a2)


def test_integrated_gradients_zero_at_baseline(model, batch):
    """IG must be zero when input equals baseline (delta is zero)."""
    X, y = batch
    baseline = X.clone()
    attr = integrated_gradients(model, X, y, baseline=baseline, steps=8)
    np.testing.assert_allclose(attr.detach().cpu().numpy(), 0.0, atol=1e-6)


def test_integrated_gradients_completeness_on_linear_model():
    """IG completeness: sum_i attr_i == f(x) - f(baseline) holds exactly for
    linear models, where grad is constant along the path."""
    torch.manual_seed(SEED)
    n_in, n_out = N_CHANS * N_TIMES, N_OUTPUTS

    class FlatLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(n_in, n_out)

        def forward(self, x):
            return self.lin(x.flatten(1))

    m = FlatLinear().eval()
    X = torch.randn(BATCH, N_CHANS, N_TIMES)
    y = torch.randint(0, n_out, (BATCH,))
    baseline = torch.zeros_like(X)

    attr = integrated_gradients(m, X, y, baseline=baseline, steps=8)
    with torch.no_grad():
        delta = (
            m(X).gather(1, y.view(-1, 1)).squeeze()
            - m(baseline).gather(1, y.view(-1, 1)).squeeze()
        )
    summed = attr.sum(dim=tuple(range(1, attr.ndim)))
    np.testing.assert_allclose(summed.numpy(), delta.numpy(), atol=1e-5)
