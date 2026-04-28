import copy

import numpy as np
import pytest
import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.visualization.attribution import (
    input_x_gradient,
    integrated_gradients,
    layer_grad_cam,
    saliency,
    select_correctly_classified,
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
    X = torch.randn(BATCH, N_CHANS, N_TIMES)
    with torch.no_grad():
        y = model(X).argmax(dim=1)
    return X, y


@pytest.mark.parametrize("fn", [saliency, input_x_gradient, integrated_gradients])
def test_input_attribution_shape(model, batch, fn):
    X, y = batch
    attr = fn(model, X, y).detach().cpu().numpy()
    assert attr.shape == (BATCH, N_CHANS, N_TIMES)


def test_layer_grad_cam_shape(model, batch):
    X, y = batch
    attr = layer_grad_cam(model, X, y, model.final_layer.conv_classifier)
    assert tuple(attr.shape) == (BATCH, N_CHANS, N_TIMES)


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


def test_select_correctly_classified_shapes(model):
    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((BATCH, N_CHANS, N_TIMES)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    X_correct, y_correct = select_correctly_classified(model, X, y)
    assert X_correct.ndim == 3
    assert X_correct.shape[1:] == (N_CHANS, N_TIMES)
    assert y_correct.shape[0] == X_correct.shape[0]


def test_select_correctly_classified_filters_wrong_class(model):
    """Module-scoped model is deepcopied since this test mutates classifier weights."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.final_layer.conv_classifier.weight[1, :] = 0.0
        m.final_layer.conv_classifier.bias[1] = -1e6

    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((BATCH, N_CHANS, N_TIMES)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])

    X_correct, y_correct = select_correctly_classified(m, X, y)
    assert torch.all(y_correct == 0)
    assert X_correct.shape[0] == int((y == 0).sum())
