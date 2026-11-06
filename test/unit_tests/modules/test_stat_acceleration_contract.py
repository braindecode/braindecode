"""Freeze reduction policy before considering fused implementations.

These tests catch changes to correction, clamp order/range, dtype promotion,
axis/storage handling, and derivatives, rather than requiring a faster kernel.
"""

import math
import warnings

import pytest
import torch

from braindecode.modules import LogPowerLayer, LogVarLayer

LAYERS = {"logvar": LogVarLayer, "logpower": LogPowerLayer}
BOUNDS = {"logvar": (1e-6, 1e6), "logpower": (1e-4, 1e4)}


pytestmark = pytest.mark.usefixtures("acceleration_caller_state")


def reference(x, kind, dim, keepdim):
    if kind == "logvar":
        reduced = torch.var(x, dim=dim, correction=1, keepdim=keepdim)
    else:
        reduced = torch.mean(x**2, dim=dim, keepdim=keepdim)
    dtype = reduced.dtype
    promoted = reduced.to(torch.promote_types(dtype, torch.float32))
    return torch.log(torch.clamp(promoted, *BOUNDS[kind])).to(dtype)


def observed(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value, error = call(), None
        except (RuntimeError, IndexError, TypeError) as exc:
            value, error = None, (type(exc), str(exc))
    return value, error, [(w.category, str(w.message)) for w in caught]


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize("length", [0, 1, 2, 7, 250, 257])
@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_log_statistics_reference_contract(kind, length, dim, keepdim, layout, dtype):
    """Wrong correction, axis, keepdim or casting changes this characterization."""
    shape = [2, 3, 4]
    shape[dim] = length
    count = math.prod(shape)
    x = torch.linspace(-2, 3, count, dtype=dtype).reshape(shape)
    if layout == "transposed":
        x = x.transpose(0, 2).contiguous().transpose(0, 2)
    elif layout == "sliced":
        x = torch.stack((x, x), dim=-1)[..., 0]
    got, error, warned = observed(lambda: LAYERS[kind](dim, keepdim)(x))
    want, ref_error, ref_warned = observed(lambda: reference(x, kind, dim, keepdim))
    assert error == ref_error
    assert warned == ref_warned
    if error is None:
        assert got.dtype == want.dtype == dtype
        expected_shape = list(shape)
        if keepdim:
            expected_shape[dim] = 1
        else:
            del expected_shape[dim]
        assert list(got.shape) == expected_shape
        torch.testing.assert_close(got, want, equal_nan=True)


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize(
    "fixture",
    ["zeros", "constant", "near_constant", "large", "nan", "inf", "mixed_inf"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_statistics_extremes_and_autograd(kind, fixture, dtype):
    values = {
        "zeros": [0.0] * 7,
        "constant": [2.0] * 7,
        "near_constant": [1.0 + i * 1e-7 for i in range(7)],
        "large": [-1e20, 1e20] * 3 + [0.0],
        "nan": [float("nan")] + [1.0] * 6,
        "inf": [float("inf")] + [1.0] * 6,
        "mixed_inf": [float("inf"), -float("inf")] + [1.0] * 5,
    }[fixture]
    x = torch.tensor(values, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    got = LAYERS[kind](-1, False)(x)
    want = reference(y, kind, -1, False)
    assert got.dtype == want.dtype == dtype
    torch.testing.assert_close(got, want, equal_nan=True)
    torch.testing.assert_close(
        torch.autograd.grad(got, x)[0],
        torch.autograd.grad(want, y)[0],
        equal_nan=True,
    )


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize("bound_index", [0, 1])
@pytest.mark.parametrize("side", [0.5, 1.0, 2.0])
def test_statistics_clamp_boundary_autograd(kind, bound_index, side):
    """Use exact representable reduction boundaries; no finite differences here."""
    bound = BOUNDS[kind][bound_index]
    if kind == "logvar":
        # [-sqrt(b),0,sqrt(b)] has unbiased variance exactly b for these fixtures.
        values = [-math.sqrt(bound * side), 0.0, math.sqrt(bound * side)]
    else:
        values = [math.sqrt(bound * side)] * 3
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    raw = x.var(correction=1) if kind == "logvar" else (x**2).mean()
    if side == 1.0:
        assert raw.item() == bound
    got = LAYERS[kind](-1, False)(x)
    want = reference(y, kind, -1, False)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(got, x)[0],
        torch.autograd.grad(want, y)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_statistics_smooth_gradcheck_gradgradcheck(kind, dim, keepdim):
    x = torch.tensor(
        [[0.2, 1.1, -0.7], [1.5, -0.4, 0.8]], dtype=torch.float64, requires_grad=True
    )
    layer = LAYERS[kind](dim, keepdim)
    assert torch.autograd.gradcheck(layer, (x,))
    assert torch.autograd.gradgradcheck(layer, (x,))


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize(
    "dim,dtype", [(3, torch.float32), (-4, torch.float32), (1, torch.int64)]
)
def test_statistics_error_contract(kind, dim, dtype):
    x = torch.ones((2, 3, 7), dtype=dtype)
    got, error, warned = observed(lambda: LAYERS[kind](dim)(x))
    want, ref_error, ref_warned = observed(lambda: reference(x, kind, dim, True))
    assert got is want is None
    assert error == ref_error
    assert warned == ref_warned


@pytest.mark.parametrize("kind", LAYERS)
def test_statistics_compiler_fast_smoke(kind):
    """Dynamo captures the real layer and its gradients (no Inductor compile)."""
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    x = torch.linspace(-2, 3, 42).reshape(2, 3, 7).requires_grad_()
    oracle_x = x.detach().clone().requires_grad_()
    layer = LAYERS[kind](-1, False)
    compiled = torch.compile(layer, backend=backend, fullgraph=True)
    actual = compiled(x)
    expected = reference(oracle_x, kind, -1, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, oracle_x.grad, rtol=0, atol=0)
    assert graphs
