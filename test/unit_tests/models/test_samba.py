"""SAMBA CPU recurrence, training, coordinate and checkpoint contracts."""

from copy import deepcopy

import pytest
import torch

from braindecode.models import SAMBA
from braindecode.models.samba import PortableMamba2


def make_model():
    return SAMBA(
        n_chans=2,
        n_outputs=3,
        n_times=72,
        chs_info=[{"loc": [-0.03, 0.08, 0.01]}, {"loc": [0.03, 0.08, 0.01]}],
        base_channels=8,
        bottleneck_channels=8,
        head_dim=4,
        d_state=3,
        expand=8,
    )


@pytest.mark.parametrize("length", [1, 63, 64, 65, 129])
def test_scan_forward_backward(length):
    torch.manual_seed(3)
    m = PortableMamba2(4, d_state=3, expand=2, headdim=4).double()
    u = torch.randn(2, length, 4, dtype=torch.double, requires_grad=True)
    a = m(u)
    b = m.recurrence_reference(u)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
    params = (u, *m.parameters())
    ga = torch.autograd.grad(a.square().sum(), params, retain_graph=True)
    gb = torch.autograd.grad(b.square().sum(), params)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-6)


def test_train_reload_and_config():
    torch.manual_seed(4)
    m = make_model()
    x = torch.randn(2, 2, 72)
    old = m.encoder.encoder1[1].mixer.in_proj.weight.detach().clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(m(x), torch.tensor([0, 1]))
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    opt.step()
    assert not torch.equal(old, m.encoder.encoder1[1].mixer.in_proj.weight)
    other = SAMBA.from_config(m.get_config())
    other.load_state_dict(m.state_dict())
    torch.testing.assert_close(other(x), m(x), atol=0, rtol=0)


def test_coordinates_and_checkpoint():
    m = make_model().eval()
    x = torch.randn(2, 2, 72)
    permuted = deepcopy(m)
    permuted.input_positions.copy_(m.input_positions.flip(0))
    torch.testing.assert_close(m(x), permuted(x.flip(1)))
    state = m.encoder.state_dict()
    m.load_pretrained_encoder(state)
    with pytest.raises(RuntimeError):
        m.load_pretrained_encoder(
            {k: v for k, v in state.items() if k != "encoder1.0.weight"}
        )
    with pytest.raises(ValueError, match="Unrecognized"):
        m.load_pretrained_encoder(dict(state, unknown=torch.ones(1)))
    bad = dict(state)
    bad["cdse.target_positions"] = state["cdse.target_positions"] + 1
    with pytest.raises(ValueError, match="montage"):
        m.load_pretrained_encoder(bad)
    with pytest.raises(ValueError):
        m(torch.randn(2, 2, 8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_gated_norm(dtype):
    from braindecode.models.samba import PortableGatedNorm

    x = torch.full((2, 4, 8), 1000.0, dtype=dtype, requires_grad=True)
    z = torch.ones_like(x, requires_grad=True)
    m = PortableGatedNorm(8).to(dtype)
    y = m(x, z)
    torch.testing.assert_close(
        y.float(), torch.ones_like(y).float(), atol=0.01, rtol=0.01
    )
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(z.grad).all()


def test_mamba_initialization():
    torch.manual_seed(44)
    m = PortableMamba2(64)
    dt = torch.nn.functional.softplus(m.dt_bias.detach())
    a = m.A_log.detach().exp()
    assert (dt >= 0.001).all() and (dt <= 0.1).all()
    assert (a >= 1).all() and (a <= 16).all()
    assert dt.std() > 0 and a.std() > 0
    assert all(p._no_weight_decay for p in (m.dt_bias, m.A_log, m.D))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_training(dtype):
    m = make_model().to(dtype)
    x = torch.randn(2, 2, 72).to(dtype)
    y = m(x)
    y.float().square().mean().backward()
    assert torch.isfinite(y).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


def test_reset_head_and_parameters():
    m = make_model().double().eval()
    m.encoder.encoder1.train()  # Preserve a deliberately mixed mode.
    old = {k: v.clone() for k, v in m.encoder.state_dict().items()}
    m.reset_head(5)
    assert m.n_outputs == 5 and m.get_config()["n_outputs"] == 5
    assert not m.final_layer.training and m.final_layer.weight.dtype == torch.double
    assert m.encoder.encoder1.training
    for key, value in m.encoder.state_dict().items():
        torch.testing.assert_close(value, old[key], atol=0, rtol=0)
    x = torch.randn(2, 2, 72, dtype=torch.double)
    restored = SAMBA.from_config(m.get_config()).double()
    restored.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(x), restored(x))
    for invalid in (0, None, -1):
        with pytest.raises(ValueError, match="positive integer"):
            m.reset_head(invalid)
    mixer = m.encoder.encoder1[1].mixer
    with torch.no_grad():
        mixer.dt_bias.zero_()
        mixer.A_log.zero_()
        mixer.D.zero_()
        mixer.norm.weight.zero_()
    m.reset_parameters()
    dt = torch.nn.functional.softplus(mixer.dt_bias.detach())
    assert (dt >= 0.001).all() and (dt <= 0.1).all()
    assert (mixer.A_log.exp() >= 1).all() and (mixer.A_log.exp() <= 16).all()
    torch.testing.assert_close(mixer.D, torch.ones_like(mixer.D))
    torch.testing.assert_close(mixer.norm.weight, torch.ones_like(mixer.norm.weight))
