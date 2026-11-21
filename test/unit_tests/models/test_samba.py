"""SAMBA: the scan over time, the coordinate-driven input and the initialization."""

import numpy as np
import pytest
import torch

from braindecode.models import SAMBA
from braindecode.models.samba import _ssd_scan

N_TIMES = 131  # not a multiple of 8: the pooling drops the tail


def _chs_info(n_chans, seed=0, loc=None):
    """Electrodes at random positions in metres; ``loc`` places them all there."""
    rng = np.random.default_rng(seed)
    positions = rng.normal(scale=0.07, size=(n_chans, 3))
    if loc is not None:
        positions[:] = loc
    return [
        {"ch_name": f"E{i}", "kind": "eeg", "loc": np.r_[positions[i], np.zeros(9)]}
        for i in range(n_chans)
    ]


def _samba(chs_info, **kwargs):
    return SAMBA(
        n_outputs=3,
        chs_info=chs_info,
        n_times=N_TIMES,
        sfreq=128.0,
        base_channels=8,
        bottleneck_channels=8,
        head_dim=4,
        d_state=3,
        **kwargs,
    ).eval()


def _recurrence(x, dt, a, b, c, d):
    """The scan, one step at a time: ``h = exp(dt * a) h + dt x b``, ``y = h c + d x``."""
    state = x.new_zeros(x.shape[0], x.shape[2], x.shape[3], b.shape[-1])
    ys = []
    for t in range(x.shape[1]):
        decay = torch.exp(dt[:, t] * a)[..., None, None]
        state = (
            decay * state
            + dt[:, t, :, None, None] * x[:, t, ..., None] * b[:, t, None, None]
        )
        ys.append((state * c[:, t, None, None]).sum(-1) + d[:, None] * x[:, t])
    return torch.stack(ys, dim=1)


@pytest.mark.parametrize("n_steps", [1, 63, 64, 65, 130])
def test_scan_in_chunks_equals_the_recurrence(n_steps):
    """The scan runs in chunks of 64 steps and the state crosses their borders:
    it decays slowly here, so that it still matters two chunks later."""
    generator = torch.Generator().manual_seed(0)

    def uniform(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.double)

    x = uniform(2, n_steps, 3, 4) - 0.5
    dt = 0.01 + 0.04 * uniform(2, n_steps, 3)
    a = -(0.1 + 0.4 * uniform(3))
    b, c = uniform(2, n_steps, 5) - 0.5, uniform(2, n_steps, 5) - 0.5
    d = uniform(3)
    torch.testing.assert_close(
        _ssd_scan(x, dt, a, b, c, d), _recurrence(x, dt, a, b, c, d)
    )


def test_output_does_not_depend_on_the_channel_order():
    """Electrodes enter through their positions: the same electrodes in another
    order, with the same weights, give the same output."""
    torch.manual_seed(0)
    chs_info, order = _chs_info(4), [2, 0, 3, 1]
    targets = _chs_info(6, seed=1)
    targets = [ch["loc"][:3].tolist() for ch in targets]
    model = _samba(chs_info, target_positions=targets)
    permuted = _samba([chs_info[i] for i in order], target_positions=targets)
    permuted.load_state_dict(model.state_dict())
    x = torch.randn(2, 4, N_TIMES)
    with torch.no_grad():
        torch.testing.assert_close(permuted(x[:, order]), model(x))


def test_loaded_weights_bring_their_targets_and_keep_the_electrodes():
    """Weights trained on one montage run on another: they set the target
    montage, and the electrodes stay those of the model."""
    torch.manual_seed(0)
    targets = [ch["loc"][:3].tolist() for ch in _chs_info(5, seed=3)]
    chs_info = _chs_info(5, seed=2)
    source = _samba(_chs_info(3, seed=1), target_positions=targets)
    expected = _samba(chs_info, target_positions=targets)
    expected.load_state_dict(source.state_dict())
    loaded = _samba(chs_info)  # its own electrodes as targets, until the load
    loaded.load_state_dict(source.state_dict())
    x = torch.randn(2, 5, N_TIMES)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), expected(x))


def test_config_keeps_the_targets_for_another_montage():
    """A model trained from scratch rebuilds from its config, and takes its
    weights, on a montage with another number of electrodes."""
    model = _samba(_chs_info(3))
    other = SAMBA.from_config({**model.get_config(), "chs_info": _chs_info(5, seed=1)})
    other.load_state_dict(model.state_dict())
    assert other(torch.randn(2, 5, N_TIMES)).shape == (2, 3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"chs_info": _chs_info(2, loc=np.nan)}, "positions"),
        ({"chs_info": _chs_info(2, loc=0.0)}, "positions"),
        (
            {"chs_info": _chs_info(2), "target_positions": [[0.0, 0.0]]},
            "target_positions",
        ),
    ],
    ids=["nan", "zeros", "target_positions"],
)
def test_positions_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _samba(**kwargs)


def test_window_shorter_than_the_pooling_is_refused():
    model = _samba(_chs_info(3))
    with pytest.raises(ValueError, match="at least 8 samples"):
        model(torch.randn(2, 3, 7))
    assert model(torch.randn(2, 3, 8)).shape == (2, 3)


def test_reset_parameters_draws_every_parameter_again():
    model = _samba(_chs_info(3))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    model.reset_parameters()
    assert all(parameter.isfinite().all() for parameter in model.parameters())
