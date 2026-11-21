"""SAMBA: the scan over time, the coordinate-driven input, the initialization and the
released weights."""

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
        _ssd_scan(x, dt, a, b, c, d, chunk_size=64), _recurrence(x, dt, a, b, c, d)
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


def test_sizes_follow_the_stack_parameters():
    """Other poolings, kernels and head widths set the weights. The shortest window
    is pool_size ** 3 and leaves one step, so the standard deviation over time (the
    second half of the features) is zero there. The heads of the first stage must
    divide base_channels * expand, and the kernels must be odd to keep the steps."""
    model = _samba(
        _chs_info(3),
        branch_kernel_sizes=(1, 5),
        pool_size=3,
        stage_kernel_size=5,
        encoder1_head_dim=32,
    )
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert shapes["encoder.input_embedding.branches.1.weight"] == (8, 3, 5)
    assert shapes["encoder.input_embedding.fuse.weight"] == (8, 2 * 8, 1)
    assert shapes["encoder.encoder2.weight"] == (8, 8, 5)
    assert shapes["encoder.encoder3.weight"] == (8, 8, 5)
    assert shapes["encoder.encoder1_mamba.A_log"] == (2,)  # 8 * 8 channels, heads of 32
    with pytest.raises(ValueError, match="at least 27 samples"):
        model(torch.randn(2, 3, 26))
    features = model(torch.randn(2, 3, 27), return_features=True)["features"]
    assert features.shape == (2, 16) and not features[:, 8:].any()
    with pytest.raises(ValueError, match="encoder1_head_dim"):
        _samba(_chs_info(3), encoder1_head_dim=48)
    for kwargs in ({"stage_kernel_size": 4}, {"branch_kernel_sizes": (1, 4)}):
        with pytest.raises(ValueError, match="must be odd"):
            _samba(_chs_info(3), **kwargs)


def test_reset_parameters_draws_every_parameter_again():
    model = _samba(_chs_info(3))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    model.reset_parameters()
    assert all(parameter.isfinite().all() for parameter in model.parameters())


# features[0, :5] (mean over time) and features[0, -5:] (std over time) of the encoders
# of Dcode-AI/SAMBA-2s@e553287 and Dcode-AI/SAMBA-100s@3a235a1 on the input of the
# test, loaded as released, with the 94 target electrodes of the weights as chs_info
@pytest.mark.network
@pytest.mark.huggingface
@pytest.mark.parametrize(
    "repo_id, revision, expected_mean, expected_std",
    [
        (
            "braindecode/samba-2s-pretrained",
            "a1ccb6c09eb1fe82f18aca6198c3c4a1cf2d676c",
            [-0.430467904, 0.292152464, 0.329552829, 0.073628992, -0.769551814],
            [0.125874534, 0.189685628, 0.611500859, 0.472888798, 0.666319072],
        ),
        (
            "braindecode/samba-100s-pretrained",
            "af77956623ef6cbb6eae7816dec3b57329deb730",
            [-0.082055420, -0.087322578, 0.000102654, 0.025375422, 0.025857089],
            [0.032365620, 0.073571838, 0.031242646, 0.063421756, 0.089154541],
        ),
    ],
    ids=["2s", "100s"],
)
def test_samba_pretrained_matches_the_release(
    repo_id, revision, expected_mean, expected_std
):
    pytest.importorskip("huggingface_hub")
    model = SAMBA.from_pretrained(repo_id, revision=revision, n_outputs=3).eval()
    x = torch.arange(94 * 256.0).reshape(1, 94, 256).mul(0.05).sin()
    with torch.no_grad():
        features = model(x, return_features=True)["features"]
    torch.testing.assert_close(
        features[0, :5], torch.tensor(expected_mean), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        features[0, -5:], torch.tensor(expected_std), rtol=0, atol=1e-5
    )
