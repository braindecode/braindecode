# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import pytest
import torch
from einops import repeat

from braindecode.models import MVPFormer
from braindecode.models.mvpformer import _MVPAttention


def _reference_repeat_kv(x, n_rep):
    # Verbatim pre-fix implementation.
    return repeat(
        x,
        "batch kv_head segment channel head_dim "
        "-> batch (kv_head group) segment channel head_dim",
        group=n_rep,
    )


def _reference_rel_shift_chan(x):
    # Verbatim pre-fix implementation (advanced indexing).
    device = x.device
    chan_size = x.shape[-1]
    if chan_size > 1:
        upper_val = torch.cat(
            [
                torch.arange(1, chan_size - i, dtype=torch.long, device=device)
                for i in range(chan_size - 1)
            ]
        )
    else:
        upper_val = torch.tensor([], dtype=torch.long, device=device)
    idxes = torch.triu_indices(chan_size, chan_size, offset=1, device=device)
    shifting_idxes = torch.zeros(chan_size, chan_size, dtype=torch.long, device=device)
    shifting_idxes[..., idxes[0], idxes[1]] = upper_val
    shifting_idxes.transpose(-2, -1)[..., idxes[0], idxes[1]] = upper_val
    shifting_idxes = (chan_size - 1 - shifting_idxes).repeat(
        x.shape[-2] // chan_size, 1
    )
    rows = torch.arange(x.size(-2), device=device).unsqueeze(1)
    return x[..., rows, shifting_idxes]


@pytest.mark.parametrize(
    "n_channels,n_segments", [(32, 1), (32, 6), (19, 3), (1, 2)]
)
def test_rel_shift_chan_is_a_gather_with_identical_values_and_grads(
    n_channels, n_segments
):
    shape = (2, 3, n_segments * n_channels, n_channels)
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(*shape, dtype=torch.float64)
    expected = _reference_rel_shift_chan(x)
    (expected * weight).sum().backward()
    expected_grad, x.grad = x.grad, None
    actual = _MVPAttention._rel_shift_chan(x)
    (actual * weight).sum().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(x.grad, expected_grad)
    # The backward must not go through index_put_, which Intel Gaudi (HPU) runs on
    # the host with wrong gradients.
    assert "Gather" in type(actual.grad_fn).__name__


def test_repeat_kv_gives_identical_values_on_permuted_singleton_keys():
    # The non-contiguous layouts that ``_rel_attn`` builds for the relative keys.
    time_key = torch.randn(2, 5, 1, 4, 8).permute(0, 3, 1, 2, 4)
    channel_key = torch.randn(2, 1, 7, 4, 8).permute(0, 3, 1, 2, 4)
    for key in (time_key, channel_key):
        assert not key.is_contiguous()
        assert torch.equal(
            _MVPAttention._repeat_kv(key, 3), _reference_repeat_kv(key, 3)
        )


def test_mvpformer_forward_and_backward_unchanged(monkeypatch):
    kwargs = dict(
        n_outputs=2,
        n_chans=6,
        n_times=2560 * 3,
        sfreq=512.0,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_head_kv=2,
        d_inner=48,
    )
    torch.manual_seed(0)
    model = MVPFormer(**kwargs).eval()
    x = torch.randn(2, 6, 2560 * 3) * 20.0

    def run():
        inputs = x.clone().requires_grad_(True)
        model.zero_grad()
        out = model(inputs)
        out.square().sum().backward()
        return (
            out.detach(),
            inputs.grad,
            [p.grad.clone() for p in model.parameters() if p.grad is not None],
        )

    new = run()
    monkeypatch.setattr(
        _MVPAttention, "_repeat_kv", staticmethod(_reference_repeat_kv)
    )
    monkeypatch.setattr(
        _MVPAttention, "_rel_shift_chan", staticmethod(_reference_rel_shift_chan)
    )
    old = run()
    assert torch.equal(new[0], old[0])
    assert torch.equal(new[1], old[1])
    assert len(new[2]) == len(old[2])
    assert all(torch.equal(a, b) for a, b in zip(new[2], old[2]))
