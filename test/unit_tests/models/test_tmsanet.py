"""Focused tests for TMSA-Net."""

import pytest
import torch
from torch import nn

from braindecode.models.tmsanet import TMSANet, _TMSAAttention


@pytest.mark.parametrize(
    "n_chans,n_times,n_outputs,embed_dim,att_drop_prob",
    [
        (22, 1000, 4, 19, 0.5),  # BCI Competition IV 2a
        (3, 1000, 2, 6, 0.5),   # BCI Competition IV 2b
        (44, 1125, 4, 10, 0.7), # HGD
    ],
)
def test_tmsanet_released_dataset_configurations(
    n_chans, n_times, n_outputs, embed_dim, att_drop_prob
):
    model = TMSANet(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        embed_dim=embed_dim,
        att_drop_prob=att_drop_prob,
    ).eval()

    x = torch.randn(2, n_chans, n_times)
    with torch.no_grad():
        out = model(x)

    n_tokens = (n_times - 50) // 15 + 1
    assert out.shape == (2, n_outputs)
    assert model.final_layer.in_features == embed_dim * n_tokens


def test_tmsanet_default_attention_preserves_reference_bottleneck():
    """The released 19-dim / 4-head attention uses a 16-dim inner space."""
    model = TMSANet(n_chans=22, n_times=1000, n_outputs=4)
    attn = model.transformer.layers[0].attention

    assert attn.head_dim == 19 // 4 == 4
    assert attn.inner_dim == 16
    assert attn.w_q.in_features == 19
    assert attn.w_q.out_features == 16
    assert attn.w_k_local.in_features == 38
    assert attn.w_k_local.out_features == 16
    assert attn.w_o.in_features == 16
    assert attn.w_o.out_features == 19


class _IdentityLocalKey(nn.Module):
    def forward(self, x):
        # Attention expects two local scales concatenated along channels.
        return torch.cat([x, x], dim=1)


def test_tmsa_attention_sums_local_and_global_branches():
    """Pin the released local-attention + global-attention combination."""
    attn = _TMSAAttention(
        embed_dim=4,
        num_heads=2,
        local_drop_prob=0.0,
        att_drop_prob=0.0,
    ).eval()
    attn.local_key = _IdentityLocalKey()

    with torch.no_grad():
        for layer in [attn.w_q, attn.w_k_global, attn.w_v, attn.w_o]:
            layer.weight.zero_()
            layer.bias.zero_()
        attn.w_q.weight.copy_(torch.eye(4))
        attn.w_k_global.weight.copy_(torch.eye(4))
        attn.w_v.weight.copy_(torch.eye(4))
        attn.w_o.weight.copy_(torch.eye(4))

        # Local projection reads the first copy of the concatenated key only,
        # making local and global attention identical.
        attn.w_k_local.weight.zero_()
        attn.w_k_local.bias.zero_()
        attn.w_k_local.weight[:, :4].copy_(torch.eye(4))

    x = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    )

    out = attn(x)

    # Recompute one branch manually. Since local/global keys are identical,
    # the module output must be exactly twice one branch before w_o.
    q = attn._reshape_heads(attn.w_q(x))
    k = attn._reshape_heads(attn.w_k_global(x))
    v = attn._reshape_heads(attn.w_v(x))
    scores = torch.matmul(q, k.transpose(-2, -1)) / (attn.head_dim**0.5)
    one_branch = torch.matmul(torch.softmax(scores, dim=-1), v)
    one_branch = one_branch.transpose(1, 2).contiguous().view(1, 2, 4)

    torch.testing.assert_close(out, 2 * one_branch)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"embed_dim": 0}, "embed_dim"),
        ({"num_heads": 0}, "num_heads"),
        ({"embed_dim": 3, "num_heads": 4}, "embed_dim"),
        ({"pool_size": 0}, "pool_size"),
        ({"pool_stride": 0}, "pool_size"),
        ({"depth": 0}, "depth"),
        ({"fc_ratio": 0}, "fc_ratio"),
    ],
)
def test_tmsanet_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TMSANet(n_chans=22, n_times=1000, n_outputs=4, **kwargs)


def test_tmsanet_rejects_window_shorter_than_pool():
    with pytest.raises(ValueError, match="n_times"):
        TMSANet(
            n_chans=22,
            n_times=49,
            n_outputs=4,
            pool_size=50,
        )
