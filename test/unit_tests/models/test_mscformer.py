# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import pytest
import torch

from braindecode.models import MSCFormer


def test_mscformer_patch_embedding_is_the_branches_in_order():
    torch.manual_seed(0)
    model = MSCFormer(n_chans=5, n_outputs=2, n_times=300).double()
    cnn = model.cnn.eval()
    x = torch.randn(3, 1, 5, 300, dtype=torch.float64)
    params = list(cnn.parameters())
    out = cnn(x)
    out.square().sum().backward()
    grads = [p.grad.clone() for p in params]
    cnn.zero_grad()
    ref = cnn.projection(torch.cat([branch(x) for branch in cnn.branches], dim=1))
    ref.square().sum().backward()
    torch.testing.assert_close(out, ref)
    for g, p in zip(grads, params):
        torch.testing.assert_close(g, p.grad)


def test_mscformer_default_attention_scale_matches_original_source():
    """MSCFormer's default attention scale must reproduce the released
    source's ``embed_dim ** -0.5`` logit scaling.

    The original MSCFormer code divides attention logits by
    ``sqrt(emb_size)`` for every head, not by ``sqrt(head_dim)`` (braindecode's
    more common default for :class:`~braindecode.modules.MultiHeadAttention`).
    The two only coincide when ``num_heads == 1``; with the paper's defaults
    (``num_heads=8``, ``emb_size=48``) they differ by ``sqrt(num_heads)``. A
    state-dict-matched parity check against the original implementation gives
    a max abs logit diff of about 2e-7 with this default, versus about 0.035
    with ``head_dim ** -0.5``.
    """
    model = MSCFormer(n_outputs=4, n_chans=22, n_times=1000)
    expected_scale = model.embed_dim**-0.5
    assert model.embed_dim == 48
    for block in model.trans.layers:
        assert block.attention.module.scale == pytest.approx(expected_scale)

    custom_scale = 0.25
    model_custom = MSCFormer(
        n_outputs=4, n_chans=22, n_times=1000, attention_scale=custom_scale
    )
    for block in model_custom.trans.layers:
        assert block.attention.module.scale == pytest.approx(custom_scale)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0])
def test_mscformer_rejects_non_positive_attention_scale(bad_scale):
    with pytest.raises(ValueError, match="attention_scale"):
        MSCFormer(n_outputs=4, n_chans=22, n_times=1000, attention_scale=bad_scale)
