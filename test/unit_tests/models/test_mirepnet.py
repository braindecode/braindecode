# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import pytest
import torch

from braindecode.models.mirepnet import MIRepNet


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"embed_dim": 0}, "embed_dim must be positive"),
        ({"n_filters_time": 0}, "n_filters_time must be positive"),
        ({"n_filters_spat": 0}, "n_filters_spat must be positive"),
        ({"filter_time_length": 0}, "filter_time_length must be positive"),
        ({"pool_time_length": 0}, "pool_time_length must be positive"),
        ({"pool_time_stride": 0}, "pool_time_stride must be positive"),
        ({"num_layers": 0}, "num_layers must be positive"),
        ({"num_heads": 0}, "num_heads must be positive"),
        (
            {"embed_dim": 15, "num_heads": 4},
            "embed_dim must be divisible by num_heads",
        ),
        (
            {"feedforward_expansion": 0},
            "feedforward_expansion must be positive",
        ),
        ({"drop_prob": -0.1}, "drop_prob must be between 0 and 1"),
        ({"att_drop_prob": 1.1}, "att_drop_prob must be between 0 and 1"),
        (
            {"feedforward_drop_prob": -0.1},
            "feedforward_drop_prob must be between 0 and 1",
        ),
        ({"attention_scale": 0}, "attention_scale must be positive or None"),
    ],
)
def test_mirepnet_edge_cases(kwargs, message):
    model_kwargs = {
        "n_chans": 3,
        "n_outputs": 2,
        "n_times": 256,
        "embed_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "feedforward_expansion": 2,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        MIRepNet(**model_kwargs)


def test_mirepnet_stem_is_the_two_convolutions():
    torch.manual_seed(0)
    model = MIRepNet(n_chans=5, n_outputs=2, n_times=200, embed_dim=16).double()
    emb = model.embedding.eval()
    x = torch.randn(3, 5, 200, dtype=torch.float64)
    params = [emb.conv1.weight, emb.conv1.bias, emb.conv2.weight, emb.conv2.bias]
    out = emb(x)
    out.square().sum().backward()
    grads = [p.grad.clone() for p in params]
    emb.zero_grad()
    ref = emb.conv2(emb.conv1(x.unsqueeze(1)))
    for layer in (emb.bn, emb.elu, emb.pool, emb.dropout, emb.projection):
        ref = layer(ref)
    ref = ref.flatten(start_dim=2).transpose(1, 2)
    ref.square().sum().backward()
    torch.testing.assert_close(out, ref)
    for g, p in zip(grads, params):
        torch.testing.assert_close(g, p.grad)
