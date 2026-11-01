# Authors: Anton Soloviev <anton@praviel.com>
# License: BSD (3-clause)

import pytest
import torch
from torch.nn import functional as F

from braindecode.models.bendr import _BENDRContextualizer


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("start_token", [-5, 0.25, None])
def test_contextualizer_unnormalized_output(training, start_token):
    """The release projects transformer outputs directly, including the token.

    Zero residual branches make the transformer an identity, so its output has
    an analytic value independent of the contextualizer's forward implementation.
    Keep a nonidentity final norm in the state dict, as in the released weights.
    """
    model = _BENDRContextualizer(
        in_features=4, hidden_feedforward=8, heads=2, layers=1,
        dropout=0, position_encoder=0, start_token=start_token,
    ).train(training)
    with torch.no_grad():
        layer = model.transformer_layers[0]
        layer.self_attn.out_proj.weight.zero_()
        layer.self_attn.out_proj.bias.zero_()
        layer.linear2.weight.zero_()
        layer.linear2.bias.zero_()
        projection = model.input_conditioning[3]
        projection.weight.zero_()
        projection.weight[:4].copy_(2 * torch.eye(4))
        projection.bias.zero_()
        model.output_layer.weight.zero_()
        model.output_layer.weight[:, :4, 0].copy_(torch.eye(4))
        model.output_layer.bias.zero_()
        model.norm.weight.fill_(0.9235)
        model.norm.bias.fill_(0.5)

    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    expected = 2 * F.layer_norm(x.transpose(1, 2), (4,)).transpose(1, 2)
    if start_token is not None:
        expected = torch.cat([torch.full((2, 4, 1), start_token), expected], dim=2)
    torch.testing.assert_close(model(x), expected)
    assert not any(p.requires_grad for p in model.norm.parameters())
