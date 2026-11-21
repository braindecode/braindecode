# Authors: The braindecode contributors.
#
# License: BSD-3

import torch

from braindecode.models.tidnet import _TemporalFilter


def test_tidnet_temporal_filter_dense_registers_parameters():
    block = _TemporalFilter(
        n_chans=4, filters=8, depth=2, temp_len=5, residual="dense"
    )
    assert isinstance(block.net, torch.nn.Sequential)
    assert len(list(block.net.parameters())) > 0
    assert set(dict(block.named_parameters())) == set(
        f"net.{k}" for k in dict(block.net.named_parameters())
    )
    x = torch.randn(2, 4, 1, 20, requires_grad=True)
    out = block(x)
    assert out.shape == (2, 4 + 2 * 8, 1, 20)
    out.sum().backward()
    assert x.grad is not None
