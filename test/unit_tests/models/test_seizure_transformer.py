import pytest
import torch

from braindecode.models import SeizureTransformer


@pytest.mark.parametrize("n_times", [256, 257])
def test_seizure_transformer_returns_dense_sample_logits(n_times):
    model = SeizureTransformer(
        n_chans=4,
        n_outputs=1,
        n_times=n_times,
        encoder_filters=(4, 8, 16, 32, 64),
        num_layers=1,
        num_heads=4,
        dim_feedforward=128,
        drop_prob=0.0,
    )

    x = torch.randn(2, 4, n_times, requires_grad=True)
    output = model(x)

    assert output.shape == (2, 1, n_times)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
