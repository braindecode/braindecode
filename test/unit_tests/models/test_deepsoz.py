"""DeepSOZ: onset zone pooling and the construction and input checks."""

import pytest
import torch

from braindecode.models import DeepSOZ

_SMALL = dict(
    n_chans=3,
    n_outputs=2,
    patch_size=16,
    num_heads=2,
    dim_feedforward=16,
    lstm_hidden_size=8,
)


def test_soz_logits_weight_the_scores_by_a_softmax_over_time_of_the_seizure_logit():
    """A seizure logit (the last output) that peaks at one second makes the SOZ
    logits that second's channel scores."""
    model = DeepSOZ(**_SMALL).eval()
    n_patches = 5
    # channel score of second t and channel c: 10 t + c
    scores = 10.0 * torch.arange(n_patches)[:, None] + torch.arange(3.0)[None]
    # (batch, n_patches, n_outputs): the seizure output peaks at second 3, the
    # other output at second 0
    logits = torch.zeros(1, n_patches, 2)
    logits[0, 3, 1] = logits[0, 0, 0] = 50.0
    model.soz_layer.register_forward_hook(lambda m, i, o: scores[None, ..., None])
    model.final_layer.register_forward_hook(lambda m, i, o: logits)
    with torch.no_grad():
        _, soz_logits = model(torch.randn(1, 3, n_patches * 16), return_soz=True)
    torch.testing.assert_close(soz_logits, scores[3][None])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(patch_size=0), "patch_size must be positive"),
        (dict(num_heads=0), "num_heads"),
        (dict(num_heads=3), "num_heads"),  # does not divide patch_size=16
        (dict(dim_feedforward=0), "dim_feedforward"),
        (dict(lstm_hidden_size=0), "lstm_hidden_size"),
        (dict(on_non_divisible="wrap"), "on_non_divisible"),
        (dict(n_times=40, on_non_divisible="error"), "not divisible"),
    ],
)
def test_invalid_construction_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DeepSOZ(**{**_SMALL, **kwargs})


def test_forward_rejects_another_number_of_channels():
    model = DeepSOZ(**_SMALL)
    with pytest.raises(ValueError, match="built for 3 channels"):
        model(torch.randn(1, 4, 32))
