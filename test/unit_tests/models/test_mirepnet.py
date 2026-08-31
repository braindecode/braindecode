# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import pytest
import torch

from braindecode.models.mirepnet import MIRepNet


@pytest.mark.parametrize(
    "kwargs, shape, message",
    [
        ({"embed_dim": 0}, None, "embed_dim must be positive"),
        ({"num_layers": 0}, None, "num_layers must be positive"),
        ({"num_heads": 0}, None, "num_heads must be positive"),
        (
            {"embed_dim": 15, "num_heads": 4},
            None,
            "embed_dim must be divisible by num_heads",
        ),
        (
            {"feedforward_expansion": 0},
            None,
            "feedforward_expansion must be positive",
        ),
        ({"drop_prob": -0.1}, None, "drop_prob must be between 0 and 1"),
        ({"drop_prob": 1.1}, None, "drop_prob must be between 0 and 1"),
        ({"n_times": 98}, None, "n_times must be at least 99"),
        ({}, (3, 256), "3 dimensions"),
        ({}, (2, 4, 256), "3 channels"),
        ({}, (2, 3, 98), "at least 99 samples"),
    ],
)
def test_mirepnet_edge_cases(kwargs, shape, message):
    model_kwargs = {
        "n_chans": 3,
        "n_outputs": 2,
        "n_times": 256,
        "embed_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "feedforward_expansion": 2,
    }
    model_kwargs.update(kwargs)

    if shape is None:
        with pytest.raises(ValueError, match=message):
            MIRepNet(**model_kwargs)
    else:
        model = MIRepNet(**model_kwargs)
        with pytest.raises(ValueError, match=message):
            model(torch.randn(*shape))
