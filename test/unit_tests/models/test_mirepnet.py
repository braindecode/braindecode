# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import pytest

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
    }
    model_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MIRepNet(**model_kwargs)
