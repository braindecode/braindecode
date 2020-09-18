from braindecode.models.modules import Expression
from braindecode.models.util import get_output_shape
from torch import nn


def test_get_output_shape_1d_model():
    model = nn.Conv1d(1, 1, 3)
    out_shape = get_output_shape(model, in_chans=1, input_window_samples=5)
    assert out_shape == (1, 1, 3,)


def test_get_output_shape_2d_model():
    model = nn.Sequential(
        Expression(lambda x: x.unsqueeze(-1)),
        nn.Conv2d(1, 1, (3, 1)))
    out_shape = get_output_shape(model, in_chans=1, input_window_samples=5)
    assert out_shape == (1, 1, 3, 1)
