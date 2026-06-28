# Authors: The braindecode contributors.
#
# License: BSD-3

import torch

from braindecode.models import SincShallowNet


def test_sinc_shallow_depthwise_batch_norm_input_is_contiguous():
    model = SincShallowNet(n_chans=22, n_outputs=4, n_times=500, sfreq=125)
    batch_norm_inputs_are_contiguous = []

    def check_input_contiguous(_module, inputs):
        batch_norm_inputs_are_contiguous.append(inputs[0].is_contiguous())

    handle = model.depthwiseconv[0].register_forward_pre_hook(check_input_contiguous)
    try:
        x = torch.randn(2, 22, 500, requires_grad=True)
        model(x).sum().backward()
    finally:
        handle.remove()

    assert batch_norm_inputs_are_contiguous == [True]
