# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
from torch.nn import ConstantPad2d

from .deep4 import Deep4Net
from .util import to_dense_prediction_model
from .shallow_fbcsp import ShallowFBCSPNet


class HybridNet(nn.Module):
    """Hybrid ConvNet model from Schirrmeister et al 2017.

    See [Schirrmeister2017]_ for details.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(self, in_chans, n_classes, input_window_samples):
        super(HybridNet, self).__init__()
        deep_model = Deep4Net(
            in_chans,
            n_classes,
            n_filters_time=20,
            n_filters_spat=30,
            n_filters_2=40,
            n_filters_3=50,
            n_filters_4=60,
            input_window_samples=input_window_samples,
            final_conv_length=2,
        )
        shallow_model = ShallowFBCSPNet(
            in_chans,
            n_classes,
            input_window_samples=input_window_samples,
            n_filters_time=30,
            n_filters_spat=40,
            filter_time_length=28,
            final_conv_length=29,
        )

        reduced_deep_model = nn.Sequential()
        for name, module in deep_model.named_children():
            if name == "conv_classifier":
                new_conv_layer = nn.Conv2d(
                    module.in_channels,
                    60,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                )
                reduced_deep_model.add_module("deep_final_conv", new_conv_layer)
                break
            reduced_deep_model.add_module(name, module)

        reduced_shallow_model = nn.Sequential()
        for name, module in shallow_model.named_children():
            if name == "conv_classifier":
                new_conv_layer = nn.Conv2d(
                    module.in_channels,
                    40,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                )
                reduced_shallow_model.add_module(
                    "shallow_final_conv", new_conv_layer
                )
                break
            reduced_shallow_model.add_module(name, module)

        to_dense_prediction_model(reduced_deep_model)
        to_dense_prediction_model(reduced_shallow_model)
        self.reduced_deep_model = reduced_deep_model
        self.reduced_shallow_model = reduced_shallow_model
        self.final_conv = nn.Conv2d(
            100, n_classes, kernel_size=(1, 1), stride=1
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        deep_out = self.reduced_deep_model(x)
        shallow_out = self.reduced_shallow_model(x)

        n_diff_deep_shallow = deep_out.size()[2] - shallow_out.size()[2]

        if n_diff_deep_shallow < 0:
            deep_out = ConstantPad2d((0, 0, -n_diff_deep_shallow, 0), 0)(
                deep_out
            )
        elif n_diff_deep_shallow > 0:
            shallow_out = ConstantPad2d((0, 0, n_diff_deep_shallow, 0), 0)(
                shallow_out
            )

        merged_out = torch.cat((deep_out, shallow_out), dim=1)
        linear_out = self.final_conv(merged_out)
        softmaxed = nn.LogSoftmax(dim=1)(linear_out)
        squeezed = softmaxed.squeeze(3)
        return squeezed
