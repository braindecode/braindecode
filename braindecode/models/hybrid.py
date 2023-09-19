# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
from torch.nn import ConstantPad2d

from .deep4 import Deep4Net
from .util import to_dense_prediction_model
from .shallow_fbcsp import ShallowFBCSPNet
from .base import EEGModuleMixin, deprecated_args


class HybridNet(EEGModuleMixin, nn.Module):
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

    def __init__(self, n_chans=None, n_outputs=None, n_times=None,
                 in_chans=None, n_classes=None, input_window_samples=None,
                 add_log_softmax=True):

        n_chans, n_outputs, n_times = deprecated_args(
            self,
            ('in_chans', 'n_chans', in_chans, n_chans),
            ('n_classes', 'n_outputs', n_classes, n_outputs),
            ('input_window_samples', 'n_times', input_window_samples, n_times),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            add_log_softmax=add_log_softmax,
        )
        self.mapping = {
            'final_conv.weight': 'final_layer.weight',
            'final_conv.bias': 'final_layer.bias'
        }

        deep_model = Deep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_filters_time=20,
            n_filters_spat=30,
            n_filters_2=40,
            n_filters_3=50,
            n_filters_4=60,
            n_times=n_times,
            final_conv_length=2,
        )
        shallow_model = ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            n_filters_time=30,
            n_filters_spat=40,
            filter_time_length=28,
            final_conv_length=29,
        )

        del n_outputs, n_chans, n_times
        del in_chans, n_classes, input_window_samples

        reduced_deep_model = nn.Sequential()
        for name, module in deep_model.named_children():
            if name == "final_layer":
                new_conv_layer = nn.Conv2d(
                    module.conv_classifier.in_channels,
                    60,
                    kernel_size=module.conv_classifier.kernel_size,
                    stride=module.conv_classifier.stride,
                )
                reduced_deep_model.add_module("deep_final_conv", new_conv_layer)
                break
            reduced_deep_model.add_module(name, module)

        reduced_shallow_model = nn.Sequential()
        for name, module in shallow_model.named_children():
            if name == "final_layer":
                new_conv_layer = nn.Conv2d(
                    module.conv_classifier.in_channels,
                    40,
                    kernel_size=module.conv_classifier.kernel_size,
                    stride=module.conv_classifier.stride,
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

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                100,
                self.n_outputs,
                kernel_size=(1, 1),
                stride=1),
            nn.LogSoftmax(dim=1) if self.add_log_softmax else nn.Identity())

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

        output = self.final_layer(merged_out)

        squeezed = output.squeeze(3)

        return squeezed
