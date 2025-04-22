# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn
from torch.nn import ConstantPad2d

from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet


class HybridNet(nn.Module):
    """Hybrid ConvNet model from Schirrmeister, R T et al (2017)  [Schirrmeister2017]_.

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

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        activation: nn.Module = nn.ELU,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.mapping = {
            "final_conv.weight": "final_layer.weight",
            "final_conv.bias": "final_layer.bias",
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
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            chs_info=chs_info,
            final_conv_length=2,
            activation_first_conv_nonlin=activation,
            activation_later_conv_nonlin=activation,
            drop_prob=drop_prob,
        )
        shallow_model = ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            chs_info=chs_info,
            n_filters_time=30,
            n_filters_spat=40,
            filter_time_length=28,
            final_conv_length=29,
            drop_prob=drop_prob,
        )
        new_conv_layer = nn.Conv2d(
            deep_model.final_layer.conv_classifier.in_channels,
            60,
            kernel_size=deep_model.final_layer.conv_classifier.kernel_size,
            stride=deep_model.final_layer.conv_classifier.stride,
        )
        deep_model.final_layer = new_conv_layer

        new_conv_layer = nn.Conv2d(
            shallow_model.final_layer.conv_classifier.in_channels,
            40,
            kernel_size=shallow_model.final_layer.conv_classifier.kernel_size,
            stride=shallow_model.final_layer.conv_classifier.stride,
        )
        shallow_model.final_layer = new_conv_layer

        deep_model.to_dense_prediction_model()
        shallow_model.to_dense_prediction_model()
        self.reduced_deep_model = deep_model
        self.reduced_shallow_model = shallow_model

        self.final_layer = nn.Sequential(
            nn.Conv2d(100, n_outputs, kernel_size=(1, 1), stride=1),
            nn.Identity(),
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
            deep_out = ConstantPad2d((0, 0, -n_diff_deep_shallow, 0), 0)(deep_out)
        elif n_diff_deep_shallow > 0:
            shallow_out = ConstantPad2d((0, 0, n_diff_deep_shallow, 0), 0)(shallow_out)

        merged_out = torch.cat((deep_out, shallow_out), dim=1)

        output = self.final_layer(merged_out)

        squeezed = output.squeeze(3)

        return squeezed
