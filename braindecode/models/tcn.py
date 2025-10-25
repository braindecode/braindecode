# Authors: Patryk Chrabaszcz
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.parametrizations import weight_norm

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Chomp1d, Ensure4d, SqueezeFinalOutput


class BDTCN(EEGModuleMixin, nn.Module):
    """Braindecode TCN from Gemein, L et al (2020) [gemein2020]_.

    .. figure:: https://ars.els-cdn.com/content/image/1-s2.0-S1053811920305073-gr3_lrg.jpg
       :align: center
       :alt: Braindecode TCN Architecture

    See [gemein2020]_ for details.

    Parameters
    ----------
    n_filters: int
        number of output filters of each convolution
    n_blocks: int
        number of temporal blocks in the network
    kernel_size: int
        kernel size of the convolutions
    drop_prob: float
        dropout probability
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

    References
    ----------
    .. [gemein2020] Gemein, L. A., Schirrmeister, R. T., Chrabąszcz, P., Wilson, D.,
       Boedecker, J., Schulze-Bonhage, A., ... & Ball, T. (2020). Machine-learning-based
       diagnostics of EEG pathology. NeuroImage, 220, 117021.
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans=None,
        n_outputs=None,
        chs_info=None,
        n_times=None,
        sfreq=None,
        input_window_seconds=None,
        # Model's parameters
        n_blocks=3,
        n_filters=30,
        kernel_size=5,
        drop_prob=0.5,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        self.base_tcn = TCN(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_blocks=n_blocks,
            n_filters=n_filters,
            kernel_size=kernel_size,
            drop_prob=drop_prob,
            activation=activation,
        )

        self.final_layer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten()
        )

    def forward(self, x):
        x = self.base_tcn(x)
        x = self.final_layer(x)
        return x


class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) from Bai et al. 2018 [Bai2018]_.

    See [Bai2018]_ for details.

    Code adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

    Parameters
    ----------
    n_filters: int
        number of output filters of each convolution
    n_blocks: int
        number of temporal blocks in the network
    kernel_size: int
        kernel size of the convolutions
    drop_prob: float
        dropout probability
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

    References
    ----------
    .. [Bai2018] Bai, S., Kolter, J. Z., & Koltun, V. (2018).
       An empirical evaluation of generic convolutional and recurrent networks
       for sequence modeling.
       arXiv preprint arXiv:1803.01271.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_blocks=3,
        n_filters=30,
        kernel_size=5,
        drop_prob=0.5,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.mapping = {
            "fc.weight": "final_layer.fc.weight",
            "fc.bias": "final_layer.fc.bias",
        }
        self.ensuredims = Ensure4d()
        t_blocks = nn.Sequential()
        for i in range(n_blocks):
            n_inputs = n_chans if i == 0 else n_filters
            dilation_size = 2**i
            t_blocks.add_module(
                "temporal_block_{:d}".format(i),
                _TemporalBlock(
                    n_inputs=n_inputs,
                    n_outputs=n_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    drop_prob=drop_prob,
                    activation=activation,
                ),
            )
        self.temporal_blocks = t_blocks

        self.final_layer = _FinalLayer(
            in_features=n_filters,
            out_features=n_outputs,
        )
        self.min_len = 1
        for i in range(n_blocks):
            dilation = 2**i
            self.min_len += 2 * (kernel_size - 1) * dilation

        # start in eval mode
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        x = self.ensuredims(x)
        # x is in format: B x C x T x 1
        (batch_size, _, time_size, _) = x.size()
        assert time_size >= self.min_len
        # remove empty trailing dimension
        x = x.squeeze(3)
        x = self.temporal_blocks(x)
        # Convert to: B x T x C
        x = x.transpose(1, 2).contiguous()

        out = self.final_layer(x, batch_size, time_size, self.min_len)

        return out


class _FinalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        self.out_fun = nn.Identity()

        self.squeeze = SqueezeFinalOutput()

    def forward(
        self, x: torch.Tensor, batch_size: int, time_size: int, min_len: int
    ) -> torch.Tensor:
        fc_out = self.fc(x.view(batch_size * time_size, x.size(2)))
        fc_out = self.out_fun(fc_out)
        fc_out = fc_out.view(batch_size, time_size, fc_out.size(1))

        out_size = 1 + max(0, time_size - min_len)
        out = fc_out[:, -out_size:, :].transpose(1, 2)
        # re-add 4th dimension for compatibility with braindecode
        return self.squeeze(out[:, :, :, None])


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        drop_prob,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = activation()
        self.dropout1 = nn.Dropout2d(drop_prob)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = activation()
        self.dropout2 = nn.Dropout2d(drop_prob)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = activation()

        init.normal_(self.conv1.weight, 0, 0.01)
        init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
