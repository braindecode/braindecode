# Authors: Patryk Chrabaszcz
#          Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD-3

from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm

from .modules import Ensure4d, Expression
from .functions import squeeze_final_output
from .base import EEGModuleMixin, deprecated_args


class TCN(EEGModuleMixin, nn.Module):
    """Temporal Convolutional Network (TCN) from Bai et al 2018.

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
    n_in_chans: int
        Alias for `n_chans`.

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
            n_blocks=None,
            n_filters=None,
            kernel_size=None,
            drop_prob=None,
            chs_info=None,
            n_times=None,
            input_window_seconds=None,
            sfreq=None,
            n_in_chans=None,
            add_log_softmax=False,
    ):
        n_chans, = deprecated_args(
            self,
            ("n_in_chans", "n_chans", n_in_chans, n_chans),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_in_chans

        self.mapping = {
            "fc.weight": "final_layer.fc.weight",
            "fc.bias": "final_layer.fc.bias"
        }
        self.ensuredims = Ensure4d()
        t_blocks = nn.Sequential()
        for i in range(n_blocks):
            n_inputs = self.n_chans if i == 0 else n_filters
            dilation_size = 2 ** i
            t_blocks.add_module("temporal_block_{:d}".format(i), TemporalBlock(
                n_inputs=n_inputs,
                n_outputs=n_filters,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                drop_prob=drop_prob
            ))
        self.temporal_blocks = t_blocks

        # Here, change to final_layer
        self.final_layer = _FinalLayer(in_features=n_filters, out_features=self.n_outputs,
                                       add_log_softmax=add_log_softmax)
        self.min_len = 1
        for i in range(n_blocks):
            dilation = 2 ** i
            self.min_len += 2 * (kernel_size - 1) * dilation

        # start in eval mode
        self.eval()

    def forward(self, x):
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
    def __init__(self, in_features, out_features, add_log_softmax=True):

        super().__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        if add_log_softmax:
            self.out_fun = nn.LogSoftmax(dim=1)
        else:
            self.out_fun = nn.Identity()

        self.squeeze = Expression(squeeze_final_output)

    def forward(self, x, batch_size, time_size, min_len):

        fc_out = self.fc(x.view(batch_size * time_size, x.size(2)))
        fc_out = self.out_fun(fc_out)
        fc_out = fc_out.view(batch_size, time_size, fc_out.size(1))

        out_size = 1 + max(0, time_size - min_len)
        out = fc_out[:, -out_size:, :].transpose(1, 2)
        # re-add 4th dimension for compatibility with braindecode
        return self.squeeze(out[:, :, :, None])


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, drop_prob):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(drop_prob)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(drop_prob)

        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()

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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def extra_repr(self):
        return 'chomp_size={}'.format(self.chomp_size)

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
