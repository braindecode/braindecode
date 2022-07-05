from math import ceil

import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm

from .modules import Ensure4d, Expression


class _BatchNormZG(nn.BatchNorm2d):
    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()


class _ConvBlock2D(nn.Module):
    """Implements Convolution block with order:
    Convolution, dropout, activation, batch-norm
    """
    def __init__(self, in_filters, out_filters, kernel, stride=(1, 1), padding=0, dilation=1,
                 groups=1, drop_prob=0.5, batch_norm=True, activation=nn.LeakyReLU, residual=False):
        super().__init__()
        self.kernel = kernel
        self.activation = activation()
        self.residual = residual

        self.conv = nn.Conv2d(in_filters, out_filters, kernel, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=not batch_norm)
        self.dropout = nn.Dropout2d(p=drop_prob)
        self.batch_norm = _BatchNormZG(out_filters) if residual else nn.BatchNorm2d(out_filters) if\
            batch_norm else lambda x: x

    def forward(self, input):
        res = input
        input = self.conv(input,)
        input = self.dropout(input)
        input = self.activation(input)
        input = self.batch_norm(input)
        return input + res if self.residual else input


class _DenseFilter(nn.Module):
    def __init__(self, in_features, growth_rate, filter_len=5, drop_prob=0.5, bottleneck=2,
                 activation=nn.LeakyReLU, dim=-2):
        super().__init__()
        dim = dim if dim > 0 else dim + 4
        if dim < 2 or dim > 3:
            raise ValueError('Only last two dimensions supported')
        kernel = (filter_len, 1) if dim == 2 else (1, filter_len)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_features),
            activation(),
            nn.Conv2d(in_features, bottleneck * growth_rate, 1),
            nn.BatchNorm2d(bottleneck * growth_rate),
            activation(),
            nn.Conv2d(bottleneck * growth_rate, growth_rate, kernel,
                      padding=tuple((k // 2 for k in kernel))),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, x):
        return torch.cat((x, self.net(x)), dim=1)


class _DenseSpatialFilter(nn.Module):
    def __init__(self, in_chans, growth, depth, in_ch=1, bottleneck=4, drop_prob=0.0,
                 activation=nn.LeakyReLU, collapse=True):
        super().__init__()
        self.net = nn.Sequential(*[
            _DenseFilter(in_ch + growth * d, growth, bottleneck=bottleneck, drop_prob=drop_prob,
                         activation=activation) for d in range(depth)
        ])
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = _ConvBlock2D(n_filters, n_filters, (in_chans, 1), drop_prob=0)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x


class _TemporalFilter(nn.Module):
    def __init__(self, in_chans, filters, depth, temp_len, drop_prob=0., activation=nn.LeakyReLU,
                 residual='netwise'):
        super().__init__()
        temp_len = temp_len + 1 - temp_len % 2
        self.residual_style = str(residual)
        net = list()

        for i in range(depth):
            dil = depth - i
            conv = weight_norm(nn.Conv2d(in_chans if i == 0 else filters, filters,
                                         kernel_size=(1, temp_len), dilation=dil,
                                         padding=(0, dil * (temp_len - 1) // 2)))
            net.append(nn.Sequential(
                conv,
                activation(),
                nn.Dropout2d(drop_prob)
            ))
        if self.residual_style.lower() == 'netwise':
            self.net = nn.Sequential(*net)
            self.residual = nn.Conv2d(in_chans, filters, (1, 1))
        elif residual.lower() == 'dense':
            self.net = net

    def forward(self, x):
        if self.residual_style.lower() == 'netwise':
            return self.net(x) + self.residual(x)
        elif self.residual_style.lower() == 'dense':
            for layer in self.net:
                x = torch.cat((x, layer(x)), dim=1)
            return x


class _TIDNetFeatures(nn.Module):
    def __init__(self, s_growth, t_filters, in_chans, input_window_samples, drop_prob, pooling,
                 temp_layers, spat_layers, temp_span, bottleneck, summary):
        super().__init__()
        self.in_chans = in_chans
        self.input_windows_samples = input_window_samples
        self.temp_len = ceil(temp_span * input_window_samples)

        def _permute(x):
            """
            Permutes data:

            from dim:
            batch, chans, time, 1

            to dim:
            batch, 1, chans, time

            """
            return x.permute([0, 3, 1, 2])

        self.temporal = nn.Sequential(
            Ensure4d(),
            Expression(_permute),
            _TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len),
            nn.MaxPool2d((1, pooling)),
            nn.Dropout2d(drop_prob),
        )
        summary = input_window_samples // pooling if summary == -1 else summary

        self.spatial = _DenseSpatialFilter(in_chans, s_growth, spat_layers, in_ch=t_filters,
                                           drop_prob=drop_prob, bottleneck=bottleneck)
        self.extract_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(int(summary)),
            nn.Flatten(start_dim=1)
        )

        self._num_features = (t_filters + s_growth * spat_layers) * summary

    @property
    def num_features(self):
        return self._num_features

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.extract_features(x)


class TIDNet(nn.Module):
    """Thinker Invariance DenseNet model from Kostas et al 2020.

    See [TIDNet]_ for details.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    in_chans : int
        Number of EEG channels.
    input_window_samples : int
        Number of samples.
    s_growth : int
        DenseNet-style growth factor (added filters per DenseFilter)
    t_filters : int
        Number of temporal filters.
    drop_prob : float
        Dropout probability
    pooling : int
        Max temporal pooling (width and stride)
    temp_layers : int
        Number of temporal layers
    spat_layers : int
        Number of DenseFilters
    temp_span : float
        Percentage of input_window_samples that defines the temporal filter length:
        temp_len = ceil(temp_span * input_window_samples)
        e.g A value of 0.05 for temp_span with 1500 input_window_samples will yield a temporal
        filter of length 75.
    bottleneck : int
        Bottleneck factor within Densefilter
    summary : int
        Output size of AdaptiveAvgPool1D layer. If set to -1, value will be calculated
        automatically (input_window_samples // pooling).

    Notes
    -----
    Code adapted from: https://github.com/SPOClab-ca/ThinkerInvariance/

    References
    ----------
    .. [TIDNet] Kostas, D. & Rudzicz, F.
        Thinker invariance: enabling deep neural networks for BCI across more
        people.
        J. Neural Eng. 17, 056008 (2020).
        doi: 10.1088/1741-2552/abb7a7.
    """
    def __init__(self, in_chans, n_classes, input_window_samples, s_growth=24, t_filters=32,
                 drop_prob=0.4, pooling=15, temp_layers=2, spat_layers=2, temp_span=0.05,
                 bottleneck=3, summary=-1):
        super().__init__()
        self.n_classes = n_classes
        self.in_chans = in_chans
        self.input_window_samples = input_window_samples
        self.temp_len = ceil(temp_span * input_window_samples)

        self.dscnn = _TIDNetFeatures(s_growth=s_growth, t_filters=t_filters, in_chans=in_chans,
                                     input_window_samples=input_window_samples,
                                     drop_prob=drop_prob, pooling=pooling, temp_layers=temp_layers,
                                     spat_layers=spat_layers, temp_span=temp_span,
                                     bottleneck=bottleneck, summary=summary)

        self._num_features = self.dscnn.num_features

        self.classify = self._create_classifier(self.num_features, n_classes)

    def _create_classifier(self, incoming, n_classes):
        classifier = nn.Linear(incoming, n_classes)
        init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        return nn.Sequential(nn.Flatten(start_dim=1), classifier, nn.LogSoftmax(dim=-1))

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        x = self.dscnn(x)
        return self.classify(x)

    @property
    def num_features(self):
        return self._num_features
