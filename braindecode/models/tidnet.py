from math import ceil

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import init
from torch.nn.utils.parametrizations import weight_norm

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Ensure4d


class TIDNet(EEGModuleMixin, nn.Module):
    """Thinker Invariance DenseNet model from Kostas et al. (2020) [TIDNet]_.

    :bdg-success:`Convolution`

    .. figure:: https://content.cld.iop.org/journals/1741-2552/17/5/056008/revision3/jneabb7a7f1_hr.jpg
        :align: center
        :alt: TIDNet Architecture

    See [TIDNet]_ for details.

    Parameters
    ----------
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
        Percentage of n_times that defines the temporal filter length:
        temp_len = ceil(temp_span * n_times)
        e.g A value of 0.05 for temp_span with 1500 n_times will yield a temporal
        filter of length 75.
    bottleneck : int
        Bottleneck factor within Densefilter
    summary : int
        Output size of AdaptiveAvgPool1D layer. If set to -1, value will be calculated
        automatically (n_times // pooling).
    in_chans :
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_window_samples :
        Alias for n_times.
    activation: nn.Module, default=nn.LeakyReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.LeakyReLU``.

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

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        s_growth: int = 24,
        t_filters: int = 32,
        drop_prob: float = 0.4,
        pooling: int = 15,
        temp_layers: int = 2,
        spat_layers: int = 2,
        temp_span: float = 0.05,
        bottleneck: int = 3,
        summary: int = -1,
        activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            chs_info=chs_info,
        )
        del n_outputs, n_chans, n_times, input_window_seconds, sfreq, chs_info

        self.temp_len = ceil(temp_span * self.n_times)

        self.dscnn = _TIDNetFeatures(
            s_growth=s_growth,
            t_filters=t_filters,
            n_chans=self.n_chans,
            n_times=self.n_times,
            drop_prob=drop_prob,
            pooling=pooling,
            temp_layers=temp_layers,
            spat_layers=spat_layers,
            temp_span=temp_span,
            bottleneck=bottleneck,
            summary=summary,
            activation=activation,
        )

        self._num_features = self.dscnn.num_features

        self.flatten = nn.Flatten(start_dim=1)

        self.final_layer = self._create_classifier(self.num_features, self.n_outputs)

    def _create_classifier(self, incoming: int, n_outputs: int):
        classifier = nn.Linear(incoming, n_outputs)
        init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        seq_clf = nn.Sequential(classifier, nn.Identity())

        return seq_clf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        x = self.dscnn(x)
        x = self.flatten(x)
        return self.final_layer(x)

    @property
    def num_features(self):
        return self._num_features


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

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        drop_prob: float = 0.5,
        batch_norm: bool = True,
        activation: type[nn.Module] = nn.LeakyReLU,
        residual: bool = False,
    ):
        super().__init__()
        self.kernel = kernel
        self.activation = activation()
        self.residual = residual

        self.conv = nn.Conv2d(
            in_filters,
            out_filters,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not batch_norm,
        )
        self.dropout = nn.Dropout2d(p=float(drop_prob))
        self.batch_norm = (
            _BatchNormZG(out_filters)
            if residual
            else nn.BatchNorm2d(out_filters)
            if batch_norm
            else nn.Identity()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = input
        input = self.conv(
            input,
        )
        input = self.dropout(input)
        input = self.activation(input)
        input = self.batch_norm(input)
        return input + res if self.residual else input


class _DenseFilter(nn.Module):
    def __init__(
        self,
        in_features: int,
        growth_rate: int,
        filter_len: int = 5,
        drop_prob: float = 0.5,
        bottleneck: int = 2,
        activation: type[nn.Module] = nn.LeakyReLU,
        dim: int = -2,
    ):
        super().__init__()
        dim = dim if dim > 0 else dim + 4
        if dim < 2 or dim > 3:
            raise ValueError("Only last two dimensions supported")
        kernel = (filter_len, 1) if dim == 2 else (1, filter_len)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_features),
            activation(),
            nn.Conv2d(in_features, bottleneck * growth_rate, 1),
            nn.BatchNorm2d(bottleneck * growth_rate),
            activation(),
            nn.Conv2d(
                bottleneck * growth_rate,
                growth_rate,
                kernel,
                padding=tuple((k // 2 for k in kernel)),
            ),
            nn.Dropout2d(p=float(drop_prob)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.net(x)), dim=1)


class _DenseSpatialFilter(nn.Module):
    def __init__(
        self,
        n_chans: int,
        growth: int,
        depth: int,
        in_ch: int = 1,
        bottleneck: int = 4,
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.LeakyReLU,
        collapse: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            *[
                _DenseFilter(
                    in_ch + growth * d,
                    growth,
                    bottleneck=bottleneck,
                    drop_prob=drop_prob,
                    activation=activation,
                )
                for d in range(depth)
            ]
        )
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = _ConvBlock2D(
                n_filters, n_filters, (n_chans, 1), drop_prob=0, activation=activation
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x


class _TemporalFilter(nn.Module):
    def __init__(
        self,
        n_chans: int,
        filters: int,
        depth: int,
        temp_len: int,
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.LeakyReLU,
        residual: str = "netwise",
    ):
        super().__init__()
        temp_len = temp_len + 1 - temp_len % 2
        self.residual_style = str(residual)
        net = list()

        for i in range(depth):
            dil = depth - i
            conv = weight_norm(
                nn.Conv2d(
                    n_chans if i == 0 else filters,
                    filters,
                    kernel_size=(1, temp_len),
                    dilation=dil,
                    padding=(0, dil * (temp_len - 1) // 2),
                )
            )
            net.append(
                nn.Sequential(conv, activation(), nn.Dropout2d(p=float(drop_prob)))
            )
        if self.residual_style.lower() == "netwise":
            self.net = nn.Sequential(*net)
            self.residual = nn.Conv2d(n_chans, filters, (1, 1))
        elif residual.lower() == "dense":
            self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        style = self.residual_style.lower()
        if style == "netwise":
            return self.net(x) + self.residual(x)
        elif style == "dense":
            for layer in self.net:
                x = torch.cat((x, layer(x)), dim=1)
            return x
        # TorchScript now knows this path always returns or errors
        else:
            # Use an assertion so TorchScript can compile it
            assert False, f"Unsupported residual style: {self.residual_style}"


class _TIDNetFeatures(nn.Module):
    def __init__(
        self,
        s_growth: int,
        t_filters: int,
        n_chans: int,
        n_times: int,
        drop_prob: float,
        pooling: int,
        temp_layers: int,
        spat_layers: int,
        temp_span: float,
        bottleneck: int,
        summary: int,
        activation: type[nn.Module] = nn.LeakyReLU,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.temp_len = ceil(temp_span * n_times)

        self.temporal = nn.Sequential(
            Ensure4d(),
            Rearrange("batch C T 1 -> batch 1 C T"),
            _TemporalFilter(
                1,
                t_filters,
                depth=temp_layers,
                temp_len=self.temp_len,
                activation=activation,
            ),
            nn.MaxPool2d((1, pooling)),
            nn.Dropout2d(p=float(drop_prob)),
        )
        summary = n_times // pooling if summary == -1 else summary

        self.spatial = _DenseSpatialFilter(
            n_chans=n_chans,
            growth=s_growth,
            depth=spat_layers,
            in_ch=t_filters,
            drop_prob=drop_prob,
            bottleneck=bottleneck,
            activation=activation,
        )
        self.extract_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(int(summary)), nn.Flatten(start_dim=1)
        )

        self._num_features = (t_filters + s_growth * spat_layers) * summary

    @property
    def num_features(self):
        return self._num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        return self.extract_features(x)
