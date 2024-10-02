from __future__ import annotations

from mne.utils import warn

import torch

from torch import nn

from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegnet import Conv2dWithConstraint
from braindecode.models.fbcnet import _valid_layers

from braindecode.models.modules import LinearWithConstraint, FilterBank


class FBMSNet(EEGModuleMixin, nn.Module):
    """FBMSNet from Liu et al (2022) [fbmsnet]_.

    0. **FilterBank Layer**: Applying filterbank to transform the input.

    1. **Temporal Convolution Block**: Utilizes mixed depthwise convolution
       (MixConv) to extract multiscale temporal features from multiview EEG
       representations. The input is split into groups corresponding to different
       views each convolved with kernels of varying sizes.
       Kernel sizes are set relative to the EEG
       sampling rate, with ratio coefficients [0.5, 0.25, 0.125, 0.0625],
       dividing the input into four groups.

    2. **Spatial Convolution Block**: Applies depthwise convolution with a kernel
       size of (n_chans, 1) to span all EEG channels, effectively learning spatial
       filters. This is followed by batch normalization and the Swish activation
       function. A maximum norm constraint of 2 is imposed on the convolution
       weights to regularize the model.

    3. **Temporal Log-Variance Block**: Computes the log-variance.

    4. **Classification Layer**: A fully connected with weight constraint.

    Notes
    -----
    This implementation is not guaranteed to be correct and has not been checked
    by the original authors; it has only been reimplemented from the paper
    description and source code [fbmsnetcode]_. There is an extra layer here to
    compute the filterbank during bash time and not on data time. This avoids
    data-leak, and allows the model to follow the braindecode convention.

    Parameters
    ----------
    n_bands : int, default=9
        Number of input channels (e.g., number of frequency bands).
    stride_factor : int, default=4
        Stride factor for temporal segmentation.
    temporal_layer : str, default='LogVarLayer'
        Temporal aggregation layer to use.
    n_filters_spat : int, default=36
        Number of output channels from the MixedConv2d layer.
    dilatability : int, default=8
        Expansion factor for the spatial convolution block.
    stride_factor : int, default=4
        Stride factor for reshaping.
    activation : nn.Module, default=nn.SiLU
        Activation function class to apply.
    verbose: bool, default False
        Verbose parameter to create the filter using mne.

    References
    ----------
    .. [fbmsnet] Liu, K., Yang, M., Yu, Z., Wang, G., & Wu, W. (2022).
        FBMSNet: A filter-bank multi-scale convolutional neural network for
        EEG-based motor imagery decoding. IEEE Transactions on Biomedical
        Engineering, 70(2), 436-445.
    .. [fbmsnetcode] Liu, K., Yang, M., Yu, Z., Wang, G., & Wu, W. (2022).
        FBMSNet: A filter-bank multi-scale convolutional neural network for
        EEG-based motor imagery decoding.
        https://github.com/Want2Vanish/FBMSNet
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans=None,
        n_outputs=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # models parameters
        n_bands=9,
        n_filters_spat: int = 36,
        temporal_layer: str = "LogVarLayer",
        n_dim: int = 3,
        stride_factor: int = 4,
        dilatability: int = 8,
        activation: nn.Module = nn.SiLU,
        verbose: bool = False,
    ):
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Parameters
        self.n_bands = n_bands
        self.n_filters_spat = n_filters_spat
        self.n_dim = n_dim
        self.stride_factor = stride_factor
        self.activation = activation
        self.dilatability = dilatability

        # Checkers
        if temporal_layer not in _valid_layers:
            raise NotImplementedError(
                f"Temporal layer '{temporal_layer}' is not implemented."
            )

        if self.n_times % self.stride_factor != 0:
            warn(
                f"Time dimension ({self.n_times}) is not divisible by"
                f" stride_factor ({self.stride_factor}). Input will be "
                f"truncated.",
                UserWarning,
            )

        # Layers
        # Following paper nomeclature
        self.spectral_filtering = FilterBank(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.n_bands,
            verbose=verbose,
        )
        # As we have an internal process to create the bands,
        # we get the values from the filterbank
        self.n_bands = self.spectral_filtering.n_bands

        # MixedConv2d Layer
        self.mix_conv = nn.Sequential(
            _MixedConv2d(
                in_channels=self.n_bands,
                out_channels=self.n_filters_spat,
                kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)],
                stride=1,
                dilation=1,
                depthwise=False,
            ),
            nn.BatchNorm2d(self.n_filters_spat),
        )

        # Spatial Convolution Block (SCB)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.n_filters_spat,
                out_channels=self.n_filters_spat * self.dilatability,
                kernel_size=(self.n_chans, 1),
                groups=self.n_filters_spat,
                max_norm=2,
                padding=0,
            ),
            nn.BatchNorm2d(self.n_filters_spat * self.dilatability),
            self.activation(),
        )

        # Temporal Aggregation Layer
        self.temporal_layer = _valid_layers[temporal_layer](dim=self.n_dim)

        self.flatten_layer = Rearrange("batch ... -> batch (...)")

        # Final fully connected layer
        self.final_layer = LinearWithConstraint(
            in_features=self.n_filters_spat * self.dilatability * self.stride_factor,
            out_features=self.n_outputs,
            max_norm=0.5,
        )

    def forward(self, x):
        """
        Forward pass of the FBMSNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_outputs).
        """
        x = self.spectral_filtering(x)
        # Mixed convolution
        x = self.mix_conv(x)
        # Spatial convolution block
        x = self.spatial_conv(x)
        batch_size, channels, _, time = x.shape
        x = x.view(batch_size, channels, self.stride_factor, time // self.stride_factor)

        x = self.temporal_layer(x)  # type: ignore[operator]
        x = self.flatten_layer(x)
        x = self.final_layer(x)
        return x


class _MixedConv2d(nn.ModuleDict):
    """Mixed Grouped Convolution for multiscale feature extraction."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[(1, 15), (1, 31), (1, 63), (1, 125)],
        stride=1,
        dilation=1,
        depthwise=False,
    ):
        super().__init__()

        num_groups = len(kernel_size)
        in_splits = self._split_channels(in_channels, num_groups)
        out_splits = self._split_channels(out_channels, num_groups)
        self.splits = in_splits

        for idx, (k, in_ch, out_ch) in enumerate(
            zip(kernel_size, in_splits, out_splits)
        ):
            conv_groups = out_ch if depthwise else 1
            padding_value = self._get_padding_value(k, stride, dilation)
            self.add_module(
                str(idx),
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride,
                    padding=padding_value,
                    dilation=dilation,
                    groups=conv_groups,
                    bias=False,
                ),
            )

    @staticmethod
    def _split_channels(num_chan, num_groups):
        split = [num_chan // num_groups for _ in range(num_groups)]
        split[0] += num_chan - sum(split)
        return split

    @staticmethod
    def _get_padding_value(kernel_size, stride, dilation):
        padding = []
        for k, s, d in zip(
            kernel_size,
            stride if isinstance(stride, tuple) else (stride, stride),
            dilation if isinstance(dilation, tuple) else (dilation, dilation),
        ):
            pad = ((s - 1) + d * (k - 1)) // 2
            padding.append(pad)
        return tuple(padding)

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x
