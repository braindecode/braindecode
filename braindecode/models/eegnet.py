# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

from typing import Dict, Optional

from einops.layers.torch import Rearrange
from mne.utils import deprecated, warn
from torch import nn

from braindecode.functional import glorot_weight_zero_bias
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import (
    Conv2dWithConstraint,
    Ensure4d,
    LinearWithConstraint,
    SqueezeFinalOutput,
)


class EEGNet(EEGModuleMixin, nn.Sequential):
    """EEGNet model from Lawhern et al. (2018) [Lawhern2018]_.

    :bdg-success:`Convolution`

    .. figure:: https://content.cld.iop.org/journals/1741-2552/15/5/056013/revision2/jneaace8cf01_hr.jpg
        :align: center
        :alt: EEGNet Architecture
        :width: 600px

    .. rubric:: Architectural Overview

    EEGNet is a compact convolutional network designed for EEG decoding with a pipeline that mirrors classical EEG processing:
    - (i) learn temporal frequency-selective filters,
    - (ii) learn spatial filters for those frequencies, and
    - (iii) condense features with depthwise-separable convolutions before a lightweight classifier.

    The architecture is deliberately small (temporal convolutional and spatial patterns) [Lawhern2018]_.

    .. rubric:: Macro Components

    - **Temporal convolution**
      Temporal convolution applied per channel; learns ``F1`` kernels that act as data-driven band-pass filters.
    - **Depthwise Spatial Filtering.**
      Depthwise convolution spanning the channel dimension with ``groups = F1``,
      yielding ``D`` spatial filters for each temporal filter (no cross-filter mixing).
    - **Norm-Nonlinearity-Pooling (+ dropout).**
      Batch normalization → ELU → temporal pooling, with dropout.
    - **Depthwise-Separable Convolution Block.**
      (a) depthwise temporal conv to refine temporal structure;
      (b) pointwise 1x1 conv to mix feature maps into ``F2`` combinations.
    - **Classifier Head.**
      Lightweight 1x1 conv or dense layer (often with max-norm constraint).

    .. rubric:: Convolutional Details

    - **Temporal.** The initial temporal convs serve as a *learned filter bank*:
      long 1-D kernels (implemented as 2-D with singleton spatial extent) emphasize oscillatory bands and transients.
      Because this stage is linear prior to BN/ELU, kernels can be analyzed as FIR filters to reveal each feature’s spectrum [Lawhern2018]_.

    - **Spatial.** The depthwise spatial conv spans the full channel axis (kernel height = #electrodes; temporal size = 1).
      With ``groups = F1``, each temporal filter learns its own set of ``D`` spatial projections—akin to CSP, learned end-to-end and
      typically regularized with max-norm.

    - **Spectral.** No explicit Fourier/wavelet transform is used. Frequency structure
      is captured implicitly by the temporal filter bank; later depthwise temporal kernels act as short-time integrators/refiners.

    .. rubric:: Additional Comments

    - **Filter-bank structure:** Parallel temporal kernels (``F1``) emulate classical filter banks; pairing them with frequency-specific spatial filters
      yields features mappable to rhythms and topographies.
    - **Depthwise & separable convs:** Parameter-efficient decomposition (depthwise + pointwise) retains power while limiting overfitting
      [Chollet2017]_ and keeps temporal vs. mixing steps interpretable.
    - **Regularization:** Batch norm, dropout, pooling, and optional max-norm on spatial kernels aid stability on small EEG datasets.
    - The v4 means the version 4 at the arxiv paper [Lawhern2018]_.


    Parameters
    ----------
    final_conv_length : int or "auto", default="auto"
        Length of the final convolution layer. If "auto", it is set based on n_times.
    pool_mode : {"mean", "max"}, default="mean"
        Pooling method to use in pooling layers.
    F1 : int, default=8
        Number of temporal filters in the first convolutional layer.
    D : int, default=2
        Depth multiplier for the depthwise convolution.
    F2 : int or None, default=None
        Number of pointwise filters in the separable convolution. Usually set to ``F1 * D``.
    depthwise_kernel_length : int, default=16
        Length of the depthwise convolution kernel in the separable convolution.
    pool1_kernel_size : int, default=4
        Kernel size of the first pooling layer.
    pool2_kernel_size : int, default=8
        Kernel size of the second pooling layer.
    kernel_length : int, default=64
        Length of the temporal convolution kernel.
    conv_spatial_max_norm : float, default=1
        Maximum norm constraint for the spatial (depthwise) convolution.
    activation : nn.Module, default=nn.ELU
        Non-linear activation function to be used in the layers.
    batch_norm_momentum : float, default=0.01
        Momentum for instance normalization in batch norm layers.
    batch_norm_affine : bool, default=True
        If True, batch norm has learnable affine parameters.
    batch_norm_eps : float, default=1e-3
        Epsilon for numeric stability in batch norm layers.
    drop_prob : float, default=0.25
        Dropout probability.
    final_layer_with_constraint : bool, default=False
        If ``False``, uses a convolution-based classification layer. If ``True``,
        apply a flattened linear layer with constraint on the weights norm as the final classification step.
    norm_rate : float, default=0.25
        Max-norm constraint value for the linear layer (used if ``final_layer_conv=False``).

    References
    ----------
    .. [Lawhern2018] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
        Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional
        neural network for EEG-based brain–computer interfaces. Journal of
        neural engineering, 15(5), 056013.
    .. [Chollet2017] Chollet, F., *Xception: Deep Learning with Depthwise Separable
        Convolutions*, CVPR, 2017.

    """

    def __init__(
        self,
        # signal's parameters
        n_chans: Optional[int] = None,
        n_outputs: Optional[int] = None,
        n_times: Optional[int] = None,
        # model's parameters
        final_conv_length: str | int = "auto",
        pool_mode: str = "mean",
        F1: int = 8,
        D: int = 2,
        F2: Optional[int | None] = None,
        kernel_length: int = 64,
        *,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool2_kernel_size: int = 8,
        conv_spatial_max_norm: int = 1,
        activation: nn.Module = nn.ELU,
        batch_norm_momentum: float = 0.01,
        batch_norm_affine: bool = True,
        batch_norm_eps: float = 1e-3,
        drop_prob: float = 0.25,
        final_layer_with_constraint: bool = False,
        norm_rate: float = 0.25,
        # Other ways to construct the signal related parameters
        chs_info: Optional[list[Dict]] = None,
        input_window_seconds=None,
        sfreq=None,
        **kwargs,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if final_conv_length == "auto":
            assert self.n_times is not None

        if not final_layer_with_constraint:
            warn(
                "Parameter 'final_layer_with_constraint=False' is deprecated and will be "
                "removed in a future release. Please use `final_layer_linear=True`.",
                DeprecationWarning,
            )

        if "third_kernel_size" in kwargs:
            warn(
                "The parameter `third_kernel_size` is deprecated "
                "and will be removed in a future version.",
            )
        unexpected_kwargs = set(kwargs) - {"third_kernel_size"}
        if unexpected_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {unexpected_kwargs}")

        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D

        if F2 is None:
            F2 = self.F1 * self.D
        self.F2 = F2

        self.kernel_length = kernel_length
        self.depthwise_kernel_length = depthwise_kernel_length
        self.pool1_kernel_size = pool1_kernel_size
        self.pool2_kernel_size = pool2_kernel_size
        self.drop_prob = drop_prob
        self.activation = activation
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_eps = batch_norm_eps
        self.conv_spatial_max_norm = conv_spatial_max_norm
        self.norm_rate = norm_rate

        # For the load_state_dict
        # When padronize all layers,
        # add the old's parameters here
        self.mapping = {
            "conv_classifier.weight": "final_layer.conv_classifier.weight",
            "conv_classifier.bias": "final_layer.conv_classifier.bias",
        }

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.add_module("ensuredims", Ensure4d())

        self.add_module("dimshuffle", Rearrange("batch ch t 1 -> batch 1 ch t"))
        self.add_module(
            "conv_temporal",
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
        )
        self.add_module(
            "bnorm_temporal",
            nn.BatchNorm2d(
                self.F1,
                momentum=self.batch_norm_momentum,
                affine=self.batch_norm_affine,
                eps=self.batch_norm_eps,
            ),
        )
        self.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                in_channels=self.F1,
                out_channels=self.F1 * self.D,
                kernel_size=(self.n_chans, 1),
                max_norm=self.conv_spatial_max_norm,
                bias=False,
                groups=self.F1,
            ),
        )

        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(
                self.F1 * self.D,
                momentum=self.batch_norm_momentum,
                affine=self.batch_norm_affine,
                eps=self.batch_norm_eps,
            ),
        )
        self.add_module("elu_1", activation())

        self.add_module(
            "pool_1",
            pool_class(
                kernel_size=(1, self.pool1_kernel_size),
            ),
        )
        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, self.depthwise_kernel_length),
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, self.depthwise_kernel_length // 2),
            ),
        )
        self.add_module(
            "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                kernel_size=(1, 1),
                bias=False,
            ),
        )

        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(
                self.F2,
                momentum=self.batch_norm_momentum,
                affine=self.batch_norm_affine,
                eps=self.batch_norm_eps,
            ),
        )
        self.add_module("elu_2", self.activation())
        self.add_module(
            "pool_2",
            pool_class(
                kernel_size=(1, self.pool2_kernel_size),
            ),
        )
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        output_shape = self.get_output_shape()
        n_out_virtual_chans = output_shape[2]

        if self.final_conv_length == "auto":
            n_out_time = output_shape[3]
            self.final_conv_length = n_out_time

        # Incorporating classification module and subsequent ones in one final layer
        module = nn.Sequential()
        if not final_layer_with_constraint:
            module.add_module(
                "conv_classifier",
                nn.Conv2d(
                    self.F2,
                    self.n_outputs,
                    (n_out_virtual_chans, self.final_conv_length),
                    bias=True,
                ),
            )

            # Transpose back to the logic of braindecode,
            # so time in third dimension (axis=2)
            module.add_module(
                "permute_back",
                Rearrange("batch x y z -> batch x z y"),
            )

            module.add_module("squeeze", SqueezeFinalOutput())
        else:
            module.add_module("flatten", nn.Flatten())
            module.add_module(
                "linearconstraint",
                LinearWithConstraint(
                    in_features=self.F2 * self.final_conv_length,
                    out_features=self.n_outputs,
                    max_norm=norm_rate,
                ),
            )
        self.add_module("final_layer", module)

        glorot_weight_zero_bias(self)


@deprecated(
    "`EEGNetv4` was renamed to `EEGNet` in v1.12; this alias will be removed in v1.14."
)
class EEGNetv4(EEGNet):
    """Deprecated alias for EEGNet."""

    pass
