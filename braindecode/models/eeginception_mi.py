# Authors: Cedric Rommel <cedric.rommel@inria.fr>
#
# License: BSD (3-clause)

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Ensure4d


class EEGInceptionMI(EEGModuleMixin, nn.Module):
    """EEG Inception for Motor Imagery, as proposed in Zhang et al. (2021) [1]_

    :bdg-success:`Convolution`

    .. figure:: https://content.cld.iop.org/journals/1741-2552/18/4/046014/revision3/jneabed81f1_hr.jpg
        :align: center
        :alt: EEGInceptionMI Architecture


    The model is strongly based on the original InceptionNet for computer
    vision. The main goal is to extract features in parallel with different
    scales. The network has two blocks made of 3 inception modules with a skip
    connection.

    The model is fully described in [1]_.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented bosed on the paper [1]_.

    Parameters
    ----------
    input_window_seconds : float, optional
        Size of the input, in seconds. Set to 4.5 s as in [1]_ for dataset
        BCI IV 2a.
    sfreq : float, optional
        EEG sampling frequency in Hz. Defaults to 250 Hz as in [1]_ for dataset
        BCI IV 2a.
    n_convs : int, optional
        Number of convolution per inception wide branching. Defaults to 5 as
        in [1]_ for dataset BCI IV 2a.
    n_filters : int, optional
        Number of convolutional filters for all layers of this type. Set to 48
        as in [1]_ for dataset BCI IV 2a.
    kernel_unit_s : float, optional
        Size in seconds of the basic 1D convolutional kernel used in inception
        modules. Each convolutional layer in such modules have kernels of
        increasing size, odd multiples of this value (e.g. 0.1, 0.3, 0.5, 0.7,
        0.9 here for ``n_convs=5``). Defaults to 0.1 s.
    activation: nn.Module
        Activation function. Defaults to ReLU activation.

    References
    ----------
    .. [1] Zhang, C., Kim, Y. K., & Eskandarian, A. (2021).
        EEG-inception: an accurate and robust end-to-end neural network
        for EEG-based motor imagery classification.
        Journal of Neural Engineering, 18(4), 046014.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        input_window_seconds=None,
        sfreq=250,
        n_convs: int = 5,
        n_filters: int = 48,
        kernel_unit_s: float = 0.1,
        activation: nn.Module = nn.ReLU,
        chs_info=None,
        n_times=None,
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

        self.n_convs = n_convs
        self.n_filters = n_filters
        self.kernel_unit_s = kernel_unit_s
        self.activation = activation

        self.ensuredims = Ensure4d()
        self.dimshuffle = Rearrange("batch C T 1 -> batch C 1 T")

        self.mapping = {
            "fc.weight": "final_layer.fc.weight",
            "tc.bias": "final_layer.fc.bias",
        }

        # ======== Inception branches ========================

        self.initial_inception_module = _InceptionModuleMI(
            in_channels=self.n_chans,
            n_filters=self.n_filters,
            n_convs=self.n_convs,
            kernel_unit_s=self.kernel_unit_s,
            sfreq=self.sfreq,
            activation=self.activation,
        )

        intermediate_in_channels = (self.n_convs + 1) * self.n_filters

        self.intermediate_inception_modules_1 = nn.ModuleList(
            [
                _InceptionModuleMI(
                    in_channels=intermediate_in_channels,
                    n_filters=self.n_filters,
                    n_convs=self.n_convs,
                    kernel_unit_s=self.kernel_unit_s,
                    sfreq=self.sfreq,
                    activation=self.activation,
                )
                for _ in range(2)
            ]
        )

        self.residual_block_1 = _ResidualModuleMI(
            in_channels=self.n_chans,
            n_filters=intermediate_in_channels,
            activation=self.activation,
        )

        self.intermediate_inception_modules_2 = nn.ModuleList(
            [
                _InceptionModuleMI(
                    in_channels=intermediate_in_channels,
                    n_filters=self.n_filters,
                    n_convs=self.n_convs,
                    kernel_unit_s=self.kernel_unit_s,
                    sfreq=self.sfreq,
                    activation=self.activation,
                )
                for _ in range(3)
            ]
        )

        self.residual_block_2 = _ResidualModuleMI(
            in_channels=intermediate_in_channels,
            n_filters=intermediate_in_channels,
            activation=self.activation,
        )

        # XXX The paper mentions a final average pooling but does not indicate
        # the kernel size... The only info available is figure1 showing a
        # final AveragePooling layer and the table3 indicating the spatial and
        # channel dimensions are unchanged by this layer... This could indicate
        # a stride=1 as for MaxPooling layers. Howevere, when we look at the
        # number of parameters of the linear layer following the average
        # pooling, we see a small number of parameters, potentially indicating
        # that the whole time dimension is averaged on this stage for each
        # channel. We follow this last hypothesis here to comply with the
        # number of parameters reported in the paper.
        self.ave_pooling = nn.AvgPool2d(
            kernel_size=(1, self.n_times),
        )

        self.flat = nn.Flatten()

        module = nn.Sequential()
        module.add_module(
            "fc",
            nn.Linear(
                in_features=intermediate_in_channels,
                out_features=self.n_outputs,
                bias=True,
            ),
        )
        module.add_module("out_fun", nn.Identity())
        self.final_layer = module

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        X = self.ensuredims(X)
        X = self.dimshuffle(X)

        res1 = self.residual_block_1(X)

        out = self.initial_inception_module(X)
        for layer in self.intermediate_inception_modules_1:
            out = layer(out)

        out = out + res1

        res2 = self.residual_block_2(out)

        for layer in self.intermediate_inception_modules_2:
            out = layer(out)

        out = res2 + out

        out = self.ave_pooling(out)
        out = self.flat(out)
        out = self.final_layer(out)
        return out


class _InceptionModuleMI(nn.Module):
    """
    Inception module.

    This module implements a inception-like architecture that processes input
    feature maps through multiple convolutional paths with different kernel
    sizes, allowing the network to capture features at multiple scales.
    It includes bottleneck layers, convolutional layers, and pooling layers,
    followed by batch normalization and an activation function.

    Parameters
    ----------
    in_channels : int
        Number of input channels in the input tensor.
    n_filters : int
        Number of filters (output channels) for each convolutional layer.
    n_convs : int
        Number of convolutional layers in the module (excluding bottleneck
        and pooling paths).
    kernel_unit_s : float, optional
        Base size (in seconds) for the convolutional kernels. The actual kernel
        size is computed as ``(2 * n_units + 1) * kernel_unit``, where ``n_units``
        ranges from 0 to ``n_convs - 1``. Default is 0.1 seconds.
    sfreq : float, optional
        Sampling frequency of the input data, used to convert kernel sizes from
        seconds to samples. Default is 250 Hz.
    activation : nn.Module class, optional
        Activation function class to apply after batch normalization. Should be
        a PyTorch activation module class like ``nn.ReLU`` or ``nn.ELU``.
        Default is ``nn.ReLU``.

    """

    def __init__(
        self,
        in_channels,
        n_filters,
        n_convs,
        kernel_unit_s=0.1,
        sfreq=250,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.n_convs = n_convs
        self.kernel_unit_s = kernel_unit_s
        self.sfreq = sfreq

        self.bottleneck = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.n_filters,
            kernel_size=1,
            bias=True,
        )

        kernel_unit = int(self.kernel_unit_s * self.sfreq)

        # XXX Maxpooling is usually used to reduce spatial resolution, with a
        # stride equal to the kernel size... But it seems the authors use
        # stride=1 in their paper according to the output shapes from Table3,
        # although this is not clearly specified in the paper text.
        self.pooling = nn.MaxPool2d(
            kernel_size=(1, kernel_unit),
            stride=1,
            padding=(0, int(kernel_unit // 2)),
        )

        self.pooling_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.n_filters,
            kernel_size=1,
            bias=True,
        )

        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=(1, (n_units * 2 + 1) * kernel_unit),
                    padding="same",
                    bias=True,
                )
                for n_units in range(self.n_convs)
            ]
        )

        self.bn = nn.BatchNorm2d(self.n_filters * (self.n_convs + 1))

        self.activation = activation()

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        X1 = self.bottleneck(X)

        X1 = [conv(X1) for conv in self.conv_list]

        X2 = self.pooling(X)
        X2 = self.pooling_conv(X2)
        # Get the target length from one of the conv branches
        target_len = X1[0].shape[-1]

        # Crop the pooling output if its length does not match
        if X2.shape[-1] != target_len:
            X2 = X2[..., :target_len]

        out = torch.cat(X1 + [X2], 1)

        out = self.bn(out)
        return self.activation(out)


class _ResidualModuleMI(nn.Module):
    """
    Residual module.

    This module performs a 1x1 convolution followed by batch normalization and an activation function.
    It is designed to process input feature maps and produce transformed output feature maps, often used
    in residual connections within neural network architectures.

    Parameters
    ----------
    in_channels : int
        Number of input channels in the input tensor.
    n_filters : int
        Number of filters (output channels) for the convolutional layer.
    activation : nn.Module, optional
        Activation function to apply after batch normalization. Should be an instance of a PyTorch
        activation module (e.g., ``nn.ReLU()``, ``nn.ELU()``). Default is ``nn.ReLU()``.


    """

    def __init__(self, in_channels, n_filters, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.activation = activation()

        self.bn = nn.BatchNorm2d(self.n_filters)
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.n_filters,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the residual module.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, ch_names, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, batch normalization, and activation function.
        """
        out = self.conv(X)
        out = self.bn(out)
        return self.activation(out)
