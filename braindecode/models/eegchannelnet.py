# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin


class ConvLayer2D(nn.Sequential):
    """Convolutional Layer Block with BatchNorm, ReLU, Conv2d, and Dropout2d.

    This block applies a 2D convolution preceded by batch normalization and ReLU activation,
    followed by dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple
        Size of the convolving kernel.
    stride : tuple
        Stride of the convolution.
    padding : tuple or int
        Zero-padding added to both sides of the input.
    dilation : tuple or int
        Spacing between kernel elements.
    dropout_rate : float, optional
        Dropout rate after the convolution. Default is 0.2.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        dropout_rate=0.2,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", activation(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
        )
        self.add_module("drop", nn.Dropout2d(dropout_rate))


class TemporalBlock(nn.Module):
    """Temporal Block consisting of multiple ConvLayer2D layers with different dilations.

    This block applies several ConvLayer2D layers to extract temporal features,
    concatenating their outputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for each ConvLayer2D layer.
    n_layers : int
        Number of layers in the temporal block.
    kernel_size : tuple
        Kernel size for the convolutional layers.
    stride : tuple
        Stride for the convolutional layers.
    dilation_list : list of tuples
        List of dilation factors for each layer.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        kernel_size,
        stride,
        dilation_list,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (
                n_layers - len(dilation_list)
            )

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        for dilation in dilation_list:
            filter_size_h = (kernel_size[0] - 1) * dilation[0] + 1
            temp_pad_h = (filter_size_h - 1) // 2
            filter_size_w = (kernel_size[1] - 1) * dilation[1] + 1
            temp_pad_w = (filter_size_w - 1) // 2
            padding.append((temp_pad_h, temp_pad_w))

        self.layers = nn.ModuleList(
            [
                ConvLayer2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding[i],
                    dilation_list[i],
                    activation=activation,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        """Forward pass through the temporal block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Concatenated output from all temporal layers.
        """
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, dim=1)
        return out


class SpatialBlock(nn.Module):
    """Spatial Block consisting of multiple ConvLayer2D layers with varying kernel sizes.

    This block applies several ConvLayer2D layers to extract spatial features,
    concatenating their outputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for each ConvLayer2D layer.
    num_spatial_layers : int
        Number of layers in the spatial block.
    stride : tuple
        Stride for the convolutional layers.
    input_height : int
        Height of the input tensor (number of channels).
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_spatial_layers,
        stride,
        input_height,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_height = max(1, input_height // (i + 1))
            kernel_list.append((kernel_height, 1))

        padding = []
        for kernel in kernel_list:
            pad_h = (kernel[0] - 1) // 2
            pad_w = (kernel[1] - 1) // 2
            padding.append((pad_h, pad_w))

        self.layers = nn.ModuleList(
            [
                ConvLayer2D(
                    in_channels,
                    out_channels,
                    kernel_list[i],
                    stride,
                    padding[i],
                    dilation=1,
                    activation=activation,
                )
                for i in range(num_spatial_layers)
            ]
        )

    def forward(self, x):
        """Forward pass through the spatial block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Concatenated output from all spatial layers.
        """
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, dim=1)
        return out


class ResidualBlock(nn.Module):
    """Residual Block consisting of two 3x3 convolutional layers with a skip connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple, optional
        Stride of the convolution. Default is 1.
    downsample : nn.Module or None, optional
        Downsampling layer to match dimensions. Default is None.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = activation(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class FeaturesExtractor(nn.Module):
    """Feature Extraction module consisting of TemporalBlock, SpatialBlock, and ResidualBlocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    temp_channels : int
        Number of channels in the temporal block.
    out_channels : int
        Number of output channels.
    input_width : int
        Width of the input tensor (number of time samples).
    input_height : int
        Height of the input tensor (number of channels).
    temporal_kernel : tuple
        Kernel size for temporal convolutions.
    temporal_stride : tuple
        Stride for temporal convolutions.
    temporal_dilation_list : list of tuples
        Dilation factors for temporal convolutions.
    num_temporal_layers : int
        Number of layers in the temporal block.
    num_spatial_layers : int
        Number of layers in the spatial block.
    spatial_stride : tuple
        Stride for spatial convolutions.
    num_residual_blocks : int
        Number of residual blocks.
    down_kernel : int
        Kernel size for downsampling convolution.
    down_stride : int
        Stride for downsampling convolution.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    """

    def __init__(
        self,
        in_channels,
        temp_channels,
        out_channels,
        input_height,
        temporal_kernel,
        temporal_stride,
        temporal_dilation_list,
        num_temporal_layers,
        num_spatial_layers,
        spatial_stride,
        num_residual_blocks,
        down_kernel,
        down_stride,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.temporal_block = TemporalBlock(
            in_channels,
            temp_channels,
            num_temporal_layers,
            temporal_kernel,
            temporal_stride,
            temporal_dilation_list,
            activation=activation,
        )

        self.spatial_block = SpatialBlock(
            temp_channels * num_temporal_layers,
            out_channels,
            num_spatial_layers,
            spatial_stride,
            input_height,
            activation=activation,
        )

        res_blocks = []
        for _ in range(num_residual_blocks):
            res_block = nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers,
                    out_channels * num_spatial_layers,
                    activation=activation,
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers,
                    out_channels * num_spatial_layers,
                    kernel_size=(down_kernel, down_kernel),
                    stride=(down_stride, down_stride),
                    padding=0,
                    dilation=1,
                    activation=activation,
                ),
            )
            res_blocks.append(res_block)
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers,
            out_channels,
            kernel_size=(down_kernel, down_kernel),
            stride=(1, 1),
            padding=0,
            dilation=1,
        )

    def forward(self, x):
        """Forward pass through the feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Extracted features.
        """
        out = self.temporal_block(x)
        out = self.spatial_block(out)

        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)
        return out


class EEGChannelNet(EEGModuleMixin, nn.Module):
    """EEG ChannelNet model from [EEGChannelNet]_.

    The model applies temporal and spatial convolutions to extract features from EEG signals,
    followed by residual blocks and fully connected layers for classification.

    Parameters
    ----------
    temp_channels : int, optional
        Number of channels in the temporal block. Default is 10.
    out_channels : int, optional
        Number of output channels. Default is 50.
    embedding_size : int, optional
        Size of the embedding vector. Default is 1000.
    temporal_dilation_list : list of tuples, optional
        List of dilations for temporal convolutions. Default is [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)].
    temporal_kernel : tuple, optional
        Size of the temporal kernel. Default is (1, 33).
    temporal_stride : tuple, optional
        Size of the temporal stride. Default is (1, 2).
    num_temporal_layers : int, optional
        Number of temporal block layers. Default is 4.
    num_spatial_layers : int, optional
        Number of spatial layers. Default is 4.
    spatial_stride : tuple, optional
        Size of the spatial stride. Default is (2, 1).
    num_residual_blocks : int, optional
        Number of residual blocks. Default is 4.
    down_kernel : int, optional
        Size of the bottleneck kernel. Default is 3.
    down_stride : int, optional
        Size of the bottleneck stride. Default is 2.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

    Notes
    -----
    This implementation is not guaranteed to be correct! it has not been checked
    by original authors. The modifications are based on derivated code from
    [CodeICASSP2025]_.

    References
    ----------
    .. [EEGChannelNet] Palazzo, S., Spampinato, C., Kavasidis, I., Giordano, D.,
       Schmidt, J., & Shah, M. (2020). Decoding brain representations by
       multimodal learning of neural activity and visual features.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       43(11), 3833-3849.
    .. [CodeICASSP2025] Code from Baselines for EEG-Music Emotion Recognition
       Grand Challenge at ICASSP 2025.
       https://github.com/SalvoCalcagno/eeg-music-challenge-icassp-2025-baselines

    """

    def __init__(
        self,
        # Braindecode's parameters
        n_chans=None,
        n_times=None,
        n_outputs=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # model's parameters
        temp_channels: int = 10,
        out_channels: int = 50,
        embedding_size: int = 1000,
        temporal_dilation_list: list[tuple[int, int]] = [
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 8),
            (1, 16),
        ],
        temporal_kernel: tuple[int, int] = (1, 33),
        temporal_stride: tuple[int, int] = (1, 2),
        num_temporal_layers: int = 4,
        num_spatial_layers: int = 4,
        spatial_stride: tuple[int, int] = (2, 1),
        num_residual_blocks: int = 4,
        down_kernel: int = 3,
        down_stride: int = 2,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        self.activation = activation

        self.encoder = FeaturesExtractor(
            in_channels=1,
            temp_channels=temp_channels,
            out_channels=out_channels,
            input_height=self.n_chans,
            temporal_kernel=temporal_kernel,
            temporal_stride=temporal_stride,
            temporal_dilation_list=temporal_dilation_list,
            num_temporal_layers=num_temporal_layers,
            num_spatial_layers=num_spatial_layers,
            spatial_stride=spatial_stride,
            num_residual_blocks=num_residual_blocks,
            down_kernel=down_kernel,
            down_stride=down_stride,
            activation=self.activation,
        )

        # Compute the encoding size by passing a dummy input through the
        # encoder
        encoding_size = self.calculate_embedding_size()

        self.embedding = nn.Sequential(
            nn.Linear(encoding_size, embedding_size), self.activation(inplace=True)
        )

        self.final_layer = nn.Linear(embedding_size, self.n_outputs)

    def forward(self, x):
        """Forward pass through the EEGChannelNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        x = x.unsqueeze(1)
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        out = self.self.embedding(out)
        out = self.final_layer(out)
        return out

    def calculate_embedding_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_chans, self.n_times)
            encoding = self.encoder(dummy_input)
            encoding_size = encoding.numel()
        return encoding_size
