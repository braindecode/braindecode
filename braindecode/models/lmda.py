"""LMDA Neural Network.
Authors: Miao Zheng Qing
         Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
"""

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class EEGDepthAttention(nn.Module):
    """
    EEG Depth Attention Module.

    This module applies depth-wise attention to the input EEG features.

    Parameters
    ----------
    n_channels : int
        Number of channels in the input data.
    n_times : int
        Number of time samples.
    kernel_size : int, optional
        Kernel size for the convolution, by default 7.

    Attributes
    ----------
    adaptive_pool : nn.AdaptiveAvgPool2d
        Adaptive average pooling layer.
    conv : nn.Conv2d
        Convolutional layer with kernel size `(kernel_size, 1)`.
    softmax : nn.Softmax
        Softmax layer applied over the depth dimension.

    """

    def __init__(self, n_channels: int, n_times: int, kernel_size: int = 7):
        super().__init__()
        self.n_channels = n_channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, n_times))
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=True,
        )
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EEG Depth Attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, n_channels, depth, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor after applying depth-wise attention.

        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.n_channels * x


class LMDANet(EEGModuleMixin, nn.Module):
    """LMDA-Net Model from Miao, Z et al (2023) [lmda]_.


    .. figure:: https://ars.els-cdn.com/content/image/1-s2.0-S1053811923003609-gr2_lrg.jpg
       :align: center
       :alt: LMDA-Net Architecture
    Overview of the Neural Network architecture.

    LMDA-Net is designed to effectively capture spectro-spatial-temporal features
    for motor imagery decoding from EEG data. It combines a benchmark feature
    extraction network, inspired by ConvNet and EEGNet, with two attention
    mechanisms: a channel attention module and a depth attention module.

    - **Channel Attention Module**: Enhances the spatial information in EEG data
      by mapping channel information to the depth dimension using tensor
      multiplication. This step helps in effectively integrating spatial
      information, based on neuroscience knowledge of the low spatial resolution
      of EEG signals.

    - **Depth Attention Module**: Further refines the extracted high-dimensional
      EEG features by enhancing interactions in the depth dimension. It applies
      semi-global pooling, a convolutional layer, and softmax activation to
      strengthen feature interactions between temporal and spatial dimensions.

    Parameters
    ----------
    depth : int, optional
        Depth parameter of the model, by default 9.
    kernel_size : int, optional
        Kernel size for temporal convolution, by default 75.
    channel_depth1 : int, optional
        Number of channels in the first convolutional layer, by default 24.
    channel_depth2 : int, optional
        Number of channels in the second convolutional layer, by default 9.
    avepool_size : int, optional
        Pooling size for average pooling, by default 5.
    activation : nn.Module, optional
        Activation function class to apply, by default `nn.GELU`.
    drop_prob : float, optional
        Dropout probability for regularization, by default 0.65.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only adaptation from PyTorch source code [lmdacode]_.

    References
    ----------
    .. [lmda] Miao, Z., Zhao, M., Zhang, X., & Ming, D. (2023). LMDA-Net: A
       lightweight multi-dimensional attention network for general EEG-based
        brain-computer interfaces and interpretability. NeuroImage, 276, 120209.
    .. [lmdacode] Miao, Z., Zhao, M., Zhang, X., & Ming, D. (2023). LMDA-Net: A
       lightweight multi-dimensional attention network for general EEG-based
       brain-computer interfaces and interpretability.
       https://github.com/MiaoZhengQing/LMDA-Code
    """

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # model related
        depth: int = 9,
        kernel_size: int = 75,
        channel_depth1: int = 24,
        channel_depth2: int = 9,
        avepool_size: int = 5,
        activation: nn.Module = nn.GELU,
        drop_prob: float = 0.65,
    ):
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        # TO-DO: normalize the variable names
        self.depth = depth
        self.kernel_size = kernel_size
        self.channel_depth1 = channel_depth1
        self.channel_depth2 = channel_depth2
        self.avepool_size = avepool_size
        self.activation = activation
        self.drop_prob = drop_prob

        # Initialize channel weights
        self.channel_weight = nn.Parameter(
            torch.randn(self.depth, 1, self.n_chans), requires_grad=True
        )
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.ensuredim = Rearrange("batch chans time -> batch 1 chans time")
        # Temporal Convolutional Layers
        self.time_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.depth,
                out_channels=self.channel_depth1,
                kernel_size=(1, 1),
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth1),
            nn.Conv2d(
                in_channels=self.channel_depth1,
                out_channels=self.channel_depth1,
                kernel_size=(1, self.kernel_size),
                groups=self.channel_depth1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth1),
            self.activation(),
        )

        # TO-DO: remove this, and calculate this manually

        # Compute dimensions after temporal convolution
        with torch.no_grad():
            dummy_input = torch.ones(1, 1, self.n_chans, self.n_times)
            x = torch.einsum("bdcw, hdc->bhcw", dummy_input, self.channel_weight)
            x_time = self.time_conv(x)
            _, c_time, _, n_times_time = x_time.size()

        # Depth-wise Attention Module
        self.depth_attention = EEGDepthAttention(
            n_channels=c_time, n_times=n_times_time, kernel_size=7
        )

        # Spatial Convolutional Layers
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_depth1,
                out_channels=self.channel_depth2,
                kernel_size=(1, 1),
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth2),
            nn.Conv2d(
                in_channels=self.channel_depth2,
                out_channels=self.channel_depth2,
                kernel_size=(self.n_chans, 1),
                groups=self.channel_depth2,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth2),
            self.activation(),
        )

        # Normalization Layers
        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, self.avepool_size)),
            nn.Dropout(p=self.drop_prob),
        )

        # TO-DO: remove this, and calculate this manually
        # Compute the number of features for the final layer
        with torch.no_grad():
            x_time = self.depth_attention(x_time)
            x = self.spatial_conv(x_time)
            x = self.norm(x)
            n_out_features = x.view(1, -1).size(1)

        # Final Classification Layer
        self.final_layers = nn.Linear(
            in_features=n_out_features, out_features=self.n_outputs
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of convolutional and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LMDA model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, n_chans, time)`.

        Returns
        -------
        torch.Tensor
            Output logits of shape `(batch_size, n_outputs)`.

        """
        x = self.ensuredim(x)
        x = torch.einsum("bdcw, hdc->bhcw", x, self.channel_weight)  # Channel weighting
        x_time = self.time_conv(x)  # Temporal convolution
        x_time = self.depth_attention(x_time)  # Depth-wise attention
        x = self.spatial_conv(x_time)  # Spatial convolution
        x = self.norm(x)  # Normalization and dropout
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.final_layers(x)
        return logits
