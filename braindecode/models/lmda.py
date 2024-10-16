import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class _EEGDepthAttention(nn.Module):
    """
    EEG Depth Attention Module.

    The depth attention module generates an output map M(F) ∈ ℝ^(D'×1×T')
    from the input feature F ∈ ℝ^(D'×C'×T'), defined as:

        M(F) = (Softmax(Conv(Pooling(F)^T)) * D')^T    (3)

    Where:
        - Pooling: Semi-global pooling applied on the spatial dimension.
        - Conv: Convolution operation for local cross-depth interaction.
        - Softmax: Softmax function to probabilize the depth information.
        - T: Transpose operation between the spatial and depth dimensions.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input data.
    num_times : int
        Number of time samples.
    kernel_size : int, optional
        Kernel size for the convolution, by default 7.
    """

    def __init__(self, num_channels: int, num_times: int, kernel_size: int = 7):
        super().__init__()
        self.num_channels = num_channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, num_times))
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
            Input tensor of shape `(batch_size, num_channels, depth, num_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor after applying depth-wise attention.

        """
        x_pool = self.adaptive_pool(x)  # D' x 1 x num_times
        x_transpose = x_pool.transpose(-2, -3)  # 1 x D' x num_times
        y = self.conv(x_transpose)  # 1 x D' x num_times
        y = self.softmax(y)  # 1 x D' x num_times
        y = y.transpose(-2, -3)  # D' x 1 x num_times
        return y * self.num_channels * x  # D' x num_channels x num_times


class LMDANet(EEGModuleMixin, nn.Module):
    """LMDA-Net Model from Miao, Z et al (2023) [lmda]_.


    .. figure:: https://ars.els-cdn.com/content/image/1-s2.0-S1053811923003609-gr2_lrg.jpg
       :align: center
       :alt: LMDA-Net Architecture
        Overview of the Neural Network architecture.

    LMDA-Net is a combination of ShallowConvNet, EEGNet and one attention
    mechanisms. The steps are:

    1. **Input Reshaping**: Reshape input for convolution operations.
    2. **Channel Weighting**: Apply learnable weights to channels to emphasize important ones.
    3. **Temporal Convolution**: Extract temporal features using two convolutional layers.
    4. **Depth-wise Attention**: Highlight important temporal features across channels.
    5. **Spatial Convolution**: Capture spatial features across channels.
    6. **Normalization and Dropout**: Reduce temporal size and regularize the model.
    7. **Flattening and Classification**: Flatten the tensor and apply a linear layer for classification

    Parameters
    ----------
    n_filters_time : int, optional
        Number of temporal filters, by default 9.
    kernel_size_time : int, optional
        Kernel size for temporal convolution, by default 75.
    channel_depth_1 : int, optional
        Number of channels in the first convolutional layer, by default 24.
    channel_depth_2 : int, optional
        Number of channels in the second convolutional layer, by default 9.
    avg_pool_size : int, optional
        Pooling size for average pooling, by default 5.
    kernel_size_attention: int, optional
       Kernel size for attention layer, by default 7.
    activation : nn.Module, optional
        Activation function class to apply, by default `nn.GELU`.
    dropout_prob : float, optional
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
        n_filters_time: int = 9,
        kernel_size_time: int = 75,
        channel_depth_1: int = 24,
        channel_depth_2: int = 9,
        kernel_size_attention: int = 7,
        avg_pool_size: int = 5,
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

        self.n_filters_time = n_filters_time
        self.kernel_size_time = kernel_size_time
        self.channel_depth_1 = channel_depth_1
        self.channel_depth_2 = channel_depth_2
        self.kernel_size_attention = kernel_size_attention
        self.avg_pool_size = avg_pool_size
        self.activation = activation
        self.drop_prob = drop_prob

        self.channel_weight = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_filters_time,
            kernel_size=(self.n_chans, 1),
            bias=False,
        )
        # nn.init.xavier_uniform_(self.channel_weight.data)

        reduced_time = self.n_times - self.kernel_size_time + 1
        embedding_size = self.channel_depth_2 * (reduced_time // avg_pool_size)

        # Layers:
        self.ensure_dim = Rearrange("batch channels time -> batch 1 channels time")
        # Temporal Convolutional Layers
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_filters_time,
                out_channels=self.channel_depth_1,
                kernel_size=(1, 1),
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth_1),
            nn.Conv2d(
                in_channels=self.channel_depth_1,
                out_channels=self.channel_depth_1,
                kernel_size=(1, self.kernel_size_time),
                groups=self.channel_depth_1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth_1),
            self.activation(),
        )

        # Depth-wise Attention Module
        self.depth_attention = _EEGDepthAttention(
            num_channels=self.channel_depth_1,
            num_times=reduced_time,
            kernel_size=self.kernel_size_attention,
        )

        # Spatial Convolutional Layers
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_depth_1,
                out_channels=self.channel_depth_2,
                kernel_size=(1, 1),
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth_2),
            nn.Conv2d(
                in_channels=self.channel_depth_2,
                out_channels=self.channel_depth_2,
                kernel_size=(1, 1),  # Previously this was `(self.n_chans, 1)`
                groups=self.channel_depth_2,
                bias=False,
            ),
            nn.BatchNorm2d(self.channel_depth_2),
            self.activation(),
        )

        # Normalization Layers
        self.normalization = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, self.avg_pool_size)),
            nn.Dropout(p=self.drop_prob),
        )

        # Final Classification Layer
        self.final_layer = nn.Linear(
            in_features=embedding_size, out_features=self.n_outputs
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
        x = self.ensure_dim(x)
        x = self.channel_weight(x)  #
        # Channel weighting
        x_time = self.temporal_conv(x)  # Temporal convolution
        x_time = self.depth_attention(x_time)  # Depth-wise attention
        x = self.spatial_conv(x_time)  # Spatial convolution
        x = self.normalization(x)  # Normalization and dropout
        x = x.view(x.size(0), -1)  # Flatten
        out = self.final_layer(x)
        return out
