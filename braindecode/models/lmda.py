# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import torch
from torch import nn, Tensor


class EEGDepthAttention(nn.Module):
    """Depth-wise Attention Mechanism for EEG Data.

    This module implements a depth-wise attention mechanism where the attention
    is applied across the depth (channel) dimension of EEG data. It uses an
    adaptive average pooling to reduce the temporal dimension followed by a
    depth-wise convolution to compute attention scores.

    Parameters
    ----------
    input_width : int
        The width of the input tensor.
    channels : int
        The number of channels in the input tensor.
    kernel_size : int, default=7
        The size of the kernel to be used in the convolutional layer.

    Attributes
    ----------
    adaptive_pool : nn.AdaptiveAvgPool2d
        Adaptive average pooling layer to reduce temporal dimension.
    conv : nn.Conv2d
        Convolutional layer to compute attention scores.
    softmax : nn.Softmax
        Softmax layer to normalize attention scores.
    """

    def __init__(self, input_width, channels, kernel_size=7):
        super(EEGDepthAttention, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, input_width))
        self.conv = nn.Conv2d(1, 1,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        x_pooled = self.adaptive_pool(x)
        x_transposed = x_pooled.transpose(-2, -3)
        y = self.conv(x_transposed)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.channels * x


class LMDA(nn.Module):
    """ LMDA-Net

    A lightweight multi-dimensional attention network for
    general EEG-based brain-computer interface paradigms and
    interpretability.

    The paper with more details about the methodological
    choices are available at the [Miao2023]_.


    Parameters
    ----------
    channels : int
        Number of EEG channels.
    samples : int
        Number of time samples in the EEG signal.
    num_classes : int
        Number of output classes for the classification task.
    depth : int, default=9
        Depth of the initial channel weighting.
    kernel : int, default=75
        Kernel size for temporal convolutions.
    channel_depth1 : int, default=24
        Number of channels after the first convolutional layer.
    channel_depth2 : int, default=9
        Number of channels after the second convolutional layer.
    ave_depth : int, default=1
        Depth for averaging in the pooling layers.
    avepool : int, default=5
        Pool size for the average pooling layer.
    final_fc_length: int | str
        The dimension of the fully connected layer.

         References
    ----------
    .. [Miao2023] Miao, Z., Zhao, M., Zhang, X. and Ming, D., 2023. LMDA-Net:
        A lightweight multi-dimensional attention network for general
        EEG-based brain-computer interfaces and interpretability.
        NeuroImage, p.120209.
    """



    def __init__(self, channels=22, samples=1125, num_classes=4, depth=9,
                 kernel=75, channel_depth1=24, channel_depth2=9,
                 avepool=5, final_fc_length="auto"):
        super(LMDA, self).__init__()
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, channels),
                                           requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1,
                      bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        self.channel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1),
                      groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2,
                      kernel_size=(channels, 1), groups=channel_depth2,
                      bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # Initialize depthAttention with dynamic width calculation
        self.depthAttention = EEGDepthAttention(samples, channel_depth1, k=7)

        if final_fc_length == "auto":
            assert self.n_times is not None
            final_fc_length = self.get_fc_size()

        self.classifier = nn.Linear(final_fc_length, num_classes)

        self._initialize_weights()

    def get_fc_size(self):

        out = self.patch_embedding(torch.ones((1, 1,
                                               self.n_chans,
                                               self.n_times)))
        size_embedding_1 = out.cpu().data.numpy().shape[1]
        size_embedding_2 = out.cpu().data.numpy().shape[2]

        return size_embedding_1 * size_embedding_2

    def forward(self, x: Tensor) -> Tensor:
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x_time = self.time_conv(x)
        x_time = self.depthAttention(x_time)
        x_channel = self.channel_conv(x_time)
        x_norm = self.norm(x_channel)
        features = torch.flatten(x_norm, 1)
        output = self.classifier(features)
        return output


    def _initialize_weights(self):
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
                nn.init.zeros_(m.bias)
