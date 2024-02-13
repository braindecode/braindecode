import torch
import torch.nn as nn


class EEGDepthAttention_old(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """

    def __init__(self, W, C, k=7):
        super(EEGDepthAttention_old, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0),
                              bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        # print('查看参数是否变化:', conv.bias)

        return y * self.C * x


class ChannelwiseAdaptiveFilter(nn.Module):
    """ChannelwiseAdaptiveFilter

    This module applies an adaptive average pooling layer to the input tensor.
    Then, a 1D convolutional layer is applied to the pooled tensor.
    Finally, the softmax function is applied to the output of the convolutional
    layer.

    Parameters
    ----------
    n_times : int
        The number of time points in the input tensor.
    n_chans : int
        The number of channels in the input tensor.
    kernel_size : int, default=7
        The size of the kernel to be used in the convolutional layer.
    """

    def __init__(self, n_times, n_chans, kernel_size=7):
        super(ChannelwiseAdaptiveFilter, self).__init__()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, n_times))
        # Maybe we can replace this with conv1d, I tried but failed
        self.conv = nn.Conv2d(1, 1,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0),
                              bias=True)

        self.softmax = nn.Softmax(dim=-2)
        self.n_chans = n_chans

    def forward(self, x):
        x_t = self.adaptive_pool(x)

        x_t = x_t.transpose(-2, -3)
        x_t = self.conv(x_t)
        x_t = self.softmax(x_t)
        x_t = x_t.transpose(-2, -3)

        return x_t * self.n_chans * x


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """

    def __init__(self, n_chans=22, n_times=1125, n_outputs=4, depth=9,
                 kernel=75, channel_depth1=24, channel_depth2=9,
                 ave_depth=1, avepool=5, drop_prob=0.5):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth

        self.channel_weight = nn.Parameter(torch.randn(depth, 1, n_chans),
                                           requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1,
                      kernel_size=(1, 1),
                      groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2,
                      kernel_size=(1, 1),
                      groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(n_chans, 1),
                      groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=drop_prob),
        )

        out = torch.ones((1, 1, n_chans, n_times))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        N, C, H, W = out.size()
        self.depthAttention = ChannelwiseAdaptiveFilter(W, C, kernel_size=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(
            n_out_time[-1] * n_out_time[-2] * n_out_time[-3], n_outputs)

        self._initialize_weights()


    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

model = LMDA(n_outputs=2, n_chans=3, n_times=875, channel_depth1=24, channel_depth2=7)
a = torch.randn(12, 1, 3, 875).float()
l2 = model(a)
print(l2.shape)
