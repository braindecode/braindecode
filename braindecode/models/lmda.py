import torch
import torch.nn as nn


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

    def __init__(self, n_chans=22, n_times=1125, n_outputs=4,
                 depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                 ave_depth=1, avepool=5, drop_prob=0.5):

        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs

        self.channel_weight = nn.Parameter(torch.randn(depth, 1, n_chans),
                                           requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1,
                      kernel_size=(1, 1),
                      groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2,
                      kernel_size=(1, 1),
                      groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2,
                      kernel_size=(n_chans, 1),
                      groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=drop_prob),
        )

        self._initialize_dynamic_layers()

    def forward(self, x):
        # Lead weight filtering
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        x_time = self.temporal_conv(x)
        x_time = self.channel_adaptive(x_time)

        x = self.channel_conv(x_time)
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

    def _initialize_dynamic_layers(self):
        """
        Dynamically initializes the ChannelwiseAdaptiveFilter and classifier
        based on simulated forward pass to infer output dimensions.
        """
        with torch.no_grad():  # Ensure operations do not track gradients
            # Simulate input tensor

            out = torch.ones((1, 1, self.n_chans, self.n_times),
                             dtype=torch.float32, requires_grad=False)

            out = torch.einsum('bdcw, hdc->bhcw', out,
                               self.channel_weight)
            out = self.temporal_conv(out)

            # Initialize ChannelwiseAdaptiveFilter with dynamic dimensions
            _, C, _, W = out.shape
            self.channel_adaptive = ChannelwiseAdaptiveFilter(W, C, kernel_size=7)

            # Continue through channel_conv and norm layers
            out = self.channel_conv(out)
            out = self.norm(out)

            # Calculate the classifier's input size dynamically
            _, C, H, W = out.shape  # Extract dynamic dimensions
            classifier_input_size = C * H * W
            self.classifier = nn.Linear(classifier_input_size,
                                        self.n_outputs)

model = LMDA(n_outputs=2, n_chans=3, n_times=875, channel_depth1=24, channel_depth2=7)
a = torch.randn(12, 1, 3, 875).float()
l2 = model(a)
print(l2.shape)
