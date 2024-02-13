import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin

class ChannelwiseAdaptiveFilter(nn.Module):
    """
    ChannelwiseAdaptiveFilter.

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
        super().__init__()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, n_times))
        # Maybe we can replace this with conv1d, I tried but failed
        self.conv = nn.Conv2d(1, 1,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0),
                              bias=True)

        self.softmax = nn.Softmax(dim=-2)
        self.n_chans = n_chans

    def forward(self, x):
        """
        Forward pass.

        We first apply the adaptive average pooling layer to the input tensor.
        Then, we apply the convolutional layer to the pooled tensor.
        Finally, we apply the softmax function to the output of the convolutional
        layer.

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        x_t = self.adaptive_pool(x)

        x_t = x_t.transpose(-2, -3)
        x_t = self.conv(x_t)
        x_t = self.softmax(x_t)
        x_t = x_t.transpose(-2, -3)

        return x_t * self.n_chans * x


class LDMNet(EEGModuleMixin, nn.Module):
    """
    LMDA-Net.

    A lightweight multi-dimensional attention network for
    general EEG-based brain-computer interface paradigms and
    interpretability.

    The paper with more details about the methodological
    choices are available at the [Miao2023]_.

    References
    ----------
    .. [Miao2023] Miao, Z., Zhao, M., Zhang, X. and Ming, D., 2023. LMDA-Net:
        A lightweight multi-dimensional attention network for general
        EEG-based brain-computer interfaces and interpretability.
        NeuroImage, p.120209.
    """

    def __init__(self, n_chans=22, n_times=1125, n_outputs=4,
                 depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                 ave_depth=1, avepool=5, drop_prob=0.5,
                 sfreq=None, chs_info=None):

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq

        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, self.n_chans),
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
                      kernel_size=(self.n_chans, 1),
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
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
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
        """
        Util function to initialize the weights of the model.

        If the layers is a Conv2d, we use xavier_uniform_ to initialize the
        weights and zeros_ to initialize the bias. If the layer is a
        BatchNorm2d, we use ones_ to initialize the weights and zeros_ to
        initialize the bias. If the layer is a Linear, we use xavier_uniform_
        to initialize the weights and zeros_ to initialize the bias.

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

    def _initialize_dynamic_layers(self):
        """
        Util function to initialize the dynamic layers of the model.

        Dynamically initializes the ChannelwiseAdaptiveFilter and classifier
        based on simulated forward pass to infer output dimensions.
        """
        with torch.no_grad():
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
