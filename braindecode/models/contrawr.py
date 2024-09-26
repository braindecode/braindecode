from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from .base import EEGModuleMixin


class ResBlock(nn.Module):
    """Convolutional Residual Block 2D.

    This block stacks two convolutional layers with batch normalization,
    max pooling, dropout, and residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int (default=1)
        Stride of the convolutional layers.
    use_downsampling : bool (default=True)
        Whether to use a downsampling residual connection.
    pooling : bool (default=True)
        Whether to use max pooling.
    kernel_size : int (default=3)
        Kernel size of the convolutional layers.
    padding : int (default=1)
        Padding of the convolutional layers.
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

    Examples
    --------
    >>> import torch
    >>> model = ResBlock2D(6, 16, 1, True, True)
    >>> input_ = torch.randn((16, 6, 28, 150))  # (batch, channel, height, width)
    >>> output = model(input_)
    >>> output.shape
    torch.Size([16, 16, 14, 75])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        use_downsampling=True,
        pooling=True,
        kernel_size=3,
        padding=1,
        drop_prob=0.5,
        activation: nn.Module = nn.ReLU,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = activation()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.use_downsampling = use_downsampling
        self.pooling = pooling
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        """

        Parameters
        ----------
        X: Tensor
            Input tensor of shape (batch_size, n_channels, n_freqs, n_times).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, n_channels, n_freqs, n_times).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_downsampling:
            residual = self.downsample(x)
            out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class ContraWR(EEGModuleMixin, nn.Module):
    """Contrast with the World Representation ContraWR from Yang et al. 2021 [Yang2021]_.

    This model is a convolutional neural network that uses a spectral
    representation with a series of convolutional layers and residual blocks.
    The model is designed to learn a representation of the EEG signal that can
    be used for sleep staging.

    Parameters
    ----------
    steps : int, optional
        Number of steps to take the frequency decomposition `hop_length`
        parameters by default 20.
    emb_size : int, optional
        Embedding size for the final layer, by default 256.
    res_channels : list[int], optional
        Number of channels for each residual block, by default [32, 64, 128].
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    .. versionadded:: 0.9

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors. The modifications are minimal and the model is expected
    to work as intended. the original code from [Code2023]_.

    References
    ----------
    .. [Yang2021] Yang, C., Xiao, C., Westover, M. B., & Sun, J. (2023).
       Self-supervised electroencephalogram representation learning for automatic
       sleep staging: model development and evaluation study. JMIR AI, 2(1), e46769.
    .. [Code2023] Yang, C., Westover, M.B. and Sun, J., 2023. BIOT
       Biosignal Transformer for Cross-data Learning in the Wild.
       GitHub https://github.com/ycq091044/BIOT (accessed 2024-02-13)
    """

    def __init__(
        self,
        n_chans: int | None = None,
        n_outputs: int | None = None,
        sfreq: int | None = None,
        emb_size: int = 256,
        res_channels: list[int] = [32, 64, 128],
        steps=20,
        activation: nn.Module = nn.ELU,
        # Another way to pass the EEG parameters
        chs_info: list[dict[Any, Any]] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds
        if not isinstance(res_channels, list):
            raise ValueError("res_channels must be a list of integers.")

        self.n_fft = int(self.sfreq)
        self.steps = steps

        res_channels = [self.n_chans] + res_channels + [emb_size]

        self.convs = nn.ModuleList(
            [
                ResBlock(
                    in_channels=res_channels[i],
                    out_channels=res_channels[i + 1],
                    stride=2,
                    use_downsampling=True,
                    pooling=True,
                )
                for i in range(len(res_channels) - 1)
            ]
        )

        self.final_layer = nn.Sequential(
            activation(),
            nn.Linear(emb_size, self.n_outputs),
        )

    def torch_stft(self, x):
        """
        Compute the Short-Time Fourier Transform (STFT) of the input tensor.

        EEG Signal is expected to be of shape (batch_size, n_channels, n_times).

        Parameters
        ----------
        X: Tensor
            Input tensor of shape (batch_size, n_channels, n_times).
        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, n_channels, n_freqs, n_times).
        """

        signal = []
        for s in range(x.shape[1]):
            spectral = torch.stft(
                x[:, s, :],
                n_fft=self.n_fft,
                hop_length=self.n_fft // self.steps,
                win_length=self.n_fft,
                normalized=True,
                center=True,
                onesided=True,
                return_complex=True,
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

    def forward(self, X):
        """
        Forward pass.

        Parameters
        ----------
        X: Tensor
            Input tensor of shape (batch_size, n_channels, n_times).
        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        X = self.torch_stft(X)

        for conv in self.convs[:-1]:
            X = conv(X)
        emb = self.convs[-1](X).squeeze(-1).squeeze(-1)
        return self.final_layer(emb)
