from __future__ import annotations

import torch
import torch.nn as nn
from mne.utils import warn

from braindecode.models.base import EEGModuleMixin


class ContraWR(EEGModuleMixin, nn.Module):
    """Contrast with the World Representation ContraWR from Yang et al (2021) [Yang2021]_.

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
    drop_prob : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.

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
        n_chans=None,
        n_outputs=None,
        sfreq=None,
        emb_size: int = 256,
        res_channels: list[int] = [32, 64, 128],
        steps=20,
        activation: nn.Module = nn.ELU,
        drop_prob: float = 0.5,
        stride_res: int = 2,
        kernel_size_res: int = 3,
        padding_res: int = 1,
        # Another way to pass the EEG parameters
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
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

        if self.input_window_seconds < 1.0:
            warning_msg = (
                "The input window is less than 1 second, which may not be "
                "sufficient for the model to learn meaningful representations."
                "changing the `n_fft` to `n_times`."
            )
            warn(warning_msg, UserWarning)
            self.n_fft = self.n_times
        else:
            self.n_fft = int(self.sfreq)

        self.steps = steps

        res_channels = [self.n_chans] + res_channels + [emb_size]

        self.torch_stft = _STFTModule(
            n_fft=self.n_fft,
            hop_length=int(self.n_fft // self.steps),
        )

        self.convs = nn.ModuleList(
            [
                _ResBlock(
                    in_channels=res_channels[i],
                    out_channels=res_channels[i + 1],
                    stride=stride_res,
                    use_downsampling=True,
                    pooling=True,
                    drop_prob=drop_prob,
                    kernel_size=kernel_size_res,
                    padding=padding_res,
                    activation=activation,
                )
                for i in range(len(res_channels) - 1)
            ]
        )
        self.adaptative_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_layer = nn.Flatten()

        self.activation_layer = activation()
        self.final_layer = nn.Linear(emb_size, self.n_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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

        for conv in self.convs:
            X = conv.forward(X)

        emb = self.adaptative_pool(X)
        emb = self.flatten_layer(emb)
        emb = self.activation_layer(emb)

        return self.final_layer(emb)


class _ResBlock(nn.Module):
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
    drop_prob : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.

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
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class _STFTModule(nn.Module):
    """
    A PyTorch module that computes the Short-Time Fourier Transform (STFT)
    of an EEG batch tensor.

    Expects input of shape (batch_size, n_channels, n_times) and returns
    (batch_size, n_channels, n_freqs, n_times).
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        center: bool = True,
        onesided: bool = True,
        return_complex: bool = True,
        normalized: bool = True,
    ):
        """
        Parameters
        ----------
        n_fft : int
            Number of FFT points (window size).
        steps : int
            Number of hops per window (i.e. hop_length = n_fft // steps).
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.one_sided = onesided
        self.return_complex = return_complex
        self.normalized = normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.ones(self.n_fft, device=x.device)

        # x: (B, C, T)
        B, C, T = x.shape
        # flatten batch & channel into one dim
        x_flat = x.reshape(B * C, T)

        # compute stft on 2D tensor
        spec_flat = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            normalized=self.normalized,
            center=self.center,
            onesided=self.one_sided,
            return_complex=self.return_complex,
        )

        F, L = spec_flat.shape[-2], spec_flat.shape[-1]
        spec = spec_flat.view(B, C, F, L)

        return torch.abs(spec)
