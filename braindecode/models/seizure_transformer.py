"""SeizureTransformer for dense, time-step EEG event prediction.

The implementation follows the architecture released by Wu et al. in
``keruiwu/SeizureTransformer`` and keeps its parameter names so released
PyTorch state dictionaries can be loaded directly.
"""

# The architecture in this file is adapted from the MIT-licensed reference
# implementation at https://github.com/keruiwu/SeizureTransformer.
# Copyright (c) 2025 Kerui Wu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin


class SeizureTransformer(EEGModuleMixin, nn.Module, license="mit"):
    r"""Time-step seizure detection with a U-Net and Transformer context.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    The model combines a five-stage temporal convolutional encoder, a residual
    CNN stack, Transformer context at the lowest temporal resolution, and a
    skip-connected decoder that returns a logit for every input sample. The
    default widths and Transformer settings follow the released challenge
    model.

    Parameters
    ----------
    n_chans : int
        Number of input EEG channels. The challenge configuration uses 19.
    n_outputs : int
        Number of dense output channels. Use 1 for binary seizure detection.
    n_times : int
        Number of samples in each input segment. The challenge uses 60 seconds
        at 256 Hz (15360 samples). The model returns the same number of samples.
    sfreq : float
        Sampling frequency in Hz.
    input_window_seconds : float or None
        Window duration. If ``None``, it is inferred from ``n_times / sfreq``.
    chs_info : list of dict or None
        Optional channel metadata used by :class:`EEGModuleMixin`.
    dim_feedforward : int
        Transformer feed-forward width. The released model uses 2048.
    num_layers : int
        Number of Transformer encoder layers. The released model uses 8.
    num_heads : int
        Number of Transformer attention heads. The released model uses 4.
    drop_prob : float
        Spatial dropout probability in the residual CNN stack. The released
        model uses 0.1.
    activation : type[nn.Module]
        Activation used in the residual CNN stack. The released model uses
        ``nn.ReLU``.
    encoder_filters : sequence of int
        Widths of the five U-Net encoder stages. The released model uses
        ``(32, 64, 128, 256, 512)``. Smaller widths are useful for tests and
        resource-constrained fine-tuning.
    transformer_dropout : float
        Dropout probability in the Transformer encoder. The reference uses
        PyTorch's default of 0.1.
    max_pos_len : int
        Maximum sequence length supported by the sinusoidal positional table.

    Notes
    -----
    The forward method returns logits with shape ``(batch, n_outputs, n_times)``.
    For binary detection, apply ``torch.sigmoid`` to obtain the reference
    implementation's per-sample probabilities. The output is not converted to
    discrete events; thresholding and event-level post-processing depend on the
    evaluation protocol.

    References
    ----------
    .. [WuEtAl2025SeizureTransformer] Wu, K., Zhao, Z. & Yener, B. Large EEG-U-Transformer
       for Time-Step Level Detection Without Pre-Training. arXiv:2504.00336, 2025.
       https://arxiv.org/abs/2504.00336
    """

    def __init__(
        self,
        n_chans: int = 19,
        n_outputs: int = 1,
        n_times: int = 15360,
        sfreq: float = 256.0,
        input_window_seconds: float | None = None,
        chs_info=None,
        dim_feedforward: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
        encoder_filters=(32, 64, 128, 256, 512),
        transformer_dropout: float = 0.1,
        max_pos_len: int = 6000,
    ):
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
            chs_info=chs_info,
        )
        del n_chans, n_outputs, n_times, sfreq, input_window_seconds, chs_info

        encoder_filters = tuple(encoder_filters)
        if len(encoder_filters) != 5:
            raise ValueError("encoder_filters must contain exactly five widths.")
        if num_heads <= 0 or encoder_filters[-1] % num_heads:
            raise ValueError(
                "The final encoder width must be divisible by num_heads; got "
                f"{encoder_filters[-1]} and {num_heads}."
            )
        if max_pos_len <= 0:
            raise ValueError(f"max_pos_len must be positive; got {max_pos_len}.")

        self.in_channels = self.n_chans
        self.in_samples = self.n_times
        self.filters = list(encoder_filters)
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        self.encoder = _Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
        )
        self.res_cnn_stack = _ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=drop_prob,
            activation=activation,
        )

        transformer_dim = self.filters[-1]
        self.position_encoding = _PositionalEncoding(
            d_model=transformer_dim,
            dropout=transformer_dropout,
            max_len=max_pos_len,
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
        )
        self.transformer_encoder_layer = transformer_layer
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )

        self.decoder_d = _Decoder(
            input_channels=transformer_dim,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=self.in_samples,
        )
        self.conv_d = nn.Conv1d(
            in_channels=self.filters[0],
            out_channels=self.n_outputs,
            kernel_size=11,
            padding=5,
        )
        # Keep the public model-zoo convention without changing reference
        # checkpoint keys (the output convolution remains named ``conv_d``).
        self.final_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dense per-sample logits for ``(batch, channels, time)`` EEG."""
        expected_shape = (self.n_chans, self.n_times)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                "SeizureTransformer expects input shaped "
                f"(batch, {self.n_chans}, {self.n_times}); got {tuple(x.shape)}."
            )

        x, skips = self.encoder(x)
        res_x = self.res_cnn_stack(x)

        x = res_x.permute(2, 0, 1)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0) + res_x

        x = self.decoder_d(x, skips)
        return self.final_layer(self.conv_d(x))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) > self.pe.size(0):
            raise ValueError(
                f"Sequence length {x.size(0)} exceeds positional encoding length "
                f"{self.pe.size(0)}. Increase max_pos_len."
            )
        return self.dropout(x + self.pe[: x.size(0)])


class _Encoder(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super().__init__()
        convs, pools, elus, paddings = [], [], [], []
        for in_ch, out_ch, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes[: len(filters)]
        ):
            convs.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            )
            padding = in_samples % 2
            paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            elus.append(nn.ELU(inplace=True))
            in_samples = (in_samples + padding) // 2
        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.elus = nn.ModuleList(elus)
        self.paddings = paddings

    def forward(self, x):
        skips = []
        for conv, pool, padding, elu in zip(
            self.convs, self.pools, self.paddings, self.elus
        ):
            x = elu(conv(x))
            skips.append(x)
            if padding:
                x = F.pad(x, (0, padding), value=torch.finfo(x.dtype).min)
            x = pool(x)
        return x, skips


class _Decoder(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, out_samples):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding:
                crops.append(len(filters) - 1 - i)
        self.crops = crops

        convs, elus = [], []
        for in_ch, out_ch, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            )
            elus.append(nn.ELU(inplace=True))
        self.convs = nn.ModuleList(convs)
        self.elus = nn.ModuleList(elus)

    def forward(self, x, skip_connections):
        for i, (conv, elu) in enumerate(zip(self.convs, self.elus)):
            x = self.upsample(x)
            if i in self.crops:
                x = x[:, :, :-1]
            x = elu(conv(x))
            if i < len(skip_connections):
                x = x + skip_connections[-(i + 1)]
        return x


class _SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        return self.dropout(x.unsqueeze(-1)).squeeze(-1)


class _ResCNNBlock(nn.Module):
    def __init__(self, filters, kernel_size, drop_rate, activation):
        super().__init__()
        self.manual_padding = kernel_size == 2
        padding = 1 if kernel_size == 3 else 0
        self.dropout = _SpatialDropout1d(drop_rate)
        self.activation = activation()
        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding=padding)

    def forward(self, x):
        y = self.dropout(self.activation(self.norm1(x)))
        if self.manual_padding:
            y = F.pad(y, (0, 1))
        y = self.conv1(y)
        y = self.dropout(self.activation(self.norm2(y)))
        if self.manual_padding:
            y = F.pad(y, (0, 1))
        return x + self.conv2(y)


class _ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate, activation):
        super().__init__()
        self.members = nn.ModuleList(
            [
                _ResCNNBlock(filters, kernel, drop_rate, activation)
                for kernel in kernel_sizes
            ]
        )

    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x
