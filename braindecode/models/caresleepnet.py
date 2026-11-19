# Authors: Jiquan Wang et al. (original implementation)
#
# License: MIT
# Adapted from https://github.com/wjq-learning/CareSleepNet (MIT).
"""CareSleepNet: a hybrid CNN-transformer sleep stager over EEG and EOG."""

from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import LearnedPositionalEncoding, TimmMlp


class CareSleepNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""CareSleepNet from Wang et al. (2024) [caresleepnet2024]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    A sequence-to-sequence sleep stager: it reads a sequence of ``n_epochs``
    consecutive sleep epochs of EEG and EOG and returns one prediction per
    epoch.

    - **Epoch encoder** (one for EEG, one for EOG). Four multi-scale dilated
      convolution blocks (three dilated convolutions and a 1x1 shortcut,
      summed and layer-normalized), each followed by max pooling; the last
      three are each followed by a post-norm transformer layer over time. The
      average over time gives one 512-dimensional vector per epoch.
    - **Cross-modal encoder.** EEG attends to EOG and EOG to EEG: in each
      pre-norm block, the queries come from one signal and the keys, values
      and residual from the other. The two outputs are summed.
    - **Sequence encoder.** A learned positional encoding and one pre-norm
      transformer layer over the epochs, then a linear classifier per epoch.

    The input is ``(batch, n_chans, n_times)`` with the epochs one after the
    other in time, ``n_times = n_epochs * epoch_samples``, or
    ``(batch, n_epochs, n_chans, epoch_samples)`` as the reference takes it.
    The last ``n_eog_chans`` channels are EOG, the others EEG. The output is
    ``(batch, n_outputs, n_epochs)``.

    Parameters
    ----------
    n_epochs : int, default=20
        Number of sleep epochs in one input sequence; ``n_times`` must be a
        multiple of it. The reference uses 20 epochs of 30 s at 100 Hz.
    n_eog_chans : int, default=2
        Number of EOG channels, at the end of the channel axis. The reference
        uses 6 EEG and 2 EOG channels on ISRUC.
    drop_prob : float, default=0.1
        Dropout rate of every dropout layer.
    activation : type[nn.Module], default=nn.GELU
        Activation of every convolution branch and transformer MLP.

    Notes
    -----
    Protocol of the reference code [caresleepnetcode]_: signals resampled to
    100 Hz (ISRUC also band-passed at 0.3-40 Hz), cut into 20-epoch
    sequences of 30 s, 5 classes (W, N1, N2, N3, REM); Adam (learning rate
    1e-4, cosine annealing over all steps), batch size 16, 60 epochs,
    gradient-norm clipping at 1, cross-entropy, 10-fold cross-validation.
    The reference hard-codes 6 EEG and 2 EOG channels and 20 epochs; this
    port makes them parameters. Every dropout rate of the reference is 0.1,
    the default ``drop_prob``.

    The epoch encoder's transformer layers are BERT layers (post-norm,
    ``eps=1e-12``, no dropout between the GELU and the second linear layer)
    with an all-ones attention mask, which is a no-op and is left out. No
    trained weights were released. The initialization follows the reference.

    `License <https://github.com/wjq-learning/CareSleepNet/blob/386e149f7eac8d93207972268f8bc82ea37361c1/LICENSE>`_

    References
    ----------
    .. [caresleepnet2024] Wang, J., Zhao, S., Jiang, H., Zhou, Y., Yu, Z.,
       Li, T., Li, S., & Pan, G. (2024). CareSleepNet: A Hybrid Deep Learning
       Network for Automatic Sleep Staging. IEEE Journal of Biomedical and
       Health Informatics, 28(12), 7392-7405.
       https://doi.org/10.1109/JBHI.2024.3426939
    .. [caresleepnetcode] Wang, J. CareSleepNet source code.
       https://github.com/wjq-learning/CareSleepNet/tree/386e149f7eac8d93207972268f8bc82ea37361c1
    """

    def __init__(
        self,
        # Signal-related parameters.
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters.
        n_epochs: int = 20,
        n_eog_chans: int = 2,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if not 0 < n_eog_chans < self.n_chans:
            raise ValueError(
                f"n_eog_chans must be between 1 and n_chans - 1 = {self.n_chans - 1}, "
                f"got {n_eog_chans}."
            )
        if self.n_times % n_epochs:
            raise ValueError(
                f"n_times ({self.n_times}) must be a multiple of n_epochs ({n_epochs})."
            )
        self.n_epochs = n_epochs
        self.n_eog_chans = n_eog_chans
        dim = 512

        self.eeg_encoder = _epoch_encoder(
            self.n_chans - n_eog_chans, drop_prob, activation
        )
        self.eog_encoder = _epoch_encoder(n_eog_chans, drop_prob, activation)
        self.eog_to_eeg = _CrossModalBlock(dim, 8, dim, drop_prob, activation)
        self.eeg_to_eog = _CrossModalBlock(dim, 8, dim, drop_prob, activation)
        self.positional_encoding = LearnedPositionalEncoding(n_epochs, dim, drop_prob)
        self.sequence_encoder = nn.TransformerEncoderLayer(
            dim,
            8,
            dim,
            drop_prob,
            activation=activation(),
            layer_norm_eps=1e-6,
            batch_first=True,
            norm_first=True,
        )
        self.sequence_norm = nn.LayerNorm(dim, eps=1e-6)
        self.final_layer = nn.Linear(dim, self.n_outputs)
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialization of the reference: BERT in the epoch encoders,
        torchvision's ViT in the cross-modal and sequence encoders."""
        for encoder in (self.eeg_encoder, self.eog_encoder):
            for layer in encoder:
                if isinstance(layer, nn.TransformerEncoderLayer):
                    for linear in (
                        layer.self_attn.out_proj,
                        layer.linear1,
                        layer.linear2,
                    ):
                        nn.init.normal_(linear.weight, std=0.02)
                        nn.init.zeros_(linear.bias)
                    nn.init.normal_(layer.self_attn.in_proj_weight, std=0.02)
        vit_linears = (
            self.eog_to_eeg.mlp.fc1,
            self.eog_to_eeg.mlp.fc2,
            self.eeg_to_eog.mlp.fc1,
            self.eeg_to_eog.mlp.fc2,
            self.sequence_encoder.linear1,
            self.sequence_encoder.linear2,
        )
        for linear in vit_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.normal_(linear.bias, std=1e-6)
        nn.init.normal_(self.positional_encoding.encoding, std=0.02)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the per-epoch classifier for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=old.weight.device, dtype=old.weight.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, n_chans, n_times)`` or ``(batch, n_epochs, n_chans,
        epoch_samples)`` to ``(batch, n_outputs, n_epochs)``."""
        if x.ndim == 3:
            x = x.reshape(x.shape[0], x.shape[1], self.n_epochs, -1).transpose(1, 2)
        batch, n_epochs, n_chans, n_samples = x.shape
        n_eeg = n_chans - self.n_eog_chans
        eeg = self.eeg_encoder(x[:, :, :n_eeg].reshape(-1, n_eeg, n_samples))
        eog = self.eog_encoder(x[:, :, n_eeg:].reshape(-1, self.n_eog_chans, n_samples))
        eeg = eeg.reshape(batch, n_epochs, -1)
        eog = eog.reshape(batch, n_epochs, -1)
        x = self.eog_to_eeg(eeg, eog) + self.eeg_to_eog(eog, eeg)
        x = self.sequence_encoder(self.positional_encoding(x))
        x = self.final_layer(self.sequence_norm(x))
        return x.transpose(1, 2)


def _epoch_encoder(
    in_chans: int, drop_prob: float, activation: type[nn.Module]
) -> nn.Sequential:
    """``(batch, in_chans, epoch_samples)`` to ``(batch, 512)``."""
    to_tokens = Rearrange("batch chans time -> batch time chans")
    to_channels = Rearrange("batch time chans -> batch chans time")
    return nn.Sequential(
        _MSDConv(in_chans, 64, 49, 12, 24, drop_prob, activation),
        nn.MaxPool1d(9, 2, 4),
        _MSDConv(64, 128, 9, 1, 4, drop_prob, activation),
        nn.MaxPool1d(2, 2),
        to_tokens,
        _bert_layer(128, drop_prob, activation),
        to_channels,
        _MSDConv(128, 256, 9, 1, 4, drop_prob, activation),
        nn.MaxPool1d(2, 2),
        to_tokens,
        _bert_layer(256, drop_prob, activation),
        to_channels,
        _MSDConv(256, 512, 9, 1, 4, drop_prob, activation),
        nn.MaxPool1d(2, 2),
        to_tokens,
        _bert_layer(512, drop_prob, activation),
        to_channels,
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
    )


def _bert_layer(
    dim: int, drop_prob: float, activation: type[nn.Module]
) -> nn.TransformerEncoderLayer:
    """BERT layer of the reference (post-norm, 8 heads, 4x MLP)."""
    layer = nn.TransformerEncoderLayer(
        dim,
        8,
        4 * dim,
        drop_prob,
        activation=activation(),
        layer_norm_eps=1e-12,
        batch_first=True,
    )
    layer.dropout = nn.Identity()  # BERT has no dropout after the GELU
    return layer


class _MSDConv(nn.Module):
    """Multi-scale dilated convolution: three dilated convolutions (dilation
    1, 2, 4) with batch norm and activation, plus a 1x1 shortcut with batch norm,
    summed, then dropout and layer norm over channels."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, 1, stride, bias=False),
            nn.BatchNorm1d(out_chans),
        )
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    in_chans,
                    out_chans,
                    kernel_size,
                    stride,
                    padding * dilation,
                    dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(out_chans),
                activation(),
            )
            for dilation in (1, 2, 4)
        )
        self.dropout = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(out_chans, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.shortcut(x)
        for branch in self.branches:
            out = out + branch(x)
        out = self.dropout(out)
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class _CrossModalBlock(nn.Module):
    """Pre-norm block whose queries come from the other signal (EEG or EOG).

    Unlike a transformer decoder block it has no self-attention, and the
    residual is the attended sequence ``x``, not the queries; no shared block
    does that.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.drop_x = nn.Dropout(drop_prob)
        self.drop_query = nn.Dropout(drop_prob)
        self.norm_x = nn.LayerNorm(dim, eps=1e-6)
        self.norm_query = nn.LayerNorm(dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=drop_prob, batch_first=True
        )
        self.drop_attention = nn.Dropout(drop_prob)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = TimmMlp(dim, mlp_dim, act_layer=activation, drop=drop_prob)
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        x = self.drop_x(x)
        query = self.norm_query(self.drop_query(query))
        x_normed = self.norm_x(x)
        attended, _ = self.attention(query, x_normed, x_normed, need_weights=False)
        x = x + self.drop_attention(attended)
        x = x + self.mlp(self.norm_mlp(x))
        return self.norm_out(x)
