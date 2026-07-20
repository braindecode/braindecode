"""BrainBERT: a self-supervised foundation model for intracranial signals.

Port of BrainBERT (Wang et al., ICLR 2023) into a braindecode-native model.
Upstream code and pretrained weights are released by the authors:

* paper: https://arxiv.org/abs/2302.14367
* code: https://github.com/czlwang/BrainBERT
* weights: released by the authors (Google Drive, see the upstream README)

BrainBERT learns representations of intracranial electrode data by masked
modelling of its **spectrogram**. The Transformer encoder and input encoding are
ported weight-for-weight from the upstream reference (bit-exact, see
``scripts/brainbert_parity_check``); the short-time Fourier transform front-end
is moved *inside* the model (a braindecode-native adaptation) and the pooling /
classification head is a braindecode-native addition. The official pretrained
checkpoint loads directly via
``BrainBERT.from_pretrained("braindecode/brainbert-pretrained")``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules.brainbert_modules import (
    _BrainBERTHead,
    _BrainBERTInputEmbedding,
    _SpecPredictionHead,
    _STFTSpectrogram,
)


class BrainBERT(EEGModuleMixin, nn.Module):
    r"""BrainBERT from Wang et al. (2023) [BrainBERT2023]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    BrainBERT is a self-supervised foundation model for intracranial neural
    signals (sEEG/iEEG). The raw signal is turned into a **spectrogram** by a
    short-time Fourier transform; each time frame is a token. A linear projection
    and a fixed sinusoidal positional encoding feed a stack of standard
    Transformer encoder layers. The model is pre-trained by masked-spectrogram
    modelling — reconstructing masked time/frequency patches — and the resulting
    per-frame representations are used for downstream decoding.

    Following the braindecode convention, the STFT is computed **inside**
    ``forward`` (via :class:`~braindecode.modules.brainbert_modules._STFTSpectrogram`)
    so the model keeps the standard ``(batch, n_chans, n_times)`` input
    signature, whereas the upstream reference consumes a pre-computed spectrogram.

    The released checkpoint expects signals sampled at **2048 Hz**,
    Laplacian-re-referenced, with ``nperseg=400``, ``noverlap=350`` and the first
    ``freq_cutoff=40`` frequency bins. The defaults below are a modest,
    ready-to-run configuration; the **released ("large") model** uses
    ``hidden_dim=768``, ``ffn_dim=3072``, ``n_heads=12`` and ``n_layers=6``
    (~43M parameters) — pass these to reproduce it.

    .. important::
       **Pre-trained weights available.** The official checkpoint is released by
       the authors and loads directly::

           model = BrainBERT.from_pretrained(
               "braindecode/brainbert-pretrained", n_outputs=2
           )

       It uses the "large" configuration above; ``n_chans`` and ``n_outputs`` may
       be changed freely, as frames are pooled and the classification head is
       task-specific.

    .. versionadded:: 1.7

    Parameters
    ----------
    hidden_dim : int, optional
        Transformer model width ``D``. Default 192. The released model uses 768.
    ffn_dim : int, optional
        Inner dimension of the Transformer feed-forward blocks. Default 384.
        The released model uses 3072.
    n_layers : int, optional
        Number of Transformer encoder layers. Default 2. Released model: 6.
    n_heads : int, optional
        Number of attention heads. Default 4. Released model: 12.
    nperseg : int, optional
        STFT window length in samples. Default 400.
    noverlap : int, optional
        STFT overlap in samples. Default 350 (hop of 50).
    freq_cutoff : int, optional
        Number of low-frequency STFT bins kept; the Transformer ``input_dim``.
        Default 40.
    stft_clip : int, optional
        Boundary frames trimmed from each end of the spectrogram. Default 5.
    activation : type[nn.Module], optional
        Transformer feed-forward activation. Default ``nn.GELU`` (as pretrained).
    drop_prob : float, optional
        Dropout probability. Default 0.1.

    References
    ----------
    .. [BrainBERT2023] Wang, C., Subramaniam, V., Yaari, A.U., Kreiman, G.,
       Katz, B., Cases, I. and Barbu, A., 2023. BrainBERT: Self-supervised
       representation learning for intracranial recordings. In International
       Conference on Learning Representations, ICLR.
       Code: https://github.com/czlwang/BrainBERT
    """

    def __init__(
        self,
        # --- BrainBERT hyper-parameters (modest defaults; "large" in docstring) ---
        hidden_dim: int = 192,
        ffn_dim: int = 384,
        n_layers: int = 2,
        n_heads: int = 4,
        nperseg: int = 400,
        noverlap: int = 350,
        freq_cutoff: int = 40,
        stft_clip: int = 5,
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.1,
        # --- braindecode mandatory signal parameters ---
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.freq_cutoff = freq_cutoff

        # braindecode-native: STFT computed inside forward (see module).
        self.spectrogram = _STFTSpectrogram(
            sfreq=self.sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            freq_cutoff=freq_cutoff,
            clip=stft_clip,
        )
        # At least one frame must survive the boundary trimming.
        self.seq_len = self.spectrogram.n_frames(self.n_times)
        if self.seq_len < 1:
            raise ValueError(
                f"n_times ({self.n_times}) is too short: it yields "
                f"{self.seq_len} spectrogram frames after trimming "
                f"{stft_clip} boundary frames on each side. Provide a longer "
                f"signal or reduce nperseg / stft_clip."
            )

        self.input_embedding = _BrainBERTInputEmbedding(
            input_dim=freq_cutoff, hidden_dim=hidden_dim, drop_prob=drop_prob
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            activation=activation(),
            dropout=drop_prob,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # kept for weight parity with the upstream pretraining checkpoint; the
        # classification path does not use it.
        self.spec_prediction_head = _SpecPredictionHead(hidden_dim, freq_cutoff)
        self.final_layer = _BrainBERTHead(hidden_dim, self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        self._n_outputs = n_outputs
        self.final_layer = _BrainBERTHead(self.hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Decode a batch of signals.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If ``True``, return the pooled encoder embedding instead of the
            class logits, as ``{"features": pooled, "cls_token": None}``
            (braindecode foundation-model convention). BrainBERT pools over
            channels and time frames and has no class token, hence
            ``cls_token`` is ``None``.

        Returns
        -------
        torch.Tensor or dict
            Class logits of shape ``(batch, n_outputs)``, or the feature dict
            ``{"features", "cls_token"}`` when ``return_features`` is set.
        """
        batch_size, n_chans, _ = x.shape

        # 1. spectrogram front-end (braindecode-native, computed here).
        spec = self.spectrogram(x)  # (batch, n_chans, n_frames, freq_cutoff)
        seq_len = spec.shape[2]
        spec = spec.reshape(batch_size * n_chans, seq_len, self.freq_cutoff)

        # 2. input encoding + Transformer over the sequence of frames.
        h = self.input_embedding(spec)
        z = self.transformer(h)  # (batch * n_chans, n_frames, hidden_dim)

        # 3. pool over channels and frames, then classify.
        z = z.reshape(batch_size, n_chans, seq_len, self.hidden_dim)
        pooled = z.mean(dim=(1, 2))
        if return_features:
            return {"features": pooled, "cls_token": None}
        return self.final_layer(pooled)
