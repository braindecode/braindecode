"""BrainBERT: a self-supervised foundation model for intracranial signals.

Port of BrainBERT (Wang et al., ICLR 2023) into a braindecode-native model.
Upstream code and pretrained weights are released by the authors:

* paper: https://arxiv.org/abs/2302.14367
* code: https://github.com/czlwang/BrainBERT
* weights: released by the authors (Google Drive, see the upstream README)

BrainBERT learns representations of intracranial electrode data by masked
modelling of its **spectrogram**. The Transformer encoder and input encoding are
ported weight-for-weight from the upstream reference (bit-exact; the gate lives
in ``test/unit_tests/models/test_brainbert.py`` and runs against a clone of the
upstream repository when ``BRAINBERT_SRC`` is set). The short-time Fourier
transform front-end is moved *inside* the model, a braindecode-native
adaptation. The official pretrained checkpoint loads directly via
``BrainBERT.from_pretrained("braindecode/brainbert-pretrained")``.

Licensing: the upstream repository ships **no LICENSE file**, so the weights are
re-hosted with their licence declared as ``unknown`` rather than assumed.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules.brainbert_modules import (
    _BrainBERTHead,
    _BrainBERTInputEmbedding,
    _SpecPredictionHead,
    _STFTSpectrogram,
)


def _as_transformer_activation(
    activation: str
    | type[nn.Module]
    | nn.Module
    | Callable[[torch.Tensor], torch.Tensor],
) -> str | Callable[[torch.Tensor], torch.Tensor]:
    """Normalise ``activation`` into something ``TransformerEncoderLayer`` takes.

    braindecode spells this parameter ``type[nn.Module]`` (see BIOT, LaBraM,
    CBraMod), so that stays the documented default. But
    :class:`~torch.nn.TransformerEncoderLayer` itself accepts a string or a
    plain callable, and instantiating those with ``activation()`` raises. The
    four accepted forms are therefore folded here: a class is instantiated, and
    a string, a module instance or a bare callable is passed straight through.
    """
    if isinstance(activation, str):
        return activation
    if isinstance(activation, type):
        if not issubclass(activation, nn.Module):
            raise ValueError(
                f"activation class must subclass nn.Module, got {activation!r}."
            )
        return activation()
    if callable(activation):
        return activation
    raise ValueError(
        "activation must be a string, an nn.Module subclass or instance, or a "
        f"callable; got {type(activation).__name__}."
    )


class BrainBERT(EEGModuleMixin, nn.Module, license="unknown"):
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
    ``idx_freq_cutoff=40`` frequency bins. The defaults below are a modest,
    ready-to-run configuration; the **released ("large") model** uses
    ``hidden_dim=768``, ``ffn_dim=3072``, ``n_heads=12`` and ``n_layers=6``
    (~43M parameters) — pass these to reproduce it.

    **Pooling follows the published downstream protocol.** Upstream feeds one
    electrode at a time and averages the ``pool_n_frames=10`` encoder outputs
    centred on the window
    (``preprocessors/spec_pretrained.py``: ``outputs[:, middle-5:middle+5].mean``),
    then applies a bare linear probe; averaging *every* frame is present in that
    file only as a commented-out alternative. This port keeps the centre-frame
    average and adds a mean over channels, which is the identity for
    ``n_chans=1``: a single-channel BrainBERT therefore reproduces the upstream
    feature exactly, while multi-channel input remains supported as the
    braindecode-native generalisation. Pass ``pool_n_frames=None`` to average
    all frames instead.

    .. important::
       **Pre-trained weights available.** The official checkpoint is released by
       the authors and loads directly::

           model = BrainBERT.from_pretrained(
               "braindecode/brainbert-pretrained", n_outputs=2
           )

       It uses the "large" configuration above; ``n_chans`` and ``n_outputs`` may
       be changed freely, as frames are pooled and the classification head is
       task-specific.

       The upstream repository ships no LICENSE file, so the re-hosted weights
       are declared ``unknown`` rather than assumed permissive; the model card
       records the SHA-256 and retrieval date of the source archive.

    .. seealso::
       :ref:`brainbert-ieeg-features` works through the three settings that
       silently change what the pretrained encoder sees — the sampling rate, the
       normalisation order and the window length — on real intracranial data,
       and reproduces the frozen-encoder linear probe of the paper.

    .. versionadded:: 1.8

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
    idx_freq_cutoff : int, optional
        Number of low-frequency STFT bins kept; the Transformer ``input_dim``.
        This is a **bin index**, not a frequency in Hz: the default 40 bins
        reach about 200 Hz at ``sfreq=2048`` with ``nperseg=400``. Default 40.
    stft_clip : int, optional
        Boundary frames trimmed from each end of the spectrogram. Default 10,
        as in the upstream ``preprocessors/stft.py`` used with the released
        checkpoint (the demo notebook uses 5; see ``stft_zscore_before_clip``).
    stft_zscore_before_clip : bool, optional
        Whether the spectrogram is z-scored before the boundary frames are
        trimmed. Default ``True``, which together with ``stft_clip=10``
        reproduces the recipe behind the published numbers. Set to ``False``
        with ``stft_clip=5`` to reproduce the upstream demo notebook instead;
        the two recipes are close (correlation 0.999 on filtered noise) but not
        equal, and they yield sequences of different length.
    pool_n_frames : int or None, optional
        Number of encoder frames, centred on the window, averaged into the
        pooled representation. Default 10, as upstream. ``None`` averages all
        frames.
    activation : type[nn.Module] or str or nn.Module or callable, optional
        Transformer feed-forward activation. Default ``nn.GELU`` (as
        pretrained). A class is instantiated; a string (``"gelu"``,
        ``"relu"``), a ready-made module or a plain callable such as
        :func:`torch.nn.functional.gelu` is forwarded to
        :class:`~torch.nn.TransformerEncoderLayer` as-is.
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
        idx_freq_cutoff: int = 40,
        stft_clip: int = 10,
        stft_zscore_before_clip: bool = True,
        pool_n_frames: int | None = 10,
        activation: str
        | type[nn.Module]
        | nn.Module
        | Callable[[torch.Tensor], torch.Tensor] = nn.GELU,
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
        self.idx_freq_cutoff = idx_freq_cutoff
        self.pool_n_frames = pool_n_frames

        # braindecode-native: STFT computed inside forward (see module).
        self.spectrogram = _STFTSpectrogram(
            sfreq=self.sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            idx_freq_cutoff=idx_freq_cutoff,
            clip=stft_clip,
            zscore_before_clip=stft_zscore_before_clip,
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
        # Refuse rather than silently fall back to a different pooling: the
        # number of frames averaged is part of the published protocol.
        if pool_n_frames is not None and self.seq_len < pool_n_frames:
            raise ValueError(
                f"n_times ({self.n_times}) yields only {self.seq_len} "
                f"spectrogram frames, fewer than the pool_n_frames="
                f"{pool_n_frames} centre frames pooled by the upstream "
                f"protocol. Provide a longer signal, or pass a smaller "
                f"pool_n_frames (None averages all frames)."
            )

        self.input_embedding = _BrainBERTInputEmbedding(
            input_dim=idx_freq_cutoff,
            hidden_dim=hidden_dim,
            drop_prob=drop_prob,
            max_len=max(5000, self.seq_len),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            activation=_as_transformer_activation(activation),
            dropout=drop_prob,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # kept for weight parity with the upstream pretraining checkpoint; the
        # classification path does not use it.
        self.spec_prediction_head = _SpecPredictionHead(hidden_dim, idx_freq_cutoff)
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
            (braindecode foundation-model convention). BrainBERT pools the
            centre frames and then the channels, and has no class token, hence
            ``cls_token`` is ``None``.

        Returns
        -------
        torch.Tensor or dict
            Class logits of shape ``(batch, n_outputs)``, or the feature dict
            ``{"features", "cls_token"}`` when ``return_features`` is set.
        """
        batch_size, n_chans, _ = x.shape

        # 1. spectrogram front-end (braindecode-native, computed here).
        spec = self.spectrogram(x)  # (batch, n_chans, n_frames, idx_freq_cutoff)
        seq_len = spec.shape[2]
        spec = spec.reshape(batch_size * n_chans, seq_len, self.idx_freq_cutoff)

        # 2. input encoding + Transformer over the sequence of frames.
        h = self.input_embedding(spec)
        z = self.transformer(h)  # (batch * n_chans, n_frames, hidden_dim)

        # 3. pool. Upstream averages the pool_n_frames encoder outputs centred
        #    on the window (spec_pretrained.py), not every frame; the mean over
        #    channels after it is the braindecode-native generalisation and is
        #    the identity when n_chans == 1.
        z = z.reshape(batch_size, n_chans, seq_len, self.hidden_dim)
        if self.pool_n_frames is not None:
            middle = seq_len // 2
            half = self.pool_n_frames // 2
            z = z[:, :, middle - half : middle - half + self.pool_n_frames, :]
        pooled = z.mean(dim=(1, 2))
        if return_features:
            return {"features": pooled, "cls_token": None}
        return self.final_layer(pooled)
