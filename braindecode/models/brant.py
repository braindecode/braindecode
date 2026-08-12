# Authors: Daoze Zhang <zhangdz@zju.edu.cn>
#          Adam Mounir <am91ris@gmail.com> (braindecode adaptation)
#
# License: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import PatchTokenizer

# Standard rhythmic-activity bands (Hz) used by Brant's frequency encoding,
# from the paper (§ Frequency encoding): theta, alpha, beta, gamma1-5.
BRANT_FREQ_BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 8.0),  # theta
    (8.0, 13.0),  # alpha
    (13.0, 30.0),  # beta
    (30.0, 50.0),  # gamma1
    (50.0, 70.0),  # gamma2
    (70.0, 90.0),  # gamma3
    (90.0, 110.0),  # gamma4
    (110.0, 128.0),  # gamma5
)


class Brant(EEGModuleMixin, nn.Module):
    r"""Brant from Zhang et al. (2023) [Brant2023]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. rubric:: Architecture Overview

    Brant models intracranial neural signals (sEEG/iEEG) in four stages
    [Brant2023]_:

    1. Split every channel into non-overlapping temporal patches.
    2. Combine a linear patch projection with learned temporal positions and a
       spectral-power embedding.
    3. Apply a temporal Transformer within each channel, followed by a spatial
       Transformer across channels at each patch index.
    4. Mean-pool the channel-patch representation and classify it with a
       Braindecode downstream head.

    .. rubric:: Macro Components

    ``Brant.patch_tokenizer``
        **Operations.** Crops an incomplete tail and reshapes the signal into
        non-overlapping patches of ``patch_size`` samples.

        **Role.** Preserves the raw samples consumed by both the linear patch
        projection and the spectral-power calculation.

    ``Brant.band_power`` and ``Brant.temporal_encoder``
        **Operations.** A periodogram is summed over the eight rhythmic bands in
        :data:`BRANT_FREQ_BANDS`. Their log powers softmax-weight learned band
        embeddings, which are added to projected patches and temporal positions
        before self-attention within each channel.

        **Role.** Fuse time- and frequency-domain information while capturing
        long-range dependencies between consecutive patches.

    ``Brant.spatial_encoder``
        **Operations.** Self-attention is applied across all channels sharing a
        patch index.

        **Role.** Capture spatial correlations without a fixed channel
        vocabulary or channel-specific parameters.

    ``Brant.final_layer``
        **Operations.** Mean-pool the encoded channel-patch grid and apply a
        three-layer MLP.

        **Role.** Adapt the upstream encoder to Braindecode classification. This
        pooling and head are not part of the masked-reconstruction pretraining
        objective.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** learned patch positions and the temporal Transformer preserve
      patch order and model long-range activity within each channel.
    - **Spatial:** the spatial Transformer attends across the channels at every
      temporal patch index.
    - **Spectral:** log power in theta, alpha, beta, and five gamma bands weights
      eight learned frequency embeddings.

    .. rubric:: Additional Mechanisms

    The upstream reconstruction projection is retained in
    ``Brant.spatial_encoder`` so converted checkpoints keep a one-to-one key
    layout, although classification does not consume its output. Brant has no
    class token, so ``return_features=True`` returns ``cls_token=None``. The
    learned temporal positions require the runtime signal length to equal the
    configured ``n_times``; channel count remains parameter-agnostic.

    The upstream model operates on signals down-sampled to **250 Hz**. The
    defaults are a modest, ready-to-run configuration. The converted checkpoint
    uses the paper architecture: ``patch_size=1500`` (6 s), ``embed_dim=2048``,
    ``ffn_dim=3072``, ``temporal_n_layers=12``, ``spatial_n_layers=5``,
    ``n_heads=16``, and ``n_times=22500`` (15 patches, 90 s).

    .. important::
       **Pre-trained weights available.** A converted upstream encoder
       checkpoint (~508M parameters, pre-trained on ~1 TB of intracranial
       recordings) is hosted on the Hugging Face Hub under the Apache-2.0
       license and loads directly::

           model = Brant.from_pretrained("braindecode/brant-pretrained", n_outputs=4)

       The checkpoint stores a two-output placeholder head. Requesting a
       different ``n_outputs`` rebuilds it automatically; for a new binary task,
       call ``model.reset_head(2)`` explicitly. ``n_chans`` may be changed because
       channels are pooled, while ``n_times=22500`` must be kept for the learned
       temporal positions. These weights are intended for medical or research
       use, following the upstream authors' release notice.

    .. versionadded:: 1.8

    Parameters
    ----------
    patch_size : int, optional
        Number of time samples per patch fed to the encoders. Default 250
        (1 s at 250 Hz). The pretrained model uses 1500 (see above).
    embed_dim : int, optional
        Model width ``D`` (patch embedding size). Default 256.
    ffn_dim : int, optional
        Inner dimension of the Transformer feed-forward blocks. Default 384.
    temporal_n_layers : int, optional
        Number of layers in the temporal Transformer encoder. Default 4.
    spatial_n_layers : int, optional
        Number of layers in the spatial Transformer encoder. Default 2.
    n_heads : int, optional
        Number of attention heads in both encoders. Default 8.
    n_freq_bands : int, optional
        Number of frequency bands used by the frequency encoding. Default 8
        (must match ``len(BRANT_FREQ_BANDS)``).
    drop_prob : float, optional
        Dropout probability. Default 0.1.
    activation : type[nn.Module], optional
        Activation used in the classification head. Default ``nn.ReLU``.

    References
    ----------
    .. [Brant2023] Zhang, D., Yuan, Z., Yang, Y., Chen, J., Wang, J. and Li, Y.,
       2023. Brant: Foundation Model for Intracranial Neural Signal. In
       Thirty-seventh Conference on Neural Information Processing Systems,
       NeurIPS. Code: https://github.com/yzz673/Brant (Apache-2.0).
    """

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # model-specific parameters
        patch_size: int = 250,
        embed_dim: int = 256,
        ffn_dim: int = 384,
        temporal_n_layers: int = 4,
        spatial_n_layers: int = 2,
        n_heads: int = 8,
        n_freq_bands: int = 8,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.temporal_n_layers = temporal_n_layers
        self.spatial_n_layers = spatial_n_layers
        self.n_heads = n_heads
        self.n_freq_bands = n_freq_bands
        self.drop_prob = drop_prob
        self._head_activation = activation

        if self.n_freq_bands != len(BRANT_FREQ_BANDS):
            raise ValueError(
                f"n_freq_bands ({self.n_freq_bands}) must equal "
                f"len(BRANT_FREQ_BANDS) ({len(BRANT_FREQ_BANDS)})."
            )
        # Number of patches per channel, fixed by the input length. The learnable
        # temporal positional encoding is sized to it, hence n_times is required.
        self.seq_len = self.n_times // self.patch_size
        if self.seq_len < 1:
            raise ValueError(
                f"n_times ({self.n_times}) must be >= patch_size "
                f"({self.patch_size}) to form at least one patch."
            )

        # Shared non-overlapping patching (same tokenizer as the other
        # transformer foundation models); non-learnable, so it is a pure reshape
        # that keeps the raw samples the band-power and the temporal input
        # embedding both consume, and adds no parameters.
        self.patch_tokenizer = PatchTokenizer(
            patch_size=self.patch_size,
            n_times=self.n_times,
            learnable=False,
            on_non_divisible="crop",
        )
        # braindecode-native: band-power computed inside forward (see module).
        self.band_power = _BandPowerFeatures(self.sfreq, BRANT_FREQ_BANDS)
        self.temporal_encoder = _BrantTemporalEncoder(
            patch_size=self.patch_size,
            d_model=self.embed_dim,
            seq_len=self.seq_len,
            n_bands=self.n_freq_bands,
            dim_feedforward=self.ffn_dim,
            n_layers=self.temporal_n_layers,
            n_heads=self.n_heads,
            drop_prob=self.drop_prob,
        )
        self.spatial_encoder = _BrantSpatialEncoder(
            d_model=self.embed_dim,
            out_dim=self.patch_size,
            dim_feedforward=self.ffn_dim,
            n_layers=self.spatial_n_layers,
            n_heads=self.n_heads,
            drop_prob=self.drop_prob,
        )
        self.final_layer = _BrantHead(self.embed_dim, self.n_outputs, activation)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        old_param = next(self.final_layer.parameters())
        self.final_layer = _BrantHead(
            self.embed_dim, n_outputs, self._head_activation
        ).to(device=old_param.device, dtype=old_param.dtype)
        self._n_outputs = n_outputs
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None:
            hub_config["n_outputs"] = n_outputs

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Decode a batch of signals.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If ``True``, return the pooled encoder embedding instead of the
            class logits, as ``{"features": pooled, "cls_token": None}``
            (braindecode foundation-model convention). Brant pools over channels
            and patches and has no class token, hence ``cls_token`` is ``None``.

        Returns
        -------
        torch.Tensor or dict
            Class logits of shape ``(batch, n_outputs)``, or the feature
            dict ``{"features", "cls_token"}`` when ``return_features`` is set.
        """
        if x.shape[-1] != self.n_times:
            raise ValueError(
                f"Brant was configured for {self.n_times} time samples, "
                f"but received {x.shape[-1]}."
            )

        batch_size, n_chans, _ = x.shape
        seq_len = self.seq_len
        d_model = self.embed_dim

        # 1. patch: split into non-overlapping patches (shared PatchTokenizer);
        #    trailing samples that do not fill a whole patch are cropped.
        patches = self.patch_tokenizer(x)

        # 2. log band-power features (computed here, not fed in as upstream).
        power = self.band_power(patches)

        # 3. temporal encoder over the seq_len consecutive patches, per channel.
        time_z = self.temporal_encoder(patches, power)
        time_z = time_z.reshape(batch_size, n_chans, seq_len, d_model)
        time_z = time_z.transpose(1, 2).reshape(batch_size * seq_len, n_chans, d_model)

        # 4. spatial encoder over the n_chans channels at each time index.
        ch_z, _ = self.spatial_encoder(time_z)
        emb = ch_z.reshape(batch_size, seq_len, n_chans, d_model).transpose(1, 2)

        # 5. pool over channels and patches, then classify.
        pooled = emb.mean(dim=(1, 2))
        if return_features:
            return {"features": pooled, "cls_token": None}
        return self.final_layer(pooled)


class _BandPowerFeatures(nn.Module):
    """Per-patch log spectral power in a set of frequency bands.

    This is the in-model counterpart of the upstream ``compute_power`` routine:
    a SciPy-compatible periodogram followed by a log-sum within each band.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input signal, in Hz.
    bands : tuple of (float, float)
        Frequency band edges ``(low, high)`` in Hz. A frequency ``f`` belongs to
        a band when ``low < f <= high``.
    """

    def __init__(self, sfreq: float, bands: tuple[tuple[float, float], ...]):
        super().__init__()
        self.sfreq = float(sfreq)
        self.bands = tuple(bands)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute log band-power of every patch.

        Parameters
        ----------
        patches : torch.Tensor
            Shape ``(batch, n_chans, seq_len, patch_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_chans, seq_len, n_bands)``.
        """
        output_dtype = patches.dtype
        # CPU FFT does not accept reduced precision, while CUDA float16 FFT is
        # restricted to power-of-two lengths (the released patch size is 1500).
        if output_dtype in (torch.float16, torch.bfloat16):
            patches = patches.float()

        n = patches.shape[-1]
        # scipy periodogram default: detrend='constant' (remove the mean).
        x = patches - patches.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(x, dim=-1)
        # one-sided power spectral density, density scaling (boxcar window).
        psd = spectrum.abs().pow(2) / (self.sfreq * n)
        psd[..., 1:] = psd[..., 1:] * 2
        if n % 2 == 0:  # do not double the Nyquist bin
            psd[..., -1] = psd[..., -1] / 2
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.sfreq, device=patches.device)

        out = []
        for low, high in self.bands:
            mask = (freqs > low) & (freqs <= high)
            band = psd[..., mask].sum(dim=-1)
            out.append(torch.log10(band + 1.0))
        return torch.stack(out, dim=-1).to(dtype=output_dtype)


class _BrantInputEmbedding(nn.Module):
    """Input encoding of Brant: linear patch projection + frequency + position."""

    def __init__(self, patch_size: int, d_model: int, seq_len: int, n_bands: int):
        super().__init__()
        self.band_encoding = nn.Parameter(torch.randn(n_bands, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        self.proj = nn.Sequential(nn.Linear(patch_size, d_model))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        """Embed raw patches together with frequency and position information."""
        batch_size, n_chans, seq_len, patch_size = data.shape
        power = self.softmax(power)
        power_emb = torch.einsum("hijk, kl->hijl", power, self.band_encoding)

        data = data.reshape(batch_size * n_chans, seq_len, patch_size)
        input_emb = self.proj(data)
        input_emb = input_emb + power_emb.reshape(batch_size * n_chans, seq_len, -1)
        return input_emb + self.positional_encoding


class _BrantTemporalEncoder(nn.Module):
    """Temporal Transformer encoder (upstream ``TimeEncoder``)."""

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        seq_len: int,
        n_bands: int,
        dim_feedforward: int,
        n_layers: int,
        n_heads: int,
        drop_prob: float,
    ):
        super().__init__()
        self.input_embedding = _BrantInputEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            seq_len=seq_len,
            n_bands=n_bands,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            batch_first=True,
        )
        self.trans_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, data: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        return self.trans_enc(self.input_embedding(data, power))


class _BrantSpatialEncoder(nn.Module):
    """Spatial Transformer encoder (upstream ``ChannelEncoder``)."""

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        dim_feedforward: int,
        n_layers: int,
        n_heads: int,
        drop_prob: float,
    ):
        super().__init__()
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            batch_first=True,
        )
        self.trans_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, time_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ch_z = self.trans_enc(time_z)
        return ch_z, self.proj_out(ch_z)


class _BrantHead(nn.Module):
    """Braindecode downstream classification head for Brant."""

    def __init__(
        self, in_dim: int, out_dim: int, activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            activation(),
            nn.Linear(in_dim // 2, in_dim // 4),
            activation(),
            nn.Linear(in_dim // 4, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)
