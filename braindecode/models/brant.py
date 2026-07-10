"""Brant: a foundation model for intracranial (sEEG/iEEG) neural signals.

Port of Brant (Zhang et al., NeurIPS 2023) into a braindecode-native model.
Upstream code and pretrained weights are released under Apache-2.0:

* paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/535915d26859036410b0533804cee788-Abstract-Conference.html
* code: https://github.com/yzz673/Brant
* weights: https://huggingface.co/Daoze/Brant

The two Transformer encoders are ported weight-for-weight from the upstream
reference (bit-exact, see ``scripts/brant_parity_check``); the classification
head and channel/patch pooling are a braindecode-native adaptation. The official
pretrained checkpoint is hosted on the Hugging Face Hub and loads directly via
``Brant.from_pretrained("braindecode/brant-pretrained")``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules.brant_modules import (
    _BandPowerFeatures,
    _BrantHead,
    _BrantSpatialEncoder,
    _BrantTemporalEncoder,
)

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

    Brant is a foundation model for intracranial neural signals (sEEG/iEEG).
    The raw signal is cut into fixed-length **patches**; two Transformer
    encoders are then stacked:

    1. a **temporal encoder** over a sequence of consecutive patches of one
       channel, capturing long-range temporal dependency, and
    2. a **spatial encoder** over the patches sharing the same time index across
       channels, capturing spatial correlation.

    Time- and frequency-domain information are combined: alongside a linear
    projection of the raw patch and a learnable temporal positional encoding,
    a **frequency encoding** is added — the log spectral power of the patch in
    each of :data:`BRANT_FREQ_BANDS` (8 rhythmic bands) softmax-weights 8
    learnable per-band embeddings. Following the braindecode convention, that
    frequency computation is performed **inside** ``forward`` so the model keeps
    the standard ``(batch, n_chans, n_times)`` input signature.

    The upstream model operates on signals down-sampled to **250 Hz**. The
    defaults below are a modest, ready-to-run configuration; the **released
    pretrained model** (paper §3.2, ~500M parameters) uses ``patch_size=1500``
    (6 s at 250 Hz), ``embed_dim=2048``, ``ffn_dim=3072``, ``temporal_n_layers=12``,
    ``spatial_n_layers=5`` and ``n_heads=16`` — pass these to reproduce it.

    .. important::
       **Pre-trained weights available.** The official checkpoint (~508M
       parameters, pre-trained on ~1 TB of intracranial recordings) is hosted on
       the Hugging Face Hub under the Apache-2.0 license and loads directly::

           model = Brant.from_pretrained("braindecode/brant-pretrained", n_outputs=2)

       It uses the paper configuration above (``patch_size=1500``,
       ``n_times=22500``); ``n_chans`` and ``n_outputs`` may be changed freely,
       as channels are pooled and the classification head is task-specific. These
       weights are released for medical or research use only.

    .. versionadded:: 1.7

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
        # --- Brant hyper-parameters (modest defaults; paper config in docstring) ---
        patch_size: int = 250,
        embed_dim: int = 256,
        ffn_dim: int = 384,
        temporal_n_layers: int = 4,
        spatial_n_layers: int = 2,
        n_heads: int = 8,
        n_freq_bands: int = 8,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
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

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.temporal_n_layers = temporal_n_layers
        self.spatial_n_layers = spatial_n_layers
        self.n_heads = n_heads
        self.n_freq_bands = n_freq_bands
        self.drop_prob = drop_prob
        self._head_activation = activation

        if n_freq_bands != len(BRANT_FREQ_BANDS):
            raise ValueError(
                f"n_freq_bands ({n_freq_bands}) must equal "
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

        # braindecode-native: band-power computed inside forward (see module).
        self.band_power = _BandPowerFeatures(self.sfreq, BRANT_FREQ_BANDS)
        self.temporal_encoder = _BrantTemporalEncoder(
            patch_size=patch_size,
            d_model=embed_dim,
            seq_len=self.seq_len,
            n_bands=n_freq_bands,
            dim_feedforward=ffn_dim,
            n_layers=temporal_n_layers,
            n_heads=n_heads,
            drop_prob=drop_prob,
        )
        self.spatial_encoder = _BrantSpatialEncoder(
            d_model=embed_dim,
            out_dim=patch_size,
            dim_feedforward=ffn_dim,
            n_layers=spatial_n_layers,
            n_heads=n_heads,
            drop_prob=drop_prob,
        )
        self.final_layer = _BrantHead(embed_dim, self.n_outputs, activation)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        self._n_outputs = n_outputs
        self.final_layer = _BrantHead(self.embed_dim, n_outputs, self._head_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of signals.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_outputs)``.
        """
        batch_size, n_chans, _ = x.shape
        seq_len = self.seq_len
        d_model = self.embed_dim

        # 1. patch: drop the tail that does not fill a whole patch.
        x = x[..., : seq_len * self.patch_size]
        patches = x.reshape(batch_size, n_chans, seq_len, self.patch_size)

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
        return self.final_layer(pooled)
