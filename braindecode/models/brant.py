"""Brant: a foundation model for intracranial (sEEG/iEEG) neural signals.

Port of Brant (Zhang et al., NeurIPS 2023) into a braindecode-native model.
Upstream code and pretrained weights are released under Apache-2.0:

* paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/535915d26859036410b0533804cee788-Abstract-Conference.html
* code: https://github.com/yzz673/Brant
* weights: https://huggingface.co/Daoze/Brant

This module is a work-in-progress skeleton (see the tracking issue
braindecode/braindecode#1097). The public contract (class name, mandatory
parameters, ``forward`` / ``reset_head`` signatures) is fixed here; the faithful
architecture and the numerical-parity check against the upstream checkpoint land
in follow-up commits on this branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin

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

    :bdg-danger:`Foundation Model`

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

    The upstream model operates on signals down-sampled to **250 Hz**; the
    default ``patch_size`` (1500 samples) corresponds to the 6 s patches used in
    the paper. Encoder sizes below are the paper's configuration (§3.2).

    .. important::
       **Pre-trained weights available.** The upstream checkpoint (>500M
       parameters, pre-trained on ~1 TB of 1000 Hz sEEG) is published on the
       Hugging Face Hub at ``Daoze/Brant`` under the Apache-2.0 license.
       Loading it into this port is tracked in braindecode/braindecode#1097.

    .. versionadded:: 1.7

    Parameters
    ----------
    patch_size : int, optional
        Number of time samples per patch fed to the encoders. Default 1500
        (~6 s at 250 Hz), matching the upstream patching.
    embed_dim : int, optional
        Model width ``D`` (patch embedding size). Default 2048 (paper §3.2).
    ffn_dim : int, optional
        Inner dimension of the Transformer feed-forward blocks. Default 3072.
    temporal_n_layers : int, optional
        Number of layers in the temporal Transformer encoder. Default 12.
    spatial_n_layers : int, optional
        Number of layers in the spatial Transformer encoder. Default 5.
    n_heads : int, optional
        Number of attention heads in both encoders. Default 16.
    n_freq_bands : int, optional
        Number of frequency bands used by the frequency encoding. Default 8
        (must match ``len(BRANT_FREQ_BANDS)``).
    drop_prob : float, optional
        Dropout probability. Default 0.1.

    References
    ----------
    .. [Brant2023] Zhang, D., Yuan, Z., Yang, Y., Chen, J., Wang, J. and Li, Y.,
       2023. Brant: Foundation Model for Intracranial Neural Signal. In
       Thirty-seventh Conference on Neural Information Processing Systems,
       NeurIPS. Code: https://github.com/yzz673/Brant (Apache-2.0).
    """

    def __init__(
        self,
        # --- Brant hyper-parameters (paper §3.2; cross-check with Brant_src) ---
        patch_size: int = 1500,
        embed_dim: int = 2048,
        ffn_dim: int = 3072,
        temporal_n_layers: int = 12,
        spatial_n_layers: int = 5,
        n_heads: int = 16,
        n_freq_bands: int = 8,
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

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.temporal_n_layers = temporal_n_layers
        self.spatial_n_layers = spatial_n_layers
        self.n_heads = n_heads
        self.n_freq_bands = n_freq_bands
        self.drop_prob = drop_prob

        # ------------------------------------------------------------------ #
        # Architecture scaffold. Each submodule is a placeholder to be replaced
        # by a faithful, weight-for-weight port of the upstream reference
        # (Brant_src on Hugging Face). Kept as attributes so the intended data
        # flow in ``forward`` is already wired.
        # ------------------------------------------------------------------ #
        # TODO(brant): patch + band-power embedding (time & frequency domains).
        self.patch_embed = nn.Identity()
        # TODO(brant): temporal Transformer encoder over consecutive patches.
        self.temporal_encoder = nn.Identity()
        # TODO(brant): spatial Transformer encoder over channels.
        self.spatial_encoder = nn.Identity()
        # TODO(brant): final classification head (replaces the pretrain head).
        self.final_layer = nn.Linear(self.embed_dim, self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)

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
        # Intended data flow (to be implemented):
        #   1. cut ``x`` into (L, C) patches of ``self.patch_size`` samples;
        #   2. input encoding = linear projection of the patch + learnable
        #      temporal positional encoding + frequency encoding (softmax over
        #      log band-power across ``BRANT_FREQ_BANDS``) -> ``self.patch_embed``;
        #   3. temporal encoder over the L consecutive patches, per channel;
        #   4. spatial encoder over the C channels at each time index;
        #   5. pool the (L, C, D) representation and apply ``self.final_layer``.
        raise NotImplementedError(
            "Brant.forward is not implemented yet — skeleton only. "
            "The faithful port and its parity check against the upstream "
            "Apache-2.0 checkpoint land in follow-up commits "
            "(braindecode/braindecode#1097)."
        )
