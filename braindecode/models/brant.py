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

    Time- and frequency-domain information are combined: alongside the raw
    patch, band-power features are computed and injected into the input
    encoding. Following the braindecode convention, that frequency computation
    is performed **inside** ``forward`` so the model keeps the standard
    ``(batch, n_chans, n_times)`` input signature.

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
        Size of the patch embedding / model width. Default 512.
    temporal_n_layers : int, optional
        Number of layers in the temporal Transformer encoder. Default 8.
    spatial_n_layers : int, optional
        Number of layers in the spatial Transformer encoder. Default 4.
    n_heads : int, optional
        Number of attention heads in both encoders. Default 8.
    ffn_ratio : int, optional
        Feed-forward expansion ratio inside the Transformer blocks. Default 4.
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
        # --- Brant hyper-parameters (defaults to be pinned from Brant_src) ---
        patch_size: int = 1500,
        embed_dim: int = 512,
        temporal_n_layers: int = 8,
        spatial_n_layers: int = 4,
        n_heads: int = 8,
        ffn_ratio: int = 4,
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
        self.temporal_n_layers = temporal_n_layers
        self.spatial_n_layers = spatial_n_layers
        self.n_heads = n_heads
        self.ffn_ratio = ffn_ratio
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
        #   1. cut ``x`` into patches of ``self.patch_size`` samples;
        #   2. embed each patch (raw + band-power) -> ``self.patch_embed``;
        #   3. temporal encoder over consecutive patches per channel;
        #   4. spatial encoder over channels at each time index;
        #   5. pool the representation and apply ``self.final_layer``.
        raise NotImplementedError(
            "Brant.forward is not implemented yet — skeleton only. "
            "The faithful port and its parity check against the upstream "
            "Apache-2.0 checkpoint land in follow-up commits "
            "(braindecode/braindecode#1097)."
        )
