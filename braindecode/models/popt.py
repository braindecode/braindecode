# Authors: Adam Mounir
#
# License: MIT (upstream PopulationTransformer code)
"""PopulationTransformer: a self-supervised aggregator over intracranial electrodes.

Port of PopulationTransformer (PopT, Chau et al. 2024) into a braindecode-native
model. Upstream code and pretrained weights are released by the authors:

* paper: https://arxiv.org/abs/2406.03044
* code: https://github.com/czlwang/PopulationTransformer
* weights: https://huggingface.co/PopulationTransformer/popt_brainbert_stft

Unlike a per-channel encoder, PopT operates on a **population** of electrodes:
its input is a set of per-electrode feature vectors (the frozen embeddings of a
channel-level foundation model such as BrainBERT) plus each electrode's integer
anatomical coordinates. A ``CLS`` token summarises the population after a stack
of Transformer encoder layers. The input embedding, spatial position encoding and
Transformer are ported weight-for-weight from the upstream reference (numerically
equivalent, max abs difference about 1e-6 with the released checkpoint, verified by the ``test_encoder_is_bit_exact_with_upstream`` parity gate in
``test/unit_tests/models/test_popt.py``); the classification head is a
braindecode-native addition. The official checkpoint loads directly via
``PopulationTransformer.from_pretrained("braindecode/popt-pretrained")``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules.popt_modules import (
    _PopTHead,
    _PopTInputEmbedding,
    _PopTSpecPredictionHead,
)


class PopulationTransformer(EEGModuleMixin, nn.Module):
    r"""PopulationTransformer (PopT) from Chau et al. (2024) [PopT2024]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    PopT is a self-supervised **population** model for intracranial recordings
    (sEEG/iEEG). It does not encode a raw time signal; instead each electrode is
    represented by a feature vector — typically the frozen embedding of a
    per-channel foundation model such as :class:`~braindecode.models.BrainBERT` —
    and PopT aggregates across electrodes. Every electrode feature is linearly
    projected and given a fixed sinusoidal **spatial** position encoding built
    from its integer anatomical coordinates (one embedding per X/Y/Z axis plus a
    sequence id). A ``CLS`` token is prepended, a stack of standard Transformer
    encoder layers mixes the population, and the ``CLS`` output is the pooled
    representation used for downstream decoding. Pre-training is by masked /
    replaced-token modelling over the electrode population.

    Following the braindecode convention, the per-electrode feature vector plays
    the role of the ``n_times`` axis, so the model keeps the standard
    ``(batch, n_chans, n_times)`` input signature: ``n_chans`` is the number of
    electrodes and ``n_times`` is the upstream feature dimension (768 for
    BrainBERT ``stft`` features). Electrode coordinates are read from
    ``chs_info`` (their ``loc``) and discretised to integer indices inside the
    model; when no positions are available the electrodes fall back to distinct
    sequential indices.

    The released ``popt_brainbert_stft`` checkpoint uses ``hidden_dim=512``,
    ``ffn_dim=2048``, ``n_heads=8``, ``n_layers=6`` on ``n_times=768`` BrainBERT
    features (~20.6M parameters). The defaults below are a modest, ready-to-run
    configuration; pass the released values to reproduce it.

    .. important::
       **Pre-trained weights available.** The official checkpoint is released by
       the authors and loads directly::

           model = PopulationTransformer.from_pretrained(
               "braindecode/popt-pretrained", n_outputs=2
           )

       It uses the released configuration above; ``n_chans`` and ``n_outputs``
       may be changed freely, as the population is pooled through the ``CLS``
       token and the classification head is task-specific.

    .. versionadded:: 1.8.2

    Parameters
    ----------
    hidden_dim : int, optional
        Transformer model width ``D``. Must be divisible by 8. Default 128. The
        released model uses 512.
    ffn_dim : int, optional
        Inner dimension of the Transformer feed-forward blocks. Default 256. The
        released model uses 2048.
    n_layers : int, optional
        Number of Transformer encoder layers. Default 2. Released model: 6.
    n_heads : int, optional
        Number of attention heads. Default 4. Released model: 8.
    max_len : int, optional
        Size of the coordinate table (largest addressable integer coordinate).
        Default 5000, as upstream.
    coord_units : {"m", "raw"}, optional
        How ``chs_info`` positions become integer coordinates. ``"m"`` (default)
        treats them as MNE metres: millimetres, rounded and shifted per axis so
        the smallest index is 0. This is **not** the coordinate space of the
        pretrained checkpoint. ``"raw"`` uses the positions as they are
        (rounded, no shift): use it with the released Brain Treebank integer
        (left, inferior, posterior) coordinates, which NEMAR nm000253 stores
        unconverted in ``x/y/z``, to match the pretrained model (upstream
        ``pt_supervised_task_coords.py``). You can also pass ``coords`` to
        :meth:`forward` directly.
    activation : type[nn.Module], optional
        Feed-forward activation, given as a class. Default :class:`~torch.nn.GELU`.
    drop_prob : float, optional
        Dropout probability. Default 0.1.

    References
    ----------
    .. [PopT2024] Chau, G., Wang, C., Talukder, S., Subramaniam, V., Soedarmadji,
       S., Yue, Y., Katz, B., & Barbu, A. (2024). Population Transformer:
       Learning Population-level Representations of Neural Activity. arXiv
       preprint arXiv:2406.03044.
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
        # model-specific parameters
        *,
        hidden_dim: int = 128,
        ffn_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 5000,
        coord_units: str = "m",
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.1,
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
        if coord_units not in ("m", "raw"):
            raise ValueError(f"coord_units must be 'm' or 'raw', got {coord_units!r}.")

        # The per-electrode feature vector plays the role of the "time" axis.
        self.input_dim = self.n_times
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.coord_units = coord_units

        self.input_embedding = _PopTInputEmbedding(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            max_len=max_len,
            drop_prob=drop_prob,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            activation=activation(),
            dropout=drop_prob,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        # kept for weight parity with the upstream pretraining checkpoint; the
        # classification path does not use it.
        self.spec_prediction_head = _PopTSpecPredictionHead(hidden_dim, self.input_dim)
        self.final_layer = _PopTHead(hidden_dim, self.n_outputs)

        # Integer electrode coordinates / sequence ids, derived once from
        # chs_info; broadcast over the batch at forward time. Buffers follow
        # device moves and are regenerated per instance (persistent=False).
        self.register_buffer(
            "electrode_coords", self._coords_from_chs_info(), persistent=False
        )

    def _coords_from_chs_info(self) -> torch.Tensor:
        """Build integer ``(n_chans, 3)`` electrode coordinates.

        Coordinates are read from ``chs_info`` (each channel's ``loc[:3]``,
        in metres as per the MNE convention unless ``coord_units='raw'``), converted
        to millimetres, rounded and shifted per axis so the smallest index is 0
        (``'raw'``: rounded as they are). If ``chs_info`` is missing or carries
        no usable positions, electrodes fall back to distinct sequential indices
        on every axis. All indices are clamped to ``[0, max_len - 1]``.
        """
        n_chans = self.n_chans
        loc = None
        # chs_info is optional; the public property raises when unset, so read
        # the underlying attribute directly.
        chs_info = getattr(self, "_chs_info", None)
        if chs_info is not None:
            try:
                loc = torch.tensor(
                    np.asarray([ch["loc"][:3] for ch in chs_info]), dtype=torch.float
                )
            except (KeyError, TypeError, ValueError):
                loc = None
        if loc is None or not torch.isfinite(loc).all() or loc.abs().sum() == 0:
            # Fallback: distinct sequential positions on each axis.
            idx = torch.arange(n_chans, dtype=torch.long)
            coords = idx.unsqueeze(1).repeat(1, 3)
        else:
            if self.coord_units == "raw":
                coords = loc.round().long()
            else:
                coords = (loc * 1000.0).round().long()
                coords = coords - coords.min(dim=0, keepdim=True).values
        return coords.clamp(0, self.max_len - 1)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        old = next(self.final_layer.parameters())
        self._set_n_outputs(n_outputs)
        self.final_layer = _PopTHead(self.hidden_dim, n_outputs).to(
            device=old.device, dtype=old.dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        seq_id: torch.Tensor | None = None,
        return_features: bool = False,
        key_padding_mask: torch.Tensor | None = None,
    ):
        """Aggregate a population of electrode features.

        Parameters
        ----------
        x : torch.Tensor
            Per-electrode features of shape ``(batch, n_chans, n_times)``, where
            ``n_times`` is the upstream feature dimension.
        coords : torch.Tensor, optional
            Integer coordinates of shape ``(batch, n_chans, 3)``. Defaults to the
            coordinates derived from ``chs_info`` at construction, broadcast over
            the batch.
        seq_id : torch.Tensor, optional
            Integer sequence ids of shape ``(batch, n_chans)``. Defaults to zero
            (single population).
        return_features : bool
            If ``True``, return ``{"features": cls, "cls_token": cls}`` (the
            pooled ``CLS`` representation) instead of the class logits.
        key_padding_mask : torch.Tensor, optional
            Boolean ``(batch, n_chans)`` mask, ``True`` for padded electrodes, so
            recordings with different electrode sets can share a batch (upstream
            ``src_key_padding_mask``). The ``CLS`` token is never masked.

        Returns
        -------
        torch.Tensor or dict
            Class logits of shape ``(batch, n_outputs)``, or the feature dict
            when ``return_features`` is set.
        """
        batch_size, n_chans, _ = x.shape
        if coords is None:
            coords = self.electrode_coords.unsqueeze(0).expand(batch_size, -1, -1)
        if seq_id is None:
            # Single population: all zeros, sized from the input so that padded
            # batches with more electrodes than at construction also work.
            seq_id = torch.zeros(batch_size, n_chans, dtype=torch.long, device=x.device)

        h = self.input_embedding(x, coords, seq_id)
        if key_padding_mask is not None:
            cls_keep = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=key_padding_mask.device
            )
            key_padding_mask = torch.cat([cls_keep, key_padding_mask.bool()], dim=1)
        # (batch, 1 + n_chans, hidden_dim)
        z = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)
        cls_token = z[:, 0, :]
        if return_features:
            return {"features": cls_token, "cls_token": cls_token}
        return self.final_layer(cls_token)
