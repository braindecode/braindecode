# Authors: Christopher Wang, Geeling Chau (original implementation)
#          Adam Mounir <am91ris@gmail.com> (braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/czlwang/PopulationTransformer
"""PopulationTransformer (PopT, Chau et al. 2024).

* paper: https://arxiv.org/abs/2406.03044
* code: https://github.com/czlwang/PopulationTransformer
* weights: https://huggingface.co/PopulationTransformer/popt_brainbert_stft
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules.popt_modules import (
    _PopTInputEmbedding,
    _PopTSpecPredictionHead,
)


class PopulationTransformer(EEGModuleMixin, nn.Module, license="mit"):
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
    ``chs_info`` (their ``loc``) and discretised to absolute integer indices
    inside the model, as upstream feeds them; when no positions are available
    the electrodes fall back to distinct sequential indices.

    The ``CLS`` output goes through a single linear layer, as in the upstream
    fine-tuning model (``PtDownstreamModel.linear_out``, one logit trained with
    binary cross-entropy there; ``n_outputs=1`` reproduces it).

    The defaults are the released ``popt_brainbert_stft`` configuration:
    ``hidden_dim=512``, ``ffn_dim=2048``, ``n_heads=8``, ``n_layers=6``, used
    on ``n_times=768`` BrainBERT features (~20M parameters).

    .. important::
       **Pre-trained weights available.** The official checkpoint is released by
       the authors and loads directly::

           model = PopulationTransformer.from_pretrained(
               "braindecode/popt-pretrained", n_outputs=2
           )

       It uses the default configuration; ``n_chans`` and ``n_outputs``
       may be changed freely, as the population is pooled through the ``CLS``
       token and the classification head is task-specific (the checkpoint
       carries no trained fine-tuning head).

    .. warning::
       **Evaluate with a time-blocked split.** The paper's downstream results
       use a random 80/10/10 split over word-aligned 5 s windows. Words are a
       fraction of a second apart, so almost every test window overlaps a
       training window, and labels that drift slowly in time leak into
       training. Re-running the paper setup (7 subjects, 3 seeds) with
       contiguous blocks of time, and dropping training windows that overlap
       the test set, pretrained PopT goes from 0.79 to 0.51 ROC-AUC on Pitch
       (chance) and from 0.89 to 0.64 on Volume. Onset (0.86 to 0.84) and
       Speech (0.90 to 0.84) hold up, and pretraining still beats training
       from scratch on Onset, Speech and Volume. The model and weights are
       not affected; the issue is only in the evaluation. When fine-tuning,
       split by blocks of time.

    .. versionadded:: 1.8.2

    Parameters
    ----------
    hidden_dim : int, optional
        Transformer model width ``D``. Must be divisible by 8. Default 512, as
        the released model.
    ffn_dim : int, optional
        Inner dimension of the Transformer feed-forward blocks. Default 2048, as
        the released model.
    n_layers : int, optional
        Number of Transformer encoder layers. Default 6, as the released model.
    n_heads : int, optional
        Number of attention heads. Default 8, as the released model.
    max_len : int, optional
        Size of the coordinate table (largest addressable integer coordinate).
        Default 5000, as upstream.
    coord_units : {"m", "raw"}, optional
        How ``chs_info`` positions become integer coordinates. ``"m"`` (default)
        treats them as MNE metres and rounds them to millimetres. ``"raw"``
        rounds the positions as they are: use it when ``x/y/z`` already hold the
        Brain Treebank integer (left, inferior, posterior) coordinates, as NEMAR
        nm000253 stores them. Either way the indices are absolute, not
        shifted, as upstream feeds them (``pt_supervised_task_coords.py``).
        They match the pretrained checkpoint only if the positions are already
        in the upstream (left, inferior, posterior) space; MNE head-frame
        positions (e.g. a standard montage) are **not** that space. Indices
        outside ``[0, max_len - 1]`` are clamped, with a warning. You can also
        pass ``coords`` to :meth:`forward` directly.
    shift_coords : bool, optional
        If ``True``, shift each axis so that its smallest index is 0. Default
        ``False``. Upstream does not shift, and the shift changes the position
        encoding, and so the output of the pretrained model; use it only for
        positions with negative values when training from scratch.
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
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        n_layers: int = 6,
        n_heads: int = 8,
        max_len: int = 5000,
        coord_units: str = "m",
        shift_coords: bool = False,
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

        self.input_dim = self.n_times
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.coord_units = coord_units
        self.shift_coords = shift_coords

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
        # Pre-training head, unused here; kept so the checkpoint loads strictly.
        self.spec_prediction_head = _PopTSpecPredictionHead(hidden_dim, self.input_dim)
        self.final_layer = nn.Linear(hidden_dim, self.n_outputs)

        self.register_buffer(
            "electrode_coords", self._coords_from_chs_info(), persistent=False
        )

    def _coords_from_chs_info(self) -> torch.Tensor:
        """Integer ``(n_chans, 3)`` coordinates from ``chs_info``, or sequential ones."""
        chs_info = getattr(self, "_chs_info", None)
        loc = extract_channel_locations_from_chs_info(
            chs_info, num_channels=self.n_chans
        )
        # The helper does not screen NaNs or partial montages.
        if loc is None or loc.shape[0] != self.n_chans or not np.isfinite(loc).all():
            idx = torch.arange(self.n_chans, dtype=torch.long)
            coords = idx.unsqueeze(1).repeat(1, 3)
        else:
            loc_t = torch.as_tensor(loc, dtype=torch.float)
            if self.coord_units == "m":
                loc_t = loc_t * 1000.0
            coords = loc_t.round().long()
            if self.shift_coords:
                coords = coords - coords.min(dim=0, keepdim=True).values
            if bool(((coords < 0) | (coords >= self.max_len)).any()):
                hint = (
                    "raise `max_len`"
                    if self.shift_coords
                    else "set `shift_coords=True` to train from scratch"
                )
                warnings.warn(
                    "Some electrode coordinates from chs_info fall outside "
                    f"[0, {self.max_len - 1}] and are clamped. The pretrained "
                    "model expects the non-negative integer (left, inferior, "
                    "posterior) indices of the upstream data; pass `coords` to "
                    f"forward, or {hint}.",
                    UserWarning,
                    stacklevel=4,
                )
        return coords.clamp(0, self.max_len - 1)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Also accept the untrained ``final_layer.{norm,fc}`` head of the HF mirror."""
        remapped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("final_layer.norm."):
                continue
            if key.startswith("final_layer.fc."):
                key = "final_layer." + key[len("final_layer.fc.") :]
            remapped[key] = value
        return super().load_state_dict(remapped, *args, **kwargs)

    def reset_head(self, n_outputs: int) -> None:
        """Swap the classification head for a new number of outputs."""
        old = self.final_layer.weight
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.hidden_dim, n_outputs).to(
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
            n_known = self.electrode_coords.shape[0]
            if n_chans != n_known:
                raise ValueError(
                    f"x has {n_chans} electrodes but the model has coordinates "
                    f"for {n_known}; pass `coords` of shape (batch, {n_chans}, 3)."
                )
            coords = self.electrode_coords.unsqueeze(0).expand(batch_size, -1, -1)
        if seq_id is None:
            seq_id = torch.zeros(batch_size, n_chans, dtype=torch.long, device=x.device)

        h = self.input_embedding(x, coords, seq_id)
        # New name (not reassigning the argument) keeps TorchScript's narrowing.
        padded_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            cls_keep = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            padded_mask = torch.cat([cls_keep, key_padding_mask.to(torch.bool)], dim=1)
        z = self.transformer_encoder(h, src_key_padding_mask=padded_mask)
        cls_token = z[:, 0, :]
        logits = self.final_layer(cls_token)
        if return_features:
            # Scripted forward stays monomorphic (same pattern as Brant).
            if torch.jit.is_scripting():
                return logits
            return {"features": cls_token, "cls_token": cls_token}
        return logits
