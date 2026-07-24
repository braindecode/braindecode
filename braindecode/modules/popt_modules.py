"""Building blocks for :class:`braindecode.models.PopulationTransformer`.

Faithful re-implementation of the upstream PopulationTransformer (PopT) reference
code (``PopulationTransformer/models/pt_model_custom.py`` and
``transformer_encoder_input.py``, https://github.com/czlwang/PopulationTransformer)
as standalone braindecode modules — no new runtime dependency (the encoder is a
stock :class:`torch.nn.TransformerEncoder`).

PopT is a *population* aggregator: it does **not** consume a raw time signal but a
set of per-electrode feature vectors (typically the frozen 768-d embeddings of a
per-channel foundation model such as BrainBERT) together with each electrode's
integer anatomical coordinates. A learnable-free sinusoidal spatial encoding
(one embedding per X/Y/Z axis plus a sequence-id embedding, each ``hidden//4``
wide) is added to a linear projection of the features; a ``CLS`` token is
prepended and, after a stack of Transformer encoder layers, its output is the
pooled population representation.

Every module here maps its parameters 1:1 to the upstream
``TransformerEncoderInput`` / ``SpecPredictionHead`` so the released checkpoint
loads weight-for-weight (see ``scripts/popt_parity_check``). The only
braindecode-native changes are (i) the ``CLS`` feature row and the electrode
coordinates are materialised **inside** the model so it keeps a standard
``forward(x)`` signature, and (ii) the classification head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PopTSpatialPositionalEncoding(nn.Module):
    """Spatial sinusoidal positional encoding (upstream
    ``MultiSubjBrainPositionalEncoding``).

    A single sinusoidal table ``pe`` of shape ``(1, max_len, hidden_dim // 4)``
    is indexed by **integer** coordinates: each of the three anatomical axes
    (X/Y/Z) and the sequence id select a ``hidden_dim // 4`` slice, and the four
    slices are concatenated into a ``hidden_dim`` position vector. The ``CLS``
    token receives ``pe[0, 0]`` tiled four times. ``pe`` is non-learnable and
    regenerated in ``__init__`` (``persistent=False``), so it stays out of the
    checkpoint.

    Parameters
    ----------
    hidden_dim : int
        Model width ``D``; must be divisible by 4.
    max_len : int
        Size of the coordinate table (largest addressable integer coordinate).
    """

    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by 4 for the "
                "spatial positional encoding (one quarter per X/Y/Z axis and "
                "the sequence id)."
            )
        pe_dim = hidden_dim // 4
        pe = torch.zeros(max_len, pe_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(
        self, seq: torch.Tensor, coords: torch.Tensor, seq_id: torch.Tensor
    ) -> torch.Tensor:
        """Add spatial position embeddings to a ``CLS``-prefixed sequence.

        Parameters
        ----------
        seq : torch.Tensor
            Projected tokens of shape ``(batch, 1 + n_electrodes, hidden_dim)``
            (position 0 is the ``CLS`` token).
        coords : torch.Tensor
            Integer coordinates of shape ``(batch, n_electrodes, 3)``.
        seq_id : torch.Tensor
            Integer sequence ids of shape ``(batch, n_electrodes)``.

        Returns
        -------
        torch.Tensor
            ``seq`` with the position embeddings added, same shape.
        """
        table = self.pe[0]  # (max_len, pe_dim)
        # (batch, n_elec, 3, pe_dim) -> (batch, n_elec, 3 * pe_dim)
        p_embed = table[coords]
        n_batch, n_elec, n_axes, pe_dim = p_embed.shape
        p_embed = p_embed.reshape(n_batch, n_elec, n_axes * pe_dim)
        # (batch, n_elec, pe_dim)
        s_embed = table[seq_id]
        elec_embed = torch.cat([p_embed, s_embed], dim=-1)  # (b, n_elec, hidden)
        # CLS position embedding: pe[0] tiled over the four quarters.
        cls_embed = table[0].repeat(n_batch, 4).unsqueeze(1)  # (b, 1, hidden)
        pos = torch.cat([cls_embed, elec_embed], dim=1)  # (b, 1 + n_elec, hidden)
        return seq + pos


class _PopTInputEmbedding(nn.Module):
    """Input encoding of PopT (upstream ``TransformerEncoderInput``).

    Prepends a ``CLS`` feature row (a vector of ones, as upstream), projects the
    per-electrode features ``input_dim -> hidden_dim``, adds the spatial position
    encoding, then applies LayerNorm and dropout. Attribute names (``in_proj``,
    ``positional_encoding``, ``layer_norm``) match upstream so weights map
    directly.

    Parameters
    ----------
    input_dim : int
        Dimension of the per-electrode input features (e.g. 768 for BrainBERT).
    hidden_dim : int
        Transformer model width ``D``.
    max_len : int
        Size of the coordinate table, forwarded to the positional encoding.
    drop_prob : float
        Dropout probability applied to the embedded tokens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_len: int = 5000,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = _PopTSpatialPositionalEncoding(
            hidden_dim, max_len=max_len
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        seq_id: torch.Tensor,
    ) -> torch.Tensor:
        """Embed per-electrode features into ``CLS``-prefixed tokens.

        Parameters
        ----------
        features : torch.Tensor
            Per-electrode features of shape ``(batch, n_electrodes, input_dim)``.
        coords : torch.Tensor
            Integer coordinates of shape ``(batch, n_electrodes, 3)``.
        seq_id : torch.Tensor
            Integer sequence ids of shape ``(batch, n_electrodes)``.

        Returns
        -------
        torch.Tensor
            Embedded tokens of shape ``(batch, 1 + n_electrodes, hidden_dim)``.
        """
        batch_size = features.shape[0]
        # Upstream prepends a CLS row of ones before the linear projection.
        cls = features.new_ones((batch_size, 1, self.input_dim))
        x = torch.cat([cls, features], dim=1)
        x = self.in_proj(x)
        x = self.positional_encoding(x, coords, seq_id)
        x = self.layer_norm(x)
        return self.dropout(x)


class _PopTSpecPredictionHead(nn.Module):
    """Feature-reconstruction head (upstream ``SpecPredictionHead``).

    Kept so the pre-training parameters map 1:1 to upstream (weight parity);
    unused by the braindecode classification path.

    Parameters
    ----------
    hidden_dim : int
        Transformer model width ``D``.
    output_dim : int
        Reconstruction target dimension (the input feature dimension upstream).
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.hidden_layer(hidden)
        h = self.act_fn(h)
        h = self.layer_norm(h)
        return self.output(h)


class _PopTHead(nn.Module):
    """braindecode-native classification head on the ``CLS`` representation.

    Upstream reads the ``CLS`` token and applies a single linear layer
    (``cls_head``, ``hidden_dim -> 1``); here a LayerNorm precedes a linear layer
    to ``n_outputs``, a minimal and standard braindecode choice that generalises
    to any number of classes.

    Parameters
    ----------
    hidden_dim : int
        Transformer model width ``D``.
    n_outputs : int
        Number of decoding classes.
    """

    def __init__(self, hidden_dim: int, n_outputs: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls_token))
