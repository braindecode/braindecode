# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: pending (the reference code is released without a license)
# Adapted from https://github.com/gmentz/seegnificant.
"""seegnificant: multi-subject decoding from sEEG, electrodes placed by position."""

import math
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    has_valid_locations,
)
from braindecode.modules import FeedForwardBlock


class SEEGnificant(EEGModuleMixin, nn.Module):
    r"""seegnificant from Mentzelopoulos et al. (2024) [Mentzelopoulos2024]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel`

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    A decoder trained jointly on sEEG subjects whose electrodes differ in number
    and placement. Each electrode is tokenized on its own, attention alternates
    between time (within an electrode) and electrodes (at a time step), and each
    electrode is told apart by its MNI position; a subject-specific head reads
    the shared representation.

    .. rubric:: Macro Components

    - **Convolutional tokenizer** (``SEEGnificant.patch_embedding``).
      *Operations:* a temporal convolution of ``kernel_size`` samples shared by
      all electrodes, batch norm, average pooling of ``pool_size`` samples with
      stride ``pool_stride``. *Role:* turn each electrode's trace into a
      sequence of ``emb_size``-dimensional tokens.
    - **Time attention** (``SEEGnificant.time_encoder``). *Operations:* a
      pre-norm transformer block over the tokens of each electrode (attention
      scores scaled by ``emb_size``, the feed-forward led by a square layer, as
      the reference).
    - **Spatial positional encoding** (``SEEGnificant.positional_encoding``).
      *Operations:* Gaussian radial basis functions of each MNI coordinate
      (centres -90 to 70 mm every 20 mm, scales 1 to 64 mm), linearly projected
      and added to every token of the electrode before each space block.
    - **Space attention** (``SEEGnificant.space_encoder``). *Operations:* a
      pre-norm transformer block over the electrodes at each time step.
    - **Read-out** (``SEEGnificant.mlp``, ``SEEGnificant.final_layer``).
      *Operations:* the tokens are flattened and projected to ``n_features``
      (the paper's neural representation, returned by ``return_features``),
      then one MLP head per subject.

    .. rubric:: Inputs

    The paper feeds broadband voltage traces z-scored per electrode and
    resampled to 400 Hz, 1.5 s windows (``n_times=600``), from at most 28
    electrodes. The flattened read-out ties the model to its ``n_chans`` and
    ``n_times``: a subject with fewer electrodes is zero-padded, and padded
    electrodes are marked by an all-zero position (the reference convention).
    In the space attention, real electrodes ignore padded ones, and a padded
    electrode attends to all electrodes uniformly (the reference's mask).

    Electrode positions are MNI coordinates in metres, passed as ``pos`` or
    read from ``chs_info``.

    .. note::
        The outputs match the reference implementation for the same weights
        and inputs. Departures: the heads end without the reference's ReLU,
        which kept its response-time outputs non-negative; padding is an
        all-zero position (the reference flags a position whose coordinates
        sum to zero).

    Parameters
    ----------
    n_subjects : int
        Number of subject-specific heads; ``forward`` then needs
        ``subject_ids`` when above 1.
    emb_size : int
        Token dimension, the number of convolution kernels.
    depth : int
        Number of time and space attention block pairs.
    num_heads : int
        Number of attention heads.
    kernel_size : int
        Temporal convolution kernel size, in samples.
    pool_size : int
        Average-pooling window, in samples.
    pool_stride : int
        Average-pooling stride, in samples.
    n_features : int
        Dimension of the shared representation.
    head_hidden : int
        Hidden width of each subject head.
    ffn_expansion : int
        Expansion of the feed-forward blocks.
    drop_prob : float
        Dropout rate.
    activation : type[nn.Module]
        Activation of the feed-forward blocks.

    References
    ----------
    .. [Mentzelopoulos2024] Mentzelopoulos, G., Chatzipantazis, E., Ramayya,
       A., Hedlund, M., Buch, V., Daniilidis, K., Kording, K. & Vitale, F.
       (2024). Neural decoding from stereotactic EEG: accounting for electrode
       variability across subjects. Advances in Neural Information Processing
       Systems 37 (NeurIPS 2024).
    """

    def __init__(
        self,
        # --- signal-related (handled by EEGModuleMixin) ---
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # --- model hyperparameters (defaults: the paper's) ---
        n_subjects: int = 1,
        emb_size: int = 2,
        depth: int = 1,
        num_heads: int = 1,
        kernel_size: int = 40,
        pool_size: int = 40,
        pool_stride: int = 5,
        n_features: int = 128,
        head_hidden: int = 16,
        ffn_expansion: int = 4,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.GELU,
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

        n_tokens = (self.n_times - kernel_size + 1 - pool_size) // pool_stride + 1
        if n_tokens < 1:
            raise ValueError(
                f"n_times={self.n_times} is shorter than kernel_size + pool_size - 1 "
                f"= {kernel_size + pool_size - 1}."
            )
        self.n_subjects = n_subjects
        self.n_features = n_features
        self.head_hidden = head_hidden
        self.drop_prob = drop_prob

        default_pos: Optional[torch.Tensor] = None
        chs_info = self._chs_info
        if has_valid_locations(chs_info):
            locations = extract_channel_locations_from_chs_info(chs_info)
            default_pos = torch.as_tensor(locations, dtype=torch.float32)
        # Montage-specific, rebuilt from chs_info: not saved with the weights.
        self.register_buffer("default_pos", default_pos, persistent=False)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, kernel_size)),
            nn.BatchNorm2d(emb_size),
            nn.AvgPool2d((1, pool_size), (1, pool_stride)),
            nn.Dropout(drop_prob),
        )
        self.time_encoder = nn.ModuleList(
            [
                _EncoderBlock(emb_size, num_heads, drop_prob, ffn_expansion, activation)
                for _ in range(depth)
            ]
        )
        self.space_encoder = nn.ModuleList(
            [
                _EncoderBlock(emb_size, num_heads, drop_prob, ffn_expansion, activation)
                for _ in range(depth)
            ]
        )
        self.positional_encoding = _MNIPositionalEncoding(emb_size, drop_prob)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size * self.n_chans * n_tokens, n_features),
            nn.ReLU(),
            nn.Dropout(drop_prob),
        )
        self.final_layer = _SubjectHeads(
            n_features, head_hidden, self.n_outputs, n_subjects, drop_prob
        )

    def reset_head(self, n_outputs):
        """Replace the subject heads for a new ``n_outputs``."""
        self._n_outputs = n_outputs
        self._update_init_kwargs(n_outputs=n_outputs)
        self.final_layer = _SubjectHeads(
            self.n_features,
            self.head_hidden,
            n_outputs,
            self.n_subjects,
            self.drop_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Decode a batch of sEEG windows.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        pos : torch.Tensor, optional
            MNI positions in metres, ``(batch, n_chans, 3)`` or ``(n_chans,
            3)``, all-zero for padded electrodes; by default those of
            ``chs_info``.
        subject_ids : torch.Tensor, optional
            ``(batch,)`` index of each window's subject head; needed with
            ``n_subjects > 1``.
        return_features : bool
            If ``True``, return ``{"features", "cls_token"}``: the shared
            representation ``(batch, n_features)`` and ``None``.

        Returns
        -------
        torch.Tensor | dict
            Outputs ``(batch, n_outputs)``, or the feature dict.
        """
        if pos is None:
            pos = self.default_pos
        if pos is None:
            raise ValueError(
                "SEEGnificant places each electrode by its MNI position: pass pos, "
                "or chs_info with the electrode positions."
            )
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if pos.shape[1] != x.shape[1] or pos.shape[2] != 3:
            raise ValueError(
                f"pos must have shape (batch, {x.shape[1]}, 3) or ({x.shape[1]}, "
                f"3), got {list(pos.shape)}."
            )
        tokens = self.patch_embedding(x.unsqueeze(1))  # (batch, emb, chans, tokens)
        batch_size, emb_size, n_chans, n_tokens = tokens.shape
        z = tokens.permute(0, 2, 3, 1)  # (batch, chans, tokens, emb)
        padded = (pos == 0).all(dim=-1)
        padded = padded[:, None].expand(batch_size, n_tokens, -1).flatten(0, 1)
        mni_mm = pos * 1000.0
        for time_block, space_block in zip(self.time_encoder, self.space_encoder):
            z = time_block(z.flatten(0, 1)).reshape(z.shape)
            z = z + self.positional_encoding(mni_mm)[:, :, None]
            z = space_block(z.transpose(1, 2).flatten(0, 1), padded)
            z = z.reshape(batch_size, n_tokens, n_chans, emb_size).transpose(1, 2)
        features = self.mlp(z.permute(0, 3, 1, 2).flatten(1))
        if return_features:
            out: Dict[str, Optional[torch.Tensor]] = {
                "features": features,
                "cls_token": None,
            }
            return out
        return self.final_layer(features, subject_ids)


class _EncoderBlock(nn.Module):
    """Pre-norm transformer block, its feed-forward led by a square layer."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        drop_prob: float,
        expansion: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(emb_size)
        self.attn = _Attention(emb_size, num_heads, drop_prob)
        self.attn_drop = nn.Dropout(drop_prob)
        self.ffn_norm = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Dropout(drop_prob),
            FeedForwardBlock(emb_size, expansion, drop_prob, activation),
        )
        self.ffn_drop = nn.Dropout(drop_prob)

    def forward(
        self, x: torch.Tensor, padded: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn_drop(self.attn(self.attn_norm(x), padded))
        return x + self.ffn_drop(self.ffn(self.ffn_norm(x)))


class _Attention(nn.Module):
    """The reference's multi-head self-attention, scores scaled by ``emb_size``.

    With ``padded``, ``(batch, seq)``, real tokens ignore padded ones and a padded
    token attends to every token uniformly, as the reference's mask does.
    """

    def __init__(self, emb_size: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.num_heads = num_heads
        self.scaling = emb_size**0.5
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(drop_prob)
        self.projection = nn.Linear(emb_size, emb_size)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, padded: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        queries, keys = self._heads(self.queries(x)), self._heads(self.keys(x))
        energy = torch.matmul(queries, keys.transpose(-1, -2))
        if padded is not None:
            query_padded = padded[:, None, :, None]
            energy = energy.masked_fill(query_padded, 0.0)
            energy = energy.masked_fill(
                ~query_padded & padded[:, None, None, :], float("-inf")
            )
        attention = self.att_drop((energy / self.scaling).softmax(dim=-1))
        out = torch.matmul(attention, self._heads(self.values(x)))
        return self.projection(out.transpose(1, 2).flatten(2))


class _MNIPositionalEncoding(nn.Module):
    """Gaussian radial basis functions of each MNI coordinate (mm), projected."""

    means: torch.Tensor
    stds: torch.Tensor

    def __init__(self, emb_size: int, drop_prob: float):
        super().__init__()
        means = torch.arange(-90.0, 80.0, 20.0)
        stds = 2.0 ** torch.arange(7.0)
        self.register_buffer(
            "means", means.repeat_interleave(len(stds)), persistent=False
        )
        self.register_buffer("stds", stds.repeat(len(means)), persistent=False)
        self.projection = nn.Linear(3 * len(means) * len(stds), emb_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, mni_mm: torch.Tensor) -> torch.Tensor:
        scaled = (mni_mm[..., None] - self.means) / self.stds
        pdf = torch.exp(-0.5 * scaled**2) / (self.stds * math.sqrt(2 * math.pi))
        return self.dropout(self.projection(pdf.flatten(-2)))


class _SubjectHeads(nn.Module):
    """One MLP head per subject, each window read by the head of its subject."""

    def __init__(
        self,
        n_features: int,
        hidden: int,
        n_outputs: int,
        n_subjects: int,
        drop_prob: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_features, hidden),
                    nn.ReLU(),
                    nn.Dropout(drop_prob),
                    nn.Linear(hidden, n_outputs),
                )
                for _ in range(n_subjects)
            ]
        )

    def forward(
        self, features: torch.Tensor, subject_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for head in self.heads:
            outputs.append(head(features))
        if subject_ids is None:
            if len(outputs) > 1:
                raise ValueError(
                    f"subject_ids is needed to pick one of the {len(outputs)} "
                    f"subject heads."
                )
            return outputs[0]
        rows = torch.arange(features.shape[0], device=features.device)
        return torch.stack(outputs, dim=1)[rows, subject_ids.long().flatten()]
