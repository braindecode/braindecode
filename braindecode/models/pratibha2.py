# Authors: Mahir Jain (mahir@mannas.ai)
#
# License: Apache-2.0
# Reference implementation and weights: https://huggingface.co/MannasAI/pratibha-2 (Apache-2.0).
"""PRATIBHA-2 (formerly MANAS-2): a dense-attention EEG foundation model."""

from functools import lru_cache, partial
from typing import Optional, Union

import numpy as np
import torch
from mne.channels import make_standard_montage
from torch import nn

from braindecode.functional import check_n_chans
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import warn_if_sfreq_differs
from braindecode.modules import (
    FeedForwardBlock,
    PreNormBlock,
    TimmAttention,
    resolve_channel_positions,
)
from braindecode.modules.channels import _reve_position_bank
from braindecode.modules.layers import RMSNorm
from braindecode.util import resolve_montage_name


class Pratibha2(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""PRATIBHA-2 (formerly MANAS-2) from Kulkarni et al. (2026) [manas2]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel`

    .. figure:: ../_static/model/pratibha2_arch.png
       :align: center
       :alt: MANAS-2 architecture
       :width: 1000px

       MANAS-2 architecture: patch and electrode-coordinate embeddings, a masked
       transformer encoder with query attention pooling, and the waveform, spectral
       and constrained-reconstruction (ConRec) decoders used for pretraining.
       Reproduced from Fig. 1 of [manas2]_ (CC BY-NC-ND 4.0).

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    A transformer encoder pretrained with masked autoencoding on clinical and
    research EEG: waveform and spectral reconstruction (Raw-Band Hybrid) with a
    constraint on the local RMS energy of the reconstructed waveform
    (Constrained Reconstruction). Every electrode-patch pair is one token and
    every layer attends densely over all tokens of the window::

        (batch, n_chans, n_times) -> per-channel z-score -> 1 s patches
        (0.1 s overlap) -> patch embedding + 4D position encoding -> dense
        transformer -> mean over tokens -> head -> (batch, n_outputs)

    .. rubric:: Macro Components

    ``Pratibha2.patch_embed``
        **Operations.** ``Linear(patch_size, embed_dim)`` on each patch.
        **Role.** One token per electrode and patch; the same weights serve
        every electrode.

    ``Pratibha2.pos_enc``
        **Operations.** Sines and cosines of ``n_freqs ** 4`` frequency
        combinations of the token's 4D position (electrode position in cm,
        patch index), projected to ``embed_dim``, plus a learned gated branch;
        the sum is RMS-normalized and added to the patch embedding.
        **Role.** Tells the attention each token's electrode and patch; this
        is how the montage enters the encoder.

    ``Pratibha2.blocks``
        **Operations.** ``depth`` pre-norm blocks (RMSNorm) of multi-head
        self-attention and a gated feed-forward network
        ``fc2(fc1(x) * (swish(g) + gelu_tanh(g)))``, ``g = fc_gate(x)``.
        **Role.** The pretrained backbone; :meth:`encode` returns its token grid.

    ``Pratibha2.final_norm``
        **Operations.** RMSNorm of every token, the last step of :meth:`encode`.
        **Role.** Normalizes the tokens that :meth:`encode` returns and the
        head pools.

    ``Pratibha2.final_layer``
        **Operations.** ``Linear -> head_activation -> Dropout -> Linear`` on
        the token mean. Not pretrained. **Role.** The classification head;
        train it for your task.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** 1 s patches with a learned linear embedding, ordered by the
      patch index of the position encoding.
    - **Spatial:** electrodes enter only through their 3D positions, so any
      montage the position table places can be read.
    - **Spectral:** learned by the band-power reconstruction target.

    .. rubric:: Additional Mechanisms

    - **Channel positions.** The pretraining electrode table, REVE's bank (part
      of the 10-05 family, all of BioSemi-128 and of GSN-HydroCel-129) plus
      PRATIBHA-2's own ``AFZ`` and ``IZ``, looked up by name (exact spelling
      first, then ignoring case, ``EEG``/``-REF``, ``T3``-style aliases, or the
      midpoint of a bipolar ``A-B``; see
      :func:`~braindecode.modules.resolve_channel_positions`). Other 10-05 names
      take their position in MNE's ``standard_1005`` montage frame, the frame of
      the table's 10-05 entries. Channel ``loc`` values are not used: they are in
      MNE's head frame, a few centimetres from the pretraining frame.
    - **Input scale.** With ``normalize_input=True`` each channel is z-scored
      within each window, with a floor on the standard deviation
      (``std_floor``), so volts and microvolts give the same output; on
      microvolt data it matches the preprocessing of the release's example,
      ``(x - mean) / (std + 1e-6)``.

    .. important::

        Resample to **200 Hz**; windows need at least ``patch_size`` samples.
        The released 92.8M-parameter encoder (``MannasAI/pratibha-2``,
        Apache-2.0, release ``863edc9``), converted to these parameter names,
        is on the Hugging Face Hub at ``braindecode/pratibha2-pretrained``; the
        head is not pretrained::

            from braindecode.models import Pratibha2
            model = Pratibha2.from_pretrained(
                "braindecode/pratibha2-pretrained",
                chs_info=raw.info["chs"], n_outputs=4, n_times=1000,
            )

    Parameters
    ----------
    patch_size : int
        Number of samples per patch (token length). 200 samples = 1 s at 200 Hz.
    patch_stride : int
        Step between consecutive patches, in samples.
    embed_dim : int
        Token embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    ffn_expansion : int
        Width multiplier of the gated feed-forward network.
    n_freqs : int
        Frequencies per axis of the Fourier position encoding.
    freq_range : tuple of float
        Lowest and highest of those frequencies, evenly spaced (``[1, 10]`` in
        release 863edc9).
    head_hidden_dim : int
        Hidden size of the classification head.
    normalize_input : bool
        If True, z-score each channel within each window before patching.
    std_floor : float
        Floor of a channel's standard deviation in that z-score, ``1e-12`` by
        default. The release's example adds 1e-6 to it instead, so volts and
        microvolts would give different tokens.
    head_activation : type[nn.Module]
        Activation of the classification head.
    drop_prob : float
        Dropout probability in the classification head.
    att_drop_prob : float
        Attention dropout probability (0 in the pretrained model).
    norm_eps : float
        Epsilon of every RMSNorm (``1e-8`` in ``model.py`` of release 863edc9).
    on_missing : str
        Channels the position table cannot place: ``"error"`` raises,
        ``"drop"`` leaves them out of the input, ``"nearest"`` takes the table
        entry closest to the channel's ``loc`` (approximate: ``loc`` is in a
        different frame).
    channel_strategy : str
        How another montage reaches the encoder; see
        :class:`~braindecode.models.base.EEGModuleMixin`. PRATIBHA-2 reads any
        montage natively, so the default ``"native"`` is usually what you want.
    channel_strategy_kwargs : dict or None
        Options of ``channel_strategy``.

    References
    ----------
    .. [manas2] Kulkarni, A., Mishra, A. R., Jain, M., Runwal, P., Saini, L.,
       Panwar, S., & Singh, S. (2026). MANAS-2: Constrained Reconstruction for
       EEG Foundation Models. arXiv:2609.13717.
       https://arxiv.org/abs/2609.13717
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: channel_positions

    channel_index: Optional[torch.Tensor]

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
        patch_size: int = 200,
        patch_stride: int = 180,
        embed_dim: int = 512,
        depth: int = 22,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        n_freqs: int = 4,
        freq_range: tuple[float, float] = (1.0, 10.0),
        head_hidden_dim: int = 128,
        normalize_input: bool = True,
        std_floor: float = 1e-12,
        head_activation: type[nn.Module] = nn.ELU,
        drop_prob: float = 0.3,
        att_drop_prob: float = 0.0,
        norm_eps: float = 1e-8,
        on_missing: str = "error",
        channel_strategy: str = "native",
        channel_strategy_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            channel_strategy=channel_strategy,
            channel_strategy_kwargs=channel_strategy_kwargs,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        warn_if_sfreq_differs("Pratibha2", self._sfreq, 200.0)
        # A window holds at least one patch; checked when its length is known.
        n_times_known = self._n_times is not None or (
            self._input_window_seconds is not None and self._sfreq is not None
        )
        if n_times_known and self.n_times < patch_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be at least patch_size ({patch_size})."
            )

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.normalize_input = normalize_input
        self.head_hidden_dim = head_hidden_dim
        self.head_activation = head_activation
        self.drop_prob = drop_prob
        self.std_floor = std_floor

        # positions: (n_kept_chans, 3) in m; kept: indices of the placed input channels.
        positions, kept = resolve_channel_positions(
            self.chs_info, _pratibha2_position_bank(), on_missing=on_missing
        )
        # channel_index: (n_kept_chans,) input channels the model reads; None reads all.
        dropped = len(kept) < len(self.chs_info)
        self.register_buffer(
            "channel_index", torch.tensor(kept) if dropped else None, persistent=False
        )
        # channel_positions: (n_kept_chans, 3), in the release's centimetres. Not
        # persistent: the pretrained weights must load onto any montage.
        positions_cm = torch.tensor(positions, dtype=torch.float32) * 100.0  # m to cm
        self.register_buffer(
            "channel_positions",
            positions_cm.to(torch.get_default_dtype()),
            persistent=False,
        )

        self.patch_embed = nn.Linear(patch_size, embed_dim, bias=False)
        self.pos_enc = _Pratibha2PositionalEncoding(
            embed_dim, n_freqs, freq_range, norm_eps
        )
        norm = partial(RMSNorm, eps=norm_eps)
        self.blocks = nn.ModuleList(
            [
                PreNormBlock(
                    embed_dim,
                    attn=TimmAttention(
                        embed_dim, num_heads, qkv_bias=True, attn_drop=att_drop_prob
                    ),
                    mlp=FeedForwardBlock(
                        embed_dim,
                        ffn_expansion,
                        drop_p=0.0,
                        activation=_SwishGelu,
                        gated=True,
                    ),
                    norm_layer=norm,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = norm(embed_dim)
        self.final_layer = self._build_head(self.n_outputs)
        self._min_n_times = self.patch_size
        self._n_times_step = self.patch_stride
        self._n_times_offset = self.patch_size
        self._snaps_in_prepare_inputs = "pad_only"

    def _build_head(self, n_outputs: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.head_hidden_dim),
            self.head_activation(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.head_hidden_dim, n_outputs),
        )

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new number of outputs.

        The encoder is left unchanged; the new head is randomly initialised.

        Parameters
        ----------
        n_outputs : int
            Number of outputs of the new head.
        """
        self._set_n_outputs(n_outputs)
        self.final_layer = self._build_head(n_outputs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG into the grid of token embeddings.

        This is the encoder output before pooling and the classification head.

        Parameters
        ----------
        x : torch.Tensor
            EEG of shape (batch, n_chans, n_times), sampled at 200 Hz.

        Returns
        -------
        torch.Tensor
            Tokens of shape (batch, n_kept_chans, n_patches, embed_dim), where
            ``n_kept_chans`` excludes channels dropped by ``on_missing="drop"``.
        """
        # x: (batch, n_chans, n_times), the montage the model was built for
        check_n_chans(x, self.n_chans, "Pratibha2")
        channel_index = self.channel_index
        if channel_index is not None:
            # leave out the channels the position table could not place
            # x: (batch, n_kept_chans, n_times)
            x = x.index_select(1, channel_index)
        if self.normalize_input:
            # z-score each channel within its window, in float32 or wider, so
            # that volts and microvolts give the same tokens:
            # x: (batch, n_kept_chans, n_times)
            x32 = x.to(torch.promote_types(x.dtype, torch.float32))
            mean = x32.mean(dim=-1, keepdim=True)
            std = x32.std(dim=-1, keepdim=True)
            x = ((x32 - mean) / std.clamp_min(self.std_floor)).to(x.dtype)
        # overlapping patches of patch_size samples:
        # (batch, n_kept_chans, n_patches, patch_size)
        patches = x.unfold(-1, self.patch_size, self.patch_stride)
        # one linear embedding shared by every patch:
        # (batch, n_kept_chans, n_patches, embed_dim)
        tokens = self.patch_embed(patches)
        batch, n_kept_chans, n_patches, _ = tokens.shape

        # position of each token, the electrode (cm) and the patch index; float32
        # or wider, as the position encoding takes angles:
        # (n_kept_chans, n_patches, 4)
        positions = self.channel_positions
        positions = positions.to(torch.promote_types(positions.dtype, torch.float32))
        time = torch.arange(n_patches, device=tokens.device, dtype=positions.dtype)
        coords = torch.cat(
            [
                # (n_kept_chans, n_patches, 3)
                positions[:, None, :].expand(-1, n_patches, -1),
                # (n_kept_chans, n_patches, 1)
                time[None, :, None].expand(n_kept_chans, -1, -1),
            ],
            dim=-1,
        )
        # add the encoding of each token's position:
        # (batch, n_kept_chans, n_patches, embed_dim)
        tokens = tokens + self.pos_enc(coords).unsqueeze(0).to(tokens.dtype)

        # one sequence of n_kept_chans * n_patches tokens per window, attended
        # densely: (batch, n_kept_chans * n_patches, embed_dim)
        tokens = tokens.reshape(batch, n_kept_chans * n_patches, self.embed_dim)
        # depth pre-norm blocks of dense attention, then the final RMSNorm:
        # (batch, n_kept_chans * n_patches, embed_dim)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        # back to the token grid: (batch, n_kept_chans, n_patches, embed_dim)
        return tokens.reshape(batch, n_kept_chans, n_patches, self.embed_dim)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Classify a batch of EEG windows.

        Tokens are averaged over electrodes and patches, then passed to the
        classification head ``final_layer``.

        Parameters
        ----------
        x : torch.Tensor
            EEG of shape (batch, n_chans, n_times), sampled at 200 Hz.
        return_features : bool
            If True, return a dict with the pooled embedding (``"features"``,
            shape (batch, embed_dim)) and the token grid (``"tokens"``, shape
            (batch, n_kept_chans, n_patches, embed_dim), see :meth:`encode`)
            instead of logits.

        Returns
        -------
        torch.Tensor or dict
            Logits of shape (batch, n_outputs), or the feature dict when
            ``return_features`` is True.
        """
        # tokens: (batch, n_kept_chans, n_patches, embed_dim)
        tokens = self.encode(x)
        # mean over electrodes and patches: (batch, embed_dim)
        features = tokens.mean(dim=(1, 2))
        if return_features:
            return {"features": features, "tokens": tokens, "cls_token": None}
        # logits: (batch, n_outputs)
        return self.final_layer(features)


class _SwishGelu(nn.Module):
    """``swish(x) + gelu_tanh(x)``, the gate activation of PRATIBHA-2.

    Elementwise: the output has the shape of ``x``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x) + nn.functional.gelu(x, approximate="tanh")


# The pretraining electrode table is REVE's bank plus these two entries (m), which
# sit 1.43 and 0.74 cm from 10-05 AFz and Iz (release 863edc9). They come last, so
# a name that matches only ignoring case (``afz``) takes them, as in the release.
_EXTRA_POSITIONS = {
    "AFZ": (0.0010058004409074783, 0.07728150114417076, 0.021591500379145145),
    "IZ": (-3.840029239654541e-05, -0.1144150011241436, -0.029236000031232834),
}


@lru_cache(maxsize=1)
def _pratibha2_position_bank() -> dict[str, np.ndarray]:
    """Name -> position (m) of the pretraining table, in MNE's montage frame.

    The table is REVE's bank plus ``_EXTRA_POSITIONS``. It is laid over MNE's
    ``standard_1005`` montage, which places the 10-05 names the table lacks in the
    frame of its 10-05 entries.
    """
    montage = make_standard_montage(resolve_montage_name("standard_1005"))
    return {
        **montage.get_positions()["ch_pos"],
        **_reve_position_bank(),
        **_EXTRA_POSITIONS,
    }


class _Pratibha2PositionalEncoding(nn.Module):
    """4D position encoding: Fourier features plus a learned gated branch.

    Kept private: no shared block computes this encoding. ``coords``
    (C, P, 4), electrode position in cm and patch index, gives (C, P, embed_dim):
    the sines and cosines of the phases ``coords @ freq_matrix`` through
    ``fourier_linear``, plus ``learned_norm(value * gate_activation(gate))`` from
    ``learned_linear``, then ``final_norm``.
    """

    def __init__(
        self,
        embed_dim: int,
        n_freqs: int,
        freq_range: tuple[float, float],
        norm_eps: float,
    ):
        super().__init__()
        # freq_matrix: (4, n_freqs ** 4), every combination of the per-axis frequencies.
        freqs = torch.linspace(*freq_range, n_freqs)
        grid = torch.cartesian_prod(freqs, freqs, freqs, freqs).reshape(-1, 4)
        self.register_buffer("freq_matrix", grid.T.contiguous(), persistent=False)
        self.fourier_linear = nn.Linear(2 * n_freqs**4, embed_dim, bias=False)
        self.learned_linear = nn.Linear(4, 2 * embed_dim, bias=False)
        self.gate_activation = _SwishGelu()
        self.learned_norm = RMSNorm(embed_dim, eps=norm_eps)
        self.final_norm = RMSNorm(embed_dim, eps=norm_eps)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # bfloat16 cannot represent the angles.
        angle_dtype = torch.promote_types(coords.dtype, torch.float32)
        # phases: (C, P, n_freqs ** 4). Autocast lowers matmul precision even
        # for two float32 operands, hence the elementwise product and sum.
        phases = (
            coords.to(angle_dtype).unsqueeze(-1) * self.freq_matrix.to(angle_dtype)
        ).sum(dim=-2)
        weight_dtype = self.fourier_linear.weight.dtype
        # fourier: (C, P, embed_dim)
        fourier = self.fourier_linear(
            torch.cat([phases.sin(), phases.cos()], dim=-1).to(weight_dtype)
        )
        # value, gate: (C, P, embed_dim) each
        value, gate = self.learned_linear(coords.to(weight_dtype)).chunk(2, dim=-1)
        # gated learned branch, RMS-normalized: (C, P, embed_dim)
        learned = self.learned_norm(value * self.gate_activation(gate))
        # sum of the two branches, RMS-normalized: (C, P, embed_dim)
        return self.final_norm(fourier + learned)
