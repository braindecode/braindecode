# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Meta Platforms, Inc. and affiliates (original DANCE)
#
# License: MIT
# Adapted from https://github.com/facebookresearch/dance (MIT).
"""``DANCE``: detect-and-classify EEG events (DETR for 1-D EEG)."""

from __future__ import annotations

import warnings

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    _disable_batch_norm_training_if_batch_size_one,  # decorates ``forward``
)
from braindecode.models.util import has_valid_locations as _has_valid_locations
from braindecode.models.util import positions_from_chs_info as _positions_from_chs_info
from braindecode.modules import ChannelMerger, Perceiver, SimpleConv
from braindecode.modules.dance_modules import DanceDetrDecoder


class DANCE(EEGModuleMixin, nn.Module):
    r"""DANCE from Lévy et al (2026) [dance]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel`

    DANCE frames EEG decoding as event *set prediction*: a long, unaligned
    window is mapped to a set of events ``(t_start, t_end, class)`` with
    normalized ``[0, 1]`` spans -- DETR for 1-D EEG.

    .. rubric:: Architecture Overview

    ``ChannelMerger`` (spatial Fourier attention over electrode positions) ->
    ``SimpleConv`` (dilated conv stack) -> ``Perceiver`` (cross-attention to a
    fixed 256-latent grid) -> a dense head (``forward`` output) and a DETR
    cross-attention decoder (``detect`` output).

    .. rubric:: Macro Components

    ``DANCE.conv.merger`` (Spatial ChannelMerger)
        **Operations.** :class:`~braindecode.modules.ChannelMerger` Fourier-embeds
        each electrode's ``(x, y)`` position (:class:`~braindecode.modules.FourierEmb`)
        and computes ``n_virtual_channels`` softmax-attention combinations of the
        input channels, mapping ``(B, n_chans, T) -> (B, n_virtual_channels, T)``.
        Nested inside ``self.conv`` (matching upstream); disabled (``None``) when
        ``chs_info`` has no usable locations. Controlled by ``merger_drop_prob``.
        **Role.** Makes the model montage-agnostic by projecting any electrode
        layout onto a fixed virtual-channel basis.

    ``DANCE.conv`` (SimpleConv dilated front-end)
        **Operations.** :class:`~braindecode.modules.SimpleConv` runs the nested
        merger, a ``1x1`` ``initial_linear`` projection, then ``conv_depth``
        residual dilated ``Conv1d`` blocks (``nn.ReLU``, dilation accumulating as
        ``int(dilation * conv_dilation_growth)`` per block, same-padding preserves
        ``T``), mapping ``-> (B, embed_dim, T)``.
        **Role.** Builds the temporal feature representation (Défossez lineage).

    ``DANCE.perceiver`` (Perceiver bottleneck)
        **Operations.** :class:`~braindecode.modules.Perceiver` Fourier-encodes the
        time axis, then cross-attends a fixed ``num_latents``-token learnable grid to
        the conv features over ``perceiver_depth`` blocks, mapping
        ``(B, T, embed_dim) -> (B, num_latents, embed_dim)``.
        **Role.** Makes the model length-agnostic (any ``T`` -> ``num_latents``
        tokens) and forms the detection time-grid (``num_latents / duration``
        tokens/s).

    ``DANCE.decoder`` (DETR cross-attention decoder)
        **Operations.** ``DanceDetrDecoder`` projects the latents to
        ``decoder_dim``, then ``decoder_depth`` self-/cross-attention layers update
        ``n_queries`` learnable event queries; per-query heads emit class logits and
        sigmoid ``start``/``end`` spans. Used only by :meth:`detect` (not
        :meth:`forward`).
        **Role.** Produces the event-set prediction ``{class, start, end}``.

    ``DANCE.final_layer`` (dense per-token head)
        **Operations.** ``nn.Linear(embed_dim, n_outputs)`` applied to every latent
        token, mapping ``(B, num_latents, embed_dim) -> (B, num_latents, n_outputs)``.
        **Role.** The :meth:`forward` output (dense per-token class logits) and the
        ``dense`` term consumed by the consistency loss.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    Temporal context comes from the dilated conv stack; spatial structure from
    the Fourier-position channel merge; the Perceiver fourier-encodes the time
    axis before cross-attention.

    .. rubric:: Additional Mechanisms

    The DETR decoder uses learnable event queries and emits per-query class
    logits plus sigmoid start/end spans. The dense head provides per-token
    class logits used as ``forward``'s output and for the consistency loss.

    .. versionadded:: 1.6.1

    Parameters
    ----------
    n_queries : int, optional
        Number of learnable event queries. The default is ``100``.
    use_channel_merger : bool, optional
        Enable the spatial Fourier ChannelMerger. Auto-disabled if ``chs_info``
        has no usable electrode locations. The default is ``True``.
    n_virtual_channels : int, optional
        Merger output channels. The default is ``270``.
    fourier_emb_dim : int, optional
        Fourier position embedding dim. The default is ``2048``.
    merger_drop_prob : float, optional
        Spatial dropout of the ChannelMerger ONLY (``self.conv.merger``);
        bans whole channels within a random radius during training. The
        default is ``0.2``.
    embed_dim : int, optional
        Conv/Perceiver feature dim. The default is ``128``.
    conv_hidden : int, optional
        SimpleConv hidden width. The default is ``512``.
    conv_depth : int, optional
        Number of dilated conv blocks. The default is ``10``.
    conv_kernel_size : int, optional
        Conv kernel size. The default is ``9``.
    conv_dilation_growth : float, optional
        Per-block dilation growth. The default is ``2.5``.
    conv_initial_linear : int, optional
        1x1 projection width. The default is ``256``.
    conv_initial_depth : int, optional
        Number of 1x1 projections. The default is ``1``.
    conv_drop_prob : float, optional
        Conv-block dropout knob, exposed for API symmetry. NOTE: the verified
        DANCE config has NO conv dropout, so this maps to no upstream weights
        and does not affect parity; kept for forward-compatibility. The
        default is ``0.2`` (the paper value).
    num_latents : int, optional
        Perceiver latent count (detection token grid). The default is ``256``.
    perceiver_depth : int, optional
        Perceiver cross-attn blocks. The default is ``6``.
    cross_attn_heads : int, optional
        Perceiver cross-attn heads. The default is ``2``.
    latent_attn_heads : int, optional
        Perceiver self-attn heads. The default is ``2``.
    cross_dim_head : int, optional
        Perceiver cross-attn head dim. The default is ``64``.
    latent_dim_head : int, optional
        Perceiver self-attn head dim. The default is ``64``.
    max_freq : float, optional
        Perceiver fourier max frequency. The default is ``10.0``.
    num_freq_bands : int, optional
        Perceiver fourier bands. The default is ``6``.
    decoder_dim : int, optional
        DETR decoder dim. The default is ``256``.
    decoder_depth : int, optional
        DETR decoder layers. The default is ``4``.
    decoder_heads : int, optional
        DETR decoder heads. The default is ``4``.
    activation : type[nn.Module], optional
        Accepted for interface symmetry but currently INERT: it is forwarded to
        ``self.decoder`` yet the decoder feed-forward hardwires GEGLU
        (:class:`~braindecode.modules.dance_modules._FeedForward`), and the
        ``SimpleConv`` front-end hardcodes ``nn.ReLU`` to match upstream. No
        submodule reads it today. The default is ``nn.GELU``.
    drop_prob : float, optional
        Dropout applied to the raw input (``self.input_drop``) and inside the
        DETR decoder (``self.decoder``). Does NOT touch the merger (use
        ``merger_drop_prob``) or the conv stack (no conv dropout upstream).
        The default is ``0.1``.

    References
    ----------
    .. [dance] Lévy, Banville, Rapin, King, Moreau, d'Ascoli (2026). DANCE:
       Detect and Classify Events in EEG. arXiv:2605.10688.
    .. [defossez2023] Défossez et al. (2023). Decoding speech from
       non-invasive brain recordings.
    .. [perceiver2021] Jaegle et al. (2021). Perceiver: General perception
       with iterative attention.
    .. [detr2020] Carion et al. (2020). End-to-end object detection with
       transformers.
    """

    # No `self.mapping` (B10): it renames keys across braindecode-published
    # checkpoints, and DANCE has none. The upstream(neuraltrain)->local parity
    # weight map is a separate concern living in scripts/dance_parity_check.py.

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        n_queries: int = 100,
        use_channel_merger: bool = True,
        n_virtual_channels: int = 270,
        fourier_emb_dim: int = 2048,
        merger_drop_prob: float = 0.2,
        embed_dim: int = 128,
        conv_hidden: int = 512,
        conv_depth: int = 10,
        conv_kernel_size: int = 9,
        conv_dilation_growth: float = 2.5,
        conv_initial_linear: int = 256,
        conv_initial_depth: int = 1,
        conv_drop_prob: float = 0.2,
        num_latents: int = 256,
        perceiver_depth: int = 6,
        cross_attn_heads: int = 2,
        latent_attn_heads: int = 2,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        max_freq: float = 10.0,
        num_freq_bands: int = 6,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 4,
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
        del n_outputs, n_chans, sfreq, n_times, input_window_seconds, chs_info

        self.num_latents = num_latents
        self.n_queries = n_queries
        self.embed_dim = embed_dim

        # Decide the merger up front (fall back to no-merger if no locations).
        if use_channel_merger and not _has_valid_locations(self.chs_info):
            warnings.warn(
                "DANCE: chs_info has no usable electrode locations "
                "('loc' missing or all-zero); disabling the ChannelMerger "
                "(use_channel_merger=False).",
                UserWarning,
            )
            use_channel_merger = False
        self.use_channel_merger = use_channel_merger

        # The merger is NESTED INSIDE SimpleConv (self.conv.merger), matching
        # upstream SimpleConvModel's call path and state_dict layout. There is
        # NO top-level self.channel_merger. The (n_chans, 2) positions buffer is
        # derived from chs_info and broadcast in forward; subject_ids is always
        # None (braindecode has no subjects; per_subject=False).
        merger = None
        if use_channel_merger:
            positions = _positions_from_chs_info(self.chs_info)
            self.register_buffer(
                "channel_positions",
                torch.as_tensor(positions, dtype=torch.float32),
                persistent=False,
            )
            merger = ChannelMerger(
                out_channels=n_virtual_channels,
                pos_dim=fourier_emb_dim,
                dropout=merger_drop_prob,
            )

        self.input_drop = nn.Dropout(drop_prob)
        self.conv = SimpleConv(
            in_channels=self.n_chans,
            out_channels=embed_dim,
            hidden=conv_hidden,
            depth=conv_depth,
            kernel_size=conv_kernel_size,
            dilation_growth=conv_dilation_growth,
            initial_linear=conv_initial_linear,
            initial_depth=conv_initial_depth,
            drop_prob=conv_drop_prob,
            activation=nn.ReLU,
            merger=merger,
        )
        # The conv stack is SAME-padded, so it preserves length for any T >= 1;
        # the dilated receptive field is not an input-length requirement. The
        # genuine minimum is one first-block kernel (dilation 1), so the guard
        # uses ``conv_kernel_size`` rather than the full receptive field.
        self._min_n_times = conv_kernel_size
        self.perceiver = Perceiver(
            input_dim=embed_dim,
            num_latents=num_latents,
            latent_dim=embed_dim,
            depth=perceiver_depth,
            cross_heads=cross_attn_heads,
            latent_heads=latent_attn_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            max_freq=max_freq,
            num_freq_bands=num_freq_bands,
        )
        self.decoder = DanceDetrDecoder(
            input_dim=embed_dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            n_queries=n_queries,
            n_outputs=self.n_outputs,
            drop_prob=drop_prob,
            activation=activation,
        )
        # final_layer LAST so it lands in the last two named_children(); init
        # weights AFTER it exists so the dense head gets the custom init too.
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-2] != self.n_chans:
            raise ValueError(
                f"expected (batch, {self.n_chans}, T) input; got {tuple(x.shape)}."
            )
        if x.shape[-1] < self._min_n_times:
            raise ValueError(
                f"n_times={x.shape[-1]} is shorter than the minimum input "
                f"length ({self._min_n_times} samples = one conv kernel)."
            )
        x = self.input_drop(x)
        # The merger is nested inside self.conv; pass positions through it.
        # subject_ids is always None (braindecode has no subjects).
        if self.conv.merger is not None:
            pos = self.channel_positions.unsqueeze(0).expand(x.size(0), -1, -1)
            x = self.conv(x, positions=pos)  # merger -> initial_linear -> blocks
        else:
            x = self.conv(x)  # (B, embed_dim, T)
        x = x.transpose(2, 1)  # (B, T, embed_dim)
        x = self.perceiver(x)  # (B, num_latents, embed_dim)
        return x

    @_disable_batch_norm_training_if_batch_size_one
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = self._encode(x)  # (B, num_latents, embed_dim)
        return self.final_layer(latents)  # (B, num_latents, n_outputs)

    @_disable_batch_norm_training_if_batch_size_one
    def detect(self, x: torch.Tensor) -> dict:
        latents = self._encode(x)  # (B, num_latents, embed_dim)
        events = self.decoder(latents)  # {class, start, end}
        events["dense"] = self.final_layer(latents)  # (B, num_latents, n_outputs)
        return events

    def reset_head(self, n_outputs: int) -> None:
        """Replace the dense head and the DETR class head for a new ``n_outputs``."""
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        old = self.final_layer
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        ch = self.decoder.class_head
        self.decoder.class_head = nn.Linear(ch.in_features, n_outputs).to(
            device=ch.weight.device, dtype=ch.weight.dtype
        )
        self._n_outputs = n_outputs
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None and "n_outputs" in init_kwargs:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None and "n_outputs" in hub_config:
            hub_config["n_outputs"] = n_outputs
