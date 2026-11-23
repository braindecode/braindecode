# Authors: Dulhan Jayalath and Oiwi Parker Jones (MEG-XL)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Adapted from https://github.com/neural-processing-lab/MEG-XL (MIT); its frozen
# BioCodec tokenizer is https://github.com/klean2050/BioCodec-release (MIT).
from __future__ import annotations

import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from braindecode.functional import check_n_chans
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import geometry_from_chs_info, warn_if_sfreq_differs
from braindecode.modules import SEANetEncoder, SpatialTemporalBlock


class MEGXL(EEGModuleMixin, nn.Module, license="mit"):
    r"""MEG-XL from Jayalath and Parker Jones (2026) [megxl]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: ../_static/model/megxl_arch.png
       :align: center
       :alt: MEG-XL architecture
       :width: 1000px

       MEG-XL pretraining: per-channel BioCodec tokenization, input embedding
       construction and temporal block masking, reproduced from [megxl]_
       (CC BY 4.0).

    .. rubric:: Architecture Overview

    A frozen BioCodec tokenizer turns every channel into discrete codes; a
    criss-cross transformer contextualizes them over time and sensors::

        (batch, n_chans, n_times) -> BioCodec (SEANet + RVQ codes) -> codebook
        embeddings -> projection + sensor embeddings -> criss-cross blocks ->
        mean over time -> (batch, n_chans * latent_dim) -> (batch, n_outputs)

    .. rubric:: Macro Components

    ``MEGXL.encoder`` and ``MEGXL.quantizer``
        **Operations.** Each channel is divided by its RMS, a causal SEANet
        encoder downsamples it by ``prod(ratios)``, and ``n_q`` residual
        codebooks pick their entry by cosine similarity. **Role.** Frozen
        BioCodec tokenizer (no gradients, no codebook updates).

    ``MEGXL.rvq_projector``
        **Operations.** The ``n_q`` codebook vectors of a token are
        concatenated and projected to ``latent_dim``.

    ``MEGXL.position_projector``, ``MEGXL.orientation_projector``, ``MEGXL.sensor_type_layer``
        **Operations.** Gaussian random Fourier features of the sensor
        position and orientation, each projected to ``latent_dim``, plus a
        gradiometer/magnetometer embedding, added to every token of the channel.
        **Role.** Montage-agnostic channel identity.

    ``MEGXL.blocks``
        **Operations.** Half of the features attend over time (RoPE), the other
        half over the channels, then a feed-forward layer (BrainOmni's block).

    ``MEGXL.final_layer``
        **Operations.** Linear layer on the time-averaged, flattened
        ``n_chans * latent_dim`` features. **Role.** Classification head (not
        pretrained).

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** ``ceil(n_times / 12)`` tokens of 240 ms at 50 Hz (the
      SEANet pads the end, as BioCodec does); RoPE attention spans the whole
      window (150 s in pretraining).
    - **Spatial:** attention over the channels, keyed by sensor geometry.
    - **Spectral:** learned implicitly by the tokenizer's convolutions.

    .. rubric:: Additional Mechanisms

    - Sensor geometry comes from ``chs_info`` as in the authors'
      preprocessing: ``loc[:3]`` centred and divided by
      ``sqrt(3 * mean(|r|^2))``, the coil x-axis of planar gradiometers or the
      coil normal of other coils as orientation. MEG coils whose name contains
      ``MAG`` are magnetometers, the other MEG coils gradiometers. EEG
      channels also get the gradiometer embedding, with zero orientation and
      positions normalised apart from the MEG ones; other channel kinds raise
      a ``ValueError``. MEG-XL is pretrained on MEG only.
    - The RoPE table holds ``n_rope_positions`` = 625 positions, those of the
      150 s pretraining window (``torch.polar`` values in the converted
      weights); a longer window rebuilds it from ``cos``/``sin``, one ulp away
      from ``torch.polar``.
    - The released attention keeps its dropout (0.1) active in eval mode; here
      dropout is off in eval mode, so the eval forward is deterministic.
    - :meth:`encode` and :meth:`encode_codes` accept a temporal code-corruption
      ``mask`` and caller-owned ``mask_value`` before metadata and context.
      They return actual temporal hidden states for native masked RVQ-code
      prediction. The mask sampler, learned mask vector and prediction head
      belong to the caller, not the default encoder checkpoint. Channel padding
      (``ch_mask``) is separate and remains unsupported.

    .. important::

        Weights converted from the released ``meg-xl-med-v2.ckpt``
        (``pnpl/MEG-XL``, step 113000, with its frozen BioCodec tokenizer) are
        on the Hugging Face Hub at ``braindecode/megxl-pretrained`` (revision
        ``0446ce5d``); the head is not pretrained::

            from braindecode.models import MEGXL
            model = MEGXL.from_pretrained("braindecode/megxl-pretrained", chs_info=raw.info["chs"], n_outputs=2)

        Input is expected at 50 Hz, preprocessed as in the authors' code
        (0.1-40 Hz band-pass, baseline correction and robust scaling per 3 s
        chunk, clipped to ``[-5, 5]``).

    .. versionadded:: 2.1

    `License <https://github.com/neural-processing-lab/MEG-XL/blob/main/LICENSE>`_

    Parameters
    ----------
    latent_dim : int
        Transformer dimension.
    num_layers : int
        Number of criss-cross blocks.
    num_heads : int
        Attention heads (even: half temporal, half spatial).
    drop_prob : float
        Dropout of the attention and feed-forward layers.
    n_rope_positions : int
        Positions cached by the rotary embedding of the temporal attention: 625
        for the 150 s pretraining window (150 s x 50 Hz / 12 samples per token).
        It sets the shape of the saved ``rotate`` tables; a longer window
        rebuilds them.
    fourier_pos_dim : int
        Size of the Fourier features of the sensor position and orientation.
    position_scale : float
        Standard deviation of the position Fourier frequencies.
    orientation_scale : float
        Standard deviation of the orientation Fourier frequencies.
    tokenizer_dim : int
        SEANet output dimension.
    n_filters : int
        Base number of SEANet filters.
    ratios : tuple of int
        SEANet downsampling ratios.
    kernel_size : int
        Kernel size of the first SEANet convolution. Frozen with the release:
        the pretrained tokenizer expects 3.
    last_kernel_size : int
        Kernel size of the last SEANet convolution. Frozen with the release: 3.
    n_residual_layers : int
        Residual blocks before each SEANet downsampling. Frozen with the
        release: 2.
    n_q : int
        Number of residual codebooks.
    codebook_size : int
        Entries per codebook.
    codebook_dim : int
        Codebook dimension.
    min_rms : float
        Floor of the per-channel RMS that the tokenizer divides by (1e-8), so that
        a flat channel is not divided by zero.

    References
    ----------
    .. [megxl] Jayalath, D., Parker Jones, O. (2026). MEG-XL: Data-Efficient
       Brain-to-Text via Long-Context Pre-Training. ICML 2026.
       https://arxiv.org/abs/2602.02494
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: final_layer.weight

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
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        drop_prob: float = 0.1,
        n_rope_positions: int = 625,
        fourier_pos_dim: int = 250,
        position_scale: float = 1.8,
        orientation_scale: float = 1.0,
        tokenizer_dim: int = 128,
        n_filters: int = 32,
        ratios: tuple[int, ...] = (3, 2, 2),
        kernel_size: int = 3,
        last_kernel_size: int = 3,
        n_residual_layers: int = 2,
        n_q: int = 6,
        codebook_size: int = 256,
        codebook_dim: int = 16,
        min_rms: float = 1e-8,
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
        # The released checkpoint runs at 50 Hz (the authors' preprocessing).
        warn_if_sfreq_differs("MEGXL", self._sfreq, 50.0)
        if num_heads % 2 or latent_dim % num_heads or (latent_dim // num_heads) % 2:
            raise ValueError(
                f"num_heads ({num_heads}) must be even and divide latent_dim "
                f"({latent_dim}) into even head dimensions."
            )
        self.latent_dim = latent_dim
        self.min_rms = min_rms  # read by tokenize
        # Derived from chs_info, so not saved: one checkpoint serves any montage.
        pos, sensor_type = geometry_from_chs_info(self.chs_info)
        pos = torch.from_numpy(pos)
        self.register_buffer("sensor_xyz", pos[:, :3], persistent=False)
        self.register_buffer("sensor_abc", pos[:, 3:], persistent=False)
        # BrainOmni codes MAG as 1; MEG-XL types sensors GRAD (0) or MAG (1).
        sensor_type = torch.from_numpy(sensor_type == 1).long()
        self.register_buffer("sensor_type", sensor_type, persistent=False)

        # BioCodec: causal SEANet, two dilated residual blocks per stage.
        self.encoder = SEANetEncoder(
            tokenizer_dim,
            n_filters,
            ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_residual_layers=n_residual_layers,
            compress=1,
            true_skip=True,
            causal=True,
        )
        self.quantizer = _BioCodecQuantizer(
            tokenizer_dim, codebook_dim, codebook_size, n_q
        )
        self.encoder.requires_grad_(False)
        self.quantizer.requires_grad_(False)

        self.rvq_projector = nn.Linear(n_q * codebook_dim, latent_dim)
        # Fixed random Fourier frequencies (saved with the weights).
        self.register_buffer(
            "position_freqs", torch.randn(3, fourier_pos_dim // 2) * position_scale
        )
        self.position_projector = nn.Linear(fourier_pos_dim, latent_dim)
        self.register_buffer(
            "orientation_freqs",
            torch.randn(3, fourier_pos_dim // 2) * orientation_scale,
        )
        self.orientation_projector = nn.Linear(fourier_pos_dim, latent_dim)
        self.sensor_type_layer = nn.Embedding(2, latent_dim)
        self.blocks = nn.ModuleList(
            [
                SpatialTemporalBlock(latent_dim, num_heads, drop_prob, n_rope_positions)
                for _ in range(num_layers)
            ]
        )
        self.flatten_sensor_batch = Rearrange(
            "batch chans coords -> (batch chans) coords"
        )
        self.restore_sensor_batch = Rearrange(
            "(batch chans) features -> batch chans features", chans=self.n_chans
        )
        self.final_layer = nn.Linear(self.n_chans * latent_dim, self.n_outputs)
        self._min_n_times = 1

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for ``n_outputs`` classes."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.n_chans * self.latent_dim, n_outputs)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """BioCodec codes ``(batch, n_chans, n_tokens, n_q)`` of ``x``."""
        batch, chans, _ = x.shape
        # (batch, n_chans, n_times) -> (batch * n_chans, 1, n_times)
        x = x.reshape(batch * chans, 1, -1)
        # Each channel is scaled to unit RMS over the window.
        # (batch * n_chans, 1, 1)
        scale = x.pow(2).mean(dim=2, keepdim=True).sqrt().clamp(min=self.min_rms)
        # (batch * n_chans, 1, n_times) -> (batch * n_chans, tokenizer_dim, n_tokens)
        # -> (batch * n_chans, n_tokens, tokenizer_dim)
        residual = self.encoder(x / scale).transpose(1, 2)
        # (batch * n_chans, n_tokens, tokenizer_dim) -> (batch * n_chans, n_tokens, n_q)
        codes = self.quantizer(residual)
        # (batch * n_chans, n_tokens, n_q) -> (batch, n_chans, n_tokens, n_q)
        return codes.reshape(batch, chans, -1, codes.shape[-1])

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contextual tokens ``(batch, n_chans, n_tokens, latent_dim)`` of ``x``.

        ``x`` is ``(batch, n_chans, n_times)``, with the same length for every
        sample and the full ``chs_info`` montage: padding a shorter recording
        would change its RMS scale, convolutions and LSTM state. ``n_tokens =
        ceil(n_times / prod(ratios))``: a partial final hop adds a boundary token,
        at index ``n_times // prod(ratios)``, which a loss on complete hops leaves
        out. ``mask`` and ``mask_value`` hide code content, see
        :meth:`encode_codes`; neighbouring codes still carry that content in
        their receptive fields.
        """
        check_n_chans(
            x,
            self.n_chans,
            "MEGXL",
            "build the model with the chs_info of this montage.",
        )
        with torch.no_grad():
            codes = self.tokenize(x)  # (batch, n_chans, n_tokens, n_q)
        return self.encode_codes(codes, mask, mask_value)

    def encode_codes(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contextual tokens ``(batch, n_chans, n_tokens, latent_dim)`` of BioCodec codes.

        Tokenizing once lets a self-supervised loss reuse the clean ``codes`` as
        targets. There is no padding mask: batch recordings of equal length.

        Parameters
        ----------
        codes : torch.Tensor
            Integer codes ``(batch, n_chans, n_tokens, n_q)`` from :meth:`tokenize`.
        mask : torch.Tensor, optional
            Boolean ``(batch, n_chans, n_tokens)``: True replaces the projected
            code content of that token by ``mask_value``, before the sensor
            metadata is added and before attention. Masked tokens still take
            part in attention.
        mask_value : torch.Tensor, optional
            Replacement vector ``(latent_dim,)``, required with ``mask``; the
            caller owns it, and any prediction head.

        Returns
        -------
        torch.Tensor
            The tokens ``(batch, n_chans, n_tokens, latent_dim)``. No parameter is
            added to the model.
        """
        # (batch, n_chans, n_tokens, n_q) -> (batch, n_chans, n_tokens, n_q, codebook_dim)
        # -> (batch, n_chans, n_tokens, n_q * codebook_dim)
        # -> (batch, n_chans, n_tokens, latent_dim)
        h = self.rvq_projector(self.quantizer.embed(codes).flatten(3))
        if mask is not None:
            if mask.dtype != torch.bool or list(mask.shape) != list(codes.shape[:3]):
                raise ValueError("mask must be boolean (batch, n_chans, n_tokens).")
            if mask_value is None:
                raise ValueError("mask_value is required with mask.")
            if list(mask_value.shape) != [self.latent_dim]:
                raise ValueError("mask_value must have shape (latent_dim,).")
            # (batch, n_chans, n_tokens, 1), (latent_dim,)
            # -> (batch, n_chans, n_tokens, latent_dim)
            h = torch.where(mask.unsqueeze(-1), mask_value, h)
        # Keep the original batched projections for bit-exact geometry gradients.
        # (n_chans, 3) -> (batch, n_chans, 3) -> (batch * n_chans, 3)
        xyz = self.sensor_xyz.expand(codes.shape[0], -1, -1)
        xyz = self.flatten_sensor_batch(xyz)
        # (n_chans, 3) -> (batch, n_chans, 3) -> (batch * n_chans, 3)
        abc = self.sensor_abc.expand(codes.shape[0], -1, -1)
        abc = self.flatten_sensor_batch(abc)
        # (n_chans,) -> (batch, n_chans)
        sensor_type = self.sensor_type.expand(codes.shape[0], -1)
        # (batch * n_chans, 3) -> (batch * n_chans, fourier_pos_dim)
        # -> (batch, n_chans, fourier_pos_dim)
        pos = _fourier_features(xyz, self.position_freqs)
        pos = self.restore_sensor_batch(pos)
        # (batch * n_chans, 3) -> (batch * n_chans, fourier_pos_dim)
        # -> (batch, n_chans, fourier_pos_dim)
        ori = _fourier_features(abc, self.orientation_freqs)
        ori = self.restore_sensor_batch(ori)
        # Each sensor embedding (batch, n_chans, latent_dim) -> (batch, n_chans, 1,
        # latent_dim) is added to every token; h stays (batch, n_chans, n_tokens,
        # latent_dim).
        h = h + self.position_projector(pos).unsqueeze(2)
        h = h + self.orientation_projector(ori).unsqueeze(2)
        h = h + self.sensor_type_layer(sensor_type).unsqueeze(2)
        for block in self.blocks:
            h = block(h)  # (batch, n_chans, n_tokens, latent_dim)
        return h

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Classify ``x`` or return the pooled features.

        There is no ``ch_mask``: the head flattens the ``n_chans * latent_dim``
        features and the sensor geometry is the fixed ``chs_info`` montage, so
        every sample needs all ``n_chans`` channels.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If True, return ``{"features": (batch, n_chans * latent_dim),
            "cls_token": None}`` instead of the logits.
        """
        # (batch, n_chans, n_times) -> (batch, n_chans, n_tokens, latent_dim)
        # -> mean over time (batch, n_chans, latent_dim)
        # -> (batch, n_chans * latent_dim)
        feat = self.encode(x).mean(dim=2).flatten(1)
        if return_features:
            out: Dict[str, Optional[torch.Tensor]] = {
                "features": feat,
                "cls_token": None,  # nosec B105
            }
            return out
        return self.final_layer(feat)  # (batch, n_outputs)


# Helpers, below the model class.


def _fourier_features(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Gaussian random Fourier features ``[sin, cos](2 pi x @ freqs)``."""
    # (n, 3) @ (3, fourier_pos_dim // 2) -> (n, fourier_pos_dim // 2)
    proj = 2 * math.pi * (x @ freqs)
    # -> (n, fourier_pos_dim)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class _BioCodecLevel(nn.Module):
    """One residual level of the BioCodec quantizer: its projections and codebook.

    ``project_in`` maps ``(..., dim)`` to ``(..., codebook_dim)``, ``project_out``
    back; ``codebook`` is ``(codebook_size, codebook_dim)``, a buffer that is zero
    until a checkpoint fills it. The attribute names are the checkpoint's keys.
    """

    def __init__(self, dim: int, codebook_dim: int, codebook_size: int):
        super().__init__()
        self.project_in = nn.Linear(dim, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, dim)
        self.register_buffer("codebook", torch.zeros(codebook_size, codebook_dim))


class _BioCodecQuantizer(nn.Module):
    """Frozen residual vector quantizer of BioCodec: codes by cosine similarity.

    Not the shared :class:`~braindecode.modules.ResidualVectorQuantizer`, which
    picks entries by Euclidean distance and carries the EMA training state.

    Steps of ``forward``, for each of the ``n_q`` levels in turn:

    1. ``project_in`` maps the residual to ``codebook_dim``; it and the codebook
       entries are L2-normalized.
    2. The code is the entry with the highest cosine similarity; the scores are
       computed in at least float32, also in a half-precision model.
    3. The chosen entry, as stored (not normalized), is mapped back by
       ``project_out`` and subtracted from the residual for the next level.

    ``forward`` maps the residual ``(n, n_tokens, dim)`` to the integer codes
    ``(n, n_tokens, n_q)``; ``embed`` maps codes ``(..., n_q)`` to their entries
    ``(..., n_q, codebook_dim)``.
    """

    def __init__(self, dim: int, codebook_dim: int, codebook_size: int, n_q: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_BioCodecLevel(dim, codebook_dim, codebook_size) for _ in range(n_q)]
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # float32 for a half-precision model, whose scores would tie too often
        work = torch.promote_types(residual.dtype, torch.float32)
        codes = []
        for level in self.layers:
            # (n, n_tokens, dim) -> (n, n_tokens, codebook_dim), unit length
            z = F.normalize(level.project_in(residual).to(work), p=2.0, dim=-1)
            # (codebook_size, codebook_dim), unit length
            entries = F.normalize(level.codebook.to(work), p=2.0, dim=1)
            # (n, n_tokens, codebook_size) cosine similarities -> (n, n_tokens)
            index = (z @ entries.t()).max(dim=-1).indices
            # (n, n_tokens) -> (n, n_tokens, codebook_dim), the stored entries
            selected = F.embedding(index, level.codebook)
            # (n, n_tokens, codebook_dim) -> (n, n_tokens, dim), taken off the residual
            residual = residual - level.project_out(selected)
            codes.append(index)
        # n_q x (n, n_tokens) -> (n, n_tokens, n_q)
        return torch.stack(codes, dim=-1)

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        levels = []
        for q, level in enumerate(self.layers):
            # (...) -> (..., codebook_dim)
            levels.append(F.embedding(codes[..., q], level.codebook))
        # n_q x (..., codebook_dim) -> (..., n_q, codebook_dim)
        return torch.stack(levels, dim=-2)
