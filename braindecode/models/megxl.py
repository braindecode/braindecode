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

from braindecode.models.base import EEGModuleMixin
from braindecode.models.brainomni import _geometry_from_chs_info, _SpatialTemporalBlock
from braindecode.models.util import warn_if_sfreq_differs
from braindecode.modules import ResidualVectorQuantizer, SEANetEncoder


class MEGXL(EEGModuleMixin, nn.Module, license="mit"):
    r"""MEG-XL from Jayalath and Parker Jones (2026) [megxl]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

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
    - The RoPE table holds the 625 positions of the 150 s pretraining window
      (``torch.polar`` values in the converted weights); a longer window
      rebuilds it from ``cos``/``sin``, one ulp away from ``torch.polar``.
    - The released attention keeps its dropout (0.1) active in eval mode; here
      dropout is off in eval mode, so the eval forward is deterministic.
    - Pretraining (temporal block masking, the ``mask_token`` and the code
      prediction head) is not ported.

    .. important::

        Weights converted from the released ``meg-xl-med-v2.ckpt``
        (``pnpl/MEG-XL``, step 113000, with its frozen BioCodec tokenizer) are
        on the Hugging Face Hub at ``braindecode/megxl-pretrained`` (revision
        ``5b5dcbcc``); the head is not pretrained::

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
    n_q : int
        Number of residual codebooks.
    codebook_size : int
        Entries per codebook.
    codebook_dim : int
        Codebook dimension.

    References
    ----------
    .. [megxl] Jayalath, D., Parker Jones, O. (2026). MEG-XL: Data-Efficient
       Brain-to-Text via Long-Context Pre-Training. ICML 2026.
       https://arxiv.org/abs/2602.02494
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        drop_prob: float = 0.1,
        fourier_pos_dim: int = 250,
        position_scale: float = 1.8,
        orientation_scale: float = 1.0,
        tokenizer_dim: int = 128,
        n_filters: int = 32,
        ratios: tuple[int, ...] = (3, 2, 2),
        n_q: int = 6,
        codebook_size: int = 256,
        codebook_dim: int = 16,
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
        warn_if_sfreq_differs("MEGXL", self._sfreq, 50)
        if num_heads % 2 or latent_dim % num_heads or (latent_dim // num_heads) % 2:
            raise ValueError(
                f"num_heads ({num_heads}) must be even and divide latent_dim "
                f"({latent_dim}) into even head dimensions."
            )
        self.latent_dim = latent_dim
        # Derived from chs_info, so not saved: one checkpoint serves any montage.
        pos, sensor_type = _geometry_from_chs_info(self.chs_info)
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
            kernel_size=3,
            last_kernel_size=3,
            n_residual_layers=2,
            compress=1,
            true_skip=True,
            causal=True,
        )
        self.quantizer = ResidualVectorQuantizer(
            dim=tokenizer_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=n_q,
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
        # RoPE table of the 150 s pretraining window (625 tokens at 50 Hz).
        self.blocks = nn.ModuleList(
            [
                _SpatialTemporalBlock(latent_dim, num_heads, drop_prob, 625)
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

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for ``n_outputs`` classes."""
        self._set_n_outputs(n_outputs)
        reference = next(self.parameters())
        self.final_layer = nn.Linear(self.n_chans * self.latent_dim, n_outputs)
        self.final_layer.to(device=reference.device, dtype=reference.dtype)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """BioCodec codes ``(batch, n_chans, n_tokens, n_q)`` of ``x``."""
        batch, chans, _ = x.shape
        # (batch, n_chans, n_times) -> (batch * n_chans, 1, n_times)
        x = x.reshape(batch * chans, 1, -1)
        # Each channel is scaled to unit RMS over the window.
        # (batch * n_chans, 1, 1)
        scale = x.pow(2).mean(dim=2, keepdim=True).sqrt().clamp(min=1e-8)
        # (batch * n_chans, 1, n_times) -> (batch * n_chans, tokenizer_dim, n_tokens)
        # -> (batch * n_chans, n_tokens, tokenizer_dim)
        residual = self.encoder(x / scale).transpose(1, 2)
        # Not the shared RVQ forward: BioCodec picks codes by cosine similarity,
        # the shared quantizer by Euclidean distance.
        codes = []
        for vq in self.quantizer.layers:
            # (batch * n_chans, n_tokens, tokenizer_dim)
            # -> (batch * n_chans, n_tokens, codebook_dim)
            z = F.normalize(vq.project_in(residual), p=2.0, dim=-1)
            # (codebook_size, codebook_dim)
            codebook = F.normalize(vq._codebook.embed, p=2.0, dim=1)
            # (batch * n_chans, n_tokens, codebook_size) -> (batch * n_chans, n_tokens)
            index = (z @ codebook.t()).max(dim=-1).indices
            # (batch * n_chans, n_tokens, tokenizer_dim)
            residual = residual - vq.project_out(F.embedding(index, vq._codebook.embed))
            codes.append(index)
        # n_q x (batch * n_chans, n_tokens) -> (batch * n_chans, n_tokens, n_q)
        # -> (batch, n_chans, n_tokens, n_q)
        return torch.stack(codes, dim=-1).reshape(batch, chans, -1, len(codes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Token features ``(batch, n_chans, n_tokens, latent_dim)``."""
        with torch.no_grad():
            codes = self.tokenize(x)  # (batch, n_chans, n_tokens, n_q)
        levels = []
        for q, vq in enumerate(self.quantizer.layers):
            # (batch, n_chans, n_tokens) -> (batch, n_chans, n_tokens, codebook_dim)
            levels.append(F.embedding(codes[..., q], vq._codebook.embed))
        # (batch, n_chans, n_tokens, n_q, codebook_dim)
        # -> (batch, n_chans, n_tokens, n_q * codebook_dim)
        # -> (batch, n_chans, n_tokens, latent_dim)
        h = self.rvq_projector(torch.stack(levels, dim=3).flatten(3))
        # Keep the original batched projections for bit-exact geometry gradients.
        # (n_chans, 3) -> (batch, n_chans, 3) -> (batch * n_chans, 3)
        xyz = self.sensor_xyz.expand(x.shape[0], -1, -1)
        xyz = self.flatten_sensor_batch(xyz)
        # (n_chans, 3) -> (batch, n_chans, 3) -> (batch * n_chans, 3)
        abc = self.sensor_abc.expand(x.shape[0], -1, -1)
        abc = self.flatten_sensor_batch(abc)
        # (n_chans,) -> (batch, n_chans)
        sensor_type = self.sensor_type.expand(x.shape[0], -1)
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
        ch_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Classify ``x`` or return the pooled features.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_features : bool
            If True, return ``{"features": (batch, n_chans * latent_dim),
            "cls_token": None}`` instead of the logits.
        ch_mask : torch.Tensor, optional
            Not supported: any value but ``None`` raises a ``ValueError``.
            The head flattens the ``n_chans * latent_dim`` features and the
            sensor geometry is the fixed ``chs_info`` montage, so every sample
            needs all ``n_chans`` channels.
        """
        if ch_mask is not None:
            raise ValueError(
                "MEGXL does not accept ch_mask: its head flattens the n_chans * "
                "latent_dim features and its sensor geometry is the chs_info "
                "montage, so every sample needs all n_chans channels."
            )
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


def _fourier_features(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Gaussian random Fourier features ``[sin, cos](2 pi x @ freqs)``."""
    # (n, 3) @ (3, fourier_pos_dim // 2) -> (n, fourier_pos_dim // 2)
    proj = 2 * math.pi * (x @ freqs)
    # -> (n, fourier_pos_dim)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
