# CSBrain: reimplementation for braindecode of the cross-scale spatiotemporal
# brain foundation model from Zhou et al. (2025). The architecture follows the
# paper (arXiv:2506.23075) and was cross-checked against the authors' reference
# implementation at https://github.com/yuchen2199/CSBrain (which ships no
# license file at the time of writing, so this file is an independent
# reimplementation in the EEGModuleMixin idiom, building on the CBraMod port
# for the shared patch-embedding design).
#
# License: BSD (3-clause)

import copy
import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock

log = logging.getLogger(__name__)


# Anatomical region identifiers for the structured sparse attention. The
# identifiers follow the five-lobe partition used by the reference
# implementation (frontal / parietal / temporal / occipital / central).
REGION_FRONTAL = 0
REGION_PARIETAL = 1
REGION_TEMPORAL = 2
REGION_OCCIPITAL = 3
REGION_CENTRAL = 4

# Map 10-20-style electrode prefixes to brain regions, longest prefix first:
# FP/AF/F/FC -> frontal, C/CP -> central, FT/T/TP -> temporal, P -> parietal,
# PO/O/I -> occipital. This reproduces the reference layouts of PhysioNet-MI,
# FACED, SHU-MI, CHB-MIT, Mumtaz2016, MentalArithmetic, ISRUC, HMC and TUSL.
# The reference is not consistent across datasets: its BCI-IV-2a, SEED-V,
# SEED-VIG and Siena layouts put FC in central and PO in parietal; pass
# ``brain_regions`` to reproduce those.
_REGION_PREFIXES: tuple[tuple[str, int], ...] = (
    ("FP", REGION_FRONTAL),
    ("AF", REGION_FRONTAL),
    ("FC", REGION_FRONTAL),
    ("FT", REGION_TEMPORAL),
    ("TP", REGION_TEMPORAL),
    ("CP", REGION_CENTRAL),
    ("PO", REGION_OCCIPITAL),
    ("F", REGION_FRONTAL),
    ("C", REGION_CENTRAL),
    ("T", REGION_TEMPORAL),
    ("P", REGION_PARIETAL),
    ("O", REGION_OCCIPITAL),
    ("I", REGION_OCCIPITAL),
)


def region_of_electrode(ch_name: str, default: int = REGION_CENTRAL) -> int:
    """Return the brain region of a 10-20-style electrode name.

    Unrecognised names (e.g. generic ``EEG 01`` labels) fall back to
    ``default`` so that montage-less recordings degrade to a single
    attention region instead of failing.
    """
    letters = "".join(c for c in ch_name.upper() if c.isalpha())
    for prefix, region in _REGION_PREFIXES:
        if letters.startswith(prefix):
            return region
    return default


def group_regions(regions: Sequence[int]):
    """Group per-channel region ids into ``(ordered_regions, sorted_indices)``.

    Regions are made contiguous and ordered by id; the original relative
    order is kept inside a region.
    """
    sorted_indices = [
        i
        for region in sorted(set(regions))
        for i, r in enumerate(regions)
        if r == region
    ]
    ordered_regions = [regions[i] for i in sorted_indices]
    return ordered_regions, sorted_indices


def order_regions(regions: Sequence[int], channel_order: Sequence[int] | None = None):
    """Return ``(ordered_regions, sorted_indices)`` for an explicit order.

    ``channel_order`` is the reference's ``sorted_indices``: a permutation of
    the channels that groups them by ascending region id and sets the
    electrode ring inside each region. ``None`` keeps the input order inside
    each region (:func:`group_regions`).
    """
    regions = list(regions)
    if channel_order is None:
        return group_regions(regions)
    sorted_indices = [int(i) for i in channel_order]
    if sorted(sorted_indices) != list(range(len(regions))):
        raise ValueError(
            f"channel_order must be a permutation of the {len(regions)} "
            f"input channels, got {sorted_indices}."
        )
    ordered_regions = [regions[i] for i in sorted_indices]
    if ordered_regions != sorted(regions):
        raise ValueError(
            "channel_order must group the channels by ascending region id, "
            f"got regions {ordered_regions} in that order."
        )
    return ordered_regions, sorted_indices


def derive_brain_regions(
    chs_info: Sequence[dict] | None, channel_order: Sequence[int] | None = None
):
    """Split channel names into regions and build the (regions, order) pair.

    Returns ``(brain_regions, sorted_indices)`` where ``brain_regions[i]`` is
    the region of the channel that ends up at position ``i`` after reordering
    by ``sorted_indices``. Unrecognised names all fall into the central
    region; if no name is recognised, the group mask lets each electrode
    attend only to itself in the inter-region attention.
    """
    names = [str(ch.get("ch_name", "")) for ch in (chs_info or [])]
    regions = [region_of_electrode(name) for name in names]
    return order_regions(regions, channel_order)


def make_area_config(brain_regions: Sequence[int]) -> dict[str, dict]:
    """Group contiguous channel indices per region for the sparse attention.

    Mirrors the reference implementation's ``generate_area_config``: regions
    must be contiguous in the (already sorted) input.
    """
    area_config: dict[str, dict] = {}
    start = 0
    for i in range(1, len(brain_regions) + 1):
        boundary = i == len(brain_regions) or brain_regions[i] != brain_regions[i - 1]
        if boundary:
            region = brain_regions[i - 1]
            area_config[f"region_{region}"] = {
                "channels": i - start,
                "slice": slice(start, i),
            }
            start = i
    return area_config


def build_region_attention_mask(
    area_config: dict[str, dict], n_channels: int
) -> Tensor:
    """Return the inter-region attention mask of the structured sparse attention.

    Channels are grouped round-robin across regions (group *g* holds the
    *g*-th electrode of every region); attention is allowed only within a
    group, so each electrode attends to at most one electrode per other
    region instead of all of them.
    """
    region_indices = [
        list(range(info["slice"].start, info["slice"].stop))
        for info in area_config.values()
    ]
    mask = torch.full((n_channels, n_channels), float("-inf"))
    n_groups = max(len(indices) for indices in region_indices)
    for g in range(n_groups):
        group = [
            indices[g % len(indices)] for indices in region_indices if len(indices) > 0
        ]
        for i in group:
            mask[i, group] = 0.0
    return mask


class CSBrain(EEGModuleMixin, nn.Module):
    r"""Cross-scale Spatiotemporal Brain Foundation Model from Zhou et al. (2025)
    [zhou2025csbrain]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: ../_static/model/csbrain_arch.png
        :align: center
        :alt: CSBrain architecture overview
        :width: 900px

    CSBrain is an EEG foundation model pre-trained with masked patch
    reconstruction, designed to decode brain activity across scales. It
    combines three mechanisms on top of CBraMod-style 200-sample patching:

    - **Cross-scale temporal embedding**: multi-scale convolutions (kernels
      1/3/5 over the patch axis) fold brief bursts and slow rhythms into the
      same token vocabulary;
    - **Region embedding**: per-region convolutions with circular padding mix
      each anatomical region's electrodes;
    - **Structured sparse attention (SSA)**: inter-window attention over
      sliding windows of 5 patches, plus inter-region attention restricted by
      a group mask so each electrode attends to at most one electrode per
      other region, avoiding spurious long-range dependencies.

    Channel names (``chs_info``) are mapped to five anatomical regions
    (frontal / parietal / temporal / occipital / central) and reordered to be
    contiguous; unrecognised names fall into the central region. Without
    ``chs_info`` (and without ``brain_regions``) the region embedding is
    skipped and the inter-region attention is unmasked.

    Parameters
    ----------
    patch_size : int, default=200
        Temporal patch size in samples (200 samples = 1 second at 200 Hz).
    dim_feedforward : int, default=800
        Dimension of the feedforward network in the encoder layers.
    n_layer : int, default=12
        Number of encoder layers.
    nhead : int, default=8
        Number of attention heads.
    activation : type[nn.Module], default=nn.GELU
        Activation function used in the encoder feedforward blocks.
    emb_dim : int, default=200
        Output dimension of the final projection applied to the encoder
        output.
    temporal_kernel_sizes : tuple[int, ...], default=(1, 3, 5)
        Kernel sizes of the cross-scale temporal embedding convolutions.
    drop_prob : float, default=0.1
        Dropout probability, applied to the backbone (patch embedding and
        encoder layers) and to the task head. The reference fine-tuning
        scripts keep the backbone at 0.1 and raise only the head dropout
        (``--dropout 0.3`` for BCIC IV-2a); to reproduce that recipe, keep
        ``drop_prob=0.1`` and raise the head dropout in a thin subclass or
        wrapper.
    brain_regions : sequence of int | None, default=None
        Explicit region id per input channel (0 frontal, 1 parietal, 2
        temporal, 3 occipital, 4 central), taking precedence over the
        name-based derivation. Pass this to reproduce a dataset-specific
        region layout, e.g. the one used by the authors' released
        fine-tuning checkpoints.
    channel_order : sequence of int | None, default=None
        Permutation of the input channels applied before the region modules
        (the reference's ``sorted_indices``). It must group the channels by
        ascending region id; inside a region it sets the electrode ring of
        the circular region convolution and the round-robin attention
        groups. ``None`` keeps the input order inside each region. Most
        reference fine-tuning models (e.g. CHB-MIT, Siena, SEED-V) use a
        hand-made topological order, so they need this to match exactly.
    head_hidden_dim : int | None, default=None
        Width of the first hidden layer of the task head. ``None`` uses
        ``n_patch * emb_dim``, the width of most reference fine-tuning heads
        (e.g. 800 for 4 s and 2000 for 10 s windows at 200 Hz). Some
        reference heads differ, e.g. SEED-V (1 s windows) uses 800; pass it
        here to load those checkpoints. The first head layer has
        ``n_chans * n_patch * emb_dim * head_hidden_dim`` weights, which grows
        quadratically with the window length by default (about 2.3e9 for 64
        channels and 30 s), so set a smaller width for long windows.
    return_encoder_output : bool, default=False
        If False (default), the projected encoder output is flattened and
        passed through the task head to produce class logits of size
        ``n_outputs``. If True, return the encoder output features.

    References
    ----------
    .. [zhou2025csbrain] Zhou, Y., Wu, J., Ren, Z., Yao, Z., Lu, W., Peng, K.,
       Zheng, Q., Song, C., Ouyang, W., & Gou, C. (2025). CSBrain: A
       Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding.
       Advances in Neural Information Processing Systems (NeurIPS 2025,
       Spotlight). https://arxiv.org/abs/2506.23075
    .. [csbraincode] Released implementation:
       https://github.com/yuchen2199/CSBrain
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        patch_size: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        activation: type[nn.Module] = nn.GELU,
        emb_dim: int = 200,
        temporal_kernel_sizes: Sequence[int] = (1, 3, 5),
        drop_prob: float = 0.1,
        brain_regions: Sequence[int] | None = None,
        channel_order: Sequence[int] | None = None,
        head_hidden_dim: int | None = None,
        return_encoder_output: bool = False,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_chans, chs_info, n_times, input_window_seconds, sfreq, n_outputs

        self.rearrange = Rearrange(
            "batch n_chans (n_patch patch_size) -> batch n_chans n_patch patch_size",
            patch_size=patch_size,
        )
        self.patch_embedding = _PatchEmbedding(patch_size, drop_prob=drop_prob)
        d_model = self.patch_embedding.d_model

        # Region structure: explicit ``brain_regions`` (one region id per input
        # channel, taking precedence over name derivation) when given, else
        # derived from channel names; without either there is no reordering,
        # no region embedding and the inter-region attention is unmasked.
        if brain_regions is not None and self._n_chans_or_none() not in (
            None,
            len(brain_regions),
        ):
            raise ValueError(
                f"brain_regions has {len(brain_regions)} entries for "
                f"{self._n_chans_or_none()} channels."
            )
        if brain_regions is not None:
            ordered_regions, sorted_indices = order_regions(
                brain_regions, channel_order
            )
        elif self._chs_info:
            ordered_regions, sorted_indices = derive_brain_regions(
                self._chs_info, channel_order
            )
        elif channel_order is not None:
            raise ValueError("channel_order needs brain_regions or chs_info.")
        else:
            ordered_regions, sorted_indices = None, None

        if sorted_indices is None:
            self.sorted_indices = None
            self.area_config = {}
        else:
            self.register_buffer(
                "sorted_indices", torch.as_tensor(sorted_indices, dtype=torch.long)
            )
            self.area_config = make_area_config(ordered_regions)

        self.temporal_embed = _CrossScaleTemporalEmbedding(
            d_model, d_model, kernel_sizes=tuple(temporal_kernel_sizes)
        )
        self.region_embed = _RegionEmbedding(d_model, d_model, self.area_config)
        encoder_layer = _CSBrainEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            activation=activation,
            area_config=self.area_config,
        )
        self.encoder = nn.ModuleList(
            [encoder_layer] + [copy.deepcopy(encoder_layer) for _ in range(n_layer - 1)]
        )
        self.proj_out = nn.Sequential(nn.Linear(d_model, emb_dim))

        self._emb_dim = emb_dim
        self._patch_size = patch_size
        self._d_model = d_model
        self._drop_prob = drop_prob
        self._head_hidden_dim = head_hidden_dim
        self._weights_init()

        if return_encoder_output:
            self.final_layer = nn.Identity()
        else:
            self.final_layer = self._make_task_head()

    def _n_chans_or_none(self) -> int | None:
        try:
            return self.n_chans
        except ValueError:
            return None

    def _n_times_or_none(self) -> int | None:
        try:
            return self.n_times
        except ValueError:
            return None

    def _make_task_head(self) -> nn.Sequential:
        # Three-layer MLP head of the reference fine-tuning models: flatten
        # (chans, patches, emb_dim) -> n_patch * emb_dim -> emb_dim -> n_outputs.
        # n_chans / n_times may come from chs_info / input_window_seconds.
        # Without them the layers are lazy and the hidden width falls back to
        # 4 * emb_dim, the reference value for 4 s windows.
        n_chans, n_times = self._n_chans_or_none(), self._n_times_or_none()
        if n_times is None or n_chans is None:
            return nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(self._head_hidden_dim or 4 * self._emb_dim),
                nn.ELU(),
                nn.Dropout(self._drop_prob),
                nn.LazyLinear(self._emb_dim),
                nn.ELU(),
                nn.Dropout(self._drop_prob),
                nn.LazyLinear(self.n_outputs),
            )
        n_patch = n_times // self._patch_size
        hidden = self._head_hidden_dim or n_patch * self._emb_dim
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_chans * n_patch * self._emb_dim, hidden),
            nn.ELU(),
            nn.Dropout(self._drop_prob),
            nn.Linear(hidden, self._emb_dim),
            nn.ELU(),
            nn.Dropout(self._drop_prob),
            nn.Linear(self._emb_dim, self.n_outputs),
        )

    def reset_head(self, n_outputs):
        self._set_n_outputs(n_outputs)
        self._update_init_kwargs(return_encoder_output=False)
        self.final_layer = self._make_task_head()

    def _weights_init(self):
        # Same rule as the reference ``_weights_init`` (and CBraMod): only the
        # Linear layers are re-initialised. The Conv2d layers keep PyTorch's
        # default init; a fan-out Kaiming init on them makes the residual
        # stream grow ~3x per layer (logits ~1e5 at 12 layers).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, mask=None, return_features=False):
        x = self.rearrange(x)
        if self.sorted_indices is not None:
            x = x[:, self.sorted_indices, :, :]
        emb = self.patch_embedding(x, mask)
        for layer in self.encoder:
            emb = self.temporal_embed(emb) + emb
            emb = self.region_embed(emb) + emb
            emb = layer(emb)
        out = self.proj_out(emb)
        if return_features:
            return {"features": out, "cls_token": None}  # nosec B105
        return self.final_layer(out)


class _PatchEmbedding(nn.Module):
    """CBraMod-style patch encoder: conv stem + FFT magnitude embedding.

    Convolutions are applied over the concatenated channel-patch sequence
    (channel-major), plus a spectral projection of each patch's rFFT
    magnitude, followed by a depthwise-convolution positional encoding.
    ``mask`` (boolean over input samples, applied per patch) replaces masked
    patches by a zero vector, supporting masked-autoencoding pretraining.
    """

    def __init__(self, patch_size: int, drop_prob: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        # (channels, kernel, stride, padding, (norm groups, norm channels))
        spec = (
            (25, 49, 25, 24, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
        )
        last_channels = 1
        proj_in_layers = []
        for channels, kernel, stride, padding, norm in spec:
            proj_in_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=last_channels,
                        out_channels=channels,
                        kernel_size=(1, kernel),
                        stride=(1, stride),
                        padding=(0, padding),
                    ),
                    nn.GroupNorm(*norm),
                    nn.GELU(),
                ]
            )
            last_channels = channels
        self.proj_in = nn.Sequential(*proj_in_layers)
        out_patch_size = patch_size
        for _, kernel, stride, padding, _ in spec:
            out_patch_size = int((out_patch_size + 2 * padding - kernel) / stride + 1)
        self.d_model = last_channels * out_patch_size
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=self.d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, self.d_model),
            nn.Dropout(drop_prob),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = (
            patch_emb.permute(0, 2, 1, 3)
            .contiguous()
            .view(bz, ch_num, patch_num, self.d_model)
        )

        flat = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(flat, dim=-1, norm="forward")
        spectral = (
            torch.abs(spectral)
            .contiguous()
            .view(bz, ch_num, patch_num, patch_size // 2 + 1)
        )
        patch_emb = patch_emb + self.spectral_proj(spectral)

        positional_embedding = self.positional_encoding(
            patch_emb.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return patch_emb + positional_embedding


class _CrossScaleTemporalEmbedding(nn.Module):
    """Multi-scale convolutions over the patch axis (kernels 1/3/5).

    The output channels are split across scales in decreasing proportion
    (each smaller scale halves), and the scales are concatenated back to
    ``dim_out`` so the result can be added residually.
    """

    def __init__(
        self, dim_in: int, dim_out: int, kernel_sizes: Sequence[int] = (1, 3, 5)
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_scale,
                    kernel_size=(kt, 1),
                    stride=(1, 1),
                    padding=((kt - 1) // 2, 0),
                )
                for kt, dim_scale in zip(kernel_sizes, dim_scales)
            ]
        )

    def forward(self, x):
        batch, chans, time, d_model = x.shape
        x = x.view(batch * chans, d_model, time, 1)
        fmaps = [conv(x) for conv in self.convs]
        x = torch.cat(fmaps, dim=1)
        return x.view(batch, chans, time, -1)


class _RegionEmbedding(nn.Module):
    """Per-region multi-scale convolutions over the electrode axis.

    Each region gets its own bank of convolutions (kernels 1/3/5 over the
    feature axis); the electrode axis is zero/circular padded so the kernel
    wraps around the region's electrode ring, mixing each region locally.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        area_config: dict,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ):
        super().__init__()
        self.dim_out = dim_out
        self.area_config = area_config
        dim_scales = [dim_out // (2 ** (i + 1)) for i in range(len(kernel_sizes) - 1)]
        dim_scales.append(dim_out - sum(dim_scales))
        self.region_blocks = nn.ModuleDict(
            {
                region_key: nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels=dim_in,
                            out_channels=dim_scale,
                            kernel_size=(k, 1),
                            padding=(0, 0),
                        )
                        for k, dim_scale in zip(kernel_sizes, dim_scales)
                    ]
                )
                for region_key in area_config
            }
        )

    def forward(self, x):
        batch, chans, T, dim_in = x.shape
        output = torch.zeros(
            (batch, chans, T, self.dim_out), device=x.device, dtype=x.dtype
        )
        for region_key, region_info in self.area_config.items():
            channel_slice = region_info["slice"]
            n_electrodes = region_info["channels"]
            x_region = x[:, channel_slice, :, :]
            x_trans = (
                x_region.permute(0, 2, 1, 3)
                .reshape(-1, n_electrodes, dim_in)
                .permute(0, 2, 1)
                .unsqueeze(-1)
            )
            fmap_outputs = []
            for conv in self.region_blocks[region_key]:
                k = conv.kernel_size[0]
                pad = (k - 1) // 2
                if n_electrodes == 1:
                    x_padded = F.pad(
                        x_trans, (0, 0, pad, pad), mode="constant", value=0
                    )
                else:
                    x_padded = F.pad(x_trans, (0, 0, pad, pad), mode="circular")
                fmap_outputs.append(conv(x_padded))
            fmap_cat = torch.cat(fmap_outputs, dim=1)
            fmap_out = (
                fmap_cat.squeeze(-1)
                .permute(0, 2, 1)
                .reshape(batch, T, n_electrodes, self.dim_out)
                .permute(0, 2, 1, 3)
            )
            output[:, channel_slice, :, :] = fmap_out
        return output


class _CSBrainEncoderLayer(nn.Module):
    """Pre-norm encoder layer with structured sparse attention.

    1. inter-window attention over sliding windows of ``min(T, 5)`` patches
    (temporal dependencies across patches, channels separate);
    2. inter-region attention over the channel axis with a region group mask
    (spatial dependencies, patches separate), enhanced with each region's
    global mean feature;
    3. feedforward block.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        area_config: dict | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.inter_window_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.inter_region_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.global_fc = nn.Linear(d_model, d_model)

        self.ff_block = FeedForwardBlock(
            emb_size=d_model,
            expansion=1,
            drop_p=dropout,
            activation=activation,
            hidden_features=dim_feedforward,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.area_config = area_config or {}
        if area_config:
            n_channels = sum(info["channels"] for info in area_config.values())
            self.register_buffer(
                "region_attn_mask",
                build_region_attention_mask(area_config, n_channels),
            )
        else:
            self.region_attn_mask = None

    def forward(self, src: Tensor) -> Tensor:
        x = src
        x = x + self.dropout1(self._inter_window_attention(self.norm1(x)))
        x = x + self.dropout2(self._inter_region_attention(self.norm2(x)))
        x = x + self.dropout3(self._ff_block(self.norm3(x)))
        return x

    def _inter_window_attention(self, x: Tensor) -> Tensor:
        batch, chans, T, Fea = x.shape
        window_size = min(T, 5)
        original_T = T
        if T % window_size != 0:
            pad_length = window_size - (T % window_size)
            x = F.pad(x, (0, 0, 0, pad_length))
            T = T + pad_length
        num_windows = T // window_size

        x = (
            x.view(batch, chans, num_windows, window_size, Fea)
            .permute(0, 3, 1, 2, 4)
            .reshape(batch * window_size * chans, num_windows, Fea)
        )
        x = self.inter_window_attn(x, x, x, need_weights=False)[0]
        x = (
            x.reshape(batch, window_size, chans, num_windows, Fea)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch, chans, T, Fea)
        )
        if T != original_T:
            x = x[:, :, :original_T, :]
        return x

    def _inter_region_attention(self, x: Tensor) -> Tensor:
        batch, chans, T, Fea = x.shape
        region_slices = [info["slice"] for info in self.area_config.values()]

        x_flat = x.permute(0, 2, 1, 3).reshape(batch * T, chans, Fea)
        global_features = torch.zeros_like(x_flat)
        for region_slice in region_slices:
            region_global = x[:, region_slice, :, :].mean(dim=1, keepdim=True)
            region_global = region_global.permute(0, 2, 1, 3).reshape(batch * T, 1, Fea)
            for idx in range(region_slice.start, region_slice.stop):
                global_features[:, idx : idx + 1, :] = region_global
        x_enhanced = x_flat + self.global_fc(global_features)

        attn_output = self.inter_region_attn(
            x_enhanced,
            x_enhanced,
            x_enhanced,
            attn_mask=self.region_attn_mask,
            need_weights=False,
        )[0]
        return attn_output.reshape(batch, T, chans, Fea).permute(0, 2, 1, 3)

    def _ff_block(self, x: Tensor) -> Tensor:
        B, C, T, Fea = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, C, Fea)
        x = self.ff_block(x)
        return x.reshape(B, T, C, Fea).permute(0, 2, 1, 3)
