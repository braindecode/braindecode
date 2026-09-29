# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: Apache-2.0
"""DIVER-1 (Han et al., 2025), adapted by Julien Gadonneix.

The reference encoder derives from Salesforce's MOIRAI / ``uni2ts``
(Copyright Salesforce, Inc.; https://www.apache.org/licenses/LICENSE-2.0).
The reference repository declares no separate code license.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import rotate_pairs
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    channel_metadata_from_chs_info,
    valid_location_mask,
)
from braindecode.modules import FeedForwardBlock, PatchTokenizer


def _check_channel_metadata(
    metadata: torch.Tensor,
    n_chans: int,
    n_modalities: int = 2,
    n_subtypes: int = 3,
) -> None:
    """Reject metadata that does not describe the channels of this batch."""
    if metadata.dim() != 2 or metadata.shape[0] != n_chans or metadata.shape[1] != 5:
        raise ValueError(
            f"chan_metadata must have shape ({n_chans}, 5), one row of "
            f"(x, y, z, modality, sub-modality) per channel of the input, got "
            f"{list(metadata.shape)}."
        )
    modality = metadata[:, 3]
    if bool(((modality < 0) | (modality >= n_modalities)).any()):
        raise ValueError(
            f"The modality column of chan_metadata must hold slots in "
            f"[0, {n_modalities}), got values from {float(modality.min())} to "
            f"{float(modality.max())}."
        )
    if bool((metadata[:, 4] >= n_subtypes).any()):
        raise ValueError(
            f"The sub-modality column of chan_metadata must hold slots below "
            f"{n_subtypes}, or a negative value for unknown, got up to "
            f"{float(metadata[:, 4].max())}."
        )


class DIVER1(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""DIVER-1 from Han et al. (2025) [Han2025]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`
    :bdg-dark-line:`Channel`

    .. figure:: ../_static/model/diver1_arch.png
        :align: center
        :alt: DIVER-1 architecture overview
        :width: 1000px

        DIVER-1 architecture, pretraining and downstream evaluation pipeline,
        reproduced from [Han2025]_.

    .. versionadded:: 1.8.2

    Encodes a ``(channel, time-patch)`` grid with any-variate attention:
    RMS-normalized queries and keys, temporal RoPE, a learned same/cross-channel
    bias and SwiGLU blocks. Each patch passes through a strided CNN; the magnitude
    spectrum of the **CNN token**, not the raw patch, is projected and added.
    STCPE adds a local positional bias by encoding sliding temporal windows and
    **averaging**, rather than summing, their overlapping outputs. These choices
    follow the reference code and Table 7 where the paper's prose differs.

    Three learned registers add a channel row, a patch column and their corner.
    They participate in attention but are discarded before the linear read-out.
    ``pooling="flatten"`` follows the paper's finetuning protocol;
    ``pooling="mean"`` allows varying channel counts with a fixed patch count.
    Pretraining masks, reconstruction heads, resampling and muP training are not
    implemented; the released checkpoints' muP attention scaling is supported.

    .. rubric:: Channel metadata

    ``chs_info`` supplies the default montage. Its ``"kind"`` must identify EEG
    or an intracranial type: SEEG/DBS imply depth electrodes, ECoG implies grids;
    strips cannot be inferred. ``"loc"`` coordinates are converted from metres
    to millimetres for PopT's sinusoidal encoding. Missing, non-finite or exactly
    zero coordinates and unknown subtypes contribute zero embeddings. Disabling
    ``use_position_emb`` removes both coordinate and type embeddings.

    For another montage, pass
    :func:`~braindecode.models.util.channel_metadata_from_chs_info`'s result to
    :meth:`forward`. All samples in a batch share this metadata. The encoder is
    channel-permutation equivariant (the flattened head is not).

    .. rubric:: Published variants and weights

    All variants use 12 layers and 32 features per head. Tiny, Small, Base,
    Large, XL and XXL have widths 256, 512, 768, 1024, 2048 and 3072 respectively.
    At 500 Hz, ``patch_size=500`` gives 1 s patches and ``patch_size=50`` gives
    0.1 s patches. Paper parameter counts include pretraining-only heads and a
    mask token; this port contains only the encoder and classification head.

    Released encoders are available as ``braindecode/DIVER-1-0.1s-tiny``
    (iEEG, width 256) and ``braindecode/DIVER-1-1s-small`` (joint EEG/iEEG,
    width 512) on Hugging Face::

        model = DIVER1.from_pretrained(
            "braindecode/DIVER-1-0.1s-tiny", chs_info=raw.info["chs"],
            n_times=500, n_outputs=2,
        )

    The classification head is initialized on load and needs fine-tuning.
    Conversion utilities are distributed with the Hub checkpoints, not the
    library. Encoder features have been checked against both released
    checkpoints on CPU, with the reference's always-active attention dropout
    disabled. This port disables attention dropout in eval mode.

    .. rubric:: License

    Code is Apache-2.0 through the reference encoder's MOIRAI / ``uni2ts``
    ancestry; released DIVER Project weights are MIT-licensed.

    Parameters
    ----------
    patch_size : int
        Number of samples per temporal patch, default 500 (1 s at the paper's
        500 Hz; the other published variant is 50).
    d_model : int
        Token embedding dimension, default 256 (the Tiny variant).
    n_layers : int
        Number of any-variate Transformer blocks.
    num_heads : int, optional
        Number of attention heads. Defaults to ``d_model // 32``, as in the
        reference implementation.
    d_ff : int, optional
        Hidden dimension of the feed-forward blocks. Defaults to
        ``4 * d_model``.
    cnn_depth : int
        Number of convolution layers in the patch encoder.
    cnn_stride : int, optional
        Stride of the first (strided) convolution of the patch encoder.
        Defaults to the reference setting: the padded patch length divided by 8
        for ``patch_size >= 100`` and by 16 below, i.e. 64 for the 1 s variant
        and 4 for the 0.1 s variant.
    cnn_kernel_size : int
        Kernel width of the first convolution of the patch encoder; must be odd.
        The remaining ``cnn_depth - 1`` convolutions use width 3 and stride 1.
    use_stcpe : bool
        Whether to add the spatio-temporal conditional positional embedding.
    stcpe_window : int
        Width (in patches) of the STCPE sliding window; must be odd.
    stcpe_ratio : int
        Bottleneck ratio of STCPE: it operates at ``d_model // stcpe_ratio``.
    use_spectral_emb : bool
        Whether to add the spectral embedding.
    use_position_emb : bool
        Whether to add the electrode coordinate and type embeddings.
    pooling : {"flatten", "mean"}
        Token aggregation before the head. ``"flatten"`` reproduces the paper's
        finetuning protocol (a linear classifier on the flattened token grid);
        ``"mean"`` averages over channels and patches first, giving a head that
        is independent of ``n_chans``, which is what lets one instance read
        montages of any size.
    mup_attention : bool
        Scale attention scores by ``1 / head_dim`` (the muP scaling the released
        checkpoints were trained with) instead of ``1 / sqrt(head_dim)``. Keep it
        ``True`` to load the pretrained weights.
    drop_prob : float
        Dropout rate used in the encoder and the spectral embedding.
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks, default
        :class:`~torch.nn.SiLU`.

    References
    ----------
    .. [Han2025] Han, D. D., Gwon, Y., Lee, A. L., Lee, T., Lee, S. J., Choi,
       J., Lee, S., Bang, J., Lee, S., Park, D. K., Yoo, S., Chung, C. K. &
       Cha, J. (2025). DIVER-1: Scaling intracranial EEG foundation models for
       transferable representations. arXiv preprint arXiv:2512.19097.
       https://arxiv.org/abs/2512.19097
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        patch_size: int = 500,
        d_model: int = 256,
        n_layers: int = 12,
        num_heads: int | None = None,
        d_ff: int | None = None,
        cnn_depth: int = 3,
        cnn_stride: int | None = None,
        cnn_kernel_size: int = 63,
        use_stcpe: bool = True,
        stcpe_window: int = 7,
        stcpe_ratio: int = 8,
        use_spectral_emb: bool = True,
        use_position_emb: bool = True,
        pooling: str = "flatten",
        mup_attention: bool = True,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.SiLU,
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

        if not self._chs_info:
            raise ValueError(
                "DIVER1 reads the electrode modality, sub-modality and "
                "coordinates from chs_info, which must therefore be given and "
                "non-empty."
            )
        if pooling not in ("flatten", "mean"):
            raise ValueError(f"pooling must be 'flatten' or 'mean', got {pooling!r}.")
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        num_heads = num_heads if num_heads is not None else max(1, d_model // 32)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        if d_model % num_heads:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if (d_model // num_heads) % 2:
            raise ValueError(
                f"The attention head dimension (d_model // num_heads = "
                f"{d_model // num_heads}) must be even for the rotary embedding."
            )
        if stcpe_window % 2 == 0:
            raise ValueError(
                f"stcpe_window must be odd so the window is centred, got "
                f"{stcpe_window}."
            )

        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_stcpe = use_stcpe
        self.stcpe_window = stcpe_window
        self.stcpe_ratio = stcpe_ratio
        self.use_spectral_emb = use_spectral_emb
        self.use_position_emb = use_position_emb
        self.pooling = pooling
        self.drop_prob = drop_prob
        self.activation = activation

        # EEGModuleMixin signal properties are unavailable inside TorchScript.
        self.n_chans_grid = self.n_chans
        # PatchTokenizer right-pads incomplete patches.
        self.n_patches = -(-self.n_times // patch_size)
        self.patch_tokenizer = PatchTokenizer(
            patch_size=patch_size, n_times=self.n_times
        )
        self.patch_cnn = _PatchCNN(
            n_patches=self.n_patches,
            patch_size=patch_size,
            d_model=d_model,
            stride=cnn_stride,
            kernel_size=cnn_kernel_size,
            depth=cnn_depth,
        )

        self.spectral_emb = (
            nn.Sequential(
                OrderedDict(
                    spectral_proj=nn.Sequential(
                        nn.Linear(d_model // 2 + 1, d_model), nn.Dropout(drop_prob)
                    )
                )
            )
            if use_spectral_emb
            else None
        )

        # A device-aware default montage; forward can supply another recording.
        self.register_buffer(
            "default_chan_metadata",
            channel_metadata_from_chs_info(self.chs_info),
            persistent=False,
        )

        self.chan_emb = _ChannelMetaEmbedding(d_model) if use_position_emb else None

        self.stcpe = (
            _STCPE(
                d_model=d_model,
                ratio=stcpe_ratio,
                window=stcpe_window,
                activation=activation,
                n_patches=self.n_patches,
                mup_attention=mup_attention,
            )
            if use_stcpe
            else None
        )

        # Extra patch column, channel row and corner: attention-only registers.
        self.patch_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.chan_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.global_register = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Fixed patch count (including registers), variable channel count.
        self.flatten_grid = Rearrange("batch chan patch dim -> batch (chan patch) dim")
        self.unflatten_grid = Rearrange(
            "batch (chan patch) dim -> batch chan patch dim", patch=self.n_patches + 1
        )

        self.encoder = _AnyVariateEncoder(
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob,
            activation=activation,
            # One extra patch position for the register column.
            max_len=max(512, self.n_patches + 1),
            mup_attention=mup_attention,
        )

        head_in_features = (
            self.n_chans * self.n_patches * d_model if pooling == "flatten" else d_model
        )
        self.final_layer = nn.Linear(head_in_features, self.n_outputs)

    def reset_head(self, n_outputs):
        """Replace the linear classification head for a new ``n_outputs``."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        self._update_init_kwargs(n_outputs=n_outputs)

    @staticmethod
    def channel_metadata(chs_info: list[dict]) -> torch.Tensor:
        """Alias for :func:`braindecode.models.util.channel_metadata_from_chs_info`."""
        return channel_metadata_from_chs_info(chs_info)

    def forward(
        self, x: torch.Tensor, chan_metadata: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode an iEEG batch into class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        chan_metadata : torch.Tensor, optional
            ``(n_chans, 5)`` electrode metadata of the recording this batch
            comes from, one row of (x, y, z, modality, sub-modality) per
            channel, as :meth:`channel_metadata` builds it from its
            ``chs_info``. Every sample of the batch shares it. Defaults to the
            montage given at construction, which only fits the
            construction-time channel count.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_outputs)``.
        """
        n_chans = x.shape[1]
        if chan_metadata is None:
            if n_chans != self.n_chans_grid:
                raise ValueError(
                    f"DIVER1 was built for {self.n_chans_grid} channels but got "
                    f"input with {n_chans}; pass this recording's metadata as "
                    f"chan_metadata, which DIVER1.channel_metadata builds from "
                    f"its chs_info."
                )
            metadata = self.default_chan_metadata
        else:
            _check_channel_metadata(chan_metadata, n_chans)
            metadata = chan_metadata.to(device=x.device, dtype=x.dtype)
        # A flattened head has weights tied to the construction-time montage.
        if self.pooling == "flatten" and n_chans != self.n_chans_grid:
            raise ValueError(
                f"pooling='flatten' ties the read-out to the "
                f"{self.n_chans_grid} channels it was built for, but got "
                f"{n_chans}. Build the model with pooling='mean' to encode "
                f"montages of any size."
            )
        # Reject a different patch count before tokenization.
        if -(-x.shape[-1] // self.patch_size) != self.n_patches:
            raise ValueError(
                f"DIVER1 was built for {self.n_patches} temporal patches of "
                f"{self.patch_size} samples but got input with {x.shape[-1]} "
                f"samples; rebuild the model for this window length."
            )

        # Reference order: CNN, spectrum, metadata, then input-conditioned STCPE.
        tokens = self.patch_tokenizer(x)
        tokens = self.patch_cnn(tokens)
        if self.spectral_emb is not None:
            # Preserve the reference's float32 FFT even for float64 tokens.
            spectrum = torch.fft.rfft(tokens.float(), dim=-1, norm="forward")
            tokens = tokens + self.spectral_emb(spectrum.abs().to(tokens.dtype))
        if self.chan_emb is not None:
            chan_emb = self.chan_emb(metadata)
            tokens = tokens + chan_emb[None, :, None, :]
        if self.stcpe is not None:
            tokens = tokens + self.stcpe(tokens)

        # Prepend the register column (per-channel), row (per-patch) and their
        # corner (global).
        batch = tokens.shape[0]
        patch_reg = self.patch_register[None].expand(batch, n_chans, -1, -1)
        tokens = torch.cat([patch_reg, tokens], dim=2)
        chan_reg = self.chan_register[None].expand(batch, -1, self.n_patches, -1)
        global_reg = self.global_register[None].expand(batch, -1, -1, -1)
        row = torch.cat([global_reg, chan_reg], dim=2)
        tokens = torch.cat([row, tokens], dim=1)

        # Registers included, so one more than the input channel and patch counts.
        _, n_rows, n_cols, _ = tokens.shape
        # Shared montage/length: one pair of ids avoids per-sample bias copies.
        var_id = torch.arange(n_rows, device=x.device).repeat_interleave(n_cols)
        time_id = torch.arange(n_cols, device=x.device).repeat(n_rows)
        tokens = self.encoder(
            self.flatten_grid(tokens),
            var_id=var_id,
            time_id=time_id,
        )
        tokens = self.unflatten_grid(tokens)
        # The reference discards register outputs, including during finetuning.
        tokens = tokens[:, 1:, 1:]

        if self.pooling == "flatten":
            pooled = tokens.flatten(start_dim=1)
        else:
            pooled = tokens.mean(dim=(1, 2))
        return self.final_layer(pooled)


class _PatchCNN(nn.Sequential):
    """Power-of-two padded patches encoded by Conv2d/GroupNorm/GELU.

    The fixed patch axis leaves the channel count free. GroupNorm uses the
    output width as its group count (gcd fallback for narrow configurations)."""

    def __init__(
        self,
        n_patches: int,
        patch_size: int,
        d_model: int,
        stride: int | None = None,
        kernel_size: int = 63,
        depth: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"cnn_kernel_size must be odd, got {kernel_size}.")
        if depth < 1:
            raise ValueError(f"cnn_depth must be at least 1, got {depth}.")

        padded = 1 << (patch_size - 1).bit_length()
        if stride is None:
            # Reference output lengths: 8 for 1 s patches, 16 for 0.1 s.
            stride = padded // (8 if patch_size >= 100 else 16)
        if stride < 1 or padded % stride:
            raise ValueError(
                f"cnn_stride ({stride}) must be a positive divisor of the padded "
                f"patch length ({padded})."
            )
        out_size = padded // stride
        if d_model % out_size:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by the number of patch-CNN "
                f"output positions ({out_size} = {padded} // {stride})."
            )
        hidden = d_model // out_size

        pad_total = padded - patch_size
        pad = (pad_total // 2, pad_total - pad_total // 2)
        n_groups = math.gcd(out_size, hidden)

        layers: list[nn.Module] = [
            nn.Conv2d(
                1,
                hidden,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, (kernel_size - 1) // 2),
            ),
            nn.GroupNorm(n_groups, hidden),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv2d(
                    hidden, hidden, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
                ),
                nn.GroupNorm(n_groups, hidden),
                nn.GELU(),
            ]
        # Named children preserve the released patch_cnn.proj_in.* keys.
        self.add_module("pad", nn.ZeroPad2d(pad))
        self.add_module(
            "fold_grid", Rearrange("batch chan patch time -> batch 1 (chan patch) time")
        )
        self.add_module("proj_in", nn.Sequential(*layers))
        self.add_module(
            "unfold_grid",
            Rearrange(
                "batch hidden (chan patch) out -> batch chan patch (hidden out)",
                patch=n_patches,
            ),
        )


class _ChannelMetaEmbedding(nn.Module):
    """Concatenate PopT coordinates and modality/subtype embeddings."""

    def __init__(self, d_model: int):
        super().__init__()
        d_type = d_model // 4
        if d_type < 1:
            raise ValueError(
                f"d_model must be at least 4 for the electrode-type embedding, "
                f"got {d_model}."
            )
        self.coord_emb = _SinusoidalCoordEmbedding(d_model - d_type)
        self.type_emb = nn.Embedding(2, d_type)
        self.subtype_emb = nn.Embedding(3, d_type)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        coords = metadata[:, :3]
        coords_known = valid_location_mask(coords).to(coords.dtype)
        subtype = metadata[:, 4].long()
        subtype_known = (subtype >= 0).to(coords.dtype).unsqueeze(-1)
        position = self.coord_emb(torch.nan_to_num(coords)) * coords_known
        modality = (
            self.type_emb(metadata[:, 3].long())
            + self.subtype_emb(subtype.clamp(min=0)) * subtype_known
        )
        return torch.cat([position, modality], dim=-1)


class _SinusoidalCoordEmbedding(nn.Module):
    """PopT coordinate encoding: interleaved sin/cos per xyz axis.

    Coordinates are in millimetres; the shortest wavelength is 256 mm.
    An even feature count per axis is padded to the requested width."""

    def __init__(
        self, d_model: int, temperature: float = 2000.0, scale: float = 1 / 256
    ):
        super().__init__()
        n_dim = 3
        self.n_feats = d_model // n_dim // 2 * 2
        self.padding = d_model - self.n_feats * n_dim
        self.scale = scale * 2.0 * math.pi
        dim_t = torch.arange(0, self.n_feats, 2, dtype=torch.float32)
        dim_t = temperature ** (dim_t / self.n_feats)
        self.register_buffer("dim_t", dim_t, persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode coordinates of shape ``(..., 3)`` into ``(..., d_model)``."""
        angles = (xyz * self.scale).unsqueeze(-1) / self.dim_t
        pairs = torch.stack([angles.sin(), angles.cos()], dim=-1)
        emb = pairs.flatten(start_dim=-3)
        return F.pad(emb, (0, self.padding))


class _STCPE(nn.Module):
    """Encode full-height sliding windows and average overlapping outputs."""

    def __init__(
        self,
        d_model: int,
        ratio: int,
        window: int,
        activation: type[nn.Module],
        n_patches: int,
        mup_attention: bool = True,
    ):
        super().__init__()
        inner_dim = d_model // ratio
        if inner_dim < 2 or inner_dim % 2:
            raise ValueError(
                f"d_model // stcpe_ratio must be even and at least 2, got "
                f"{inner_dim} (d_model={d_model}, stcpe_ratio={ratio})."
            )
        # d_model // 256 heads as in Table 7, walked down until the head
        # dimension is a whole even number (the rotary embedding needs it).
        inner_heads = max(1, inner_dim // 32)
        while inner_heads > 1 and (
            inner_dim % inner_heads or (inner_dim // inner_heads) % 2
        ):
            inner_heads -= 1
        self.window = window
        self.n_patches = n_patches
        # Full-height windows, padded in time to include partial overlaps.
        self.stride = (1, 1)
        self.padding = (0, window - 1)
        self.n_windows = n_patches + window - 1
        # Fold features into batch; each temporal window spans every channel.
        self.fold_features = Rearrange(
            "batch chan patch dim -> (batch dim) 1 chan patch"
        )
        self.unfold_features = Rearrange(
            "(batch dim) 1 chan patch -> batch chan patch dim", dim=inner_dim
        )
        # One sequence per window for the inner encoder, and back again.
        self.windows_to_batch = Rearrange(
            "(batch dim) (chan window) n_win -> (batch n_win) (chan window) dim",
            dim=inner_dim,
            window=window,
        )
        self.batch_to_windows = Rearrange(
            "(batch n_win) (chan window) dim -> (batch dim) (chan window) n_win",
            n_win=self.n_windows,
            window=window,
        )
        self.down = nn.Linear(d_model, inner_dim)
        self.encoder = _AnyVariateEncoder(
            d_model=inner_dim,
            n_layers=1,
            num_heads=inner_heads,
            d_ff=4 * inner_dim,
            drop_prob=0.0,
            activation=activation,
            max_len=window,
            mup_attention=mup_attention,
        )
        self.up = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a token grid to a positional bias of the same shape."""
        n_chans = x.shape[1]
        kernel_size = (n_chans, self.window)
        grid_size = (n_chans, self.n_patches)

        x = self.down(x)
        flat = self.fold_features(x)
        unfolded = F.unfold(
            flat,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        windows = self.windows_to_batch(unfolded)

        var_id = torch.arange(n_chans, device=x.device).repeat_interleave(self.window)
        time_id = torch.arange(self.window, device=x.device).repeat(n_chans)
        encoded = self.encoder(windows, var_id=var_id, time_id=time_id)
        encoded = self.batch_to_windows(encoded)

        folded = F.fold(
            encoded,
            output_size=grid_size,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        # Fold in the working dtype: a scalar window divisor changes BF16
        # accumulation for large windows (e.g. 259 overlaps accumulate to 256).
        overlap = F.fold(
            torch.ones(
                1,
                n_chans * self.window,
                self.n_windows,
                dtype=x.dtype,
                device=x.device,
            ),
            output_size=grid_size,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        averaged = folded / overlap
        return self.up(self.unfold_features(averaged))


class _AnyVariateEncoder(nn.Module):
    """Any-variate blocks with shared temporal RoPE and a final RMSNorm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_ff: int,
        drop_prob: float,
        activation: type[nn.Module],
        max_len: int,
        mup_attention: bool = True,
    ):
        super().__init__()
        # Share parameter-free RoPE; learned channel biases remain per-layer.
        self.rotary = _RotaryEmbedding(d_model // num_heads, max_len)
        self.layers = nn.ModuleList(
            [
                _AnyVariateEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    drop_prob=drop_prob,
                    activation=activation,
                    rotary=self.rotary,
                    mup_attention=mup_attention,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-5)

    def forward(
        self, x: torch.Tensor, var_id: torch.Tensor, time_id: torch.Tensor
    ) -> torch.Tensor:
        """Encode ``(batch, seq, d_model)`` tokens tagged by channel and patch.

        ``var_id`` and ``time_id`` are 1D tensors of length ``seq`` holding the
        channel and temporal-patch index of every token.
        """
        for layer in self.layers:
            x = layer(x, var_id=var_id, time_id=time_id)
        return self.norm(x)


class _AnyVariateEncoderLayer(nn.Module):
    """Pre-norm attention and bias-free SwiGLU residual block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        drop_prob: float,
        activation: type[nn.Module],
        rotary: _RotaryEmbedding,
        mup_attention: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.self_attn = _AnyVariateAttention(
            d_model=d_model,
            num_heads=num_heads,
            drop_prob=drop_prob,
            rotary=rotary,
            mup_attention=mup_attention,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.ffn = FeedForwardBlock(
            d_model,
            expansion=4,
            drop_p=drop_prob,
            activation=activation,
            hidden_features=d_ff,
            gated=True,
            bias=False,
            output_drop_p=drop_prob,
        )

    def forward(
        self, x: torch.Tensor, var_id: torch.Tensor, time_id: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), var_id, time_id))
        return x + self.ffn(self.norm2(x))


class _AnyVariateAttention(nn.Module):
    """QK-normalized attention with temporal RoPE and learned channel bias.

    The reference uses one head per query group, i.e. ordinary multi-head
    attention. The two bias slots distinguish same from different channels."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        drop_prob: float,
        rotary: _RotaryEmbedding,
        mup_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # None lets SDPA use its default 1 / sqrt(head_dim).
        self.scale = 1.0 / self.head_dim if mup_attention else None
        self.drop_prob = drop_prob
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.channel_bias = nn.Embedding(2, num_heads)
        self.split_heads = Rearrange(
            "batch seq (heads dim) -> batch heads seq dim", heads=num_heads
        )
        self.merge_heads = Rearrange("batch heads seq dim -> batch seq (heads dim)")

    def forward(
        self, x: torch.Tensor, var_id: torch.Tensor, time_id: torch.Tensor
    ) -> torch.Tensor:
        query = self.rotary(self.q_norm(self.split_heads(self.q_proj(x))), time_id)
        key = self.rotary(self.k_norm(self.split_heads(self.k_proj(x))), time_id)
        value = self.split_heads(self.v_proj(x))

        # Bias slot 1 for same-channel pairs, slot 0 otherwise.
        same_channel = var_id.unsqueeze(-1) == var_id.unsqueeze(-2)
        weight = self.channel_bias.weight  # (2, num_heads)
        bias = torch.where(
            same_channel[None, None],
            weight[1].view(1, -1, 1, 1),
            weight[0].view(1, -1, 1, 1),
        ).to(query.dtype)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.drop_prob if self.training else 0.0,
            scale=self.scale,
        )
        out = self.merge_heads(out)
        return self.out_proj(out)


class _RotaryEmbedding(nn.Module):
    """Interleaved RoPE indexed by explicit temporal positions."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}.")
        theta = 1.0 / torch.pow(
            base, torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), theta)
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_id: torch.Tensor) -> torch.Tensor:
        """Rotate ``(batch, heads, seq, head_dim)`` by the angle of each position."""
        cos = self.cos[position_id].to(x.dtype)
        sin = self.sin[position_id].to(x.dtype)
        direct = cos * x
        rotated = rotate_pairs(x)
        return direct + sin * rotated
