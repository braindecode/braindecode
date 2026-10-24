# Authors: Chris Warner, Jonas Mago, Jon Huml
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: Apache-2.0

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import scaled_dot_product_attention
from braindecode.functional.attention import _hpu_lazy
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    warn_if_sfreq_differs,
)
from braindecode.modules import PatchTokenizer
from braindecode.modules.layers import RMSNorm


class ZUNA(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""ZUNA from Warner et al. (2026) [Warner2026ZUNA]_.

    :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel` :bdg-info:`Attention/Transformer`

    .. versionadded:: 1.7

    .. figure:: ../_static/model/zuna_arch.png
       :align: center
       :alt: ZUNA pretraining encoder-decoder architecture
       :width: 1000px

    ZUNA was introduced as a diffusion autoencoder for masked EEG channel
    reconstruction and super-resolution [Warner2026ZUNA]_. ZUNA1.1 retains
    that objective and adds query-key normalization, sandwich normalization,
    pretraining windows from 0.5 to 30 seconds, and eight channel and time
    dropout schemes [Warner2026ZUNA11]_.

    This Braindecode class contains the ZUNA1.1 encoder followed by a
    classification head. It does not contain the diffusion decoder, channel
    masking, or reconstruction sampler shown in the figure. :meth:`forward`
    returns logits of shape ``(batch, n_outputs)``. With
    ``return_features=True``, it returns the per-channel encoder features used
    by the classification head.

    Signal size and sampling frequency follow the standard Braindecode model
    arguments. Supply either ``n_times`` or both ``input_window_seconds`` and
    ``sfreq``; ZUNA does not set these values implicitly. ZUNA1.1 was trained
    at 256 Hz with 32-sample tokens (another ``sfreq`` warns). This
    implementation does not resample, filter, or normalize the input. Channel
    coordinates are read from ``chs_info`` (in metres, MNE head frame) when the
    model is constructed and stored in fixed rotary position buffers; a channel
    without a location takes the standard 10-05 position of its name, with a
    warning.

    .. rubric:: Architecture Overview

    :class:`~braindecode.modules.PatchTokenizer` splits every channel into
    non-overlapping patches of ``fine_time_pts`` samples. The resulting
    ``n_chans * (n_times // fine_time_pts)`` patches are serialized in
    channel-major order. A learned register is placed before each patch, and
    both are projected to ``dim`` before entering the transformer blocks.

    Self-attention is bidirectional. Four-dimensional rotary positions encode
    each token's discretized scalp coordinates ``(x, y, z)`` and coarse-time
    index. The encoder reads the register positions, projects them to
    ``latent_dim``, restores the channel and patch axes, and averages over the
    patch axis. A linear layer maps the concatenated channel features to
    ``n_outputs``.

    .. rubric:: Macro Components

    - ``ZUNA.patch_embedding`` (:class:`torch.nn.Sequential`)

      **Operations**: split ``(batch, channel, time)`` into patches with
      :class:`~braindecode.modules.PatchTokenizer`, then rearrange
      ``(channel, temporal_patch)`` into one token axis.

      **Role**: produce one continuous-valued token for every channel and time
      patch.

    - ``ZUNA.encoder`` (:class:`torch.nn.Module`)

      **Operations**: ``tok_embeddings`` (linear patch embedding) →
      interleave ``registers`` → ``n_layers`` × ``_TransformerBlock``
      (RMS-normed multi-head attention with 4D RoPE and QK-norm, SwiGLU
      feed-forward, sandwich norm) → ``norm`` → ``output`` linear projection
      to ``latent_dim`` per token.

      **Role**: encode channel-time patches while retaining their spatial and
      temporal coordinates.

    - ``ZUNA.final_layer`` (:class:`torch.nn.Sequential`)

      **Operations**: rearrange the ``(n_chans, latent_dim)`` channel features
      into one axis → linear map to ``n_outputs``.

      **Role**: produce task logits from the pooled encoder features.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal**: fixed-size patches provide local samples, and the fourth
      rotary axis identifies each patch's coarse-time index. The patch axis is
      averaged after encoding.
    - **Spatial**: the first three rotary axes contain bucketed 3D coordinates
      from ``chs_info``. A model instance uses the montage supplied at
      construction.
    - **Spectral**: there is no Fourier or filter-bank stage. The encoder
      receives raw time-domain patches.

    .. rubric:: Additional Mechanisms

    - **Register tokens**: one learned register is paired with every patch.
      Only register outputs enter the latent projection. With
      ``registers=False`` the encoder runs on the patch tokens alone (half the
      tokens) and the patch outputs enter the latent projection instead.
    - **ZUNA1.1 normalization**: query-key RMS normalization and optional
      post-attention and post-feed-forward RMS normalization match the changes
      introduced in ZUNA1.1.
    - **Fixed positional grid**: channel and coarse-time rotary values are
      computed during construction. Inputs passed to :meth:`forward` must have
      the configured channel count and window length.

    Parameters
    ----------
    dim : int, optional
        Transformer embedding dimension. The default is ``1024``.
    n_layers : int, optional
        Number of transformer blocks. The default is ``16``.
    n_heads : int, optional
        Number of attention heads per block. The default is ``8``.
    head_dim : int, optional
        Dimension of each attention head. It must be divisible by eight. The
        default is ``64``.
    fine_time_pts : int, optional
        Number of fine time points per token (the encoder input dimension).
        ``n_times`` must be divisible by this value unless ``on_non_divisible``
        is ``"pad"`` or ``"crop"``. The default is ``32``, or 0.125 seconds for
        data sampled at 256 Hz.
    latent_dim : int, optional
        Per-token output dimension of the encoder. The default is ``32``.
    max_seqlen : int, optional
        Minimum length of the rotary frequency table. The table is extended to
        ``max(pos_bins, the number of temporal patches)`` when the window needs
        more rows, so longer windows than the release's 8192 samples are
        accepted (positions past the pretraining range are extrapolated by the
        rotary embedding). The default is ``256``.
    rope_theta : float, optional
        Base period of the rotary positional embedding. The default is
        ``10000.0``.
    pos_bins : int, optional
        Number of buckets per spatial coordinate. The default is ``50``.
    pos_half_range : float, optional
        Half-range (in metres) used to normalise channel coordinates before
        bucketing. Coordinates at or beyond this range are clipped to the first
        or last bucket. The default is ``0.12``.
    norm_eps : float, optional
        Epsilon of the RMS normalization layers. The default is ``1e-5``.
    multiple_of : int, optional
        Feed-forward hidden dimension is rounded up to a multiple of this
        value. The default is ``256``.
    ffn_dim_multiplier : float | None, optional
        Multiplier applied before rounding the feed-forward hidden dimension.
        The default is ``None``.
    sandwich_norm : bool, optional
        Apply RMS normalization after attention and the feed-forward layer. The
        default is ``True``.
    qk_norm : bool, optional
        Apply RMS normalization to queries and keys. The default is ``True``.
    activation : type[nn.Module], optional
        Feed-forward activation class. The default is
        :class:`torch.nn.SiLU`.
    on_non_divisible : {"pad", "crop", "error"}, optional
        How to handle an ``n_times`` that is not a multiple of ``fine_time_pts``,
        passed to :class:`braindecode.modules.PatchTokenizer`: ``"error"`` raises,
        ``"pad"`` right-pads the last patch with zeros, ``"crop"`` drops the
        trailing samples. The default is ``"pad"``.
    registers : bool, optional
        Place a learned register before each patch, as in ZUNA1.1. ``False``
        drops the registers: the sequence holds the patch tokens only (half the
        tokens, so about half the memory and a large share of the compute per
        step), each keeps its rotary position, and the head reads the patch
        outputs. This changes the model (it is not the published ZUNA1.1
        encoder), but the parameters are the same, so pretrained weights load
        unchanged; the register parameter is then not used and does not
        require gradients. The default is ``True``.

        .. versionadded:: 1.8.2

    Notes
    -----
    The full ZUNA1.1 pretraining model has an encoder and a rectified-flow
    decoder. Its encoder latent is regularized with a maximum mean discrepancy
    loss. The decoder and both pretraining losses are outside this class.

    In the paper's downstream experiments, token latents were averaged and the
    encoder was fine-tuned with the classifier. A frozen encoder with a linear
    head performed worse. The authors also found that checkpoints with lower
    reconstruction error were not necessarily better for classification
    [Warner2026ZUNA11]_.

    References
    ----------
    .. [Warner2026ZUNA] Warner, C., Mago, J., Huml, J. R., Osman, M., and
       Millidge, B. (2026). ZUNA: Flexible EEG Superresolution with
       Position-Aware Diffusion Autoencoders. arXiv:2602.18478.
       https://arxiv.org/abs/2602.18478
    .. [Warner2026ZUNA11] Warner, C., Mago, J., Huml, J. R., and Millidge, B.
       (2026). ZUNA1.1: A More Flexible EEG Foundation Model for Denoising and
       Super-Resolution. arXiv:2607.27308.
       https://arxiv.org/abs/2607.27308
    """

    def __init__(
        self,
        # braindecode parameters
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        chs_info: Optional[list[dict]] = None,
        n_times: Optional[int] = None,
        input_window_seconds: Optional[float] = None,
        sfreq: Optional[float] = None,
        # model-specific parameters
        *,
        dim: int = 1024,
        n_layers: int = 16,
        n_heads: int = 8,
        head_dim: int = 64,
        fine_time_pts: int = 32,
        latent_dim: int = 32,
        max_seqlen: int = 256,
        rope_theta: float = 10000.0,
        pos_bins: int = 50,
        pos_half_range: float = 0.12,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
        on_non_divisible: str = "pad",
        registers: bool = True,
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
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

        # Variables
        self.num_channels = self.n_chans
        self.latent_dim = latent_dim
        if on_non_divisible == "pad":
            # PatchTokenizer right-pads the last partial patch at forward time.
            coarse_time_points = -(-self.n_times // fine_time_pts)
        else:
            coarse_time_points = self.n_times // fine_time_pts
        rotary_axis_dim = head_dim // 4

        # Checks
        warn_if_sfreq_differs("ZUNA", self._sfreq, 256)
        channel_positions = torch.as_tensor(
            _channel_positions(self._chs_info, self.num_channels), dtype=torch.float32
        )
        if channel_positions.abs().max() > 1.0:  # a head is ~0.1 m across
            warnings.warn(
                "ZUNA expects channel locations in metres (MNE head frame); got "
                f"coordinates up to {float(channel_positions.abs().max()):g}, which "
                f"are clipped to +/-{pos_half_range} m.",
                UserWarning,
                stacklevel=2,
            )
        if head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by eight for 4D RoPE.")

        # Layers
        self.patch_embedding = nn.Sequential(
            PatchTokenizer(
                patch_size=fine_time_pts,
                n_times=self.n_times,
                on_non_divisible=on_non_divisible,
            ),
            Rearrange(
                "batch channel temporal_patch sample "
                "-> batch (channel temporal_patch) sample"
            ),
        )

        normalized_positions = (channel_positions + pos_half_range) / (
            2 * pos_half_range
        )
        channel_position_indices = (
            (normalized_positions * pos_bins).long().clamp(0, pos_bins - 1)
        )
        channel_position_indices = channel_position_indices.repeat_interleave(
            coarse_time_points, dim=0
        )
        coarse_time_indices = torch.arange(coarse_time_points).repeat(self.num_channels)
        token_position_indices = torch.cat(
            (channel_position_indices, coarse_time_indices.unsqueeze(1)), dim=1
        )

        rotary_frequency_table = _build_rotary_frequency_table(
            # RoPE is analytic and the tables are non-persistent buffers: extend the
            # table to the window instead of capping it (rows < max_seqlen unchanged).
            torch.arange(
                max(max_seqlen, pos_bins, coarse_time_points), dtype=torch.float32
            ),
            axis_dim=rotary_axis_dim,
            theta=rope_theta,
        )
        token_rotary_frequencies = rearrange(
            rotary_frequency_table[token_position_indices],
            "token coordinate rotary_frequency -> token (coordinate rotary_frequency)",
        )
        if registers:  # each register takes the position of the patch after it
            token_rotary_frequencies = token_rotary_frequencies.repeat_interleave(
                2, dim=0
            )

        self.encoder = _ZUNAEncoder(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            input_dim=fine_time_pts,
            output_dim=latent_dim,
            rotary_frequencies=token_rotary_frequencies,
            norm_eps=norm_eps,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            sandwich_norm=sandwich_norm,
            qk_norm=qk_norm,
            activation=activation,
            registers=registers,
        )
        self.final_layer = nn.Sequential(
            Rearrange("batch channel latent -> batch (channel latent)"),
            nn.Linear(self.num_channels * self.latent_dim, self.n_outputs),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        return_features: bool = False,
    ):
        patch_tokens = self.patch_embedding(input_tensor)
        token_latents = self.encoder(patch_tokens)
        structured_latents = token_latents.reshape(
            token_latents.shape[0], self.num_channels, -1, self.latent_dim
        )
        features = structured_latents.mean(dim=2)
        logits = self.final_layer(features)

        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {
                "features": features,
                "cls_token": None,  # nosec B105
            }
        return logits

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new number of outputs."""
        self._n_outputs = n_outputs
        self._update_init_kwargs(n_outputs=n_outputs)
        self.final_layer = nn.Sequential(
            Rearrange("batch channel latent -> batch (channel latent)"),
            nn.Linear(self.num_channels * self.latent_dim, n_outputs),
        )


def _channel_positions(chs_info, n_chans):
    """``(n_chans, 3)`` locations from ``chs_info``; a channel without a usable
    ``loc`` takes the standard 10-05 position of its name, as MNE places it."""
    locs = extract_channel_locations_from_chs_info(
        chs_info, num_channels=n_chans, fill_missing=True
    )
    if locs is None:
        raise ValueError("ZUNA requires channel locations in chs_info.")
    missing = np.flatnonzero(~np.isfinite(locs).all(1) | ~locs.any(1))
    if len(missing) == 0:
        return locs
    import mne

    names = [str(chs_info[i].get("ch_name", "")) for i in missing]
    unknown = [n for n in names if not n]
    if not unknown:
        montage = (
            "colin27_1005"  # standard_1005's name from MNE 1.13
            if "colin27_1005" in mne.channels.get_builtin_montages()
            else "standard_1005"
        )
        info = mne.create_info(names, 256.0, "eeg")
        info.set_montage(
            montage, match_case=False, match_alias=True, on_missing="ignore"
        )
        filled = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=locs.dtype)
        ok = np.isfinite(filled).all(1) & filled.any(1)
        unknown = [n for n, k in zip(names, ok) if not k]
    if unknown:
        raise ValueError(
            "ZUNA requires finite 3D locations for every channel; "
            f"{unknown} have none and are not in the standard 10-05 montage."
        )
    warnings.warn(
        f"ZUNA: {len(names)} channel(s) without a location (e.g. {names[0]}) take "
        "the standard 10-05 position of their name.",
        UserWarning,
        stacklevel=3,
    )
    locs[missing] = filled
    return locs


def _build_rotary_frequency_table(
    positions: torch.Tensor, *, axis_dim: int, theta: float
) -> torch.Tensor:
    """Build ZUNA's per-axis rotary frequencies with native PyTorch."""
    embedding_dim = max(axis_dim, 4)
    inverse_frequencies = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                embedding_dim,
                2,
                device=positions.device,
                dtype=torch.float32,
            )
            / embedding_dim
        )
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * inverse_frequencies
    return angles.repeat_interleave(2, dim=-1)[:, :axis_dim]


class _RotaryPositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        # Layers
        self.merge_rotary_pairs = Rearrange(
            "batch sequence head frequency_pair complex_component "
            "-> batch sequence head (frequency_pair complex_component)"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rotary_cosine: torch.Tensor,
        rotary_sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting() and query.device.type == "hpu":
            fused = _hpu_fused_rotary(query, key, rotary_cosine, rotary_sine)
            if fused is not None:
                return fused
        rotary_cosine = rotary_cosine.unsqueeze(0).unsqueeze(2)
        rotary_sine = rotary_sine.unsqueeze(0).unsqueeze(2)

        batch_size, sequence_length, num_heads, head_dim = query.shape
        work = torch.promote_types(query.dtype, torch.float32)
        # Each (real, imaginary) pair (a, b) rotates to (-b, a): swap the pair
        # here; the sign is in ``rotary_sine`` (see _ZUNAEncoder).
        query_pairs = query.to(work).reshape(
            batch_size, sequence_length, num_heads, head_dim // 2, 2
        )
        key_pairs = key.to(work).reshape(
            batch_size, sequence_length, num_heads, head_dim // 2, 2
        )
        rotated_query = self.merge_rotary_pairs(query_pairs.flip(-1))
        rotated_key = self.merge_rotary_pairs(key_pairs.flip(-1))

        query = (query.to(work) * rotary_cosine + rotated_query * rotary_sine).type_as(
            query
        )
        key = (key.to(work) * rotary_cosine + rotated_key * rotary_sine).type_as(key)
        return query, key


def _define_hpu_pairwise_rotary():
    """Gaudi's fused pairwise RoPE kernel as a custom op.

    A custom op (not an ``autograd.Function``) so that a compiled region keeps it
    as one opaque node. The kernel has no gradient of its own; the rotation is
    orthogonal, so the gradient is the same kernel with the opposite angle. Only
    called on HPU tensors; defining it imports nothing from the Gaudi bridge.
    """
    if hasattr(torch.ops.braindecode, "zuna_pairwise_rotary"):  # module reloaded
        return torch.ops.braindecode.zuna_pairwise_rotary

    # Explicit schema: this module's postponed annotations are strings.
    @torch.library.custom_op(
        "braindecode::zuna_pairwise_rotary",
        mutates_args=(),
        schema="(Tensor x, Tensor cosine, Tensor sine) -> Tensor",
    )
    def rotary(x, cosine, sine):
        from habana_frameworks.torch.hpex.kernels import (
            RotaryPosEmbeddingMode,
            apply_rotary_pos_emb,
        )

        return apply_rotary_pos_emb(
            x.contiguous(), cosine, sine, None, 0, RotaryPosEmbeddingMode.PAIRWISE
        )

    @rotary.register_fake
    def _(x, cosine, sine):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1], inputs[2])

    def backward(ctx, grad_output):
        cosine, sine = ctx.saved_tensors
        return rotary(grad_output, cosine, -sine), None, None

    rotary.register_autograd(backward, setup_context=setup_context)
    return rotary


_hpu_pairwise_rotary = _define_hpu_pairwise_rotary()


@torch.jit.unused
def _hpu_fused_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_cosine: torch.Tensor,
    rotary_sine: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """``_RotaryPositionEmbedding`` in HPU eager mode (and compiled regions).

    Gaudi's fused pairwise RoPE kernel. Not in lazy mode, where the graph
    compiler fuses the formula itself and the kernel was slower (2.47 -> 2.86 s
    per train step, 21 channels x 1280 samples, batch 64): None there.

    The formula runs as about ten element-wise ops on float32 copies of the
    queries and keys (a quarter of a ZUNA block's train-step time at 107,520
    tokens per batch in eager mode); the kernel does it in one pass, also inside
    a compiled region. Same float32 arithmetic, so the outputs are identical;
    the gradient is rounded to the input dtype once instead of once per term.
    """
    if not torch.compiler.is_compiling() and _hpu_lazy():
        return None
    work = torch.promote_types(query.dtype, torch.float32)
    cosine = rotary_cosine.unsqueeze(0).unsqueeze(2)
    # The kernel applies the pair sign itself: pass sin(angle), which is
    # ``rotary_sine`` at the imaginary component of every pair.
    sine = rotary_sine[:, 1::2].repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)
    return (
        _hpu_pairwise_rotary(query.to(work), cosine, sine).type_as(query),
        _hpu_pairwise_rotary(key.to(work), cosine, sine).type_as(key),
    )


class _Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        head_dim: int,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()

        # Variables
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Layers
        self.wq = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, embedding_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.rotary_embedding = _RotaryPositionEmbedding()

    def forward(
        self,
        input_tensor: torch.Tensor,
        rotary_cosine: torch.Tensor,
        rotary_sine: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = input_tensor.shape
        attention_shape = (
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )
        query = self.q_norm(self.wq(input_tensor).reshape(attention_shape))
        key = self.k_norm(self.wk(input_tensor).reshape(attention_shape))
        value = self.wv(input_tensor).reshape(attention_shape)
        query, key = self.rotary_embedding(query, key, rotary_cosine, rotary_sine)

        attention_output = scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.n_heads * self.head_dim
        )
        return self.wo(attention_output)


class _FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        # Variables
        hidden_dim = int(8 * embedding_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)

        # Layers
        self.w1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.activation = activation()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(input_tensor)) * self.w3(input_tensor))


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        head_dim: int,
        norm_eps: float,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        # Layers
        self.attention = _Attention(
            embedding_dim,
            n_heads,
            head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
        )
        self.feed_forward = _FeedForward(
            embedding_dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            activation=activation,
        )
        self.attention_norm = RMSNorm(embedding_dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(embedding_dim, eps=norm_eps)
        self.attention_norm_post = (
            RMSNorm(embedding_dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        )
        self.ffn_norm_post = (
            RMSNorm(embedding_dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        rotary_cosine: torch.Tensor,
        rotary_sine: torch.Tensor,
    ) -> torch.Tensor:
        # Residual stream in float32 at least; the sub-layers run in the
        # weights' dtype.
        dtype = self.attention.wq.weight.dtype
        residual_dtype = torch.promote_types(dtype, torch.float32)
        input_tensor = input_tensor.to(residual_dtype)
        hidden_states = input_tensor + self.attention_norm_post(
            self.attention(
                self.attention_norm(input_tensor).to(dtype), rotary_cosine, rotary_sine
            ).to(residual_dtype)
        )
        return hidden_states + self.ffn_norm_post(
            self.feed_forward(self.ffn_norm(hidden_states).to(dtype)).to(residual_dtype)
        )


class _ZUNAEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        input_dim: int,
        output_dim: int,
        rotary_frequencies: torch.Tensor,
        norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
        registers: bool = True,
    ):
        super().__init__()
        self.use_registers = registers

        # Layers
        self.tok_embeddings = nn.Linear(input_dim, dim)
        # Kept without registers too, so the pretrained state dict loads as is.
        self.registers = nn.Parameter(torch.zeros(1, input_dim))
        self.registers.requires_grad_(registers)
        self.layers = nn.ModuleList(
            _TransformerBlock(
                dim,
                n_heads,
                head_dim,
                norm_eps,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                sandwich_norm=sandwich_norm,
                qk_norm=qk_norm,
                activation=activation,
            )
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, output_dim, bias=False)

        # Buffers
        self.register_buffer(
            "rotary_cosine",
            rotary_frequencies.cos(),
            persistent=False,
        )
        # Sine with the pair rotation's sign folded in: -sin on the real, +sin
        # on the imaginary component of every pair (exact, saves a negation).
        pair_sign = torch.tensor([-1.0, 1.0]).repeat(rotary_frequencies.shape[-1] // 2)
        self.register_buffer(
            "rotary_sine",
            rotary_frequencies.sin() * pair_sign,
            persistent=False,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_registers:
            hidden_states = self.tok_embeddings(patch_tokens)
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states, self.rotary_cosine, self.rotary_sine
                )
            return self.output(self.norm(hidden_states).to(self.output.weight.dtype))
        batch_size, sequence_length, patch_size = patch_tokens.shape
        register_tokens = self.registers.expand(batch_size, sequence_length, -1)
        interleaved_tokens = torch.stack(
            (register_tokens, patch_tokens), dim=2
        ).reshape(batch_size, 2 * sequence_length, patch_size)

        hidden_states = self.tok_embeddings(interleaved_tokens)
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.rotary_cosine, self.rotary_sine)
        register_latents = hidden_states.reshape(batch_size, sequence_length, 2, -1)[
            :, :, 0
        ]
        return self.output(self.norm(register_latents).to(self.output.weight.dtype))
