# Original authors (Zyphra/ZUNA): Chris Warner, Jonas Mago, Jon Huml
# Braindecode adaptation: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Ports the encoder-side inference path from https://github.com/Zyphra/zuna.
# The upstream repository is released under the Apache License 2.0; this file
# therefore inherits Apache-2.0 and is NOT covered by braindecode's BSD-3
# license.
#
# License: Apache-2.0

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info


def _is_tracing() -> bool:
    # torch.compiler.is_compiling only exists from torch 2.1.
    compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return torch.jit.is_tracing() or bool(compiling and compiling())


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------
class ZUNA(EEGModuleMixin, nn.Module):
    r"""ZUNA from Warner et al (2026) [Warner2026]_.

    :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel` :bdg-info:`Attention/Transformer`

    .. figure:: ../_static/model/zuna_arch.png
       :align: center
       :alt: ZUNA encoder-decoder architecture
       :width: 1000px

    ZUNA is a position-aware diffusion autoencoder for EEG superresolution.

    Every architecture hyperparameter is a constructor argument and defaults to the
    published ``Zyphra/ZUNA1.1`` config, so the defaults reproduce the pretrained
    encoder while smaller configurations can be built for training from
    scratch. To download the pretrained encoder checkpoint from
    Hugging Face (requires ``pip install 'braindecode[hub]'``)::

        # Defaults to the upstream ZUNA1.1 classifier checkpoint; ``n_chans``
        # and ``n_outputs`` are montage- and task-dependent and must be given.
        ZUNA.from_pretrained(n_chans=19, n_outputs=4)

    Inputs must be EEG windows sampled at 256 Hz and may span 0.5 to 30.0
    seconds. The default constructor shape remains ``n_times=1280`` (i.e. 5 seconds), but the
    encoder accepts any runtime window length in the supported range as long
    as it lands on the coarse-time token grid
    (``fine_time_pts=32`` samples, i.e. 0.125 s, by default). Channel
    coordinates are resolved by :meth:`forward` in this order, and any of the
    three sources is sufficient:

    1. ``channel_positions`` passed to :meth:`forward`.
    2. ``chs_info`` provided at construction (via
       :func:`braindecode.models.util.extract_channel_locations_from_chs_info`,
       cached at construction time).
    3. ``channel_names`` looked up in an MNE standard montage (defaults to
       ``"standard_1005"``; pass ``montage=None`` to disable).

    :meth:`forward` returns ``(batch, n_outputs)`` logits by default, or a
    dict of intermediate latents when ``return_features=True``.

    .. rubric:: Architecture Overview

    Each channel's window is cut into non-overlapping patches of
    ``fine_time_pts`` samples (0.125 s at 256 Hz), giving a sequence of
    ``n_chans * (n_times // fine_time_pts)`` tokens. Tokens are linearly
    embedded, interleaved with learned register tokens, and processed by a
    stack of ``n_layers`` transformer blocks whose attention is rotated by a
    4D rotary embedding over the token's discretised scalp coordinates
    ``(x, y, z)`` and its coarse-time index. The register slots are projected
    to per-token latents, mean-pooled over time per channel, and classified.

    .. rubric:: Macro Components

    - ``ZUNA.encoder`` (:class:`torch.nn.Module`)

      **Operations**: ``tok_embeddings`` (linear patch embedding) →
      interleave ``registers`` → ``n_layers`` × ``_TransformerBlock``
      (RMS-normed grouped-query attention with 4D RoPE and QK-norm, SwiGLU
      feed-forward, sandwich norm) → ``norm`` → ``output`` linear projection
      to ``latent_dim`` per token.

      **Role**: pretrained, position-aware EEG token encoder.

    - ``ZUNA.final_layer`` (:class:`torch.nn.Sequential`)

      **Operations**: flatten the ``(n_chans, latent_dim)`` channel embedding
      → linear map to ``n_outputs``.

      **Role**: randomly initialised classification head fine-tuned on the
      downstream task.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal**: 0.125-s patch tokens plus a coarse-time rotary axis; the
      time dimension is mean-pooled after encoding.
    - **Spatial**: three rotary axes carry each channel's bucketed 3D scalp
      coordinates, making the encoder montage-agnostic.
    - **Spectral**: no explicit frequency decomposition; spectral structure
      is learned from the raw 256 Hz patches.

    .. rubric:: Additional Mechanisms

    - **Register tokens**: one learned register interleaved per data token;
      only register slots are read out, decoupling readout from input tokens.
    - **Grouped-query attention**: ``n_kv_heads`` may be smaller than
      ``n_heads``; grouping uses SDPA's native ``enable_gqa`` and therefore
      requires ``torch>=2.5`` (the default config has no grouping).
    - **Variable-length windows**: any 0.5-30 s window divisible by
      ``fine_time_pts`` is accepted at runtime without re-instantiation.

    .. versionadded:: 1.7

    Parameters
    ----------
    n_outputs : int | None
        Number of output classes / regression targets.
    n_chans : int | None
        Number of EEG channels. Inferred from ``chs_info`` if not given.
    chs_info : list of dict | None
        MNE-style channel info; also used to extract coordinates.
    n_times : int | None
        Number of samples per window. If ``None``, inferred from
        ``input_window_seconds`` and ``sfreq``, or defaults to ``1280`` when
        neither is specified. Must correspond to 0.5 to 30.0 seconds at
        256 Hz and be divisible by ``fine_time_pts``.
    input_window_seconds : float | None
        Window length in seconds. If ``None``, inferred from ``n_times`` and
        ``sfreq``, or from the default ``n_times`` and ``sfreq`` when neither
        is specified. Must be in the ZUNA1.1 training range of 0.5 to 30.0
        seconds.
    sfreq : float | None
        Sampling frequency in Hz. ZUNA1.1 expects ``256.0`` Hz inputs.
    dim : int
        Transformer embedding dimension of the encoder.
    n_layers : int
        Number of transformer blocks in the encoder.
    n_heads : int
        Number of attention heads per block.
    n_kv_heads : int | None
        Number of key/value attention heads. Defaults to ``n_heads``.
    head_dim : int
        Dimension of each attention head. Must be divisible by ``rope_dim``.
    fine_time_pts : int
        Number of fine time points per token (the encoder input dimension).
        ``n_times`` must be divisible by this value. The pretrained ZUNA1.1
        encoder uses ``32`` samples, equivalent to 0.125-second coarse-time
        tokens at 256 Hz.
    latent_dim : int
        Per-token latent dimension produced by the encoder (the encoder
        output dimension).
    max_seqlen : int
        Size of the precomputed rotary table; must cover both ``pos_bins``
        and the largest ``n_times // fine_time_pts`` that the instance should
        accept. The default ``256`` covers 30-second windows at 256 Hz.
    rope_theta : float
        Base period of the rotary positional embedding.
    rope_dim : int
        Number of rotary axes (4D RoPE over ``x, y, z, coarse_time``).
    pos_bins : int
        Number of discretisation bins per spatial axis for channel
        coordinates.
    pos_half_range : float
        Half-range (in metres) used to normalise channel coordinates before
        bucketing (scalp-radius normalisation).
    norm_eps : float
        Epsilon of the RMS normalisation layers.
    multiple_of : int
        Feed-forward hidden dimension is rounded up to a multiple of this
        value.
    ffn_dim_multiplier : float | None
        Optional multiplier applied to the feed-forward hidden dimension.
    sandwich_norm : bool
        Whether to apply the ZUNA1.1 post-attention and post-FFN RMS norms.
    qk_norm : bool
        Whether to apply ZUNA1.1 query/key RMS norms inside attention.
    drop_prob : float
        Accepted for braindecode API symmetry; the published encoder has no
        dropout, so the value is not wired into the pretrained architecture.
    activation : type[nn.Module]
        Accepted for braindecode API symmetry; the encoder uses the fixed
        SiLU feed-forward activation baked into the pretrained weights.

    References
    ----------
    .. [Warner2026] Warner, C., Mago, J., Huml, J.R. and Millidge, B.,
       2026. ZUNA1.1: A more flexible EEG foundation model for Denoising and
       Super-resolution. arXiv preprint arXiv:2607.27308.
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
        n_kv_heads: Optional[int] = None,
        head_dim: int = 64,
        fine_time_pts: int = 32,
        latent_dim: int = 32,
        max_seqlen: int = 256,
        rope_theta: float = 10000.0,
        rope_dim: int = 4,
        pos_bins: int = 50,
        pos_half_range: float = 0.12,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ):
        if rope_dim != 4:
            raise ValueError(
                "rope_dim must be 4 (x, y, z, coarse-time axes); "
                f"got rope_dim={rope_dim}."
            )
        # ZUNA1.1 checkpoint contract: 256 Hz windows, default 5 s (1280
        # samples). EEGModuleMixin checks n_times/input_window_seconds
        # consistency and derives one from the other.
        sfreq = 256.0 if sfreq is None else float(sfreq)
        if n_times is None:
            n_times = (
                round(input_window_seconds * sfreq)
                if input_window_seconds is not None
                else round(5.0 * sfreq)
            )
        self._validate_signal_window(
            n_times,
            sfreq=sfreq,
            fine_time_pts=fine_time_pts,
            max_seqlen=max_seqlen,
            pos_bins=pos_bins,
        )

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, input_window_seconds

        self._latent_dim = latent_dim
        self._fine_time_pts = fine_time_pts
        self._pos_bins = pos_bins
        self._pos_half_range = pos_half_range
        self.drop_prob = drop_prob
        self.activation = activation

        self.encoder = _ZUNAEncoder(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            input_dim=fine_time_pts,
            output_dim=latent_dim,
            max_seqlen=max_seqlen,
            rope_theta=rope_theta,
            rope_dim=rope_dim,
            norm_eps=norm_eps,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            sandwich_norm=sandwich_norm,
            qk_norm=qk_norm,
        )
        self.final_layer = self._make_final_layer(self.n_outputs)

        # Pure functions of construction-time state; never invalidated.
        self._montage_positions: dict = {}
        self._tok_idx_cache: dict = {}

        # Cache positions resolved from chs_info. Discard partial or
        # non-finite extractions so forward falls back to channel_names /
        # explicit positions instead of silently corrupting RoPE buckets.
        cached = extract_channel_locations_from_chs_info(self._chs_info)
        positions = None
        if cached is not None:
            positions = torch.as_tensor(cached, dtype=torch.float32)
            if (
                positions.shape[0] != self.n_chans
                or not torch.isfinite(positions).all()
            ):
                positions = None
        self.register_buffer("_cached_positions", positions, persistent=False)

    @staticmethod
    def _validate_signal_window(
        n_times: int,
        *,
        sfreq: float,
        fine_time_pts: int,
        max_seqlen: int,
        pos_bins: int,
    ) -> None:
        msg = (
            "ZUNA1.1 expects 256 Hz EEG windows from 0.5 to 30.0 seconds, "
            "with n_times divisible by fine_time_pts."
        )
        if not math.isclose(sfreq, 256.0):
            raise ValueError(f"{msg} Got sfreq={sfreq}.")
        if fine_time_pts <= 0 or n_times % fine_time_pts != 0:
            raise ValueError(
                f"{msg} Got n_times={n_times} and fine_time_pts={fine_time_pts}."
            )
        window_seconds = n_times / sfreq
        if not 0.5 <= window_seconds <= 30.0:
            raise ValueError(
                f"{msg} Got {window_seconds:g} seconds "
                f"({n_times} samples at {sfreq:g} Hz)."
            )
        coarse_time = n_times // fine_time_pts
        if max(pos_bins, coarse_time) > max_seqlen:
            raise ValueError(
                f"max_seqlen ({max_seqlen}) must be at least "
                f"max(pos_bins ({pos_bins}), n_times // fine_time_pts "
                f"({coarse_time}))."
            )

    def _validate_runtime_window(self, n_times: int) -> None:
        self._validate_signal_window(
            n_times,
            sfreq=self.sfreq,
            fine_time_pts=self._fine_time_pts,
            max_seqlen=self.encoder.freqs_cis.shape[0],
            pos_bins=self._pos_bins,
        )

    def _make_final_layer(self, n_outputs: int) -> nn.Module:
        return nn.Sequential(
            Rearrange("batch chans latent -> batch (chans latent)"),
            nn.Linear(self.n_chans * self._latent_dim, n_outputs),
        )

    def _resolve_positions(
        self,
        channel_positions: Optional[torch.Tensor],
        channel_names: Optional[list[str]],
        montage: Optional[str],
        device: torch.device,
    ) -> torch.Tensor:
        # Positions are only used for fp32 bucketing in _make_tok_idx, so
        # they stay fp32 regardless of the model/input dtype.
        if channel_positions is not None:
            pos = torch.as_tensor(channel_positions, dtype=torch.float32, device=device)
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError("channel_positions must have shape (n_chans, 3).")
            return pos
        if self._cached_positions is not None:
            return self._cached_positions.to(device=device)
        if channel_names is None:
            raise ValueError("ZUNA requires channel coordinates or names.")
        if montage is None:
            raise ValueError("ZUNA requires a montage to resolve channel names.")
        key = (tuple(channel_names), montage)
        pos = self._montage_positions.get(key)
        if pos is None:
            import mne

            ch_pos = mne.channels.make_standard_montage(montage).get_positions()[
                "ch_pos"
            ]
            missing = [n for n in channel_names if n not in ch_pos]
            if missing:
                raise ValueError(
                    f"Channel names {missing} not found in MNE montage {montage!r}."
                )
            pos = torch.stack(
                [torch.as_tensor(ch_pos[n], dtype=torch.float32) for n in channel_names]
            )
            self._montage_positions[key] = pos
        return pos.to(device=device)

    def _make_tok_idx(self, positions: torch.Tensor, coarse_time: int) -> torch.Tensor:
        # Discretise channel coords into [0, pos_bins) per axis, then
        # interleave with a per-token coarse-time index. Bucketing is run in
        # fp32 so model dtype (e.g. fp16) does not perturb bucket boundaries.
        normalised = (positions + self._pos_half_range) / (2 * self._pos_half_range)
        xyz = (normalised * self._pos_bins).long().clamp_(0, self._pos_bins - 1)
        xyz = repeat(xyz, "c d -> (c t) d", t=coarse_time)
        t = repeat(
            torch.arange(coarse_time, device=positions.device),
            "t -> (c t) 1",
            c=self.n_chans,
        )
        return torch.cat((xyz, t), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor] = None,
        channel_names: Optional[list[str]] = None,
        montage: Optional[str] = "standard_1005",
        return_features: bool = False,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3 or x.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected (batch, {self.n_chans}, n_times); "
                f"got shape {tuple(x.shape)}."
            )
        b, n_chans, n_times = x.shape
        self._validate_runtime_window(n_times)
        coarse_time = n_times // self._fine_time_pts
        tokens = rearrange(x, "b c (t p) -> b (c t) p", p=self._fine_time_pts)

        # Without explicit per-call positions, tok_idx is a pure function of
        # (window length, montage source), so cache it — except mid-trace,
        # where a cached fake tensor would break torch.export.
        cacheable = channel_positions is None and not _is_tracing()
        cache_key = (
            coarse_time,
            x.device,
            None if channel_names is None else (tuple(channel_names), montage),
        )
        tok_idx = self._tok_idx_cache.get(cache_key) if cacheable else None
        if tok_idx is None:
            positions = self._resolve_positions(
                channel_positions, channel_names, montage, x.device
            )
            if positions.shape[0] != n_chans:
                raise ValueError(
                    f"Expected {n_chans} channel positions, got {positions.shape[0]}."
                )
            tok_idx = self._make_tok_idx(positions, coarse_time)
            if cacheable:
                if len(self._tok_idx_cache) >= 64:
                    self._tok_idx_cache.clear()
                self._tok_idx_cache[cache_key] = tok_idx
        token_latents = self.encoder(tokens, tok_idx)
        structured = rearrange(token_latents, "b (c t) d -> b c t d", c=n_chans)
        features = structured.mean(dim=2)

        if return_features:
            return {
                "features": features,
                # No CLS token in ZUNA; key required by the foundation-model
                # contract. Not a credential (Bandit B105 false positive).
                "cls_token": None,  # nosec B105
                "token_latents": token_latents,
                "structured_latents": structured,
            }
        return self.final_layer(features)

    def get_output_shape(self) -> tuple[int, int]:
        # The mixin's forward-pass implementation feeds zeros without channel
        # positions, which _resolve_positions rejects — so answer statically.
        return (1, self.n_outputs)

    def reset_head(self, n_outputs):
        """Replace the classification head for a new number of outputs."""
        self._n_outputs = n_outputs
        ref = next(self.parameters())
        self.final_layer = self._make_final_layer(n_outputs).to(
            device=ref.device, dtype=ref.dtype
        )
        # Sync the captured init config so get_config()/push_to_hub()
        # round-trips rebuild the head with the new size.
        for cfg_name in ("_braindecode_init_kwargs", "_hub_mixin_config"):
            cfg = getattr(self, cfg_name, None)
            if cfg is not None and "n_outputs" in cfg:
                cfg["n_outputs"] = n_outputs

    @staticmethod
    def _normalise_encoder_key(key: str) -> str:
        key = key.removeprefix("model.").removeprefix("encoder.")
        if key.endswith(".norm.weight"):
            key = f"{key.removesuffix('.norm.weight')}.weight"
        return key

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Upstream Zyphra/ZUNA nests encoder weights under
        # ``model.encoder.*`` and bundles decoder weights we don't use.
        if any(k.startswith("model.encoder.") for k in state_dict):
            state_dict = {
                self._normalise_encoder_key(k): v
                for k, v in state_dict.items()
                if k.removeprefix("model.").startswith("encoder.")
            }
            encoder_keys = self.encoder.state_dict().keys()
            if not any(k in encoder_keys for k in state_dict):
                raise ValueError(
                    "No upstream ZUNA keys matched the encoder after "
                    "remapping; the checkpoint layout is not the expected "
                    "'model.encoder.*'."
                )
            return self.encoder.load_state_dict(state_dict, strict=strict, **kwargs)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    #: Default Hugging Face repo for :meth:`from_pretrained`.
    _HF_DEFAULT_REPO = "Zyphra/ZUNA1.1"
    _HF_DEFAULT_FILENAME = "classifier/model-00001-of-00001.safetensors"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load pretrained ZUNA weights, defaulting to the ZUNA1.1 classifier.

        ``pretrained_model_name_or_path`` defaults to ``"Zyphra/ZUNA1.1"``, and
        ``filename`` defaults to the classifier checkpoint under
        ``"classifier/model-00001-of-00001.safetensors"``. Only the encoder is
        loaded; decoder weights in the upstream checkpoint are ignored, and the
        Braindecode classification head is randomly initialised. ``n_chans`` (or
        ``chs_info``) and ``n_outputs`` are montage- and task-dependent and must
        be supplied.
        """
        model_id = args[0] if args else kwargs.get("pretrained_model_name_or_path")
        if model_id is None:
            model_id = cls._HF_DEFAULT_REPO
            kwargs["pretrained_model_name_or_path"] = model_id
        if model_id == cls._HF_DEFAULT_REPO and kwargs.get("filename") is None:
            kwargs["filename"] = cls._HF_DEFAULT_FILENAME
        return super().from_pretrained(*args, **kwargs)


# ---------------------------------------------------------------------------
# Rotary embedding (4D over channel position + coarse time)
# ---------------------------------------------------------------------------
def _precompute_freqs_cis(rot_dim: int, end: int, theta: float) -> torch.Tensor:
    # Complex rotary table e^{i * t * freq}: (end, rot_dim / 2), complex64.
    freqs = 1.0 / (
        theta ** (torch.arange(0, rot_dim, 2)[: rot_dim // 2].float() / rot_dim)
    )
    angles = torch.outer(torch.arange(end).float(), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def _apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (B, L, n_heads, head_dim); freqs_cis: complex (L, head_dim/2).
    # Rotation via native complex multiply: (x0 + i x1) * e^{i theta}.
    pairs = rearrange(x.float(), "b l h (d two) -> b l h d two", two=2)
    rotated = torch.view_as_complex(pairs.contiguous()) * rearrange(
        freqs_cis, "l d -> 1 l 1 d"
    )
    return rearrange(
        torch.view_as_real(rotated), "b l h d two -> b l h (d two)"
    ).type_as(x)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    ``torch.nn.RMSNorm`` is only available from PyTorch 2.4, but braindecode
    supports ``torch>=2.0``; this shippable equivalent (same approach as
    :class:`~braindecode.models.REVE` and ``CodeBrain``) keeps the model
    importable on older PyTorch while preserving the ``.weight`` parameter
    name so upstream ZUNA checkpoints still load.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # type_as(weight) keeps half-precision models (model.half()) feeding
        # their linears half inputs; fp32 models are unchanged.
        return self._norm(x.float()).type_as(self.weight) * self.weight


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: Optional[int] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        self.heads_per_group = n_heads // self.n_kv_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = _RMSNorm(head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(head_dim, eps=norm_eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        xq = self.q_norm(rearrange(self.wq(x), "b l (h d) -> b l h d", h=self.n_heads))
        xk = self.k_norm(
            rearrange(self.wk(x), "b l (h d) -> b l h d", h=self.n_kv_heads)
        )
        xq = _apply_rotary(xq, freqs_cis)
        xk = _apply_rotary(xk, freqs_cis)
        xv = rearrange(self.wv(x), "b l (h d) -> b l h d", h=self.n_kv_heads)
        # SDPA expects (B, n_heads, L, head_dim). Each batch element is its
        # own document — no mask needed. GQA configs (n_kv_heads < n_heads)
        # use SDPA's native grouping and therefore need torch>=2.5.
        q, k, v = (rearrange(t, "b l h d -> b h l d") for t in (xq, xk, xv))
        if self.heads_per_group > 1:
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.wo(rearrange(out, "b h l d -> b l (h d)"))


class _FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden = int(8 * dim / 3)
        if ffn_dim_multiplier is not None:
            hidden = int(ffn_dim_multiplier * hidden)
        hidden = multiple_of * math.ceil(hidden / multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        norm_eps: float,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.attention = _Attention(
            dim,
            n_heads,
            head_dim,
            n_kv_heads=n_kv_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
        )
        self.feed_forward = _FeedForward(
            dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier
        )
        self.attention_norm = _RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = _RMSNorm(dim, eps=norm_eps)
        self.attention_norm_post = (
            _RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        )
        self.ffn_norm_post = (
            _RMSNorm(dim, eps=norm_eps) if sandwich_norm else nn.Identity()
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Residual stream is kept in fp32; submodule outputs are cast back up
        # in case autocast ran them in half precision.
        x = x.float()
        h = x + self.attention_norm_post(
            self.attention(self.attention_norm(x), freqs_cis).float()
        )
        return h + self.ffn_norm_post(self.feed_forward(self.ffn_norm(h)).float())


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class _ZUNAEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        n_layers: int = 16,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        head_dim: int = 64,
        input_dim: int = 32,
        output_dim: int = 32,
        max_seqlen: int = 256,
        rope_theta: float = 10000.0,
        rope_dim: int = 4,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        sandwich_norm: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        if head_dim % rope_dim != 0:
            raise ValueError("head_dim must be divisible by rope_dim.")
        self.tok_embeddings = nn.Linear(input_dim, dim)
        self.registers = nn.Parameter(torch.zeros(1, input_dim))
        self.layers = nn.ModuleList(
            _TransformerBlock(
                dim,
                n_heads,
                head_dim,
                norm_eps,
                n_kv_heads=n_kv_heads,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                sandwich_norm=sandwich_norm,
                qk_norm=qk_norm,
            )
            for _ in range(n_layers)
        )
        self.norm = _RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, output_dim, bias=False)
        self.register_buffer(
            "freqs_cis",
            _precompute_freqs_cis(head_dim // rope_dim, max_seqlen, rope_theta),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L, input_dim); tok_idx: (L, rope_dim).
        b, seq_len, _ = tokens.shape

        # Interleave one register token per source token, doubling the length.
        regs = self.registers.expand(b, seq_len, -1)
        tokens = rearrange(
            torch.stack((regs, tokens), dim=2), "b l two d -> b (l two) d"
        )

        # 4D RoPE: concatenate the per-axis complex phases along head_dim/2.
        # ``tok_idx`` (L, rope_dim) gathers (L, rope_dim, head_dim/(2*rope_dim))
        # from the complex table; merging axes gives (L, head_dim/2).
        tok_idx = repeat(tok_idx, "l d -> (l two) d", two=2)
        freqs_cis = rearrange(self.freqs_cis[tok_idx], "l a d -> l (a d)")

        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = rearrange(h, "b (l two) d -> b l two d", two=2)[:, :, 0]  # register slot
        return self.output(self.norm(h))
