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
from torch import nn
from torch.nn import functional as F

import braindecode.models.base as bd_base
from braindecode.models.util import extract_channel_locations_from_chs_info

# All ``__init__`` defaults below reproduce the published Zyphra/ZUNA
# ``config_infer.yaml``. Change them together with the upstream checkpoint, or
# pretrained weights will silently produce non-comparable embeddings.
_SIGNAL_ERROR = (
    "ZUNA requires inputs with 1280 time steps: 5 seconds sampled at 256 Hz."
)


# ---------------------------------------------------------------------------
# Rotary embedding (4D over channel position + coarse time)
# ---------------------------------------------------------------------------
def _precompute_freqs_cis(rot_dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, rot_dim, 2)[: rot_dim // 2].float() / rot_dim)
    )
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(end, rot_dim // 2, 2, 2)


def _apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (B, L, n_heads, head_dim); freqs_cis: (L, head_dim/2, 2, 2).
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2)
    freqs = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1], 2, 2).float()
    return (x_ * freqs).sum(-1).flatten(-2).type_as(x)


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
        return self._norm(x.float()).type_as(x) * self.weight


class _Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        self.wq = nn.Linear(dim, inner, bias=False)
        self.wk = nn.Linear(dim, inner, bias=False)
        self.wv = nn.Linear(dim, inner, bias=False)
        self.wo = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        shape = (b, seq_len, self.n_heads, self.head_dim)
        xq = _apply_rotary(self.wq(x).view(shape), freqs_cis)
        xk = _apply_rotary(self.wk(x).view(shape), freqs_cis)
        xv = self.wv(x).view(shape)
        # SDPA expects (B, n_heads, L, head_dim). Each batch element is its
        # own document — no mask needed.
        out = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        )
        return self.wo(out.transpose(1, 2).reshape(b, seq_len, -1))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, multiple_of: int = 256):
        super().__init__()
        hidden = multiple_of * math.ceil(int(8 * dim / 3) / multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, norm_eps: float):
        super().__init__()
        self.attention = _Attention(dim, n_heads, head_dim)
        self.feed_forward = _FeedForward(dim)
        self.attention_norm = _RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = _RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        return x + self.feed_forward(self.ffn_norm(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class _ZUNAEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        n_layers: int = 16,
        n_heads: int = 8,
        head_dim: int = 64,
        input_dim: int = 32,
        output_dim: int = 32,
        max_seqlen: int = 50,
        rope_theta: float = 10000.0,
        rope_dim: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if head_dim % rope_dim != 0:
            raise ValueError("head_dim must be divisible by rope_dim.")
        self.tok_embeddings = nn.Linear(input_dim, dim)
        self.registers = nn.Parameter(torch.zeros(1, input_dim))
        self.layers = nn.ModuleList(
            _TransformerBlock(dim, n_heads, head_dim, norm_eps) for _ in range(n_layers)
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
        regs = self.registers.expand(b, seq_len, -1).unsqueeze(2)
        tokens = torch.cat([regs, tokens.unsqueeze(2)], dim=2).reshape(
            b, 2 * seq_len, -1
        )

        # 4D RoPE: stack per-axis rotation matrices along head_dim/2. With
        # ``tok_idx`` of shape (L, rope_dim), the gather yields
        # (L, rope_dim, head_dim/(2*rope_dim), 2, 2); flatten gives the
        # expected (L, head_dim/2, 2, 2).
        tok_idx = tok_idx.repeat_interleave(2, dim=0)
        freqs_cis = self.freqs_cis[tok_idx].flatten(1, 2)

        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = h.reshape(b, seq_len, 2, -1)[:, :, 0]  # take the register slot
        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------
class ZUNA(bd_base.EEGModuleMixin, nn.Module):
    r"""ZUNA from Warner et al (2026) [Warner2026]_.

    :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel` :bdg-info:`Attention/Transformer`

    .. figure:: ../_static/model/zuna_arch.png
       :align: center
       :alt: ZUNA encoder-decoder architecture
       :width: 1000px

    ZUNA is a position-aware diffusion autoencoder for EEG superresolution,
    wrapped here with a Braindecode classification head.

    Ports the inference path of the public ``Zyphra/ZUNA`` encoder. Every
    architecture hyperparameter is a constructor argument and defaults to the
    published ``Zyphra/ZUNA`` config, so the defaults reproduce the pretrained
    encoder while smaller configurations can be built for training from
    scratch or research. To download the upstream encoder checkpoint from
    Hugging Face (requires ``pip install 'braindecode[hub]'``)::

        ZUNA.from_pretrained(
            "Zyphra/ZUNA", filename="model-00001-of-00001.safetensors"
        )

    Inputs must be 5-second EEG windows sampled at 256 Hz
    (``n_times=1280``). Channel coordinates are resolved by :meth:`forward`
    in this order, and any of the three sources is sufficient:

    1. ``channel_positions`` passed to :meth:`forward`.
    2. ``chs_info`` provided at construction (via
       :func:`braindecode.models.util.extract_channel_locations_from_chs_info`,
       cached at construction time).
    3. ``channel_names`` looked up in an MNE standard montage (defaults to
       ``"standard_1005"``; pass ``montage=None`` to disable).

    :meth:`forward` returns ``(batch, n_outputs)`` logits by default, or a
    dict of intermediate latents when ``return_features=True``.

    .. versionadded:: 1.6

    Parameters
    ----------
    n_outputs : int | None
        Number of output classes / regression targets.
    n_chans : int | None
        Number of EEG channels. Inferred from ``chs_info`` if not given.
    chs_info : list of dict | None
        MNE-style channel info; also used to extract coordinates.
    n_times : int | None
        Number of samples per window (must be ``1280``).
    input_window_seconds : float | None
        Window length in seconds (must be ``5.0``).
    sfreq : float | None
        Sampling frequency in Hz (must be ``256.0``).
    dim : int
        Transformer embedding dimension of the encoder.
    n_layers : int
        Number of transformer blocks in the encoder.
    n_heads : int
        Number of attention heads per block.
    head_dim : int
        Dimension of each attention head. Must be divisible by ``rope_dim``.
    fine_time_pts : int
        Number of fine time points per token (the encoder input dimension).
        ``n_times`` must be divisible by this value.
    latent_dim : int
        Per-token latent dimension produced by the encoder (the encoder
        output dimension).
    max_seqlen : int
        Size of the precomputed rotary table; must cover both ``pos_bins``
        and ``n_times // fine_time_pts``.
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
    drop_prob : float
        Accepted for braindecode API symmetry; the published encoder has no
        dropout, so the value is not wired into the pretrained architecture.
    activation : type[nn.Module]
        Accepted for braindecode API symmetry; the encoder uses the fixed
        SiLU feed-forward activation baked into the pretrained weights.

    References
    ----------
    .. [Warner2026] Warner, C., Mago, J., Huml, J.R., Osman, M. and
       Millidge, B., 2026. ZUNA: Flexible EEG Superresolution with
       Position-Aware Diffusion Autoencoders. arXiv preprint arXiv:2602.18478.
    """

    def __init__(
        self,
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        chs_info: Optional[list[dict]] = None,
        n_times: Optional[int] = 1280,
        input_window_seconds: Optional[float] = None,
        sfreq: Optional[float] = None,
        dim: int = 1024,
        n_layers: int = 16,
        n_heads: int = 8,
        head_dim: int = 64,
        fine_time_pts: int = 32,
        latent_dim: int = 32,
        max_seqlen: int = 50,
        rope_theta: float = 10000.0,
        rope_dim: int = 4,
        pos_bins: int = 50,
        pos_half_range: float = 0.12,
        norm_eps: float = 1e-5,
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ):
        n_times = 1280 if n_times is None else n_times
        input_window_seconds = (
            5.0 if input_window_seconds is None else input_window_seconds
        )
        sfreq = 256.0 if sfreq is None else sfreq
        if (n_times, sfreq, input_window_seconds) != (1280, 256.0, 5.0):
            raise ValueError(_SIGNAL_ERROR)
        if n_times % fine_time_pts != 0:
            raise ValueError(
                f"n_times ({n_times}) must be divisible by fine_time_pts "
                f"({fine_time_pts})."
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
            head_dim=head_dim,
            input_dim=fine_time_pts,
            output_dim=latent_dim,
            max_seqlen=max_seqlen,
            rope_theta=rope_theta,
            rope_dim=rope_dim,
            norm_eps=norm_eps,
        )
        self.final_layer = self._make_final_layer(self.n_outputs)

        # Cache positions resolved from chs_info, if any.
        cached = extract_channel_locations_from_chs_info(self._chs_info)
        self.register_buffer(
            "_cached_positions",
            torch.as_tensor(cached, dtype=torch.float32)
            if cached is not None
            else None,
            persistent=False,
        )

    def _make_final_layer(self, n_outputs: int) -> nn.Module:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_chans * self._latent_dim, n_outputs),
        )

    def _resolve_positions(
        self,
        channel_positions: Optional[torch.Tensor],
        channel_names: Optional[list[str]],
        montage: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if channel_positions is not None:
            pos = torch.as_tensor(channel_positions, dtype=dtype, device=device)
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError("channel_positions must have shape (n_chans, 3).")
            return pos
        if self._cached_positions is not None:
            return self._cached_positions.to(device=device, dtype=dtype)
        if channel_names is None:
            raise ValueError("ZUNA requires channel coordinates or names.")
        if montage is None:
            raise ValueError("ZUNA requires a montage to resolve channel names.")
        import mne

        ch_pos = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
        missing = [n for n in channel_names if n not in ch_pos]
        if missing:
            raise ValueError(
                f"Channel names {missing} not found in MNE montage {montage!r}."
            )
        return torch.stack(
            [
                torch.as_tensor(ch_pos[n], dtype=dtype, device=device)
                for n in channel_names
            ]
        )

    def _make_tok_idx(self, positions: torch.Tensor, coarse_time: int) -> torch.Tensor:
        # Discretise channel coords into [0, pos_bins) per axis, then
        # interleave with a per-token coarse-time index. Bucketing is run in
        # fp32 so model dtype (e.g. fp16) does not perturb bucket boundaries.
        positions = positions.float()
        normalised = (positions + self._pos_half_range) / (2 * self._pos_half_range)
        xyz = (normalised * self._pos_bins).long().clamp_(0, self._pos_bins - 1)
        xyz = xyz.repeat_interleave(coarse_time, dim=0)
        t = torch.arange(coarse_time, device=positions.device).repeat(self.n_chans)
        return torch.cat((xyz, t.unsqueeze(1)), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor] = None,
        channel_names: Optional[list[str]] = None,
        montage: Optional[str] = "standard_1005",
        return_features: bool = False,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(
                f"Expected (batch, n_chans, n_times); got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {x.shape[1]}.")
        if x.shape[2] != self.n_times:
            raise ValueError(_SIGNAL_ERROR)

        b, n_chans, n_times = x.shape
        coarse_time = n_times // self._fine_time_pts
        tokens = x.reshape(b, n_chans * coarse_time, self._fine_time_pts)

        positions = self._resolve_positions(
            channel_positions, channel_names, montage, x.device, x.dtype
        )
        if positions.shape[0] != n_chans:
            raise ValueError(
                f"Expected {n_chans} channel positions, got {positions.shape[0]}."
            )

        tok_idx = self._make_tok_idx(positions, coarse_time)
        token_latents = self.encoder(tokens, tok_idx)
        structured = token_latents.reshape(b, n_chans, coarse_time, self._latent_dim)
        features = structured.mean(dim=2)

        if return_features:
            return {
                "features": features,
                "cls_token": None,
                "token_latents": token_latents,
                "structured_latents": structured,
            }
        return self.final_layer(features)

    def get_output_shape(self) -> tuple[int, int]:
        return (1, self.n_outputs)

    def reset_head(self, n_outputs):
        """Replace the classification head for a new number of outputs."""
        self._n_outputs = n_outputs
        ref = next(self.parameters())
        self.final_layer = self._make_final_layer(n_outputs).to(
            device=ref.device, dtype=ref.dtype
        )
        # Keep the captured init config in sync so get_config()/from_config()
        # and push_to_hub() round-trips rebuild the head with the new size
        # instead of the value frozen at construction time.
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None and "n_outputs" in init_kwargs:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None and "n_outputs" in hub_config:
            hub_config["n_outputs"] = n_outputs

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Upstream Zyphra/ZUNA nests encoder weights under
        # ``model.encoder.*`` and bundles decoder weights we don't use.
        if any(k.startswith("model.encoder.") for k in state_dict):
            state_dict = {
                k.removeprefix("model.").removeprefix("encoder."): v
                for k, v in state_dict.items()
                if k.removeprefix("model.").startswith("encoder.")
            }
            return self.encoder.load_state_dict(state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
