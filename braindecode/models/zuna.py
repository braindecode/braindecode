# Authors: Chris Warner, Jonas Mago, Jon Huml
#
# License: BSD (3-clause)
#
# This code ports the encoder-side inference path from:
# https://github.com/Zyphra/zuna

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

import braindecode.models.base as bd_base
from braindecode.models.util import extract_channel_locations_from_chs_info


# ZUNA requires flex_attention from >=torch 2.5 during sequence packing
try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
        noop_mask,
    )

    HAS_FLEX = True
except ImportError:
    BlockMask = None
    create_block_mask = None
    flex_attention = None
    noop_mask = None
    HAS_FLEX = False


class InitStdFactor(Enum):
    DISABLED = "disabled"
    GLOBAL_DEPTH = "global_depth"
    CURRENT_DEPTH = "current_depth"
    DIM_RATIO = "dim_ratio"


@dataclass
class _ZUNAEncoderArgs:
    dim: int = 1024
    n_layers: int = 16
    head_dim: int = 64
    n_heads: Optional[int] = 8
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"
    max_seqlen: int = 50
    rope_dim: int = 4
    tok_idx_type: str = "{x,y,z,tc}"
    encoder_input_dim: int = 32
    encoder_output_dim: int = 32
    encoder_latent_downsample_factor: int = 1
    encoder_sliding_window: int = 65536
    encoder_hidden_dim: Optional[int] = None
    stft_global_sigma: float = 0.1
    dropout_type: str = "zeros"


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seqlen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seqlen, n_kv_heads * n_rep, head_dim)
    )


def _precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def _reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int
) -> torch.Tensor:
    ndim = x.ndim
    expected_shape = (x.shape[seq_dim], x.shape[-3], 2, 2)
    if freqs_cis.shape != expected_shape:
        raise ValueError(
            f"freqs_cis has shape {tuple(freqs_cis.shape)}, expected "
            f"{expected_shape} for rotary embedding."
        )
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def _apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_, seq_dim).float()
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class _RotaryEmbedding(nn.Module):
    def __init__(self, theta: float, head_dim: int, max_seqlen: int, rope_dim: int):
        super().__init__()
        if head_dim % rope_dim != 0:
            raise ValueError("head_dim must be divisible by rope_dim.")
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_dim = rope_dim
        self.register_buffer(
            "freqs_cis",
            _precompute_freqs_cis(head_dim // rope_dim, max_seqlen, theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = _precompute_freqs_cis(
            self.head_dim // self.rope_dim, self.max_seqlen, self.theta
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.freqs_cis[:seqlen]


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)
        return (output.float() * self.weight.float()).type_as(x)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class _Attention(nn.Module):
    def __init__(self, args: _ZUNAEncoderArgs):
        super().__init__()
        self.dim = args.dim
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // self.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.rope_dim = args.rope_dim

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor],
        mask: Optional[BlockMask],
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        output_shape = xq.shape

        xq = xq.view(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        if self.rope_dim == 1:
            freqs = freq_cis[tok_idx] if tok_idx is not None else freq_cis[:seq_len]
            xq, xk = _apply_rotary_emb(xq, xk, 1, freqs)
        elif self.rope_dim == 4:
            if tok_idx is None:
                raise ValueError("4D RoPE requires tok_idx.")
            freq_parts = [freq_cis[tok_idx[:, i]] for i in range(self.rope_dim)]
            xq, xk = _apply_rotary_emb(xq, xk, 1, torch.cat(freq_parts, dim=1))
        elif self.rope_dim != 0:
            raise ValueError(f"Unsupported rope_dim={self.rope_dim}.")

        xk = _repeat_kv(xk, self.heads_per_group)
        xv = _repeat_kv(xv, self.heads_per_group)
        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
        out = flex_attention(xq, xk, xv, block_mask=mask)
        out = out.transpose(1, 2).contiguous()
        return self.wo(out.reshape(output_shape))

    def reset_parameters(self, init_std: Optional[float], factor: float):
        init_std = init_std or self.dim**-0.5
        for w in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(
                w.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std
            )
        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


class _FeedForward(nn.Module):
    def __init__(self, args: _ZUNAEncoderArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * math.ceil(hidden_dim / args.multiple_of)
        self.dim = args.dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, init_std: Optional[float], factor: float):
        in_std = init_std or self.dim**-0.5
        out_std = (init_std or self.hidden_dim**-0.5) / factor
        for w in (self.w1, self.w3):
            nn.init.trunc_normal_(
                w.weight, mean=0.0, std=in_std, a=-3 * in_std, b=3 * in_std
            )
        nn.init.trunc_normal_(
            self.w2.weight, mean=0.0, std=out_std, a=-3 * out_std, b=3 * out_std
        )


class _TransformerBlock(nn.Module):
    def __init__(self, args: _ZUNAEncoderArgs):
        super().__init__()
        self.attention = _Attention(args)
        self.feed_forward = _FeedForward(args)
        self.attention_norm = _RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = _RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor],
        mask: Optional[BlockMask],
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freq_cis, tok_idx, mask)
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std: Optional[float], factor: float):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


def _create_document_mask(
    lengths: torch.Tensor, sliding_window: int, device: torch.device
):
    lengths_cpu = lengths.detach().cpu()
    doc_end = lengths_cpu.cumsum(0).tolist()
    doc_start = [0] + doc_end[:-1]
    total = int(lengths_cpu.sum().item())
    doc_bounds = list(zip(doc_start, doc_end))

    def mask_mod(b, h, q_idx, kv_idx):
        valid = torch.zeros_like(noop_mask(b, h, q_idx, kv_idx))
        for start, end in doc_bounds:
            in_doc = (
                (q_idx >= start)
                & (q_idx < end)
                & (kv_idx >= start)
                & (kv_idx < end)
            )
            in_window = (q_idx - kv_idx).abs() <= sliding_window
            valid = valid | (in_doc & in_window)
        return valid

    return create_block_mask(
        mask_mod,
        None,
        None,
        total,
        total,
        device=device,
        _compile=device.type == "cuda",
    )


class _ZUNAEncoder(nn.Module):
    def __init__(self, args: _ZUNAEncoderArgs):
        super().__init__()
        if args.encoder_hidden_dim is not None:
            args = _ZUNAEncoderArgs(
                **{**args.__dict__, "dim": args.encoder_hidden_dim}
            )
        self.dim = args.dim
        self.downsample_factor = args.encoder_latent_downsample_factor
        self.sliding_window = args.encoder_sliding_window
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.dropout_type = args.dropout_type
        self.dropout_vec = (
            nn.Parameter(args.stft_global_sigma * torch.rand(1, args.encoder_input_dim))
            if args.dropout_type == "learnable"
            else None
        )

        self.rope_embeddings = _RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            rope_dim=args.rope_dim,
        )
        self.layers = nn.ModuleList(
            [_TransformerBlock(args) for _ in range(args.n_layers)]
        )
        self.tok_embeddings = nn.Linear(args.encoder_input_dim, args.dim)
        self.norm = _RMSNorm(args.dim, eps=args.norm_eps)
        self.registers = nn.Parameter(torch.zeros(1, args.encoder_input_dim))
        self.output = nn.Linear(args.dim, args.encoder_output_dim, bias=False)

    def _interleave_registers(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch, seqlen, dim = x.shape
        df = self.downsample_factor
        num_groups = math.ceil(seqlen / df)
        new_seqlen = num_groups * df
        if new_seqlen > seqlen:
            x = torch.cat([x, x.new_zeros(batch, new_seqlen - seqlen, dim)], dim=1)
        x = x.reshape(batch, num_groups, df, dim)
        regs = self.registers.expand(batch, num_groups, -1).unsqueeze(2)
        token_values = torch.cat([regs, x], dim=2).reshape(batch, -1, dim)
        return token_values.contiguous(), num_groups

    def forward(
        self,
        token_values: torch.Tensor,
        seq_lens: torch.Tensor,
        tok_idx: torch.Tensor,
    ) -> torch.Tensor:
        original_seqlen = token_values.shape[1]
        token_values, num_groups = self._interleave_registers(token_values)
        do_idx = token_values.sum(dim=2).eq(0).squeeze(0)
        if self.dropout_vec is not None and do_idx.any():
            token_values[:, do_idx, :] = self.dropout_vec

        h = self.tok_embeddings(token_values)
        mask = _create_document_mask(
            seq_lens * (self.downsample_factor + 1),
            self.sliding_window,
            device=token_values.device,
        )
        tok_idx = tok_idx.repeat_interleave(repeats=self.downsample_factor + 1, dim=1)
        tok_idx = tok_idx.squeeze(0)
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen)

        for layer in self.layers:
            h = layer(h, freq_cis=freq_cis, tok_idx=tok_idx, mask=mask)

        h = h.reshape(
            h.shape[0], num_groups, self.downsample_factor + 1, h.shape[-1]
        )
        registers = h[:, :, 0, :][:, :original_seqlen, :].contiguous()
        return self.output(self.norm(registers))


def _resolve_channel_positions(
    channel_positions: Optional[torch.Tensor],
    channel_names: Optional[list[str]],
    chs_info: Optional[list[dict]],
    device: torch.device,
    dtype: torch.dtype,
    montage: Optional[str],
) -> torch.Tensor:
    if channel_positions is not None:
        pos = torch.as_tensor(channel_positions, dtype=dtype, device=device)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("channel_positions must have shape (n_chans, 3).")
        return pos

    chs_positions = extract_channel_locations_from_chs_info(chs_info)
    if chs_positions is not None:
        return torch.as_tensor(chs_positions, dtype=dtype, device=device)

    if channel_names is not None and montage is not None:
        import mne

        mne_montage = mne.channels.make_standard_montage(montage)
        pos_dict = mne_montage.get_positions()["ch_pos"]
        missing = [name for name in channel_names if name not in pos_dict]
        if missing:
            raise ValueError(
                f"Channel names {missing} not found in MNE standard montage "
                f"{montage!r}."
            )
        return torch.stack(
            [
                torch.as_tensor(pos_dict[name], dtype=dtype, device=device)
                for name in channel_names
            ]
        )

    if channel_names is None:
        raise ValueError("ZUNA requires channels coordinates or names")
    raise ValueError("ZUNA requires a montage to resolve channel names")


def _discretize_channel_positions(
    channel_positions: torch.Tensor,
    num_bins: int,
    extremes_type: str,
) -> torch.Tensor:
    if extremes_type == "twelves":
        extremes = channel_positions.new_tensor(
            [[-0.12, -0.12, -0.12], [0.12, 0.12, 0.12]]
        )
    elif extremes_type == "thirteens":
        extremes = channel_positions.new_tensor(
            [[-0.13, -0.13, -0.13], [0.13, 0.13, 0.13]]
        )
    else:
        raise ValueError(
            "chan_pos_xyz_extremes_type must be 'twelves' or 'thirteens'."
        )
    normalized = (channel_positions - extremes[0]) / (extremes[1] - extremes[0])
    return torch.clamp((normalized * num_bins).long(), 0, num_bins - 1)


def _zuna_signal_error_message() -> str:
    n_times, sample_frequency, input_window_seconds = _zuna_expected_signal()
    return (
        "ZUNA requires inputs with "
        f"{n_times} time steps: {input_window_seconds:g} seconds sampled at "
        f"{sample_frequency:g} Hz."
    )


def _validate_zuna_n_times(n_times: Optional[int]) -> None:
    expected_n_times, _, _ = _zuna_expected_signal()
    if n_times is not None and n_times != expected_n_times:
        raise ValueError(_zuna_signal_error_message())


def _zuna_expected_signal() -> tuple[int, float, float]:
    return 1280, 256.0, 5.0


def _zuna_hf_files() -> tuple[str, str, str]:
    return "Zyphra/ZUNA", "model-00001-of-00001.safetensors", "config.json"


def _zuna_channel_position_config() -> tuple[int, str]:
    return 50, "twelves"


def _zuna_published_config() -> dict:
    """Architecture config baked from the public ``Zyphra/ZUNA`` Hub repo.

    Kept in-tree so model construction does not need a network call.
    """
    return {
        "dim": 1024,
        "n_layers": 16,
        "head_dim": 64,
        "encoder_input_dim": 32,
        "encoder_output_dim": 32,
        "encoder_latent_downsample_factor": 1,
        "encoder_sliding_window": 65536,
        "max_seqlen": 50,
        "rope_dim": 4,
        "rope_theta": 10000.0,
        "tok_idx_type": "{x,y,z,tc}",
        "dropout_type": "zeros",
        "stft_global_sigma": 0.1,
    }


def _build_zuna_encoder_args(model_config: dict) -> _ZUNAEncoderArgs:
    if model_config.get("encoder_latent_downsample_factor", 1) != 1:
        raise ValueError("ZUNA currently requires encoder_latent_downsample_factor=1.")
    if model_config.get("tok_idx_type", "{x,y,z,tc}") != "{x,y,z,tc}":
        raise ValueError('ZUNA currently requires tok_idx_type="{x,y,z,tc}".')
    if model_config.get("rope_dim", 4) != 4:
        raise ValueError("ZUNA currently requires rope_dim=4.")

    encoder_arg_names = _ZUNAEncoderArgs.__dataclass_fields__
    encoder_config = {
        key: value for key, value in model_config.items() if key in encoder_arg_names
    }
    return _ZUNAEncoderArgs(**encoder_config)


class ZUNA(bd_base.EEGModuleMixin, nn.Module):
    r"""Encoder-only ZUNA model (see Github repo for reconstruction).

    ZUNA tokenizes EEG into channel-by-coarse-time tokens and, in this version, returns latent
    encodings for downstream probes. The
    encoder architecture is fixed to the public ``Zyphra/ZUNA`` Hugging Face
    config (see :func:`_zuna_published_config`); the constructor only exposes
    Braindecode signal metadata and channel-coordinate resolution options.

    ZUNA only accepts ``n_times=1280`` at
    ``sfreq=256`` Hz (5 s windows). Other shapes raise exceptions at construction or
    forward.

    Channel coordinates are resolved by :func:`forward` in this priority
    order, and any of the three sources is sufficient:

    1. ``channel_positions`` passed to :meth:`forward` (highest priority).
    2. ``chs_info`` provided at construction (extracted via
       :func:`braindecode.models.util.extract_channel_locations_from_chs_info`).
    3. ``channel_names`` looked up in an MNE standard montage (defaults to
       ``"standard_1005"``; pass ``montage=None`` in :meth:`forward` to
       disable this fallback).

    Parameters
    ----------
    n_outputs : int | None
        Declared output count carried on the module for downstream probes;
        ZUNA itself has no classification head (see :meth:`reset_head`).
    n_chans : int | None
        Number of EEG channels. Inferred from ``chs_info`` when not given.
    chs_info : list of dict | None
        MNE-style channel info. Used both to infer ``n_chans`` and to supply
        ``loc`` coordinates when ``channel_positions`` is not passed to
        :meth:`forward`.
    n_times : int | None
        Number of samples per window. Must be ``1280`` (the only value the
        encoder accepts, for now...); defaults to ``1280``.
    input_window_seconds : float | None
        Window length in seconds. Must be ``5.0``.
    sfreq : float | None
        Sampling frequency in Hz. Must be ``256.0``.
    """

    def __init__(
        self,
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        chs_info: Optional[list[dict]] = None,
        n_times: Optional[int] = 1280,
        input_window_seconds: Optional[float] = None,
        sfreq: Optional[float] = None,
    ):
        if not HAS_FLEX:
            raise ImportError(
                "ZUNA requires Pytorch with torch.nn.attention.flex_attention."
            )
        expected_n_times, expected_sfreq, expected_window = _zuna_expected_signal()
        n_times = expected_n_times if n_times is None else n_times
        input_window_seconds = (
            expected_window if input_window_seconds is None else input_window_seconds
        )
        sfreq = expected_sfreq if sfreq is None else sfreq
        if (
            n_times != expected_n_times
            or sfreq != expected_sfreq
            or input_window_seconds != expected_window
        ):
            raise ValueError(_zuna_signal_error_message())

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        model_config = _zuna_published_config()
        self.num_fine_time_pts = model_config["encoder_input_dim"]
        self.encoder_output_dim = model_config["encoder_output_dim"]
        (
            self.num_bins_discretize_xyz_chan_pos,
            self.chan_pos_xyz_extremes_type,
        ) = _zuna_channel_position_config()
        self.encoder = _ZUNAEncoder(_build_zuna_encoder_args(model_config))
        self._init_weights()
        self.final_layer = nn.Identity()

    def _init_weights(self):
        self.encoder.rope_embeddings.reset_parameters()
        for depth, layer in enumerate(self.encoder.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.encoder.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.encoder.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.encoder.init_std_factor]
            layer.init_weights(self.encoder.init_base_std, factor)

    def load_pretrained_weights(self) -> None:
        """Download and load the public ``Zyphra/ZUNA`` encoder weights.

        Network call is deliberately *not* performed in ``__init__``; users
        opt in by calling this method (or instantiate via
        :meth:`from_pretrained` for a braindecode-format checkpoint).
        """
        if not bd_base.HAS_HF_HUB:
            raise ImportError(
                "load_pretrained_weights() requires huggingface_hub. "
                "Install with: pip install 'braindecode[hub]'"
            )
        from safetensors.torch import load_file as safe_load

        repo_id, weights_filename, _ = _zuna_hf_files()
        weights_path = bd_base.huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=weights_filename, token=False
        )
        state_dict = safe_load(weights_path, device="cpu")
        encoder_state = {}
        for key, value in state_dict.items():
            key = key.removeprefix("model.")
            if key.startswith("encoder."):
                encoder_state[key.removeprefix("encoder.")] = value
        self.encoder.load_state_dict(encoder_state, strict=True)

    def _apply_channel_mask(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor],
        dropped_channels: Optional[list[Union[int, str]]],
        channel_names: Optional[list[str]],
    ) -> torch.Tensor:
        if channel_mask is not None and dropped_channels is not None:
            raise ValueError("Pass either channel_mask or dropped_channels, not both.")
        if channel_mask is None and dropped_channels is None:
            return x

        if dropped_channels is not None:
            mask = torch.ones(self.n_chans, dtype=torch.bool, device=x.device)
            drop_indices = []
            for channel in dropped_channels:
                if isinstance(channel, str):
                    if channel_names is None:
                        raise ValueError(
                            "String dropped_channels require channel_names."
                        )
                    if channel not in channel_names:
                        raise ValueError(f"Unknown dropped channel {channel!r}.")
                    drop_indices.append(channel_names.index(channel))
                else:
                    drop_indices.append(int(channel))
            invalid = [
                index for index in drop_indices if index < 0 or index >= self.n_chans
            ]
            if invalid:
                raise ValueError(
                    f"dropped_channels contains invalid indices {invalid}."
                )
            if drop_indices:
                mask[drop_indices] = False
            channel_mask = mask

        mask = torch.as_tensor(channel_mask, dtype=torch.bool, device=x.device)
        if mask.ndim == 1:
            if mask.shape[0] != self.n_chans:
                raise ValueError("channel_mask must have shape (n_chans,).")
            mask = mask.view(1, self.n_chans, 1)
        elif mask.ndim == 2:
            if mask.shape != (x.shape[0], self.n_chans):
                raise ValueError("channel_mask must have shape (batch, n_chans).")
            mask = mask.unsqueeze(-1)
        else:
            raise ValueError(
                "channel_mask must have shape (n_chans,) or (batch, n_chans)."
            )
        return x * mask.to(dtype=x.dtype)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("ZUNA expects input shape (batch, n_chans, n_times).")
        expected_n_times, _, _ = _zuna_expected_signal()
        if x.shape[-1] != expected_n_times:
            raise ValueError(_zuna_signal_error_message())
        if x.shape[-1] % self.num_fine_time_pts != 0:
            raise ValueError(
                "n_times must be divisible by "
                f"num_fine_time_pts={self.num_fine_time_pts}."
            )
        batch, n_chans, n_times = x.shape
        if n_chans != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {n_chans}.")
        coarse_time = n_times // self.num_fine_time_pts
        return x.reshape(batch, n_chans, coarse_time, self.num_fine_time_pts).reshape(
            batch, n_chans * coarse_time, self.num_fine_time_pts
        )

    def _make_tok_idx(
        self, channel_positions: torch.Tensor, coarse_time: int
    ) -> torch.Tensor:
        channel_positions_discrete = _discretize_channel_positions(
            channel_positions,
            self.num_bins_discretize_xyz_chan_pos,
            self.chan_pos_xyz_extremes_type,
        )
        chan_pos = channel_positions_discrete.repeat_interleave(coarse_time, dim=0)
        t_coarse = torch.arange(coarse_time, device=channel_positions.device)
        t_coarse = t_coarse.repeat(self.n_chans).unsqueeze(1)
        return torch.cat((chan_pos, t_coarse), dim=1).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor] = None,
        channel_names: Optional[list[str]] = None,
        montage: Optional[str] = "standard_1005",
        channel_mask: Optional[torch.Tensor] = None,
        dropped_channels: Optional[list[Union[int, str]]] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            EEG signal of shape ``(batch, n_chans, n_times)``.
        channel_positions : torch.Tensor | None
            Channel coordinates of shape ``(n_chans, 3)``.
        channel_names : list[str] | None
            Channel names used to resolve coordinates from a standard montage
            when neither ``channel_positions`` nor ``chs_info`` coordinates are
            available.
        montage : str | None
            MNE standard montage name used only as a forward-time fallback for
            ``channel_names``. Set to ``None`` to disable name-based lookup.
        channel_mask : torch.Tensor | None
            Boolean mask of channels to keep, shaped ``(n_chans,)`` or
            ``(batch, n_chans)``. Masked channels are zeroed before encoding.
        dropped_channels : list[int | str] | None
            Channel indices or names to zero before encoding.
        return_features : bool
            If ``True`` return a dict with ``"features"`` (time-pooled
            per-channel latents, the canonical probe input), ``"cls_token"``
            (``None``), and the ZUNA-specific ``"token_latents"`` /
            ``"structured_latents"`` tensors. Default returns ``"features"``
            directly so the encoder slots into the standard braindecode
            ``forward(x)`` contract.
        """
        x = self._apply_channel_mask(x, channel_mask, dropped_channels, channel_names)
        tokens = self._tokenize(x)
        batch, seq_len, _ = tokens.shape
        coarse_time = x.shape[-1] // self.num_fine_time_pts
        channel_positions = _resolve_channel_positions(
            channel_positions,
            channel_names,
            self._chs_info,
            device=x.device,
            dtype=x.dtype,
            montage=montage,
        )
        if channel_positions.shape[0] != self.n_chans:
            raise ValueError(
                f"Expected {self.n_chans} channel positions, "
                f"got {channel_positions.shape[0]}."
            )
        tok_idx = self._make_tok_idx(channel_positions, coarse_time)

        # Pack the batch as ``batch`` independent documents so the encoder
        # runs once. The document mask isolates each sample's attention.
        packed_tokens = tokens.reshape(1, batch * seq_len, -1)
        packed_tok_idx = tok_idx.repeat(1, batch, 1)
        seq_lens = torch.full((batch,), seq_len, dtype=torch.long, device=x.device)

        packed_latents = self.encoder(packed_tokens, seq_lens, packed_tok_idx)
        token_latents = packed_latents.reshape(
            batch, seq_len, self.encoder_output_dim
        )
        structured_latents = token_latents.reshape(
            batch, self.n_chans, coarse_time, self.encoder_output_dim
        )
        time_pooled_latents = structured_latents.mean(dim=2)

        if return_features:
            return {
                "features": time_pooled_latents,
                "cls_token": None,
                "token_latents": token_latents,
                "structured_latents": structured_latents,
            }
        return time_pooled_latents

    def get_output_shape(self) -> tuple[int, int, int]:
        return (1, self.n_chans, self.encoder_output_dim)

    def reset_head(self, n_outputs):
        """Update ``n_outputs`` metadata; ZUNA has no classification head.

        ZUNA is a feature extractor: :meth:`forward` returns time-pooled
        per-channel latents (shape ``(batch, n_chans, encoder_output_dim)``)
        and the model exposes :class:`torch.nn.Identity` as ``final_layer``.
        Wire a downstream probe yourself to obtain logits; this method only
        records the declared output count on the module.
        """
        self._n_outputs = n_outputs
