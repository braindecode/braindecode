# Authors: Mahir Jain (mahir@mannas.ai)
#
# License: Apache-2.0
"""AXON: an axis-factorized EEG foundation model."""

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.util import resolve_montage_name

# Positions are stored in the model in centimetres, because that is the unit the
# pretrained spatial encoder saw. MNE ``chs_info[i]["loc"]`` is in metres.
_METRES_TO_CM = 100.0


class AXON(EEGModuleMixin, nn.Module):
    r"""AXON, an axis-factorized EEG foundation model from Jain et al. (2026) [axon2026]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel`

    AXON (**AX**\ is-factorized **O**\ perator **N**\ etwork) is a transformer
    encoder pretrained with masked autoencoding on clinical and research EEG.
    Each recording window is cut into tokens, one per electrode per one-second
    patch, so the tokens form a grid of electrodes by time steps. Instead of
    dense self-attention over all tokens, every layer runs two attention paths
    in parallel:

    - a **temporal path**, in which each token attends to the tokens of its own
      electrode across time, and
    - a **spatial path**, in which each token attends to the tokens of all
      electrodes at the same time step.

    A small gate predicts, for every token, two weights that sum to one, and
    the token's update is the weighted sum of the two path outputs. On a full
    electrode-by-time grid any two tokens are connected after two layers.

    .. rubric:: Temporal windows

    The temporal path is computed twice: once over all time steps of the
    electrode and once restricted to time steps at most ``temporal_window``
    patches away (a band of up to ``2 * temporal_window + 1`` patches). A
    second per-token gate mixes the two. When a window has at most
    ``temporal_window + 1`` patches the two branches are identical.

    .. rubric:: Channel positions

    AXON is montage-agnostic: electrodes are identified only by their 3D scalp
    position, taken from ``chs_info[i]["loc"][:3]`` (MNE head coordinates, in
    metres). Channels without a valid position are looked up by name in MNE's
    10-20 and then 10-05 standard montages, in head coordinates.
    Channel order does not matter.

    .. rubric:: Expected input

    - sampling rate **200 Hz** (resample beforehand);
    - windows of at least one patch (``patch_size`` samples, 1 s);
    - by default each channel is z-scored within each window inside the model
      (``normalize_input=True``), as during the reference evaluation. The
      z-score is scale-free, so data in volts (MNE's default) or microvolts
      give the same output.

    .. rubric:: Pretrained weights

    The pretrained encoder has 118.6M parameters. It is loaded with
    :meth:`from_pretrained`. The classification head is not pretrained and
    must be trained for your task::

        import mne
        from braindecode.models import AXON
        from braindecode.util import resolve_montage_name

        raw = mne.io.read_raw_edf("recording.edf", preload=True)
        raw.set_montage(resolve_montage_name("standard_1020"), match_case=False)
        raw.resample(200)
        model = AXON.from_pretrained(
            "NeuroDX/axon-eeg",
            chs_info=raw.info["chs"],
            n_outputs=4,
            n_times=800,
        )

    For linear probing, freeze everything except ``final_layer``. The
    reference fine-tuning recipe also kept the two gates frozen.

    Parameters
    ----------
    patch_size : int
        Number of samples per patch (token length). 200 samples = 1 s at 200 Hz.
    patch_stride : int
        Step between consecutive patches, in samples.
    embed_dim : int
        Token embedding dimension.
    depth : int
        Number of AXON blocks.
    num_heads : int
        Number of attention heads in each path.
    ffn_expansion : int
        Width multiplier of the gated feed-forward network.
    temporal_window : int
        Radius, in patches, of the restricted temporal branch.
    gate_reduction : int
        Hidden size of the gate MLPs is ``max(16, embed_dim // gate_reduction)``.
    head_hidden_dim : int
        Hidden size of the classification head.
    normalize_input : bool
        If True, z-score each channel within each window before patching.
    activation : type[nn.Module]
        Activation of the gate MLPs and of the spatial position encoder.
    head_activation : type[nn.Module]
        Activation of the classification head.
    drop_prob : float
        Dropout probability in the classification head.
    att_drop_prob : float
        Attention dropout probability (0 in the pretrained model).

    References
    ----------
    .. [axon2026] Jain, M., Runwal, P., Mishra, A. R., Kulkarni, A., Lahiri, J. B.,
       Singh, S., & Panwar, S. (2026). Adaptive Anisotropic Attention for
       Axis-Structured Signals. arXiv:2609.08788.
       https://arxiv.org/abs/2609.08788
    """

    def __init__(
        self,
        # signal parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # model parameters
        patch_size: int = 200,
        patch_stride: int = 180,
        embed_dim: int = 512,
        depth: int = 22,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        temporal_window: int = 5,
        gate_reduction: int = 4,
        head_hidden_dim: int = 128,
        normalize_input: bool = True,
        activation: type[nn.Module] = nn.GELU,
        head_activation: type[nn.Module] = nn.ELU,
        drop_prob: float = 0.3,
        att_drop_prob: float = 0.0,
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

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be even.")
        if self._sfreq is not None and abs(float(self._sfreq) - 200.0) > 1e-6:
            warnings.warn(
                f"AXON was pretrained on 200 Hz EEG, got sfreq={self._sfreq}. "
                "Resample the data to 200 Hz to use the pretrained weights.",
                stacklevel=2,
            )
        if self._n_times is not None and self._n_times < patch_size:
            raise ValueError(
                f"n_times ({self._n_times}) must be at least patch_size ({patch_size})."
            )

        self.embed_dim = embed_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_activation = head_activation
        self.drop_prob = drop_prob

        # Electrode positions (C, 3) in cm, resolved once from chs_info.
        positions = _resolve_channel_positions(self.chs_info) * _METRES_TO_CM
        self.encoder = _AXONEncoder(
            channel_positions=torch.as_tensor(positions, dtype=torch.float32),
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            temporal_window=temporal_window,
            gate_reduction=gate_reduction,
            normalize_input=normalize_input,
            activation=activation,
            att_drop_prob=att_drop_prob,
        )
        self.final_layer = self._build_head(self.n_outputs)

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
        self._n_outputs = n_outputs
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
            Tokens of shape (batch, n_chans, n_patches, embed_dim).
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
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
            (batch, n_chans, n_patches, embed_dim)) instead of logits.

        Returns
        -------
        torch.Tensor or dict
            Logits of shape (batch, n_outputs), or the feature dict when
            ``return_features`` is True.
        """
        tokens = self.encoder(x)
        features = tokens.mean(dim=(1, 2))
        logits = self.final_layer(features)

        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {
                "features": features,
                "tokens": tokens,
                "cls_token": None,  # nosec B105
            }
        return logits


class _AXONEncoder(nn.Module):
    """Patch embedding, position encoding and the stack of AXON blocks.

    Maps EEG (B, C, T) to the token grid (B, C, P, D).
    """

    def __init__(
        self,
        channel_positions: torch.Tensor,
        patch_size: int,
        patch_stride: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ffn_expansion: int,
        temporal_window: int,
        gate_reduction: int,
        normalize_input: bool,
        activation: type[nn.Module],
        att_drop_prob: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.temporal_window = temporal_window
        self.normalize_input = normalize_input
        # Not persistent: the pretrained weights must load onto any montage.
        self.register_buffer("channel_positions", channel_positions, persistent=False)

        self.patch_embed = nn.Linear(patch_size, embed_dim, bias=False)
        self.spatial_pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            activation(),
            nn.LayerNorm(embed_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _AXONBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    gate_reduction=gate_reduction,
                    activation=activation,
                    att_drop_prob=att_drop_prob,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.RMSNorm(embed_dim, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            # Per-channel z-score within each window, in float32 so that a flat
            # channel cannot become 0/0 in float16. Scale-free, so volts (MNE)
            # and microvolts give the same result; on microvolt data this matches
            # the reference evaluation's ``(x - mean) / (std + 1e-6)``.
            x32 = x.float()
            mean = x32.mean(dim=-1, keepdim=True)
            std = x32.std(dim=-1, keepdim=True)
            x = ((x32 - mean) / std.clamp_min(1e-12)).to(x.dtype)
        patches = x.unfold(-1, self.patch_size, self.patch_stride)  # (B, C, P, S)
        tokens = self.patch_embed(patches)  # (B, C, P, D)
        n_patches = tokens.shape[2]

        spatial = self.spatial_pos_embed(self.channel_positions)  # (C, D)
        temporal = _sinusoidal_encoding(
            n_patches, self.embed_dim, tokens.device, spatial.dtype
        )
        position = spatial.unsqueeze(1) + temporal.unsqueeze(0)  # (C, P, D)
        tokens = tokens + position.unsqueeze(0).to(tokens.dtype)

        index = torch.arange(n_patches, device=tokens.device)
        band = (index.unsqueeze(0) - index.unsqueeze(1)).abs() <= self.temporal_window

        for block in self.blocks:
            tokens = block(tokens, band)
        return self.final_norm(tokens)


def _sinusoidal_encoding(
    n_positions: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Fixed sinusoidal encoding of positions 0..n_positions-1, shape (n, dim)."""
    position = torch.arange(n_positions, device=device, dtype=dtype).unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(n_positions, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


class _TokenGate(nn.Module):
    """Per-token softmax gate over ``n_paths`` paths, read through a stop-gradient."""

    def __init__(
        self,
        embed_dim: int,
        n_paths: int,
        reduction: int,
        activation: type[nn.Module],
        init_bias: Sequence[float],
    ):
        super().__init__()
        hidden = max(16, embed_dim // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            activation(),
            nn.Linear(hidden, n_paths),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor(list(init_bias), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The gate reads the token but sends no gradient back into the encoder.
        return torch.softmax(self.mlp(x.detach()), dim=-1)


class _GatedFeedForward(nn.Module):
    """Feed-forward block ``out(a * (swish(g) + gelu_tanh(g)))`` with ``[a, g] = in(x)``."""

    def __init__(self, embed_dim: int, expansion: int):
        super().__init__()
        hidden = expansion * embed_dim
        self.in_proj = nn.Linear(embed_dim, 2 * hidden)
        self.out_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        gate = gate * torch.sigmoid(gate) + F.gelu(gate, approximate="tanh")
        return self.out_proj(value * gate)


class _AxisAttention(nn.Module):
    """Temporal and spatial attention paths mixed by a per-token gate.

    Input and output have shape (B, C, P, D).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        gate_reduction: int,
        activation: type[nn.Module],
        att_drop_prob: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.att_drop_prob = att_drop_prob

        self.temporal_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.temporal_out = nn.Linear(embed_dim, embed_dim)
        self.spatial_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.spatial_out = nn.Linear(embed_dim, embed_dim)
        # Axis gate: [temporal, spatial]; initialised to lean slightly temporal.
        self.axis_gate = _TokenGate(
            embed_dim, 2, gate_reduction, activation, init_bias=(0.5, 0.0)
        )
        # Scale gate: [restricted window, full window]; initialised to favour full.
        self.scale_gate = _TokenGate(
            embed_dim, 2, gate_reduction, activation, init_bias=(0.0, 1.0)
        )

    def _attend(
        self, projected: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Multi-head attention within each sequence; ``projected`` is (N, L, 3D)."""
        n_seq, length, _ = projected.shape
        query, key, value = projected.chunk(3, dim=-1)
        query = query.reshape(n_seq, length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.reshape(n_seq, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(n_seq, length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        drop = self.att_drop_prob if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=drop
        )
        return attended.transpose(1, 2).reshape(n_seq, length, self.embed_dim)

    def forward(self, x: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        batch, n_chans, n_patches, dim = x.shape
        axis_weights = self.axis_gate(x)  # (B, C, P, 2)
        scale_weights = self.scale_gate(x)  # (B, C, P, 2)

        # Temporal path: one sequence of P tokens per electrode.
        per_channel = self.temporal_qkv(x.reshape(batch * n_chans, n_patches, dim))
        restricted = self.temporal_out(self._attend(per_channel, band)).reshape(
            batch, n_chans, n_patches, dim
        )
        full = self.temporal_out(self._attend(per_channel, None)).reshape(
            batch, n_chans, n_patches, dim
        )
        temporal = scale_weights[..., 0:1] * restricted + scale_weights[..., 1:2] * full

        # Spatial path: one sequence of C tokens per time step.
        per_step = self.spatial_qkv(
            x.transpose(1, 2).reshape(batch * n_patches, n_chans, dim)
        )
        spatial = (
            self.spatial_out(self._attend(per_step, None))
            .reshape(batch, n_patches, n_chans, dim)
            .transpose(1, 2)
        )
        return axis_weights[..., 0:1] * temporal + axis_weights[..., 1:2] * spatial


class _AXONBlock(nn.Module):
    """Pre-norm block: axis attention then gated feed-forward, both residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_expansion: int,
        gate_reduction: int,
        activation: type[nn.Module],
        att_drop_prob: float,
    ):
        super().__init__()
        self.norm_attn = nn.RMSNorm(embed_dim, eps=1e-8)
        self.attn = _AxisAttention(
            embed_dim, num_heads, gate_reduction, activation, att_drop_prob
        )
        self.norm_ffn = nn.RMSNorm(embed_dim, eps=1e-8)
        self.ffn = _GatedFeedForward(embed_dim, ffn_expansion)

    def forward(self, x: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), band)
        return x + self.ffn(self.norm_ffn(x))


def _valid_position(loc: Any) -> Optional[np.ndarray]:
    """Return the first three entries of an MNE ``loc`` if they are usable."""
    if loc is None:
        return None
    position = np.asarray(loc, dtype=np.float64).reshape(-1)[:3]
    if position.shape[0] < 3 or not np.all(np.isfinite(position)):
        return None
    if np.allclose(position, 0.0):
        return None
    return position


def _standard_head_positions(ch_names: List[str]) -> Dict[str, np.ndarray]:
    """Head-frame positions (metres) of ``ch_names`` in MNE standard montages."""
    import mne

    found: Dict[str, np.ndarray] = {}
    remaining = list(dict.fromkeys(ch_names))
    for montage_name in ("standard_1020", "standard_1005"):
        if not remaining:
            break
        montage = mne.channels.make_standard_montage(resolve_montage_name(montage_name))
        known = {name.lower() for name in montage.ch_names}
        names = [name for name in remaining if name.lower() in known]
        if not names:
            continue
        info = mne.create_info(names, sfreq=1.0, ch_types="eeg")
        info.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
        for name, ch in zip(names, info["chs"]):
            position = _valid_position(ch["loc"])
            if position is not None:
                found[name] = position
        remaining = [name for name in remaining if name not in found]
    return found


def _resolve_channel_positions(chs_info: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Positions (C, 3) in metres from ``loc``, falling back to standard montages."""
    positions: List[Optional[np.ndarray]] = [
        _valid_position(ch.get("loc")) for ch in chs_info
    ]
    missing = [
        str(ch.get("ch_name", f"channel {i}"))
        for i, (ch, pos) in enumerate(zip(chs_info, positions))
        if pos is None
    ]
    if missing:
        lookup = _standard_head_positions(missing)
        unknown = [name for name in missing if name not in lookup]
        if unknown:
            raise ValueError(
                "AXON needs a 3D position for every channel. No valid 'loc' in "
                f"chs_info and no match in MNE standard montages for: {unknown}. "
                "Set a montage on your data (e.g. raw.set_montage(...)) first."
            )
        for i, ch in enumerate(chs_info):
            if positions[i] is None:
                positions[i] = lookup[str(ch.get("ch_name", f"channel {i}"))]
    return np.stack(positions).astype(np.float32)
