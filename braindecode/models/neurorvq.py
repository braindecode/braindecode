# Authors: Konstantinos Barmpas et al. (original implementation)
#          Braindecode contributors (adaptation)
#
# License: CC BY-NC 4.0
"""NeuroRVQ EEG foundation model."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from braindecode.models.base import HAS_HF_HUB, EEGModuleMixin, huggingface_hub

_PRETRAINED_REPO_ID = "ntinosbarmpas/NeuroRVQ"
_PRETRAINED_REVISION = "d944b87f44ae0ba2923b2f10d0518f23f6803b76"
_PRETRAINED_FILENAME = (
    "pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt"
)


# Channel order used to train NeuroRVQ-EEG v1. Keep in sync with the released
# inference module: https://github.com/KonstantinosBarmpas/NeuroRVQ
NEURORVQ_CHANNELS = (
    "a1",
    "a2",
    "af3",
    "af4",
    "af7",
    "af8",
    "afz",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "ccp1",
    "ccp2",
    "ccp3",
    "ccp4",
    "ccp5",
    "ccp6",
    "ccp7",
    "ccp8",
    "cfc1",
    "cfc2",
    "cfc3",
    "cfc4",
    "cfc5",
    "cfc6",
    "cfc7",
    "cfc8",
    "cp1",
    "cp2",
    "cp3",
    "cp4",
    "cp5",
    "cp6",
    "cpz",
    "cz",
    "eog",
    "f1",
    "f10",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "fc1",
    "fc2",
    "fc3",
    "fc4",
    "fc5",
    "fc6",
    "fcz",
    "fp1",
    "fp2",
    "fpz",
    "ft7",
    "ft8",
    "fz",
    "iz",
    "loc",
    "o1",
    "o2",
    "oz",
    "p08",
    "p1",
    "p10",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "po1",
    "po10",
    "po2",
    "po3",
    "po4",
    "po7",
    "po8",
    "po9",
    "poz",
    "pz",
    "roc",
    "sp1",
    "sp2",
    "t1",
    "t10",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "t9",
    "tp10",
    "tp7",
    "tp8",
    "tp9",
)


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: Callable[[int], nn.Module] | None,
        attn_drop: float,
        proj_drop: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        inner_dim = head_dim * num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None
        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            if self.v_bias is None:
                raise RuntimeError("q_bias and v_bias must be initialized together.")
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)
            )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).type_as(v)
            k = self.k_norm(k).type_as(v)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: Callable[[int], nn.Module] | None,
        drop: float,
        attn_drop: float,
        drop_path: float,
        init_values: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, qkv_bias, qk_norm, attn_drop, drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop)
        self.gamma_1 = (
            nn.Parameter(init_values * torch.ones(dim)) if init_values > 0 else None
        )
        self.gamma_2 = (
            nn.Parameter(init_values * torch.ones(dim)) if init_values > 0 else None
        )

    def forward(self, x: Tensor) -> Tensor:
        attn = self.attn(self.norm1(x))
        if self.gamma_1 is not None:
            attn = self.gamma_1 * attn
        x = x + self.drop_path(attn)
        mlp = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            mlp = self.gamma_2 * mlp
        return x + self.drop_path(mlp)


class _MultiScaleTemporalConv(nn.Module):
    """Released four-branch EEG patch embedding."""

    def __init__(self, out_chans: int = 8, activation: type[nn.Module] = nn.GELU):
        super().__init__()
        for i, kernel in enumerate((21, 15, 9, 5), start=1):
            setattr(
                self,
                f"conv1_{i}",
                nn.Conv2d(1, out_chans, (1, kernel), padding=(0, kernel // 2)),
            )
            setattr(self, f"norm1_{i}", nn.GroupNorm(4, out_chans))
            setattr(self, f"pool1_{i}", nn.AvgPool2d((1, 2)))
        for i, kernel in enumerate((9, 7, 5, 3), start=1):
            setattr(
                self,
                f"conv2_{i}",
                nn.Conv2d(out_chans, out_chans, (1, kernel), padding=(0, kernel // 2)),
            )
            setattr(self, f"norm2_{i}", nn.GroupNorm(4, out_chans))
            setattr(self, f"pool2_{i}", nn.AvgPool2d((1, 4)))
        self.gelu1 = activation()
        self.gelu2 = activation()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # x: batch x channels x patches x patch_size, in channel-major order.
        batch, channels, patches, patch_size = x.shape
        x = x.reshape(batch, channels * patches, patch_size).unsqueeze(1)
        outputs = []
        for i in range(1, 5):
            branch = getattr(self, f"pool1_{i}")(
                self.gelu1(getattr(self, f"norm1_{i}")(getattr(self, f"conv1_{i}")(x)))
            )
            branch = getattr(self, f"pool2_{i}")(
                self.gelu2(
                    getattr(self, f"norm2_{i}")(getattr(self, f"conv2_{i}")(branch))
                )
            )
            # Match the reference reshape B,C,channel*patch,time -> B,token,time*C.
            branch = branch.permute(0, 2, 3, 1).reshape(batch, channels * patches, -1)
            outputs.append(branch)
        return tuple(outputs)


class NeuroRVQ(EEGModuleMixin, nn.Module, license="cc-by-nc-4.0"):
    r"""NeuroRVQ-EEG foundation model from Barmpas et al. [neurorvq]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    The model combines four temporal convolution scales with a shared
    Transformer encoder and learned channel/time embeddings. The released EEG
    checkpoint contains a 6M-parameter masked-token foundation model; this
    class adapts its pretrained encoder to Braindecode's window classifier API.

    Inputs are ``(batch, channels, time)``. Windows must be sampled at 200 Hz,
    contain whole 200-sample patches, and use channels supported by the
    pretrained model when ``channel_names`` or ``chs_info`` are provided. The
    source EEG preprocessing example applies a 0.5--45 Hz band-pass, resamples
    to 200 Hz, and clips amplitudes to +/-500 before inference. This model does
    not preprocess signals.

    The upstream implementation and checkpoint are licensed CC BY-NC 4.0.
    This is a non-commercial research license.

    Load the published EEG foundation checkpoint (the task-specific
    classification head remains randomly initialized)::

        model = NeuroRVQ(
            n_chans=3, n_outputs=4, n_times=800, sfreq=200,
            channel_names=("f3", "f4", "cz"),
        )
    model.load_pretrained_weights()

    Parameters
    ----------
    n_outputs : int
        Number of task-specific output classes.
    n_chans : int
        Number of EEG channels in each input window.
    chs_info : list of dict or None
        MNE channel metadata. Channel names are used to select the pretrained
        spatial embedding slots when ``channel_names`` is not provided.
    n_times : int
        Number of input time samples. Must be divisible by 200 and no greater
        than ``patch_size * max_patches``.
    sfreq : float or None
        Sampling frequency. If provided or inferable, it must be 200 Hz.
    channel_names : tuple of str, list of str, or None
        Ordered electrode names. Names are case-insensitive and must occur in
        the released 104-channel montage. If omitted, names are read from
        ``chs_info``; without either, randomly initialized models use the first
        entries in the released channel order. Loading released pretrained
        weights requires explicit ``channel_names`` or ``chs_info`` to avoid
        silently assigning input electrodes to the wrong spatial embeddings.
    patch_size : int, default=200
        Samples per temporal patch. NeuroRVQ-EEG v1 requires 200.
    max_patches : int, default=256
        Maximum number of patches represented by the pretrained temporal
        embedding table.
    depth : int, default=12
        Number of shared Transformer blocks applied to each temporal branch.
    num_heads : int, default=10
        Number of attention heads.
    mlp_ratio : float, default=4.0
        Transformer feed-forward expansion ratio.
    out_chans : int, default=8
        Channels per temporal convolution branch. Must be divisible by four.
    qkv_bias : bool, default=True
        Whether to use the released query/value attention biases.
    qk_norm : bool, default=True
        Whether to normalize query and key vectors with LayerNorm.
    init_values : float, default=1e-5
        LayerScale initialization value used by the released EEG checkpoint.
    drop_prob : float, default=0.0
        Dropout probability in the patch embedding, feed-forward blocks, and
        positional embedding path.
    attn_drop_rate : float, default=0.0
        Attention-probability dropout.
    drop_path_rate : float, default=0.0
        Maximum stochastic-depth probability across Transformer depth.
    activation : type[nn.Module], default=nn.GELU
        Activation in the multi-scale temporal patch embedding. GELU reproduces
        the released model.

    Input shape
    -----------
    ``(batch, n_chans, n_times)``.

    Output shape
    ------------
    By default, ``(batch, n_outputs)`` logits. With ``return_features=True``,
    returns a dictionary whose ``"features"`` entry has shape
    ``(batch, 4 * embed_dim * n_chans * (n_times // patch_size))``.

    .. versionadded:: 1.8

    References
    ----------
    .. [neurorvq] Barmpas et al. (2025). NeuroRVQ: Multi-Scale Biosignal
       Tokenization for Generative Foundation Models. https://arxiv.org/abs/2510.13068
    .. [neurorvqcode] https://github.com/KonstantinosBarmpas/NeuroRVQ
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        channel_names: tuple[str, ...] | list[str] | None = None,
        patch_size: int = 200,
        max_patches: int = 256,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        out_chans: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 1e-5,
        drop_prob: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
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
        if patch_size != 200:
            raise ValueError(
                "NeuroRVQ-EEG v1 requires patch_size=200 (200 Hz sampling)."
            )
        if self.n_times % patch_size:
            raise ValueError("n_times must be divisible by patch_size (200 samples).")
        if self.n_times // patch_size > max_patches:
            raise ValueError(f"n_times supports at most {max_patches} patches.")
        model_sfreq = self._sfreq
        if model_sfreq is None and self._input_window_seconds is not None:
            model_sfreq = self.n_times / self._input_window_seconds
        if model_sfreq is not None and model_sfreq != 200:
            raise ValueError(
                "NeuroRVQ-EEG v1 was trained at 200 Hz; resample before calling the model."
            )
        for name, value in (
            ("depth", depth),
            ("num_heads", num_heads),
            ("out_chans", out_chans),
            ("max_patches", max_patches),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if out_chans % 4:
            raise ValueError("out_chans must be divisible by 4 for GroupNorm.")
        embed_dim = out_chans * (patch_size // 8)
        if embed_dim % num_heads:
            raise ValueError(
                "The derived embedding width must be divisible by num_heads."
            )
        if init_values is None:
            raise ValueError(
                "init_values must be a number; the reference requires a numeric value."
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.num_patches = self.n_times // patch_size
        self._has_explicit_channel_mapping = (
            channel_names is not None or self._chs_info is not None
        )
        self.channel_names = self._resolve_channel_names(channel_names)
        channel_to_index = {name: i for i, name in enumerate(NEURORVQ_CHANNELS)}
        unknown = [name for name in self.channel_names if name not in channel_to_index]
        if unknown:
            raise ValueError(f"Unsupported NeuroRVQ channel name(s): {unknown}.")
        self.register_buffer(
            "spatial_embedding_ix",
            torch.tensor([channel_to_index[name] for name in self.channel_names]),
            persistent=False,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = _MultiScaleTemporalConv(
            out_chans=out_chans, activation=activation
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(len(NEURORVQ_CHANNELS) + 1, embed_dim)
        )
        self.time_embed = nn.Parameter(torch.zeros(max_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        norm = (lambda dim: nn.LayerNorm(dim, eps=1e-6)) if qk_norm else None
        self.blocks = nn.ModuleList(
            [
                _Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    norm,
                    drop_prob,
                    attn_drop_rate,
                    drop_paths[i],
                    init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity()
        head_dim = embed_dim * 4 * self.n_chans * self.num_patches
        self.fc_norm = nn.LayerNorm(head_dim)
        self.final_layer = nn.Linear(head_dim, self.n_outputs)
        self._init_weights()

    def _resolve_channel_names(self, channel_names):
        if channel_names is None and self._chs_info is not None:
            channel_names = [ch["ch_name"] for ch in self._chs_info]
        if channel_names is None:
            # This fallback is useful for training from scratch. For pretrained
            # inference pass channel_names or chs_info to select spatial slots.
            channel_names = NEURORVQ_CHANNELS[: self.n_chans]
        if len(channel_names) != self.n_chans:
            raise ValueError("channel_names must have one entry per input channel.")
        normalized = tuple(str(name).strip().lower() for name in channel_names)
        if len(set(normalized)) != len(normalized):
            raise ValueError("channel_names must be unique.")
        return normalized

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            for i, block in enumerate(self.blocks, start=1):
                block.attn.proj.weight.div_(math.sqrt(2.0 * i))
                block.mlp.fc2.weight.div_(math.sqrt(2.0 * i))

    def _features(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(
                f"Expected input shape (batch, {self.n_chans}, {self.n_times}), got {tuple(x.shape)}."
            )
        batch = x.shape[0]
        x = x.reshape(batch, self.n_chans, self.num_patches, self.patch_size)
        branches = self.patch_embed(x)
        tokens = []
        spatial_ix = self.spatial_embedding_ix.repeat_interleave(self.num_patches)
        spatial_ix = F.pad(spatial_ix, (1, 0), value=0)
        spatial = self.pos_embed[spatial_ix].unsqueeze(0)
        temporal_ix = torch.arange(
            self.max_patches - self.num_patches, self.max_patches, device=x.device
        ).repeat(self.n_chans)
        temporal = self.time_embed[temporal_ix].unsqueeze(0)
        cls = self.cls_token.expand(batch, -1, -1)
        for branch in branches:
            branch = torch.cat((cls, branch), dim=1)
            branch = branch + spatial
            branch[:, 1:] = branch[:, 1:] + temporal
            branch = self.pos_drop(branch)
            for block in self.blocks:
                branch = block(branch)
            tokens.append(self.norm(branch[:, 1:]))
        return torch.cat(tokens, dim=-1).flatten(1)

    def forward(self, x: Tensor, return_features: bool = False):
        features = self.fc_norm(self._features(x))
        if return_features:
            return {"features": features, "cls_token": None}  # nosec B105
        return self.final_layer(features)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(
            self.fc_norm.normalized_shape[0],
            n_outputs,
            device=self.final_layer.weight.device,
            dtype=self.final_layer.weight.dtype,
        )

    def load_pretrained_weights(self, checkpoint_path: str | None = None):
        """Load the released NeuroRVQ-EEG v1 foundation checkpoint.

        Parameters
        ----------
        checkpoint_path : str or None
            Local checkpoint path. If ``None``, download the upstream file at
            the pinned revision from Hugging Face Hub.

        Returns
        -------
        self : NeuroRVQ
            The model with the pretrained encoder weights loaded. The
            classifier head is left at its current initialization.

        Notes
        -----
        The source checkpoint also contains masked-token prediction heads that
        are not used by this downstream classifier. They are intentionally
        ignored; all shared encoder tensors must load successfully. Pretrained
        spatial embeddings are electrode-specific, so ``channel_names`` or
        ``chs_info`` must have been provided when constructing the model.
        """
        if not self._has_explicit_channel_mapping:
            raise ValueError(
                "Loading pretrained NeuroRVQ weights requires channel_names or "
                "chs_info so input electrodes map to the released spatial "
                "embedding slots. The implicit first-N channel fallback is only "
                "supported for randomly initialized training."
            )
        if checkpoint_path is None:
            if not HAS_HF_HUB:
                raise ImportError(
                    "Loading NeuroRVQ weights from the Hub requires "
                    "huggingface_hub. Install braindecode[hub] or pass a local "
                    "checkpoint_path."
                )
            checkpoint_path = huggingface_hub.hf_hub_download(
                repo_id=_PRETRAINED_REPO_ID,
                filename=_PRETRAINED_FILENAME,
                revision=_PRETRAINED_REVISION,
            )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        incompatible = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            "fc_norm.weight",
            "fc_norm.bias",
            "final_layer.weight",
            "final_layer.bias",
        }
        ignored_unexpected = {
            key
            for key in incompatible.unexpected_keys
            if key == "mask_token" or key.startswith(("norm_pre.", "head_pre_"))
        }
        unexpected = set(incompatible.unexpected_keys) - ignored_unexpected
        if set(incompatible.missing_keys) != expected_missing or unexpected:
            raise RuntimeError(
                "The NeuroRVQ checkpoint does not match the expected encoder "
                f"schema (missing={incompatible.missing_keys}, "
                f"unexpected={sorted(unexpected)})."
            )
        return self
