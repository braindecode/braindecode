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

from braindecode.models.base import EEGModuleMixin
from braindecode.models.labram import _Attention
from braindecode.modules import MLP, DropPath

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

# The four released packages (ntinosbarmpas/NeuroRVQ, NeuroRVQ_{EEG,ECG,EMG,PPG})
# differ only in these values: sampling rate, patch size, temporal-embedding
# length, the two rows of temporal kernel sizes, the fine-tuning head pooling,
# the residual codebooks per scale and the channel list.
_MODALITIES = {
    "eeg": dict(
        sfreq=200,
        patch_size=200,
        max_patches=256,
        kernels=((21, 15, 9, 5), (9, 7, 5, 3)),
        head_pooling="flatten",
        num_quantizers=8,
        channels=NEURORVQ_CHANNELS,
    ),
    "ecg": dict(
        sfreq=200,
        patch_size=40,
        max_patches=600,
        kernels=((21, 15, 9, 5), (9, 7, 5, 3)),
        head_pooling="mean",
        num_quantizers=8,
        channels=(
            "avf",
            "avl",
            "avr",
            "i",
            "ii",
            "iii",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "vx",
            "vy",
            "vz",
        ),
    ),
    "emg": dict(
        sfreq=1000,
        patch_size=200,
        max_patches=256,
        kernels=((51, 17, 8, 5), (25, 9, 4, 3)),
        head_pooling="mean",
        num_quantizers=16,
        channels=tuple(sorted(f"c{i}" for i in range(1, 17))),
    ),
    "ppg": dict(
        sfreq=100,
        patch_size=80,
        max_patches=12,
        kernels=((41, 31, 17, 9), (17, 13, 9, 5)),
        head_pooling="mean",
        num_quantizers=8,
        channels=("ppg_c1",),
    ),
}


def _modality(model, modality: str) -> dict:
    """Return the preset of ``modality`` after checking the sampling rate."""
    if modality not in _MODALITIES:
        raise ValueError(f"modality must be one of {sorted(_MODALITIES)}.")
    preset = _MODALITIES[modality]
    model_sfreq = model._sfreq
    if model_sfreq is None and model._input_window_seconds is not None:
        model_sfreq = model.n_times / model._input_window_seconds
    if model_sfreq is not None and model_sfreq != preset["sfreq"]:
        raise ValueError(
            f"NeuroRVQ-{modality.upper()} was trained at {preset['sfreq']} Hz; "
            "resample before calling the model."
        )
    return preset


def _channel_slots(model, channel_names, channels: tuple[str, ...]):
    """Normalized channel names and their slots in the pretrained ``channels``."""
    if channel_names is None and model._chs_info is not None:
        channel_names = [ch["ch_name"] for ch in model._chs_info]
    if channel_names is None:
        # Fallback for training from scratch; pass channel_names or chs_info
        # with pretrained weights so electrodes map to their spatial slots.
        channel_names = channels[: model.n_chans]
    if len(channel_names) != model.n_chans:
        raise ValueError("channel_names must have one entry per input channel.")
    names = tuple(str(name).strip().lower() for name in channel_names)
    if len(set(names)) != len(names):
        raise ValueError("channel_names must be unique.")
    unknown = [name for name in names if name not in channels]
    if unknown:
        raise ValueError(f"Unsupported NeuroRVQ channel name(s): {unknown}.")
    return names, torch.tensor([channels.index(name) for name in names])


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        # LaBraM's ``_Attention`` calls ``qk_norm(head_dim, eps=1e-6)``.
        qk_norm: Callable[..., nn.Module] | None,
        drop: float,
        attn_drop: float,
        drop_path: float,
        init_values: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        # Single trailing dropout (vs. the released module's two, after the
        # activation and after fc2): identical forward since drop_prob=0.0 by
        # default, and both placements are no-ops whenever drop=0.
        self.mlp = MLP(
            in_features=dim,
            hidden_features=(int(dim * mlp_ratio),),
            out_features=dim,
            drop=drop,
        )
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
    """Released four-branch patch embedding; ``kernels`` = (first, second) rows."""

    def __init__(self, out_chans: int, activation: type[nn.Module], kernels):
        super().__init__()
        for i, kernel in enumerate(kernels[0], start=1):
            setattr(
                self,
                f"conv1_{i}",
                nn.Conv2d(1, out_chans, (1, kernel), padding=(0, kernel // 2)),
            )
            setattr(self, f"norm1_{i}", nn.GroupNorm(4, out_chans))
            setattr(self, f"pool1_{i}", nn.AvgPool2d((1, 2)))
        for i, kernel in enumerate(kernels[1], start=1):
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
    r"""NeuroRVQ foundation model for EEG, ECG and EMG from Barmpas et al. [neurorvq]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    The model combines four temporal convolution scales with a shared
    Transformer encoder and learned channel/time embeddings; this class adapts
    the pretrained encoder to Braindecode's window classifier API.

    Inputs are ``(batch, channels, time)``, sampled at the modality's rate
    (EEG 200 Hz, ECG 200 Hz, EMG 1000 Hz, PPG 100 Hz), made of whole patches
    (EEG 200, ECG 40, EMG 200, PPG 80 samples), with channel names from the
    modality's pretrained list. This model does not preprocess signals; the
    authors' EEG example applies a 0.5--45 Hz band-pass, resamples to 200 Hz
    and clips amplitudes to +/-500.

    The upstream implementation and checkpoints are licensed CC BY-NC 4.0.
    This is a non-commercial research license.

    `License <https://github.com/KonstantinosBarmpas/NeuroRVQ/blob/main/LICENSE>`_

    .. important::
       **Pre-trained Weights Available**

       The released foundation models (`ntinosbarmpas/NeuroRVQ
       <https://huggingface.co/ntinosbarmpas/NeuroRVQ>`_, revision ``d944b87``,
       CC BY-NC 4.0) are hosted as ``braindecode/neurorvq-eeg-pretrained``,
       ``braindecode/neurorvq-ecg-pretrained`` and
       ``braindecode/neurorvq-emg-pretrained``; the tokenizers are in
       ``braindecode/neurorvq-tokenizer-{eeg,ecg,emg,ppg}-pretrained``
       (:class:`NeuroRVQTokenizer`). The classification head is not
       pretrained:

       .. code-block:: python

           from braindecode.models import NeuroRVQ

           model = NeuroRVQ.from_pretrained(
               "braindecode/neurorvq-eeg-pretrained", chs_info=raw.info["chs"], n_outputs=4
           )

    Parameters
    ----------
    n_outputs : int
        Number of task-specific output classes.
    n_chans : int
        Number of channels in each input window.
    chs_info : list of dict or None
        MNE channel metadata. Channel names are used to select the pretrained
        spatial embedding slots when ``channel_names`` is not provided.
    n_times : int
        Number of input time samples. Must be divisible by ``patch_size`` and
        no greater than ``patch_size * max_patches``.
    sfreq : float or None
        Sampling frequency. If provided or inferable, it must be the
        modality's rate.
    channel_names : tuple of str, list of str, or None
        Ordered channel names. Names are case-insensitive and must occur in
        the modality's pretrained list (EEG: 104 electrodes; ECG: ``i``,
        ``ii``, ``iii``, ``avr``, ``avl``, ``avf``, ``v1``-``v6``, ``vx``,
        ``vy``, ``vz``; EMG: ``c1``-``c16``; PPG: ``ppg_c1``). If omitted,
        names are read from ``chs_info``; without either, the first entries
        of that list are used, which only suits training from scratch.
    modality : {"eeg", "ecg", "emg", "ppg"}, default="eeg"
        Released configuration: sampling rate, temporal kernel sizes, channel
        list and the defaults of ``patch_size``, ``max_patches`` and
        ``head_pooling``. No PPG foundation model was released.
    patch_size : int or None, default=None
        Samples per temporal patch; ``None`` uses the modality's value.
    max_patches : int or None, default=None
        Length of the temporal embedding table (EEG 256, ECG 600, EMG 256,
        PPG 12); ``None`` uses the modality's value.
    head_pooling : {"flatten", "mean"} or None, default=None
        How the classification head reads the tokens: ``"flatten"``
        concatenates every token, ``"mean"`` averages them. ``None`` uses the
        authors' released choice (EEG flatten, the others mean).
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
    ``(batch, 4 * embed_dim * n_chans * (n_times // patch_size))`` with the
    flatten head and ``(batch, 4 * embed_dim)`` with the mean head, where
    ``embed_dim = out_chans * patch_size // 8``.

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
        modality: str = "eeg",
        patch_size: int | None = None,
        max_patches: int | None = None,
        head_pooling: str | None = None,
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
        preset = _modality(self, modality)
        patch_size = patch_size or preset["patch_size"]
        max_patches = max_patches or preset["max_patches"]
        head_pooling = head_pooling or preset["head_pooling"]
        if head_pooling not in ("flatten", "mean"):
            raise ValueError("head_pooling must be 'flatten' or 'mean'.")
        if self.n_times % patch_size:
            raise ValueError(f"n_times must be divisible by patch_size ({patch_size}).")
        if self.n_times // patch_size > max_patches:
            raise ValueError(f"n_times supports at most {max_patches} patches.")
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
        self.head_pooling = head_pooling
        self.num_patches = self.n_times // patch_size
        self.channel_names, slots = _channel_slots(
            self, channel_names, preset["channels"]
        )
        self.register_buffer("spatial_embedding_ix", slots, persistent=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = _MultiScaleTemporalConv(
            out_chans, activation, preset["kernels"]
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(len(preset["channels"]) + 1, embed_dim)
        )
        self.time_embed = nn.Parameter(torch.zeros(max_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        # LaBraM's attention builds ``qk_norm(head_dim, eps=1e-6)``.
        norm = nn.LayerNorm if qk_norm else None
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
        head_dim = embed_dim * 4
        if head_pooling == "flatten":
            head_dim *= self.n_chans * self.num_patches
        self.fc_norm = nn.LayerNorm(head_dim)
        self.final_layer = nn.Linear(head_dim, self.n_outputs)
        self._init_weights()

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
                # block.mlp[2] is the MLP's output Linear (``mlp.fc2`` upstream).
                block.mlp[2].weight.div_(math.sqrt(2.0 * i))

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
        tokens = torch.cat(tokens, dim=-1)
        return tokens.flatten(1) if self.head_pooling == "flatten" else tokens.mean(1)

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
