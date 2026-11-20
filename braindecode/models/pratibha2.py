# Authors: Mahir Jain (mahir@mannas.ai)
#
# License: Apache-2.0
# Reference implementation and weights: https://huggingface.co/MannasAI/pratibha-2 (Apache-2.0).
"""PRATIBHA-2 (formerly MANAS-2): a dense-attention EEG foundation model."""

import re
from functools import lru_cache, partial
from typing import Optional

import numpy as np
import torch
from mne.channels import make_standard_montage
from torch import nn

from braindecode.models.axon import _GatedFeedForward
from braindecode.models.base import EEGModuleMixin
from braindecode.models.reve import _reve_position_bank
from braindecode.models.util import warn_if_sfreq_differs
from braindecode.modules import (
    PreNormBlock,
    TimmAttention,
    resolve_channel_positions,
)
from braindecode.modules.layers import RMSNorm
from braindecode.util import resolve_montage_name

# Parameter names of the original release -> names here.
_ORIGINAL_KEYS = (
    (r"patch_embed\.linear\.", "encoder.patch_embed."),
    (r"pos_enc\.learned_linear\.0\.", "encoder.pos_enc.learned_linear."),
    (r"pos_enc\.learned_linear\.2\.", "encoder.pos_enc.learned_norm."),
    (r"pos_enc\.(fourier_linear|final_norm)\.", r"encoder.pos_enc.\1."),
    (r"encoder\.layers\.(\d+)\.pre_attn_norm\.", r"encoder.blocks.\1.norm1."),
    (r"encoder\.layers\.(\d+)\.attn\.qkv_proj\.", r"encoder.blocks.\1.attn.qkv."),
    (r"encoder\.layers\.(\d+)\.attn\.out_proj\.", r"encoder.blocks.\1.attn.proj."),
    (r"encoder\.layers\.(\d+)\.pre_ffn_norm\.", r"encoder.blocks.\1.norm2."),
    (r"encoder\.layers\.(\d+)\.ffn\.", r"encoder.blocks.\1.mlp."),
)


class Pratibha2(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""PRATIBHA-2 (formerly MANAS-2), an EEG foundation model from Kulkarni et al. (2026) [manas2]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel`

    PRATIBHA-2 is a transformer encoder pretrained with masked autoencoding on
    clinical and research EEG. Pretraining combines waveform and spectral
    reconstruction (Raw-Band Hybrid) with a constraint on the local RMS energy of
    the reconstructed waveform (Constrained Reconstruction). Each window is cut
    into tokens, one per electrode per one-second patch (0.1 s overlap), and
    every layer applies dense self-attention over all tokens of the window.

    .. rubric:: Position encoding

    Every token gets a 4D position: the electrode's 3D scalp position (in
    centimetres) and the patch index. A Fourier branch (sines and cosines of
    256 frequency combinations) and a learned branch are summed and
    RMS-normalized, and the result is added to the patch embedding.

    .. rubric:: Channel positions

    Positions come from the electrode table used in pretraining, REVE's bank
    plus its own ``AFZ`` and ``IZ`` (MNE montage frame, no download), looked up
    by channel name: an exact match first, then the name ignoring case, an
    ``EEG`` prefix or ``-REF`` suffix, legacy 10-20 aliases (``T3`` -> ``T7``),
    or the midpoint of a bipolar pair ``A-B`` (see
    :func:`~braindecode.modules.resolve_channel_positions`). Other 10-05 names
    take their position in MNE's ``standard_1005`` montage frame, the frame of
    the table's 10-05 entries. Channel ``loc`` values are not used: they are in
    MNE's head frame, a few centimetres from the pretraining frame.

    .. rubric:: Expected input

    - sampling rate **200 Hz** (resample beforehand);
    - windows of at least one patch (``patch_size`` samples, 1 s);
    - by default each channel is z-scored within each window inside the model
      (``normalize_input=True``). The z-score is scale-free, so data in volts
      (MNE's default) or microvolts give the same output; on microvolt data it
      matches the reference preprocessing ``(x - mean) / (std + 1e-6)``.

    .. important::
       **Pretrained weights.** The 92.8M-parameter encoder is on the Hugging
       Face Hub (`MannasAI/pratibha-2 <https://huggingface.co/MannasAI/pratibha-2>`_,
       `License <https://www.apache.org/licenses/LICENSE-2.0>`_) and is loaded
       with :meth:`from_pretrained`, which maps the parameter names of that
       release. The classification head is not pretrained and must be trained
       for your task::

           import mne
           from braindecode.models import Pratibha2

           raw = mne.io.read_raw_edf("recording.edf", preload=True)
           raw.resample(200)
           model = Pratibha2.from_pretrained(
               "MannasAI/pratibha-2",
               chs_info=raw.info["chs"],
               n_outputs=4,
               n_times=1000,
           )

       For linear probing, freeze everything except ``final_layer``.

    Parameters
    ----------
    patch_size : int
        Number of samples per patch (token length). 200 samples = 1 s at 200 Hz.
    patch_stride : int
        Step between consecutive patches, in samples.
    embed_dim : int
        Token embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    ffn_expansion : int
        Width multiplier of the gated feed-forward network.
    n_freqs : int
        Frequencies per axis of the Fourier position encoding, which projects
        the sines and cosines of ``n_freqs ** 4`` frequency combinations to
        ``embed_dim``.
    head_hidden_dim : int
        Hidden size of the classification head.
    normalize_input : bool
        If True, z-score each channel within each window before patching.
    head_activation : type[nn.Module]
        Activation of the classification head.
    drop_prob : float
        Dropout probability in the classification head.
    att_drop_prob : float
        Attention dropout probability (0 in the pretrained model).
    on_missing : str
        Channels the position table cannot place: ``"error"`` raises,
        ``"drop"`` leaves them out of the input, ``"nearest"`` takes the table
        entry closest to the channel's ``loc`` (approximate: ``loc`` is in a
        different frame); see
        :func:`~braindecode.modules.resolve_channel_positions`.
    channel_strategy : str
        How another montage reaches the encoder; see
        :class:`~braindecode.models.base.EEGModuleMixin`. PRATIBHA-2 reads any
        montage natively, so the default ``"native"`` is usually what you want.
    channel_strategy_kwargs : dict or None
        Options of ``channel_strategy``.

    References
    ----------
    .. [manas2] Kulkarni, A., Mishra, A. R., Jain, M., Runwal, P., Saini, L.,
       Panwar, S., & Singh, S. (2026). MANAS-2: Constrained Reconstruction for
       EEG Foundation Models. arXiv:2609.13717.
       https://arxiv.org/abs/2609.13717
    """

    channel_index: Optional[torch.Tensor]

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
        n_freqs: int = 4,
        head_hidden_dim: int = 128,
        normalize_input: bool = True,
        head_activation: type[nn.Module] = nn.ELU,
        drop_prob: float = 0.3,
        att_drop_prob: float = 0.0,
        on_missing: str = "error",
        channel_strategy: str = "native",
        channel_strategy_kwargs: Optional[dict] = None,
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

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        warn_if_sfreq_differs("Pratibha2", self._sfreq, 200.0)
        try:
            n_times = self.n_times  # also inferred from input_window_seconds
        except ValueError:
            n_times = None
        if n_times is not None and n_times < patch_size:
            raise ValueError(
                f"n_times ({n_times}) must be at least patch_size ({patch_size})."
            )

        self.embed_dim = embed_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_activation = head_activation
        self.drop_prob = drop_prob
        self.on_missing = on_missing

        positions, kept = resolve_channel_positions(
            self.chs_info, _pratibha2_position_bank(), on_missing=on_missing
        )
        self._positions_cm = (positions.astype(np.float32) * 100.0).tolist()
        self._kept_indices = list(kept) if len(kept) < len(self.chs_info) else []
        self.register_buffer("channel_index", None, persistent=False)

        self.encoder = _Pratibha2Encoder(
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            n_freqs=n_freqs,
            normalize_input=normalize_input,
            att_drop_prob=att_drop_prob,
        )
        self.final_layer = self._build_head(self.n_outputs)
        self._restore_montage()
        self.register_load_state_dict_post_hook(self._restore_montage)

    def _restore_montage(self, _module=None, _incompatible_keys=None):
        # The target montage is independent of the loaded checkpoint.
        anchor = self.encoder.patch_embed.weight
        self.encoder.channel_positions = anchor.new_tensor(self._positions_cm)
        if self._kept_indices:
            self.channel_index = anchor.new_tensor(self._kept_indices, dtype=torch.long)

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
        self._set_n_outputs(n_outputs)
        self.final_layer = self._build_head(n_outputs)

    @staticmethod
    def _convert_original_keys(state_dict, prefix):
        for key in [k for k in state_dict if k.startswith(prefix)]:
            name = key[len(prefix) :]
            if name == "pos_enc.freq_matrix":
                del state_dict[key]
                continue
            for pattern, new in _ORIGINAL_KEYS:
                if re.match(pattern, name):
                    state_dict[prefix + re.sub(pattern, new, name, count=1)] = (
                        state_dict.pop(key)
                    )
                    break

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # The mixin records provenance from the names it receives.
        converted = dict(state_dict)
        self._convert_original_keys(converted, "")
        return super().load_state_dict(converted, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix:  # A parent PyTorch module owns the load.
            self._convert_original_keys(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
            Tokens of shape (batch, n_kept_chans, n_patches, embed_dim), where
            ``n_kept_chans`` excludes channels dropped by ``on_missing="drop"``.
        """
        if x.dim() != 3 or x.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected EEG with {self.n_chans} channels in the configured montage."
            )
        channel_index = self.channel_index
        if channel_index is not None:
            x = x.index_select(1, channel_index)
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
            (batch, n_kept_chans, n_patches, embed_dim), see :meth:`encode`)
            instead of logits.

        Returns
        -------
        torch.Tensor or dict
            Logits of shape (batch, n_outputs), or the feature dict when
            ``return_features`` is True.
        """
        tokens = self.encode(x)
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


# The pretraining electrode table is REVE's bank plus these two entries (m), which
# sit 1.43 and 0.74 cm from 10-05 AFz and Iz (release 863edc9). They come last, so
# a name that matches only ignoring case (``afz``) takes them, as in the release.
_EXTRA_POSITIONS = {
    "AFZ": (0.0010058004409074783, 0.07728150114417076, 0.021591500379145145),
    "IZ": (-3.840029239654541e-05, -0.1144150011241436, -0.029236000031232834),
}


@lru_cache(maxsize=1)
def _pratibha2_position_bank() -> dict[str, np.ndarray]:
    """Name -> position (m) of the pretraining table, in MNE's montage frame.

    The table is REVE's bank plus ``_EXTRA_POSITIONS``. It is laid over MNE's
    ``standard_1005`` montage, which places the 10-05 names the table lacks in the
    frame of its 10-05 entries.
    """
    montage = make_standard_montage(resolve_montage_name("standard_1005"))
    return {
        **montage.get_positions()["ch_pos"],
        **_reve_position_bank(),
        **_EXTRA_POSITIONS,
    }


class _Pratibha2Encoder(nn.Module):
    """Patch embedding, position encoding and the stack of transformer blocks.

    Maps EEG (B, C, T) to the token grid (B, C, P, D).
    """

    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ffn_expansion: int,
        n_freqs: int,
        normalize_input: bool,
        att_drop_prob: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.normalize_input = normalize_input
        # Not persistent: the pretrained weights must load onto any montage.
        self.register_buffer("channel_positions", torch.empty(0, 3), persistent=False)

        self.patch_embed = nn.Linear(patch_size, embed_dim, bias=False)
        self.pos_enc = _Pratibha2PositionalEncoding(embed_dim, n_freqs)
        norm = partial(RMSNorm, eps=1e-8)
        self.blocks = nn.ModuleList(
            [
                PreNormBlock(
                    embed_dim,
                    attn=TimmAttention(
                        embed_dim, num_heads, qkv_bias=True, attn_drop=att_drop_prob
                    ),
                    mlp=_GatedFeedForward(embed_dim, ffn_expansion),
                    norm_layer=norm,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = norm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x32 = x.to(torch.promote_types(x.dtype, torch.float32))
            mean = x32.mean(dim=-1, keepdim=True)
            std = x32.std(dim=-1, keepdim=True, unbiased=x32.size(-1) > 1)
            x = ((x32 - mean) / std.clamp_min(1e-12)).to(x.dtype)
        patches = x.unfold(-1, self.patch_size, self.patch_stride)  # (B, C, P, S)
        tokens = self.patch_embed(patches)  # (B, C, P, D)
        batch, n_chans, n_patches, _ = tokens.shape

        positions = self.channel_positions
        positions = positions.to(torch.promote_types(positions.dtype, torch.float32))
        time = torch.arange(n_patches, device=tokens.device, dtype=positions.dtype)
        coords = torch.cat(
            [
                positions[:, None, :].expand(-1, n_patches, -1),
                time[None, :, None].expand(n_chans, -1, -1),
            ],
            dim=-1,
        )  # (C, P, 4)
        tokens = tokens + self.pos_enc(coords).unsqueeze(0).to(tokens.dtype)

        tokens = tokens.reshape(batch, n_chans * n_patches, self.embed_dim)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        return tokens.reshape(batch, n_chans, n_patches, self.embed_dim)


class _Pratibha2PositionalEncoding(nn.Module):
    """4D position encoding: Fourier features plus a learned gated branch.

    Kept private: the released weights depend on this layout.
    """

    def __init__(self, embed_dim: int, n_freqs: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.register_buffer("freq_matrix", torch.empty(0), persistent=False)
        self.fourier_linear = nn.Linear(2 * n_freqs**4, embed_dim, bias=False)
        self.learned_linear = nn.Linear(4, 2 * embed_dim, bias=False)
        self.learned_norm = RMSNorm(embed_dim, eps=1e-8)
        self.final_norm = RMSNorm(embed_dim, eps=1e-8)
        self._restore_frequencies()
        self.register_load_state_dict_post_hook(self._restore_frequencies)

    def _restore_frequencies(self, _module=None, _incompatible_keys=None):
        anchor = self.fourier_linear.weight
        freqs = torch.linspace(
            1.0, 10.0, self.n_freqs, dtype=torch.float32, device=anchor.device
        )
        self.freq_matrix = (
            torch.cartesian_prod(freqs, freqs, freqs, freqs).T.contiguous().to(anchor)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # bfloat16 cannot represent the angles.
        angle_dtype = torch.promote_types(coords.dtype, torch.float32)
        # Autocast lowers matmul precision even for two float32 operands.
        phases = (
            coords.to(angle_dtype).unsqueeze(-1) * self.freq_matrix.to(angle_dtype)
        ).sum(dim=-2)
        weight_dtype = self.fourier_linear.weight.dtype
        fourier = self.fourier_linear(
            torch.cat([phases.sin(), phases.cos()], dim=-1).to(weight_dtype)
        )
        value, gate = self.learned_linear(coords.to(weight_dtype)).chunk(2, dim=-1)
        gate = gate * torch.sigmoid(gate) + nn.functional.gelu(gate, approximate="tanh")
        learned = self.learned_norm(value * gate)
        return self.final_norm(fourier + learned)
