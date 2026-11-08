# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: MIT
# Adapted from https://github.com/LiuyinYang1101/CORTEG (MIT).
# Copyright (c) 2026 Liuyin Yang
"""CORTEG: an ST-EEGFormer adapted to intracranial recordings."""

import math
from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from braindecode.functional import check_ch_mask, masked_mean
from braindecode.models.base import EEGModuleMixin
from braindecode.models.steegformer import (
    _ChannelPositionalEmbed,
    _init_weights,
    _TemporalPositionalEncoding,
    _TransformerEncoderBlock,
)
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    has_valid_locations,
)
from braindecode.modules import PatchTokenizer

# Head-frame positions (m) of the first 142 slots of ST-EEGFormer's channel
# vocabulary, the slots the soft branch attends over (the reference's table).
# fmt: off
_VOCAB_POSITIONS = (
    (-0.036158, -0.009984, 0.089752), (0.000325, -0.081115, 0.082615),
    (0.067118, -0.010900, 0.063580), (0.067914, 0.049830, 0.016367),
    (0.083888, 0.001946, 0.008501), (0.000108, -0.114892, 0.014657),
    (-0.029437, 0.083917, -0.006990), (-0.074692, 0.004303, 0.045307),
    (0.078520, -0.060432, 0.012902), (0.071096, -0.062624, 0.047328),
    (0.037672, -0.009624, 0.088412), (0.051836, 0.054305, 0.040814),
    (0.015095, -0.118018, -0.006933), (0.035712, 0.077726, 0.021956),
    (0.000376, 0.027390, 0.088668), (0.079006, -0.028986, 0.049628),
    (0.085549, -0.045545, -0.007130), (0.043895, -0.109127, -0.013170),
    (-0.034062, 0.026011, 0.079987), (0.079534, 0.019936, 0.024438),
    (-0.080280, -0.013760, 0.029160), (0.073056, -0.073068, -0.002540),
    (0.081815, 0.015417, -0.011330), (0.067888, -0.075904, 0.028091),
    (-0.073009, -0.073766, -0.040998), (0.000312, 0.058512, 0.066462),
    (-0.023582, 0.069917, 0.047293), (0.078903, -0.060955, -0.023805),
    (0.025558, 0.070556, 0.047827), (0.073895, -0.074390, -0.041220),
    (0.019420, -0.065595, 0.092405), (-0.086076, -0.044990, -0.067986),
    (0.077002, 0.005336, 0.045350), (-0.082355, 0.000826, 0.008579),
    (0.034784, 0.026438, 0.078808), (0.026437, -0.092295, 0.063199),
    (-0.017195, 0.084849, 0.010027), (-0.054840, 0.068572, -0.010590),
    (0.054988, -0.098091, -0.035541), (0.055743, 0.069657, -0.010755),
    (-0.015820, -0.065600, 0.091164), (-0.072434, -0.073453, -0.002487),
    (-0.027496, 0.056931, 0.060342), (0.018923, 0.085597, 0.011443),
    (-0.054910, -0.098045, -0.035465), (-0.084076, 0.014567, -0.050429),
    (0.038384, -0.047073, 0.090695), (0.000004, -0.118565, -0.023078),
    (-0.018219, 0.009094, 0.092529), (-0.077215, 0.018643, 0.024460),
    (-0.072434, -0.073453, -0.002487), (-0.079592, -0.046551, 0.030949),
    (0.083322, -0.046101, 0.031206), (-0.065238, 0.036428, 0.036144),
    (0.029514, 0.057602, 0.059540), (0.085794, -0.045009, -0.068031),
    (-0.042862, -0.108073, -0.013151), (-0.050800, 0.064041, 0.023089),
    (0.036782, -0.100849, 0.036397), (-0.019862, -0.108942, 0.029760),
    (0.029872, 0.084896, -0.007080), (-0.084161, -0.016019, -0.009346),
    (0.066612, -0.046637, 0.065580), (0.000216, -0.102178, 0.050608),
    (-0.085565, -0.030629, 0.011153), (-0.084161, -0.016019, -0.009346),
    (0.085794, -0.025009, -0.068031), (0.055114, -0.028386, 0.080474),
    (0.085080, -0.015020, -0.009490), (0.065014, -0.087806, -0.018952),
    (-0.060182, 0.022716, 0.055544), (-0.050244, 0.053111, 0.042192),
    (-0.064466, 0.048035, 0.016921), (-0.086076, -0.024990, -0.067986),
    (-0.053007, -0.078788, 0.055940), (0.062293, 0.023723, 0.055630),
    (0.018787, 0.009248, 0.091562), (0.067128, 0.037800, 0.035296),
    (0.078053, 0.032982, 0.004483), (0.020220, -0.028148, 0.098172),
    (0.050674, -0.064482, 0.076130), (0.073056, -0.073068, -0.002540),
    (-0.084125, -0.001847, -0.029794), (0.054609, -0.089640, 0.037035),
    (-0.063556, -0.047009, 0.065624), (-0.035513, -0.047292, 0.091315),
    (-0.033701, 0.076837, 0.021227), (0.084113, 0.014365, -0.050538),
    (-0.014850, -0.117987, -0.006920), (-0.078160, -0.060757, -0.023824),
    (-0.067272, -0.076291, 0.028382), (0.029742, -0.114260, -0.029256),
    (-0.018354, -0.028322, 0.098220), (0.085080, -0.015020, -0.009490),
    (-0.052928, -0.028906, 0.080304), (-0.029413, -0.112449, 0.008839),
    (-0.048424, -0.099341, 0.021599), (-0.064597, -0.087656, -0.019014),
    (-0.054010, -0.089899, 0.037332), (-0.028620, -0.080525, 0.075436),
    (0.000231, 0.080771, 0.035417), (0.049820, -0.099446, 0.021727),
    (-0.036511, -0.100853, 0.037167), (0.029843, -0.112156, 0.008800),
    (-0.068115, -0.062975, 0.047252), (-0.015424, 0.043660, 0.077682),
    (0.051885, 0.007798, 0.073507), (-0.074500, 0.031300, 0.004846),
    (0.017592, 0.044054, 0.077788), (0.045853, 0.041623, 0.060647),
    (0.000401, -0.009167, 0.100244), (-0.084830, -0.046022, -0.007056),
    (0.000112, 0.088247, -0.001713), (0.084123, -0.001808, -0.029638),
    (-0.054840, -0.097528, 0.002792), (-0.046914, -0.064691, 0.075296),
    (0.055667, -0.078560, 0.056561), (0.031920, -0.080487, 0.076716),
    (0.073043, 0.044422, -0.012000), (0.000386, -0.047318, 0.099432),
    (-0.051051, 0.007177, 0.074377), (-0.044410, 0.040762, 0.061690),
    (-0.080775, 0.014120, -0.011135), (-0.029818, -0.114570, -0.029216),
    (0.086000, -0.029820, 0.011248), (0.052397, 0.065071, 0.022862),
    (-0.076407, -0.029731, 0.049217), (0.083456, -0.012776, 0.029208),
    (-0.024648, -0.092292, 0.062076), (0.055667, -0.097625, 0.002730),
    (-0.065358, -0.011632, 0.064358), (0.020294, -0.108914, 0.028944),
    (-0.076680, -0.060832, 0.012880), (-0.070263, 0.042474, -0.011420),
    (-0.085894, -0.015829, -0.048283), (-0.085619, -0.046515, -0.045707),
    (0.085560, -0.016361, -0.048271), (0.086162, -0.047035, -0.045869),
    (-0.013664, -0.109266, 0.032856), (0.013651, -0.109106, 0.030936),
    (-0.012047, -0.092607, 0.065508), (0.013923, -0.092694, 0.066958),
)
# fmt: on


class CORTEG(EEGModuleMixin, nn.Module):
    r"""CORTEG from Yang et al. (2026) [Yang2026b]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    .. figure:: https://raw.githubusercontent.com/LiuyinYang1101/CORTEG/main/docs/assets/figure1_overview_scheme.png
        :align: center
        :width: 1000px

        CORTEG overview, reproduced from the official repository
        (``LiuyinYang1101/CORTEG``).

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    A scalp-EEG foundation model, :class:`~braindecode.models.STEEGFormer`,
    adapted to intracranial (ECoG) recordings. The transformer stays the
    pretrained one; around it, a spatial adapter places each electrode by its
    3D position, and a second token stream carries the high-gamma activity the
    scalp model was never trained on. The paper trains only the spatial
    adapter, the high-gamma patch embedding, LoRA on the last
    ``n_merge_blocks`` blocks with their LayerNorms, and the head (about 297 K
    parameters for the small backbone); this module builds the network, and
    LoRA and freezing are left to the training code.

    .. rubric:: Macro Components

    - **Dual-stream tokenizer** (``CORTEG.patch_embed``,
      ``CORTEG.patch_embed_hi``). *Operations:* the low-frequency stream ``x``
      is cut into patches of ``patch_size`` samples, the high-gamma stream
      ``x_hi`` into patches of ``hi_patch_size``, each linearly embedded and
      the high-gamma tokens average-pooled to the low-frequency count. The
      high-gamma embedding starts from the pretrained low-frequency one,
      linearly interpolated to its patch size.
    - **KNNSoftFourier spatial adapter** (``CORTEG.channel_adapter``).
      *Operations:* a soft branch maps each electrode position to softmax
      weights over the 142 positioned slots of the pretrained channel
      embedding (``CORTEG.channel_pos``, kept frozen) and returns their
      weighted sum; the last bias starts from a Gaussian ``knn_k``-nearest-
      neighbour kernel over the slots' scalp positions, so the soft branch
      starts as a nearest-electrode interpolation. A Fourier residual (``F``
      log-spaced frequencies per axis, zero-initialized output) adds a
      position-specific correction. *Role:* replace the scalp channel
      embedding, which knows no intracranial electrode.
    - **Transformer encoder** (``CORTEG.encoder``). *Operations:* the
      ``depth`` pretrained pre-norm ViT blocks of ST-EEGFormer, over a CLS
      token and the (time-patch, channel) tokens of the low-frequency stream.
    - **Late fusion** (``merge_strategy``). *Operations:* ``"average"``
      replaces the low-frequency tokens by their mean with the high-gamma
      tokens before the last ``n_merge_blocks`` blocks; ``"layerwise_gate"``
      adds the high-gamma tokens after every block, scaled per block and per
      sample by a small gate network that starts at zero.
    - **Read-out** (``CORTEG.norm``, ``CORTEG.final_layer``). *Operations:*
      layer norm, mean over all tokens (CLS included), linear head.

    .. rubric:: Inputs

    The paper decomposes the raw signal into two streams, both z-scored per
    channel with training-set statistics:

    - ``x``: the low-frequency stream, resampled to 128 Hz (the 16-sample
      patches of the pretrained model then span 125 ms);
    - ``x_hi``: the 70-200 Hz Hilbert envelope at 200 Hz, so that its
      25-sample patches line up with those of ``x``.

    With MNE, ``raw.copy().filter(70, 200).apply_hilbert(envelope=True)
    .resample(200)`` gives ``x_hi``. Without ``x_hi`` the model reads ``x``
    alone, as the paper's low-frequency-only ablation.

    Electrode positions are 3D coordinates in metres (the paper's MNI
    coordinates), passed as ``pos`` or read from ``chs_info``.

    .. rubric:: Pre-trained weights

    The backbone loads from the re-hosted ST-EEGFormer checkpoints; the
    high-gamma embedding is warm-started from the loaded weights, and the
    adapter and head keep their initialization::

        model = CORTEG.from_pretrained(
            "braindecode/STEEGFormer-small", n_outputs=5, n_times=128,
            drop_path=0.1, chs_info=chs_info,
        )

    The paper trains with ``drop_path=0.1``; the checkpoint's configuration
    sets ``0`` unless it is passed.

    The authors' Stanford model (finger flexion, nine patients pooled) is
    re-hosted with its LoRA factors folded in::

        model = CORTEG.from_pretrained(
            "braindecode/corteg-stanford-pooled", chs_info=chs_info
        )

    .. note::
        The outputs match the reference implementation for the same weights
        and inputs, for both merge strategies. ``ch_mask`` is an addition:
        the reference batches one patient at a time.

    Parameters
    ----------
    patch_size : int
        Patch size of the low-frequency stream (16 for the released
        backbones).
    hi_patch_size : int
        Patch size of the high-gamma stream.
    embed_dim : int
        Token embedding dimension (512 for the small backbone).
    depth : int
        Number of Transformer encoder blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden-to-embedding ratio of the MLP blocks.
    drop_prob : float
        Dropout rate.
    drop_path : float
        Stochastic-depth rate (max of a linear schedule over ``depth``).
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks.
    n_chans_pos : int
        Size of the pretrained channel vocabulary (145 for small/base/large).
    merge_strategy : {"average", "layerwise_gate"}
        Fusion of the high-gamma stream: the paper's default mean before the
        last ``n_merge_blocks`` blocks (the released Stanford adapter), or the
        layer-wise gate (its BrainTreebank results).
    n_merge_blocks : int
        Number of blocks after the ``"average"`` fusion.
    knn_k : int
        Number of nearest scalp slots of the soft branch's initial kernel.
    adapter_hidden : int
        Hidden width of the spatial adapter's MLPs.
    n_fourier_freqs : int
        Number of Fourier frequencies per axis of the residual branch.
    gate_bottleneck : int
        Hidden width of the layer-wise gate.
    on_non_divisible : {"pad", "crop", "error"}
        How a stream whose length is not a multiple of its patch size is
        handled (right zero-pad, crop, or raise); the reference crops.

    References
    ----------
    .. [Yang2026b] Yang, L., Sun, Q., Van Dyck, B., Calvo Merino, E. &
       Van Hulle, M. M. (2026). CORTEG: Foundation models enable
       cross-modality representation transfer from scalp to intracranial
       brain recordings. Advances in Neural Information Processing Systems
       (NeurIPS 2026). https://arxiv.org/abs/2605.10337
    """

    def __init__(
        self,
        # --- signal-related (handled by EEGModuleMixin) ---
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # --- model hyperparameters (defaults: small backbone) ---
        patch_size: int = 16,
        hi_patch_size: int = 25,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        drop_path: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        n_chans_pos: int = 145,
        merge_strategy: str = "average",
        n_merge_blocks: int = 4,
        knn_k: int = 8,
        adapter_hidden: int = 128,
        n_fourier_freqs: int = 32,
        gate_bottleneck: int = 16,
        on_non_divisible: str = "pad",
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

        if merge_strategy not in ("average", "layerwise_gate"):
            raise ValueError(
                "merge_strategy must be 'average' or 'layerwise_gate', got "
                f"{merge_strategy!r}."
            )
        if not 1 <= n_merge_blocks <= depth:
            raise ValueError(
                f"n_merge_blocks must be in [1, depth={depth}], got {n_merge_blocks}."
            )
        if n_chans_pos < len(_VOCAB_POSITIONS):
            raise ValueError(
                f"n_chans_pos must be at least {len(_VOCAB_POSITIONS)}, the "
                f"positioned slots of the vocabulary, got {n_chans_pos}."
            )
        self.embed_dim = embed_dim
        self.depth = depth
        self.merge_strategy = merge_strategy
        self.merge_at = depth - n_merge_blocks
        self.knn_k = knn_k

        default_pos: Optional[torch.Tensor] = None
        chs_info = self._chs_info
        if has_valid_locations(chs_info):
            locations = extract_channel_locations_from_chs_info(chs_info)
            default_pos = torch.as_tensor(locations, dtype=torch.float32)
        # Montage-specific, rebuilt from chs_info: not saved with the weights.
        self.register_buffer("default_pos", default_pos, persistent=False)
        self.register_buffer(
            "vocab_positions",
            torch.tensor(_VOCAB_POSITIONS, dtype=torch.float32),
            persistent=False,
        )

        # The ST-EEGFormer backbone, under the names of STEEGFormer's modules
        # so that its checkpoints load as they are.
        self.patch_embed = PatchTokenizer(
            patch_size=patch_size,
            n_times=self.n_times,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible=on_non_divisible,
            projection="linear",
            output_order="patch_channel",
        )
        self.temporal_pos = _TemporalPositionalEncoding(embed_dim)
        self.channel_pos = _ChannelPositionalEmbed(n_chans_pos, embed_dim)
        # The soft branch's codebook, frozen as in the paper.
        self.channel_pos.embedding.weight.requires_grad_(False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.encoder = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    embed_dim, num_heads, drop_prob, int(mlp_ratio), dpr[i], activation
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # x_hi of this length gives one high-gamma token per low-frequency token.
        self.patch_embed_hi = PatchTokenizer(
            patch_size=hi_patch_size,
            n_times=self.patch_embed.n_patches(self.n_times) * hi_patch_size,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible=on_non_divisible,
            projection="linear",
            output_order="patch_channel",
        )
        self.channel_adapter = _KNNSoftFourierAdapter(
            len(_VOCAB_POSITIONS), embed_dim, adapter_hidden, n_fourier_freqs
        )
        self.layerwise_gate: Optional[_LayerwiseGate] = (
            _LayerwiseGate(embed_dim, depth, gate_bottleneck)
            if merge_strategy == "layerwise_gate"
            else None
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

        self._init_parameters()
        # A checkpoint without a high-gamma embedding warm-starts it.
        self.register_load_state_dict_pre_hook(self._fill_hi_embed)

    def _init_parameters(self) -> None:
        """Model-specific initialization (see ``reset_parameters``)."""
        for module in (self.patch_embed, self.encoder, self.norm, self.final_layer):
            module.apply(_init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        with torch.no_grad():
            proj = self.patch_embed_hi.proj
            assert proj is not None
            proj.weight.copy_(self._hi_weight(self.patch_embed.proj.weight))
            proj.bias.copy_(self.patch_embed.proj.bias)
        prior = None
        if self.default_pos is not None:
            prior = _knn_log_prior(self.default_pos, self.vocab_positions, self.knn_k)
        self.channel_adapter.init_parameters(prior)
        if self.layerwise_gate is not None:
            self.layerwise_gate.init_parameters()

    def _hi_weight(self, lo_weight: torch.Tensor) -> torch.Tensor:
        hi_patch_size = self.patch_embed_hi.patch_size
        return F.interpolate(lo_weight[None], size=hi_patch_size, mode="linear")[0]

    def _fill_hi_embed(self, module, state_dict, prefix, *_):
        lo, hi = f"{prefix}patch_embed.proj.", f"{prefix}patch_embed_hi.proj."
        if f"{lo}weight" in state_dict and f"{hi}weight" not in state_dict:
            state_dict[f"{hi}weight"] = self._hi_weight(state_dict[f"{lo}weight"])
            state_dict[f"{hi}bias"] = state_dict[f"{lo}bias"]

    def reset_head(self, n_outputs):
        """Replace the linear head for a new ``n_outputs``."""
        self._n_outputs = n_outputs
        self._update_init_kwargs(n_outputs=n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        _init_weights(self.final_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_hi: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode a batch of intracranial recordings.

        Parameters
        ----------
        x : torch.Tensor
            Low-frequency stream, ``(batch, n_chans, n_times)``.
        x_hi : torch.Tensor, optional
            High-gamma stream of the same channels, ``(batch, n_chans,
            n_times_hi)``. Without it the model reads ``x`` alone.
        pos : torch.Tensor, optional
            Electrode positions in metres, ``(batch, n_chans, 3)`` or
            ``(n_chans, 3)``; by default those of ``chs_info``.
        ch_mask : torch.Tensor, optional
            ``(batch, n_chans)`` mask of real channels, see
            :func:`~braindecode.functional.check_ch_mask`. Padded channels
            take part in no attention and no pooling.
        return_features : bool
            If ``True``, return ``{"features", "cls_token", "tokens"}``: the
            layer-normalized patch tokens ``(batch, n_patches * n_chans,
            embed_dim)``, the CLS token, and the patch tokens as ``(batch,
            n_chans, n_patches, embed_dim)``; padded channels' tokens are zero.

        Returns
        -------
        torch.Tensor | dict
            Outputs ``(batch, n_outputs)``, or the feature dict.
        """
        mask = check_ch_mask(ch_mask, x)
        if pos is None:
            pos = self.default_pos
        if pos is None:
            raise ValueError(
                "CORTEG places each electrode by its 3D position: pass pos, or "
                "chs_info with the electrode positions."
            )
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if pos.shape[1] != x.shape[1] or pos.shape[2] != 3:
            raise ValueError(
                f"pos must have shape (batch, {x.shape[1]}, 3) or ({x.shape[1]}, "
                f"3), got {list(pos.shape)}."
            )
        if mask is not None:
            # Padding may hold NaN, which the embeddings would spread.
            x = x.masked_fill(~mask[:, :, None], 0.0)
            pos = pos.masked_fill(~mask[:, :, None], 0.0)
            if x_hi is not None:
                x_hi = x_hi.masked_fill(~mask[:, :, None], 0.0)

        tokens = self.patch_embed(x)  # (batch, n_patches, n_chans, embed_dim)
        batch_size, n_patches, n_chans, _ = tokens.shape
        codebook = self.channel_pos.embedding.weight[: self.vocab_positions.shape[0]]
        channel_emb = self.channel_adapter(pos, codebook)
        embedding = self.temporal_pos(n_patches) + channel_emb[:, None]
        lo = (tokens + embedding).flatten(1, 2)
        hi: Optional[torch.Tensor] = None
        if x_hi is not None:
            hi_tokens = self.patch_embed_hi(x_hi)
            if hi_tokens.shape[1] != n_patches:
                pooled = F.adaptive_avg_pool1d(
                    hi_tokens.permute(0, 2, 3, 1).flatten(0, 1), n_patches
                )
                hi_tokens = pooled.reshape(
                    batch_size, n_chans, self.embed_dim, n_patches
                ).permute(0, 3, 1, 2)
            hi = (hi_tokens + embedding).flatten(1, 2)

        token_mask: Optional[torch.Tensor] = None
        key_mask: Optional[torch.Tensor] = None
        attn_mask: Optional[torch.Tensor] = None
        if mask is not None:
            token_mask = mask.repeat(1, n_patches)  # tokens are (patch, channel)
            key_mask = F.pad(token_mask, (1, 0), value=True)
            attn_mask = key_mask[:, None, None, :]

        cls = self.cls_token + self.temporal_pos.cls_token_encoding()
        z = self.pos_drop(torch.cat([cls.expand(batch_size, -1, -1), lo], dim=1))
        gates: Optional[torch.Tensor] = None
        if hi is not None and self.layerwise_gate is not None:
            gates = self.layerwise_gate(
                masked_mean(lo, token_mask, [1]), masked_mean(hi, token_mask, [1])
            )
        for i, block in enumerate(self.encoder):
            if hi is not None and gates is None and i == self.merge_at:
                z = torch.cat([z[:, :1], 0.5 * (z[:, 1:] + hi)], dim=1)
            z = block(z, attn_mask)
            if hi is not None and gates is not None:
                z = z + gates[:, i, None, None] * F.pad(hi, (0, 0, 1, 0))
        z = self.norm(z)

        if return_features:
            patches = z[:, 1:]
            if token_mask is not None:
                patches = patches.masked_fill(~token_mask[:, :, None], 0.0)
            per_channel = patches.reshape(
                batch_size, n_patches, n_chans, self.embed_dim
            ).transpose(1, 2)
            return {"features": patches, "cls_token": z[:, 0], "tokens": per_channel}
        return self.final_layer(masked_mean(z, key_mask, [1]))


def _knn_log_prior(
    positions: torch.Tensor, vocab_positions: torch.Tensor, k: int
) -> torch.Tensor:
    """Log Gaussian k-NN weight of each vocabulary slot, averaged over ``positions``.

    The bandwidth is the median distance of the ``k`` nearest slots.
    """
    distances, slots = torch.cdist(positions, vocab_positions).topk(
        k, dim=-1, largest=False
    )
    sigma = distances.median().clamp(min=1e-6)
    weights = torch.exp(-0.5 * (distances / sigma) ** 2)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    dense = torch.zeros(len(positions), len(vocab_positions)).scatter_(
        1, slots, weights
    )
    return torch.log(dense + 1e-8).mean(dim=0)


class _KNNSoftFourierAdapter(nn.Module):
    """Channel embedding of a 3D position: a soft branch over the pretrained
    codebook plus a Fourier-feature residual."""

    def __init__(self, n_slots: int, embed_dim: int, hidden: int, n_freqs: int):
        super().__init__()
        self.soft = _SoftCodebookBranch(n_slots, hidden)
        self.residual = _FourierBranch(embed_dim, hidden, n_freqs)

    def init_parameters(self, prior: Optional[torch.Tensor]) -> None:
        """Start the soft branch at ``prior`` (uniform if ``None``) and the
        residual at zero."""
        with torch.no_grad():
            bias = self.soft.net[-1].bias
            bias.copy_(torch.zeros_like(bias) if prior is None else prior)
            self.soft.scale.fill_(1.0)
        nn.init.zeros_(self.residual.net[-1].weight)

    def forward(self, pos: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return self.soft(pos, codebook) + self.residual(pos)


class _SoftCodebookBranch(nn.Module):
    """Softmax attention of a position over the codebook slots."""

    def __init__(self, n_slots: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_slots),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pos: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.matmul(self.net(pos).softmax(dim=-1), codebook)


class _FourierBranch(nn.Module):
    """MLP on sinusoidal features of a position, at log-spaced frequencies."""

    freqs: torch.Tensor

    def __init__(self, embed_dim: int, hidden: int, n_freqs: int):
        super().__init__()
        self.register_buffer(
            "freqs", torch.logspace(0, math.log10(50.0), n_freqs), persistent=False
        )
        self.net = nn.Sequential(
            nn.Linear(6 * n_freqs, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim, bias=False),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        angles = (pos[..., None] * self.freqs).flatten(-2)
        return self.net(torch.cat([angles.sin(), angles.cos()], dim=-1))


class _LayerwiseGate(nn.Module):
    """One tanh gate per block and sample, from the mean low and high tokens."""

    def __init__(self, embed_dim: int, depth: int, bottleneck: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, depth),
        )

    def init_parameters(self) -> None:
        """Start every gate closed, at the low-frequency-only model."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([lo, hi], dim=-1)))
