# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: MIT (see NOTICE.txt)
#
# Adapted from https://github.com/PierreGtch/eeg-fm-masking (commit 38a089e, MIT):
# eeg_fm_masking/oeb/wrapper.py, models.py, modules.py, transformer.py, functions.py.
"""Encoder shared by the 58 eeg-fm-masking checkpoints (Guetschel et al., 2026)."""

import math
import numbers
import warnings
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from mne.io.constants import FIFF
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.signal_jepa import _pos_encode_time
from braindecode.models.util import has_valid_locations, warn_if_sfreq_differs

_NORMALIZATIONS = ("median_std_clip", "none")
_PRETEXTS = ("mae", "jepa")
_MASK_RADII = ("one", "6cm", "9cm", "12cm", "all")
_MASK_LENGTHS = (1, 2, 4, 8, 16, 33)


def _xyz_div_term(n_dim):
    """Frequencies of the coordinate encoding, as ``pos_encode_continuous_batched``.

    Computed once, on the CPU in the default dtype, as the reference does at each
    call; stored as a buffer, it then follows the model dtype exactly (a float32
    to float64 cast is exact).
    """
    # on the CPU whatever the default-device context, which would change the last bits
    freqs = torch.arange(0, n_dim, 2, device="cpu")
    return torch.exp((1 - freqs / n_dim) * 2 * math.pi)


def _pos_encode_xyz(ch_pos, x_min: float, x_max: float, div_term):
    """Sinusoidal encoding of the coordinates, ``(..., 3) -> (..., 3, n_dim)``.

    The order of the operations is the one of the reference implementation
    (``pos_encode_continuous_batched``): a float64 intermediate or another
    order changes the float32 results by up to 5.7e-5.
    """
    out = ch_pos.new_empty(ch_pos.shape + (2 * div_term.shape[0],))
    xx = ((ch_pos - x_min) / (x_max - x_min)).unsqueeze(-1)
    out[..., 0::2] = torch.sin(xx * div_term)
    out[..., 1::2] = torch.cos(xx * div_term)
    return out


class _GaussianRandomProjection(nn.Module):
    """Fixed Gaussian random projection ``(..., n_features) -> (..., n_components)``.

    The matrix has shape ``(n_components, n_features)`` and entries drawn from
    :math:`\\mathcal{N}(0, 1/n_components)`, the distribution of
    :class:`sklearn.random_projection.GaussianRandomProjection` and of the
    ``_make_projection_matrix`` of OpenEEGBench. It is drawn in float32 on the CPU
    with a dedicated generator, so the global random state and the default device
    are not involved: the same seed gives the same matrix for a given PyTorch
    version and CPU type (the Gaussian sampler is vectorised differently across
    CPU architectures, so the last bits can differ between platforms). It is
    then moved to the default device and stored in the default dtype (a no-op
    on the CPU in float32). It is a persistent buffer, not a parameter: it is
    saved in the ``state_dict`` and never trained.
    """

    def __init__(self, n_features: int, n_components: int, seed: int = 0):
        super().__init__()
        self.n_features, self.n_components = n_features, n_components
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.empty(
            n_components, n_features, dtype=torch.float32, device="cpu"
        )
        projection.normal_(0.0, 1.0 / math.sqrt(n_components), generator=generator)
        # cast once here (not in forward, which would copy the matrix at each call);
        # the draw stays on the CPU, only the result follows the default device
        self.register_buffer(
            "projection",
            projection.to(
                device=torch.get_default_device(), dtype=torch.get_default_dtype()
            ),
        )

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, n_components={self.n_components}"

    def forward(self, x):
        return F.linear(x, self.projection)  # x @ projection.T, as in OpenEEGBench


class _LinearPatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, patch_overlap):
        super().__init__()
        self.patch_size, self.patch_step = patch_size, patch_size - patch_overlap
        self.linear = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        # (B, C, T) -> (B, C, P, D); the linear layer acts on the unfold view.
        return self.linear(x.unfold(2, self.patch_size, self.patch_step))


class _PositionalEncoder(nn.Module):
    def __init__(self, ch_pos, embed_dim, pos_half_range, max_n_times):
        super().__init__()
        self.coord_dim = self.time_dim = embed_dim // 4
        self.pos_half_range, self.max_n_times = pos_half_range, max_n_times
        self.register_buffer("ch_pos", ch_pos, persistent=False)
        self.register_buffer(
            "div_term",
            _xyz_div_term(self.coord_dim).to(torch.get_default_device()),
            persistent=False,
        )
        self.register_buffer(
            "encoding_time",
            # formula values until the weights are loaded: the checkpoints all store
            # the same table, which overwrites them. Computed on the CPU, then moved
            # to the default device.
            # 33 = (6000 - 200) // 180 + 1 patches: the 30 s pre-training window
            _pos_encode_time(33, self.time_dim, max_n_times).to(
                torch.get_default_device()
            ),
        )

    def forward(self, n_patches: int, batch_size: int):
        """Return the additive encoding, ``(batch_size, n_chans, n_patches, embed_dim)``."""
        h = self.pos_half_range
        # The sin/cos run on the batch-expanded positions, like the reference
        # implementation: torch.sin can differ in the last bit for the same
        # value depending on its place in the flattened tensor and on the number
        # of threads, so another layout breaks the bit-for-bit parity.
        ch_pos = self.ch_pos[None].expand(batch_size, -1, -1)
        spat = _pos_encode_xyz(ch_pos, -h, h, self.div_term).flatten(-2)  # (B, C, 3d)
        if n_patches <= self.encoding_time.shape[0]:
            # the stored table: a recomputation differs in the last bits depending on
            # the platform's math library, so the checkpoint's values are used
            enc_t = self.encoding_time[:n_patches]
        else:  # beyond the stored table: recomputed for this call, not stored
            enc_t = _pos_encode_time(
                n_patches, self.time_dim, self.max_n_times, self.encoding_time.device
            ).to(self.encoding_time.dtype)
        c = spat.shape[1]
        # the concatenation equals the original zero-padded sum, bitwise
        return torch.cat(
            [
                spat[:, :, None].expand(-1, -1, n_patches, -1),
                enc_t[None, None].expand(batch_size, c, -1, -1),
            ],
            -1,
        )


class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_prob):
        super().__init__()
        self.num_heads, self.drop_prob = num_heads, drop_prob
        self.packed_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        q, k, v = self.packed_proj(x).chunk(3, dim=-1)
        q, k, v = (
            t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2) for t in (q, k, v)
        )
        a = F.scaled_dot_product_attention(  # default scale, never a manual softmax
            q, k, v, dropout_p=self.drop_prob if self.training else 0.0
        )
        return self.out_proj(a.transpose(1, 2).flatten(-2))


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, drop_prob, activation):
        super().__init__()
        self.self_attn = _MultiHeadAttention(embed_dim, num_heads, drop_prob)
        self.linear1 = nn.Linear(embed_dim, 2 * dim_feedforward, bias=False)
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim, bias=False)
        self.norm1, self.norm2 = nn.RMSNorm(embed_dim), nn.RMSNorm(embed_dim)
        self.dropout1, self.dropout2 = nn.Dropout(drop_prob), nn.Dropout(drop_prob)
        self.activation = activation()

    def forward(self, x):
        x = x + self.dropout1(self.self_attn(self.norm1(x)))
        gate, up = self.linear1(self.norm2(x)).chunk(2, dim=-1)  # GELU on the gate
        return x + self.dropout2(self.linear2(self.dropout(self.activation(gate) * up)))


class _TransformerEncoder(nn.Module):
    def __init__(self, depth, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [_TransformerEncoderLayer(**layer_kwargs) for _ in range(depth)]
        )

    def forward(self, z):
        # (B, C, P, D); the tokens are channel-major.
        c, t = z.shape[1], z.shape[2]
        z = z.flatten(1, 2)
        for layer in self.layers:
            z = layer(z)
        return z.unflatten(1, (c, t))


class _ContextualEncoder(nn.Module):
    def __init__(self, pos_encoder, transformer):
        super().__init__()
        self.pos_encoder, self.transformer = pos_encoder, transformer

    def forward(self, local_features):
        pos = self.pos_encoder(local_features.shape[2], local_features.shape[0])
        # as the reference, whose encoding is written into a tensor of the features'
        # dtype: under autocast the residual stream stays in the autocast dtype
        # (a no-op in float32 and float64)
        return self.transformer(local_features + pos.to(local_features.dtype))


class Guetschel2026(EEGModuleMixin, nn.Module, license="mit"):
    r"""Encoder of the EEG masking-geometry study from Guetschel et al. (2026) [guetschel2026]_.

    Official website of the study: https://pierregtch.github.io/eeg-fm-masking/

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-dark-line:`Channel`

    .. versionadded:: 1.8.2

    .. figure:: ../_static/model/guetschel2026_arch.png
        :align: center
        :alt: Figure 1 of Guetschel et al. (2026): shared MAE/JEPA pre-training pipeline (A) and the block-masking geometries (B)
        :width: 1000px

        Figure 1 of [guetschel2026]_. (A) MAE and JEPA share one pre-training
        pipeline; this class is its tokeniser and encoder. (B) The masks are
        blocks of spatial radius :math:`r` and temporal length :math:`L`.

    Which mask should an EEG foundation model learn from? The study answers
    with a controlled sweep: one backbone, pre-trained 58 times on the same data
    with the same recipe, changing only the pretext (MAE or JEPA) and the
    geometry of the mask. This class is that backbone: all 58 checkpoints load
    into it.

    Both pretexts agree on the best mask, blocks of radius 9 cm and length
    2 patches. With it, the frozen features reach the level of REVE-Base under
    a linear probe on the 12 datasets of OpenEEGBench, with 12.7 M parameters
    and a fraction of REVE's pre-training compute.

    .. rubric:: Architecture

    The backbone follows REVE-Small, with a simpler positional encoding:

    - ``feature_encoder`` **Patch embedding.** Each channel is cut into
      overlapping 1 s patches (200 samples, 20 of overlap), and one linear
      layer embeds each patch into 512 features.
    - ``model.pos_encoder`` **Positional encoding.** Fixed sinusoids of the
      electrode position :math:`(x, y, z)` and of the patch index are added to
      the embeddings. Any montage works, as long as the channels have
      positions.
    - ``model.transformer`` **Transformer encoder.** Four pre-norm layers
      (RMSNorm, 8 heads, GEGLU feed-forward) attend over all the
      channel × patch tokens.
    - ``final_layer`` **Head.** Flatten, then a linear layer, with an optional
      fixed random projection in between (see below). It is not part of the
      checkpoints.

    .. rubric:: The 58 checkpoints

    Each checkpoint is a Hugging Face repository, named after its three
    parameters (:meth:`hub_repo_id` builds the name)::

        PierreGtch/eeg-fm-masking_{pretext}_r{radius}_L{length}

    ================  ==========================================================
    Pretext           ``mae`` or ``jepa``
    Radius :math:`r`  ``one`` (a single channel), ``6cm``, ``9cm``, ``12cm``,
                      ``all`` (every channel)
    Length :math:`L`  1, 2, 4, 8, 16 or 33 patches (33 is the whole 30 s
                      window)
    ================  ==========================================================

    Every combination exists except :math:`r` = all with :math:`L` = 33, which
    would mask the whole window: 2 × 29 = 58 checkpoints.

    - **Recommended:** ``mae_r9cm_L2`` or ``jepa_r9cm_L2``. Many other masks
      are nearly as good.
    - **To avoid** for downstream use: masks that are too local
      (:math:`r` = one), too global (:math:`r` = all) or, in most cases, too
      long (:math:`L` = 8, 16). JEPA collapses at :math:`r` = all. These
      checkpoints are released to study how the mask shapes the
      representations.
    - **Intermediate epochs:** ``model.safetensors`` is the end of epoch 10,
      the one evaluated in the paper; the folders ``epoch_01/`` to
      ``epoch_09/`` hold the earlier epochs of the same run.

    `License <https://github.com/PierreGtch/eeg-fm-masking/blob/main/LICENSE>`_
    (MIT, the code). The weights are released under CC-BY-4.0 (`collection
    <https://huggingface.co/collections/PierreGtch/eeg-fm-masking-6ab912b6a03bba1348fc7366>`_,
    `project page <https://pierregtch.github.io/eeg-fm-masking/>`_).

    .. rubric:: Usage

    .. code-block:: python

        from braindecode.models import Guetschel2026

        raw.set_montage("standard_1020")  # channel positions, in metres
        model = Guetschel2026.from_pretrained(
            Guetschel2026.hub_repo_id("mae", "9cm", 2),
            chs_info=raw.info["chs"],
            n_times=1000,  # 5 s at 200 Hz
            n_outputs=4,
            # filename="epoch_05/model.safetensors",  # an intermediate epoch
        )

        # features: (batch, n_chans, n_patches, 512)
        features = model(x, return_features=True)["features"]

        # linear probing: train only the head, which starts from random weights
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("final_layer.")

    .. rubric:: Random projection head

    The paper probes the frozen features as OpenEEGBench does: it projects the
    flattened features to 5000 dimensions with a Gaussian random projection,
    then fits a linear model on top. Pass ``random_projection=5000`` to put the
    same projection between the flatten and the linear layer of the head:

    - The projection is drawn once, from ``random_projection_seed``, with the
      same distribution as scikit-learn's
      :class:`~sklearn.random_projection.GaussianRandomProjection`. It is
      stored as a buffer: saved with the model, never trained.
    - It is large: ``random_projection`` × ``n_chans`` × ``n_patches`` × 512
      values, about 0.9 GB in float32 for 5000 components, 22 channels and 4 s
      windows.

    .. warning::

       **Input requirements**

       - **Sampling rate:** 200 Hz, as in pre-training.
       - **Units:** volts, without standardisation. The model scales each
         window itself, as in pre-training (microvolts, then division by the
         median channel standard deviation, clipped at 15).
       - **Channel positions:** in metres, in the MNE head frame, as given by
         ``raw.set_montage(...)``. Every channel must be 5 to 20 cm from the
         origin, which catches positions in centimetres or millimetres. For
         channels with standard names but no positions, pass
         ``channel_strategy="exact"``.
       - **Window:** at least 200 samples; trailing samples that do not fill a
         patch are dropped.

    .. note::

       **Differences from the reference implementation.** The backbone gives
       bit-identical features. Around it:

       - the head takes the actual number of overlapping patches,
         ``(n_times - 200) // 180 + 1``. The original wrapper sizes its head for
         ``n_times // 200`` patches, which ignores the overlap, so that head
         only fits short windows (not 2000 or 6000 samples, for instance);
         OpenEEGBench replaces it, so the paper's results do not depend on it;
       - the random projection head is new;
       - channel positions are validated;
       - attention dropout is off in eval mode.

    Parameters
    ----------
    embed_dim : int, default=512
        Width of the tokens. Must be divisible by 8 and by ``num_heads``.
    depth : int, default=4
        Number of transformer layers.
    num_heads : int, default=8
        Number of attention heads.
    dim_feedforward : int, default=1365
        Width of each half of the GEGLU feed-forward block.
    patch_size : int, default=200
        Number of samples of a patch (1 s at 200 Hz).
    patch_overlap : int, default=20
        Number of samples shared by two consecutive patches.
    pos_half_range : float, default=0.15
        Half range, in metres, of the electrode coordinates mapped to
        :math:`[0, 1]` before the sinusoidal encoding.
    activation : type[nn.Module], default=nn.GELU
        Activation of the gate of the feed-forward block.
    drop_prob : float, default=0.0
        Dropout probability of the attention and feed-forward branches.
    normalization : {"median_std_clip", "none"}, default="median_std_clip"
        Input scaling. ``"median_std_clip"`` divides each window by the median
        over channels of the channel standard deviations and clips it at
        ``clip_sigma``; ``"none"`` only multiplies by ``input_scale``. Use
        ``"none"`` with ``input_scale=1.0`` for data you already scaled.
    input_scale : float, default=1e6
        Factor applied to the input first (volts to microvolts).
    clip_sigma : float, default=15.0
        Clipping bound of ``"median_std_clip"``.
    random_projection : int or None, default=None
        Size of the fixed random projection inserted in the head (5000 in the
        paper), or ``None`` for no projection. Memory grows linearly with it
        (see "Random projection head" above).
    random_projection_seed : int, default=0
        Seed of the random projection.

    Raises
    ------
    ValueError
        If the channel positions are missing, not in metres (every channel
        must be 5 to 20 cm from the origin) or not in the MNE head frame, if
        an architecture argument is invalid, or if the signal-related
        parameters are missing and cannot be inferred.

    References
    ----------
    .. [guetschel2026] Guetschel, P., Aristimunha, B., El Ouahidi, Y.,
       Delorme, A., Moreau, T., & Tangermann, M. (2026).
       What masking geometry works best for EEG foundation models?
       arXiv:2609.33487. https://arxiv.org/abs/2609.33487
       Code: https://github.com/PierreGtch/eeg-fm-masking.
       Project page: https://pierregtch.github.io/eeg-fm-masking/.
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # backbone; the defaults are the ones of the 58 released checkpoints
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1365,
        patch_size: int = 200,
        patch_overlap: int = 20,
        pos_half_range: float = 0.15,
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.0,
        # input scaling, applied in forward as in the original wrapper
        normalization: str = "median_std_clip",
        input_scale: float = 1e6,
        clip_sigma: float = 15.0,
        # head; a projection buffer holds random_projection x n_chans x n_patches x 512
        # floats (~0.9 GB for 5000 components at 22 channels, 4 s)
        random_projection: int | None = None,
        random_projection_seed: int = 0,
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

        if normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_NORMALIZATIONS}, got {normalization!r}."
            )
        if embed_dim % 8 != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by 8: each of the four "
                "positional encodings needs an even width."
            )
        if (
            isinstance(num_heads, bool)
            or not isinstance(num_heads, numbers.Integral)
            or num_heads < 1
        ):
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads!r}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if not (math.isfinite(pos_half_range) and pos_half_range > 0):
            raise ValueError(
                f"pos_half_range must be a positive finite number, got {pos_half_range!r}."
            )
        if not 0 <= patch_overlap < patch_size:
            raise ValueError(
                f"patch_overlap ({patch_overlap}) must satisfy "
                f"0 <= patch_overlap < patch_size ({patch_size})."
            )
        if random_projection is not None and (
            isinstance(random_projection, bool)
            or not isinstance(random_projection, numbers.Integral)
            or random_projection < 1
        ):
            raise ValueError(
                "random_projection must be None or an integer >= 1, "
                f"got {random_projection!r}."
            )
        if isinstance(random_projection_seed, bool) or not isinstance(
            random_projection_seed, numbers.Integral
        ):
            raise ValueError(
                f"random_projection_seed must be an integer, got {random_projection_seed!r}."
            )
        if self.n_times < patch_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be at least patch_size ({patch_size})."
            )
        if self._sfreq is not None or self._input_window_seconds is not None:
            # sfreq given or derived from n_times / input_window_seconds
            warn_if_sfreq_differs("Guetschel2026", self.sfreq, 200.0)
        if not has_valid_locations(self.chs_info):
            raise ValueError(
                "Guetschel2026 requires channel locations: a finite loc[:3] in "
                "metres for every channel (raw.set_montage(...)). For standard "
                "10-05 names without positions, pass channel_strategy='exact'; "
                "other names (e.g. EGI 'E1') need positions in metres."
            )

        # loc[:3] must be head-frame coordinates; a missing coord_frame counts as head
        other_frame = [
            ch.get("ch_name", f"#{i}")
            for i, ch in enumerate(self.chs_info)
            if ch.get("coord_frame", FIFF.FIFFV_COORD_HEAD)
            not in (FIFF.FIFFV_COORD_HEAD, "head")
        ]
        if other_frame:
            raise ValueError(
                "Guetschel2026 requires channel positions in the MNE head frame, but "
                f"{len(other_frame)} channels are in another coordinate frame: "
                f"{other_frame[:5]}. Set the montage with raw.set_montage(...) or "
                "transform the positions to the head frame."
            )
        ch_pos_f64 = np.array([ch["loc"][:3] for ch in self.chs_info], dtype=np.float64)
        # Every electrode lies 5 to 20 cm from the head-frame origin (MNE montages:
        # 0.064-0.146 m; REVE's 32,713 pre-training channels: 0.078-0.133 m), so
        # centimetres, millimetres or a placeholder at the origin fall outside.
        # Checked on the float64 positions as given: the bounds are inclusive.
        dist = np.linalg.norm(ch_pos_f64, axis=1)
        bad = np.flatnonzero((dist < 0.05) | (dist > 0.20))
        if bad.size:
            listed = ", ".join(
                f"{self.chs_info[i].get('ch_name', f'#{i}')!r} ({dist[i]:.3g} m)"
                for i in bad[:5]
            )
            raise ValueError(
                "Guetschel2026 requires channel positions in METRES (MNE head "
                f"frame), 5 to 20 cm from the origin; {bad.size} of {len(dist)} "
                f"are not: {listed}{', ...' if bad.size > 5 else ''}. Convert "
                "centimetres or millimetres to metres. For standard 10-05 names "
                "with an all-zero loc, pass channel_strategy='exact' (it does not "
                "replace given positions; other names need positions in metres)."
            )
        # float32 BEFORE the encoding: float64 then cast gives a 4.2e-5 difference.
        ch_pos_np = ch_pos_f64.astype(np.float32)
        ch_pos = torch.from_numpy(ch_pos_np).to(torch.get_default_device())
        self.normalization: str = normalization
        self.input_scale: float = input_scale
        self.clip_sigma: float = clip_sigma
        self.patch_size: int = patch_size
        self.patch_step: int = patch_size - patch_overlap
        self.embed_dim: int = embed_dim
        self.random_projection: int | None = (
            None if random_projection is None else int(random_projection)
        )
        self.random_projection_seed: int = int(random_projection_seed)
        # store the normalised values (e.g. a NumPy integer) in the saved config
        self._update_init_kwargs(
            random_projection=self.random_projection,
            random_projection_seed=self.random_projection_seed,
        )
        max_n_times = int(600.0 * (200.0 / self.patch_step))  # 600 s at 200 Hz: 666
        self.n_patches: int = (self.n_times - patch_size) // self.patch_step + 1

        # The attribute names are the key prefixes of the released checkpoints
        # (feature_encoder.*, model.*): do not rename them.
        self.feature_encoder: nn.Module = _LinearPatchEmbedding(
            embed_dim, patch_size, patch_overlap
        )
        self.model: nn.Module = _ContextualEncoder(
            _PositionalEncoder(ch_pos, embed_dim, pos_half_range, max_n_times),
            _TransformerEncoder(
                depth,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                drop_prob=drop_prob,
                activation=activation,
            ),
        )
        self.final_layer: nn.Sequential = self._build_head(self.n_outputs)

    @classmethod
    def hub_repo_id(cls, pretext: str, mask_radius: str, mask_length: int) -> str:
        """Hub repository of one of the 58 released checkpoints.

        Parameters
        ----------
        pretext : {"mae", "jepa"}
            Pre-training objective.
        mask_radius : {"one", "6cm", "9cm", "12cm", "all"}
            Radius of the masked blocks of channels.
        mask_length : {1, 2, 4, 8, 16, 33}
            Length of the masked blocks in patches. ``("all", 33)`` was never
            trained, because it would mask the whole window.

        Returns
        -------
        str
            ``"PierreGtch/eeg-fm-masking_{pretext}_r{mask_radius}_L{mask_length}"``.

        Raises
        ------
        ValueError
            If a value is not in the lists above, or for ``("all", 33)``.

        Examples
        --------
        >>> Guetschel2026.hub_repo_id("mae", "9cm", 2)
        'PierreGtch/eeg-fm-masking_mae_r9cm_L2'
        """
        if pretext not in _PRETEXTS:
            raise ValueError(f"pretext must be one of {_PRETEXTS}, got {pretext!r}.")
        if mask_radius not in _MASK_RADII:
            raise ValueError(
                f"mask_radius must be one of {_MASK_RADII}, got {mask_radius!r}."
            )
        if (
            isinstance(mask_length, bool)
            or not isinstance(mask_length, numbers.Integral)
            or mask_length not in _MASK_LENGTHS
        ):
            raise ValueError(
                f"mask_length must be one of {_MASK_LENGTHS}, got {mask_length!r}."
            )
        if (mask_radius, mask_length) == ("all", 33):
            raise ValueError(
                "No checkpoint was trained with mask_radius='all' and mask_length=33: "
                "it would mask the whole window. Use another radius or length."
            )
        return f"PierreGtch/eeg-fm-masking_{pretext}_r{mask_radius}_L{int(mask_length)}"

    def _build_head(self, n_outputs: int) -> nn.Sequential:
        n_features = self.n_chans * self.n_patches * self.embed_dim
        if self.random_projection is None:
            return nn.Sequential(nn.Flatten(), nn.Linear(n_features, n_outputs))
        if self.random_projection > n_features:
            warnings.warn(
                f"random_projection={self.random_projection} exceeds the {n_features} "
                "flattened features: the projection expands rather than reduces them. "
                "Consider random_projection=None (what OpenEEGBench does in this "
                "case) or a smaller value.",
                UserWarning,
                stacklevel=4,  # _build_head, __init__, the init wrapper, caller
            )
        return nn.Sequential(
            nn.Flatten(),
            _GaussianRandomProjection(
                n_features, self.random_projection, self.random_projection_seed
            ),
            nn.Linear(self.random_projection, n_outputs),
        )

    def reset_head(self, n_outputs: int) -> None:
        """Replace the last linear layer, keeping the random projection.

        Parameters
        ----------
        n_outputs : int
            New number of outputs.
        """
        self._set_n_outputs(n_outputs)
        old = self.final_layer[-1]
        self.final_layer[-1] = nn.Linear(
            old.in_features, n_outputs, device=old.weight.device, dtype=old.weight.dtype
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load a state dict whose backbone matches this model exactly.

        Arguments are passed on to :meth:`torch.nn.Module.load_state_dict`.
        Keys under ``final_layer.*`` (and ``channel_layer.*``) may be absent or
        extra, because the pretrained checkpoints carry no head. Any other
        missing or unexpected key raises an error.

        Raises
        ------
        RuntimeError
            If a backbone key is missing or unexpected, even with
            ``strict=False`` (:meth:`from_pretrained` loads with
            ``strict=False``).
        """
        result = super().load_state_dict(state_dict, *args, **kwargs)

        def backbone(keys):
            return [
                k for k in keys if not k.startswith(("final_layer.", "channel_layer."))
            ]

        missing, unexpected = (
            backbone(result.missing_keys),
            backbone(result.unexpected_keys),
        )
        if missing or unexpected:
            # from_pretrained loads with strict=False: never keep random backbone
            # weights, nor silently drop pretrained ones (e.g. depth=2 on a
            # 4-layer checkpoint).
            raise RuntimeError(
                "Guetschel2026: the state dict does not match the backbone (missing: "
                f"{missing}, unexpected: {unexpected}). Only the head (final_layer.*) "
                "may differ; check that the checkpoint is one of the eeg-fm-masking "
                "repositories and that the architecture arguments are the defaults."
            )
        return result

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        # Same operations, same order as ``scale_signal`` of the reference code.
        dtype = x.dtype
        x = x.to(torch.promote_types(dtype, torch.float32)) * self.input_scale
        if self.normalization == "median_std_clip":
            std = x.std(dim=-1, keepdim=True, correction=0)
            median = std.median(dim=-2, keepdim=True).values  # lower median
            x = x / (median + 1e-6)
            x = x.clamp(-self.clip_sigma, self.clip_sigma)
        return x.to(dtype)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``, in volts.
        return_features : bool, default=False
            If ``True``, return the features instead of the logits.

        Returns
        -------
        torch.Tensor or dict
            The logits, of shape ``(batch, n_outputs)``, or, with
            ``return_features=True``, ``{"features": z, "cls_token": None}``
            with ``z`` of shape ``(batch, n_chans, n_patches, embed_dim)``.
        """
        if x.shape[-1] < self.patch_size:
            raise ValueError(
                f"Input has {x.shape[-1]} samples, fewer than patch_size "
                f"({self.patch_size})."
            )
        if x.shape[-2] != self.n_chans:
            # one channel would otherwise broadcast over every configured electrode
            raise ValueError(
                f"The model was built for {self.n_chans} channels, but the input has "
                f"{x.shape[-2]}. Pass the matching chs_info at construction, or use a "
                "channel_strategy to map another montage."
            )
        n_patches = (x.shape[-1] - self.patch_size) // self.patch_step + 1
        if not return_features and n_patches != self.n_patches:
            raise ValueError(
                f"The head expects {self.n_patches} patches (n_times={self.n_times}), "
                f"but the input has {x.shape[-1]} samples ({n_patches} patches). "
                "Use return_features=True for other window lengths, or build the "
                f"model with n_times={x.shape[-1]}."
            )
        z = self.model(self.feature_encoder(self._scale(x)))  # (B, C, P, embed_dim)
        if return_features:
            return {"features": z, "cls_token": None}  # nosec B105
        return self.final_layer(z)
