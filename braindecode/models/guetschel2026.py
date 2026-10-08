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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.signal_jepa import _pos_encode_time
from braindecode.models.util import has_valid_locations

_N_STORED_TIME_PATCHES = 33  # (6000 - 200) // 180 + 1: the 30 s pre-training window
_PRETRAIN_SFREQ = 200.0
_MAX_SECONDS = 600.0
_SCALE_EPS = 1e-6
_NORMALIZATIONS = ("median_std_clip", "none")

_HUB_NAMESPACE = "PierreGtch"
_PRETEXTS = ("mae", "jepa")
_MASK_RADII = ("one", "6cm", "9cm", "12cm", "all")
_MASK_LENGTHS = (1, 2, 4, 8, 16, 33)
# Bounds (metres) of the distance of every electrode to the origin of the MNE head
# frame. The MNE built-in montages (34 in MNE 1.13) lie between 0.064 and 0.146 m,
# and the 32,713 REVE pre-training channels between 0.078 and 0.133 m.
_MIN_CH_DIST = 0.05
_MAX_CH_DIST = 0.20
_MAX_LISTED_CHANNELS = 5


def _pos_encode_xyz(ch_pos, x_min, x_max, n_dim):
    """Sinusoidal encoding of the coordinates, ``(..., 3) -> (..., 3, n_dim)``.

    The order of the operations is the one of the reference implementation
    (``pos_encode_continuous_batched``): a float64 intermediate or another
    order changes the float32 results by up to 5.7e-5.
    """
    out = ch_pos.new_empty(ch_pos.shape + (n_dim,))
    div_term = torch.exp(
        (1 - torch.arange(0, n_dim, 2, device=out.device) / n_dim) * 2 * math.pi
    )
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
            "encoding_time",
            # formula values until the weights are loaded: the checkpoints all store
            # the same table, which overwrites them. Computed on the CPU, then moved
            # to the default device.
            _pos_encode_time(_N_STORED_TIME_PATCHES, self.time_dim, max_n_times).to(
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
        spat = _pos_encode_xyz(ch_pos, -h, h, self.coord_dim).flatten(-2)  # (B, C, 3d)
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


def _check_channel_distances(chs_info, ch_pos) -> None:
    """Raise if an electrode is not within 5 to 20 cm of the head-frame origin.

    The positions must be in metres. Centimetres or millimetres are far above
    the range and a wrong unit or a placeholder at the origin (an all-zero
    ``loc``; MNE marks an unknown position with NaN or zeros, and NaN is
    rejected earlier) far below it.

    ``ch_pos`` is the ``(C, 3)`` array of the positions, in float64: the bounds
    are inclusive for the positions as given, not as rounded to float32.
    """
    dist = np.linalg.norm(np.asarray(ch_pos, dtype=np.float64), axis=1)
    bad = np.flatnonzero((dist < _MIN_CH_DIST) | (dist > _MAX_CH_DIST))
    if bad.size == 0:
        return
    listed = ", ".join(
        f"{chs_info[i].get('ch_name', f'#{i}')!r} ({dist[i]:.3g} m, "
        f"{'below' if dist[i] < _MIN_CH_DIST else 'above'} the range)"
        for i in bad[:_MAX_LISTED_CHANNELS]
    )
    if bad.size > _MAX_LISTED_CHANNELS:
        listed += f", ... ({bad.size - _MAX_LISTED_CHANNELS} more)"
    raise ValueError(
        "Guetschel2026 requires channel positions in METRES (MNE head frame, e.g. "
        "raw.info['chs'][i]['loc'][:3] after raw.set_montage(...)): the distance of "
        f"every channel to the origin must be between {_MIN_CH_DIST * 100:g} cm and "
        f"{_MAX_CH_DIST * 100:g} cm ({_MIN_CH_DIST:g} to {_MAX_CH_DIST:g} m), but "
        f"{bad.size} of {len(dist)} channel positions are outside it: {listed}. "
        "Values far above the range look like centimetres or millimetres; values "
        "far below it look like a wrong unit or a placeholder position at the "
        "origin (MNE marks an unknown position with an all-zero or NaN loc). "
        "Convert the positions to metres. A channel whose loc is all zeros (no "
        "position) can instead be left as it is: pass the same chs_info with "
        "channel_strategy='exact', which looks the missing positions up by "
        "channel name (standard 10-05 names, standard_1005) and "
        "keeps the positions that are given. Channels whose name is not a "
        "standard_1005 name (e.g. 'E1' of an EGI net) cannot be looked up and "
        "stay at the origin: they need positions in metres. Positions in a "
        "wrong unit are not replaced by 'exact': convert them, or pass chs_info "
        "with standard 10-05 channel names only (loc missing or all zeros) "
        "together with channel_strategy='exact' to use the standard_1005 "
        "positions."
    )


class Guetschel2026(EEGModuleMixin, nn.Module, license="mit"):
    r"""Encoder of the EEG masking-geometry study from Guetschel et al. (2026) [guetschel2026]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-dark-line:`Channel`

    .. versionadded:: 1.8.2

    .. figure:: ../_static/model/guetschel2026_arch.png
        :align: center
        :alt: Figure 1 of Guetschel et al. (2026): shared MAE/JEPA pre-training pipeline (A) and the block-masking geometries (B)
        :width: 1000px

        Figure 1 of [guetschel2026]_. **(A)** Framework axis: the MAE (green)
        and JEPA (orange) branches share one pipeline (linear tokeniser,
        masking, transformer encoder, transformer decoder); this class keeps
        only the tokeniser and the encoder. **(B)** Masking axis: block masks
        parameterised by a spatial radius ``r``, a temporal length ``L`` and a
        target ratio :math:`\rho`. They recover random patches (``L=1``,
        ``r="one"``), temporal blocks (``r="all"``), spatial blocks (``L=33``)
        and spatio-temporal blocks. The 29 ``(L, r)`` configurations at
        :math:`\rho = 0.55`, times the 2 pretexts, give the 58 checkpoints.

    The backbone is the REVE-Small architecture of REVE with minor
    simplifications (12,687,872 parameters). It was pre-trained 58 times, with
    a masked autoencoder (MAE) or a joint-embedding predictive architecture
    (JEPA) and with 29 different spatio-temporal mask geometries (a radius
    :math:`r` and a length :math:`L`, about 55 % of the tokens masked). This
    class is the backbone: the 58 released checkpoints all load into it, with
    the default arguments (see the box below). All 58 checkpoints were
    pre-trained on the same data: the open-licence subset of the REVE
    pre-training corpus (4.4 TB, about 34,000 h of EEG; the datasets are listed
    in the paper's appendix), for 10 epochs. The study finds that a block mask
    of radius 9 cm and length 2 patches is the best for both pretexts, that a
    poorly chosen mask costs JEPA 2.4 times more than MAE, and that the
    resulting frozen features reach the level of REVE-Base with 12.7 M
    parameters [guetschel2026]_.

    The signal is cut into overlapping 1 s patches (``patch_size=200`` samples
    at 200 Hz, a step of 180) that a single linear layer embeds. A fixed,
    additive sinusoidal encoding of the electrode position
    :math:`(x, y, z)` and of the patch index gives the transformer its
    spatio-temporal context, so that any montage with channel locations can be
    used. Four pre-norm transformer layers (RMSNorm, 8 heads, GEGLU
    feed-forward of width 1365, no bias, no final norm) then attend over all
    ``n_chans * n_patches`` tokens. :meth:`forward` returns the logits of a head
    that flattens all the tokens, or, with ``return_features=True``, the
    ``(batch, n_chans, n_patches, embed_dim)`` features themselves.

    .. rubric:: Macro Components

    - ``Guetschel2026.feature_encoder`` **Patch embedding**

      The input is scaled (see the warning below), unfolded into overlapping
      patches of ``patch_size`` samples and projected by one
      :class:`~torch.nn.Linear` to ``embed_dim``.

    - ``Guetschel2026.model.pos_encoder`` **Positional encoding**

      Sinusoidal encoding of :math:`x`, :math:`y` and :math:`z` (each of width
      ``embed_dim // 4``, computed from the channel locations) and of the patch
      index (width ``embed_dim // 4``). It is added to the patch embeddings.

    - ``Guetschel2026.model.transformer`` **Transformer encoder**

      ``depth`` pre-norm layers with multi-head self-attention
      (:func:`torch.nn.functional.scaled_dot_product_attention`) and a GEGLU
      feed-forward block, over the channel-major sequence of tokens.

    - ``Guetschel2026.final_layer`` **Classification head**

      :class:`~torch.nn.Flatten`, then a fixed Gaussian random projection to
      ``random_projection`` features (see below), then :class:`~torch.nn.Linear`
      to ``n_outputs``. With ``random_projection=None``, only Flatten and
      Linear. The released checkpoints contain no head.

    .. important::
       **Pre-trained weights: 58 checkpoints, one backbone**

       The checkpoints are hosted on the Hugging Face Hub in the repositories
       ``PierreGtch/eeg-fm-masking_{pretext}_r{radius}_L{length}`` (see the
       `collection <https://huggingface.co/collections/PierreGtch/eeg-fm-masking-6ab912b6a03bba1348fc7366>`_
       and the `project page <https://pierregtch.github.io/eeg-fm-masking/>`_).
       They share this architecture and the default arguments of this class and
       differ only in the pretext (``mae`` or ``jepa``) and in the mask
       geometry used for the pre-training: the radius ``r`` of the blocks of
       masked channels (``one`` channel, ``6cm``, ``9cm``, ``12cm`` or
       ``all`` channels) and their length ``L`` in patches. Each cell of the
       table gives the ``r{radius}_L{length}`` suffix; both ``mae_`` and
       ``jepa_`` exist for every cell but the last, which gives
       2 x 29 = 58 checkpoints.

       .. list-table::
          :header-rows: 1
          :stub-columns: 1

          * - radius / length
            - L=1
            - L=2
            - L=4
            - L=8
            - L=16
            - L=33
          * - one channel
            - ``rone_L1``
            - ``rone_L2``
            - ``rone_L4``
            - ``rone_L8``
            - ``rone_L16``
            - ``rone_L33``
          * - 6 cm
            - ``r6cm_L1``
            - ``r6cm_L2``
            - ``r6cm_L4``
            - ``r6cm_L8``
            - ``r6cm_L16``
            - ``r6cm_L33``
          * - 9 cm
            - ``r9cm_L1``
            - ``r9cm_L2`` (recommended)
            - ``r9cm_L4``
            - ``r9cm_L8``
            - ``r9cm_L16``
            - ``r9cm_L33``
          * - 12 cm
            - ``r12cm_L1``
            - ``r12cm_L2``
            - ``r12cm_L4``
            - ``r12cm_L8``
            - ``r12cm_L16``
            - ``r12cm_L33``
          * - all channels
            - ``rall_L1``
            - ``rall_L2``
            - ``rall_L4``
            - ``rall_L8``
            - ``rall_L16``
            - not trained (would mask the whole window)

       **Recommended:** ``PierreGtch/eeg-fm-masking_mae_r9cm_L2`` or
       ``PierreGtch/eeg-fm-masking_jepa_r9cm_L2``. :meth:`hub_repo_id` builds
       and validates the repository name from the three parameters.

       The paper advises against the extremes of the grid. With ``rone_*`` the
       pretext is too easy. With ``rall_*``, MAE stays below the REVE baseline,
       and JEPA suffers *bias-inflation collapse*: its encoder drifts towards a
       lookup table of the channel positions, the part of the features that
       depends on the input shrinks, and the downstream score falls from about
       epoch 3 onwards. For ``jepa_rall_*`` the score peaks around epochs 3-4,
       so ``model.safetensors`` (epoch 10) sits well below the best
       intermediate checkpoints of the run. These checkpoints are released
       to reproduce that finding [guetschel2026]_.

       ``model.safetensors`` is the end of epoch 10, the checkpoint behind the
       paper's main results. The folders ``epoch_01/`` to ``epoch_09/`` hold
       intermediate checkpoints of the same run (used for the paper's training
       trajectories), loaded with
       ``filename="epoch_05/model.safetensors"``. The JEPA files hold the
       student encoder. The weights are released under CC-BY-4.0, the code
       under the MIT license.

       .. code-block:: python

           import mne
           import torch
           from braindecode.models import Guetschel2026

           raw = mne.io.read_raw_edf("recording.edf", preload=True)
           raw.resample(200.0)
           # the model needs channel locations ("colin27_1020" on MNE >= 1.13)
           raw.set_montage("standard_1020")

           repo_id = Guetschel2026.hub_repo_id("mae", "9cm", 2)
           # -> "PierreGtch/eeg-fm-masking_mae_r9cm_L2"
           model = Guetschel2026.from_pretrained(
               repo_id,
               chs_info=raw.info["chs"],
               n_times=1000,  # 5 s at 200 Hz
               n_outputs=4,
               sfreq=200.0,
               # filename="epoch_05/model.safetensors",  # an intermediate epoch
               # revision="<commit sha>",  # pin the weights for reproducibility
           )

           # The features: (batch, n_chans, n_patches, 512), n_patches = 5 here.
           x = torch.randn(8, len(raw.ch_names), 1000) * 1e-5  # volts
           features = model(x, return_features=True)["features"]

       The head is randomly initialized (the checkpoints have none), so train
       it before using the logits. For a linear probe, freeze the encoder and
       train the head only:

       .. code-block:: python

           for name, p in model.named_parameters():
               p.requires_grad = name.startswith("final_layer.")

       and for fine-tuning, train everything, for instance with
       :class:`~braindecode.classifier.EEGClassifier`:

       .. code-block:: python

           from braindecode import EEGClassifier

           clf = EEGClassifier(model, lr=1e-4, max_epochs=20, batch_size=64)
           clf.fit(train_set, y=None)

       The names built by :meth:`hub_repo_id` cover the whole grid:

       .. code-block:: python

           for pretext in ("mae", "jepa"):
               for radius in ("one", "6cm", "9cm", "12cm", "all"):
                   for length in (1, 2, 4, 8, 16, 33):
                       if (radius, length) == ("all", 33):
                           continue  # never trained
                       # includes the rone_* / rall_* checkpoints the paper advises against
                       repo_id = Guetschel2026.hub_repo_id(pretext, radius, length)
                       model = Guetschel2026.from_pretrained(
                           repo_id, chs_info=chs_info, n_times=1000, n_outputs=2
                       )

       If your channels have names but no coordinates, look them up by name
       with ``channel_strategy="exact"`` (see :doc:`/user_guide/channel_strategies`):

       .. code-block:: python

           model = Guetschel2026.from_pretrained(
               repo_id,
               chs_info=[{"ch_name": "C3"}, {"ch_name": "Cz"}, {"ch_name": "C4"}],
               n_times=1000,
               n_outputs=2,
               channel_strategy="exact",
           )

       Loading is not strict by default: the missing head is expected, but a
       checkpoint that lacks any backbone weight raises a ``RuntimeError``
       instead of silently keeping random weights.

    .. rubric:: Random projection head

    The head reuses the random-projection step of the OpenEEGBench ridge probe:
    the flattened features (``n_chans * n_patches * embed_dim``) are projected
    to ``random_projection`` features (5000 by default) by a fixed Gaussian
    random projection, which is never trained, then fed to a
    :class:`~torch.nn.Linear` layer. This reduces the dimension only when
    ``n_chans * n_patches * embed_dim`` exceeds ``random_projection``: for few
    channels or short windows (for instance 3 channels x 1 s = 1,536 features)
    it expands the features instead. OpenEEGBench applies the projection only
    when the features outnumber its ``max_features`` and uses the raw features
    otherwise. This head always applies it, so pass ``random_projection=None``
    to match OpenEEGBench on such small inputs. OpenEEGBench fits a closed-form
    ridge regression on the projected frozen features instead of training a
    linear layer, and the paper reports means over 5 projection seeds, so
    training this head by gradient descent does not reproduce the paper's
    numbers exactly.

    The projection matrix has shape
    ``(random_projection, n_chans * n_patches * embed_dim)`` with entries drawn
    from :math:`\mathcal{N}(0, 1/\mathrm{random\_projection})`, the
    distribution of :class:`sklearn.random_projection.GaussianRandomProjection`
    and of ``_make_projection_matrix`` in `OpenEEGBench
    <https://github.com/braindecode/OpenEEGBench/blob/3e4d034ae3e009deeda9c987d05a37e18fd87e15/open_eeg_bench/ridge_probe.py#L62>`_
    (the values differ for a given seed). It is generated in float32 on the CPU
    from ``random_projection_seed``, without touching the global random state, and
    stored as a buffer (in the default dtype): it is saved in the
    ``state_dict`` and by ``save_pretrained``, and never trained. Loading a
    checkpoint without head therefore gives the same projection for the same
    seed, with the same PyTorch version and CPU type; save the model
    (``save_pretrained``) to keep the projection bit-exact across machines.

    **Memory cost.** The matrix has ``random_projection * n_chans * n_patches
    * embed_dim`` entries in the default dtype: in float32, about 0.9 GB for
    22 channels and 4 s (45,056 features) and 2.1 GB for 19 channels and 10 s
    with the default 5000 components, and twice that in float64. Lower
    ``random_projection`` for long windows or many channels, or pass
    ``random_projection=None`` for the plain ``Flatten`` then ``Linear`` head
    of the original wrapper, which is the head to use to reproduce its logits.
    :meth:`reset_head` replaces only the last linear layer and keeps the
    projection.

    .. warning::
       **Input.** The model was pre-trained at 200 Hz (a warning is raised if
       ``sfreq`` differs) on 30 s windows (33 patches) of 32 channels, randomly
       sampled from each recording and zero-padded when fewer were available
       (paper, Sec. 3); any number of channels can be used at inference.

       - The signal is expected in volts, not standardized: the model
         multiplies it by ``input_scale`` (:math:`10^6`, so microvolts), divides
         it by the median over channels of the standard deviations of each
         channel and clips it at ``clip_sigma``, as the original wrapper does.
         For data that you already scaled, use ``normalization="none"`` and
         ``input_scale=1.0``.
       - ``chs_info`` must hold the electrode positions in ``loc[:3]`` (metres,
         MNE head frame), for any montage; they are read at construction. To
         use another montage for each call, use a ``channel_strategy``.
         The distance of every channel to the origin must be between 5 cm and
         20 cm, otherwise a ``ValueError`` is raised: this catches positions in
         centimetres or millimetres (far above), and in a wrong unit or at the
         origin (far below). All the built-in MNE montages (34 in MNE 1.13)
         and the REVE pre-training positions are inside this range. A channel
         without a position (a NaN or all-zero ``loc`` in MNE) is rejected
         too: pass the same ``chs_info`` with ``channel_strategy="exact"`` to
         look the missing positions up by name (standard 10-05 names,
         ``standard_1005``, e.g. ``"Fz"``; the positions that are given are
         kept). Channels whose name is not a standard_1005 name (e.g. ``"E1"``
         of an EGI net) cannot be looked up and stay at the origin: they need
         positions in metres. ``channel_strategy`` does not replace non-zero
         positions: positions in a wrong unit must be converted to metres, or
         replaced by the standard ones by passing ``chs_info`` with standard
         10-05 channel names only.
       - The pre-training corpus includes PhysioNet-MI (EEGMMIDB, 48.5 h):
         results on that dataset are not an evaluation on unseen data (paper,
         Limitations). The other eleven OpenEEGBench datasets were not seen
         during pre-training.
       - The window needs at least ``patch_size`` (200) samples. Trailing
         samples that do not fill a patch are dropped.

    .. note::
        Differences from the reference implementation (the backbone features
        are otherwise identical, bit for bit, in float32 and float64):

        - The head uses the true number of overlapping patches,
          ``(n_times - patch_size) // (patch_size - patch_overlap) + 1``. The
          original wrapper sizes its head for ``n_times // patch_size``
          patches. That count is wrong for every window of 1820 samples or more
          (for instance 2000 or 6000 samples) and for some shorter ones (for
          instance 380 to 399), and the original head fails on all of them.
        - Attention dropout is off in eval mode.
        - The scaling runs in at least float32, so that float16 inputs do not
          overflow.
        - The random projection head (``random_projection=None`` gives the
          original head).
        - Every channel position must be finite, not all zero, and between
          5 cm and 20 cm from the origin (metres, MNE head frame). The
          original checks only for NaN, so it accepts positions in the wrong
          unit, infinite values and all-zero rows, up to a montage with no
          positions at all, which it encodes as the origin; this class raises
          a ``ValueError`` for all of them (with ``channel_strategy="exact"``,
          NaN and all-zero rows are looked up by name instead, see above).

    Parameters
    ----------
    embed_dim : int, default=512
        Width of the tokens. Must be divisible by 8 and by ``num_heads``; the
        three coordinates and the patch index each get ``embed_dim // 4``
        positional features.
    depth : int, default=4
        Number of transformer layers.
    num_heads : int, default=8
        Number of attention heads.
    dim_feedforward : int, default=1365
        Width of each half of the GEGLU feed-forward block.
    patch_size : int, default=200
        Number of samples of a patch (1 s at 200 Hz).
    patch_overlap : int, default=20
        Number of samples shared by two consecutive patches; the step is
        ``patch_size - patch_overlap``.
    pos_half_range : float, default=0.15
        Half range in metres of the electrode coordinates, which are mapped
        from ``[-pos_half_range, pos_half_range]`` to ``[0, 1]`` before the
        sinusoidal encoding.
    activation : type[nn.Module], default=nn.GELU
        Activation class applied to the gate of the feed-forward block.
    drop_prob : float, default=0.0
        Dropout probability (attention and feed-forward residual branches).
    normalization : {"median_std_clip", "none"}, default="median_std_clip"
        Input scaling. ``"median_std_clip"`` divides each window by the lower
        median over channels of the per-channel standard deviations and clips
        it at ``clip_sigma``; ``"none"`` only multiplies by ``input_scale``.
    input_scale : float, default=1e6
        Factor applied to the input before the normalization (volts to
        microvolts).
    clip_sigma : float, default=15.0
        Clipping bound of ``"median_std_clip"``.
    random_projection : int or None, default=5000
        Number of features of the fixed Gaussian random projection between the
        flatten and the linear layer of the head (see the section above), or
        ``None`` for no projection.
    random_projection_seed : int, default=0
        Seed of the random projection matrix.

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
        # head
        random_projection: int | None = 5000,
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
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
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
        try:
            sfreq_known = self.sfreq
        except ValueError:
            sfreq_known = None
        if sfreq_known is not None and not math.isclose(sfreq_known, _PRETRAIN_SFREQ):
            warnings.warn(
                f"Guetschel2026 was pre-trained at {_PRETRAIN_SFREQ:g} Hz but sfreq is "
                f"{sfreq_known:g} Hz; resample the data to {_PRETRAIN_SFREQ:g} Hz.",
                UserWarning,
                stacklevel=3,  # __init__, the track_model_init_kwargs wrapper, caller
            )
        if not has_valid_locations(self.chs_info):
            raise ValueError(
                "Guetschel2026 requires channel locations: every chs_info entry needs "
                "a finite loc[:3] (metres, MNE head frame) and not all of them zero. "
                "Call raw.set_montage(...) before taking raw.info['chs'], or pass "
                "channel_strategy='exact' to look the missing positions up by "
                "channel name (standard 10-05 names, standard_1005). Channels "
                "whose name is not a standard_1005 name (e.g. 'E1' of an EGI net) "
                "cannot be looked up: they need positions in metres."
            )

        ch_pos_f64 = np.array([ch["loc"][:3] for ch in self.chs_info], dtype=np.float64)
        _check_channel_distances(self.chs_info, ch_pos_f64)
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
        max_n_times = int(_MAX_SECONDS * (_PRETRAIN_SFREQ / self.patch_step))  # 666
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
        if isinstance(mask_length, bool) or mask_length not in _MASK_LENGTHS:
            raise ValueError(
                f"mask_length must be one of {_MASK_LENGTHS}, got {mask_length!r}."
            )
        if (mask_radius, mask_length) == ("all", 33):
            raise ValueError(
                "No checkpoint was trained with mask_radius='all' and mask_length=33: "
                "it would mask the whole window. Use another radius or length."
            )
        return f"{_HUB_NAMESPACE}/eeg-fm-masking_{pretext}_r{mask_radius}_L{int(mask_length)}"

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
        """Load a state dict, allowing only the head to be missing.

        Arguments are passed on to :meth:`torch.nn.Module.load_state_dict`.
        Keys under ``final_layer.*`` may be absent, because the pretrained
        checkpoints carry no head. Any other missing key raises an error.

        Raises
        ------
        RuntimeError
            If a backbone key is missing, even with ``strict=False``
            (:meth:`from_pretrained` loads with ``strict=False``).
        """
        result = super().load_state_dict(state_dict, *args, **kwargs)
        missing = [k for k in result.missing_keys if not k.startswith("final_layer.")]
        if missing:
            # from_pretrained loads with strict=False: never keep random backbone weights.
            raise RuntimeError(
                f"Guetschel2026: the state dict misses backbone weights {missing}. "
                "Only the head (final_layer.*) may be absent; check that the checkpoint "
                "is one of the eeg-fm-masking repositories and that the architecture "
                "arguments are the defaults."
            )
        return result

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        # Same operations, same order as ``scale_signal`` of the reference code.
        dtype = x.dtype
        x = x.to(torch.promote_types(dtype, torch.float32)) * self.input_scale
        if self.normalization == "median_std_clip":
            std = x.std(dim=-1, keepdim=True, correction=0)
            median = std.median(dim=-2, keepdim=True).values  # lower median
            x = x / (median + _SCALE_EPS)
            x = x.clamp(-self.clip_sigma, self.clip_sigma)
        return x.to(dtype)

    def forward(self, x: torch.Tensor, return_features: bool = False):
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
        z = self.model(self.feature_encoder(self._scale(x)))  # (B, C, P, embed_dim)
        if return_features:
            return {"features": z, "cls_token": None}
        return self.final_layer(z)
