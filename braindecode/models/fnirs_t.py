# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
# Adapted from https://github.com/wzhlearning/fNIRS-Transformer (MIT).
"""fNIRS-T and fNIRS-PreT: Transformer Models for fNIRS Classification."""

from __future__ import annotations

from typing import Literal

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order
from braindecode.modules import (
    FeedForwardBlock,
    MultiHeadAttention,
    PreNormTransformerBlock,
)


class FNIRST(EEGModuleMixin, nn.Module, license="mit"):
    r"""fNIRS-T from Wang et al. (2022).

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. figure:: https://braindecode.org/dev/_static/model/fnirst.png
       :align: center
       :alt: fNIRS-T and fNIRS-PreT architecture

       Figure of the authors' repository (``fig/model.png``,
       wzhlearning/fNIRS-Transformer at ``6ebe987f``, MIT): the
       :math:`H \times W \times C` input (channels, samples, HbO/HbR) feeds a
       "Spatial-level" and a "Channel-level" branch, each a "Transformer
       Encoder" with a CLS token. Bottom: fNIRS-T takes filtered,
       baseline-corrected segments; fNIRS-PreT replaces those steps by a
       "Preprocessing Module" (:class:`FNIRSPreT`). The paper (IEEE JBHI
       2022) is not open access; its caption is not reproduced.

    From the abstract :footcite:`wang2022fnirst`: "Inspired by the success of
    Transformer based on self-attention mechanism in the fields of natural
    language processing and computer vision, we propose an fNIRS
    classification network based on Transformer, named fNIRS-T. We explore
    the spatial-level and channel-level representation of fNIRS signals to
    improve data utilization and network representation capacity." On the
    three-class finger- and foot-tapping task (30 subjects), "fNIRS-T, CNN,
    and LSTM obtain 75.49%, 72.89%, and 61.94% on test sets".

    In the authors' code the two levels read the same signals at two
    spatial scales: a convolution turns the time course of each fNIRS
    channel (channel level) or of a group of consecutive channels (spatial
    level) into one token, and a transformer lets these tokens attend to
    each other; time never becomes a token axis. Only the abstract, the
    README, the figure and the code (``model.py``, ``KFold_Train.py``,
    ``LOSO_Train.py``) are public, and the description below comes from
    them.

    .. rubric:: Architecture Overview

    In the figure's symbols, :math:`H` = ``n_chans // 2`` fNIRS channels,
    :math:`W` = ``n_times``, :math:`C` = 2 chromophores, :math:`H_S = H -
    k_s + 1` spatial tokens, :math:`H_C = H` channel tokens and :math:`D` =
    ``embed_dim``::

        input: H HbO rows and H HbR rows             (batch, 2 H, W)
        to_chromophore_planes                        (batch, 2, H, W)
        Spatial level (to_patch_embedding, transformer_patch):
          Conv_S, Rearrange, Linear, LayerNorm       (batch, H_S, D)
          prepend CLS, add positions, encoder        (batch, H_S + 1, D)
        Channel level (to_channel_embedding, transformer_channel):
          Conv_C, Rearrange, Linear, LayerNorm       (batch, H_C, D)
          prepend CLS, add positions, encoder        (batch, H_C + 1, D)
        CLS token of each level, Concat              (batch, 2 D)
        final_norm, final_layer                      (batch, n_outputs)

    .. rubric:: Macro Components

    - ``to_chromophore_planes`` **Input.** The authors' input is "[B, 2,
      fNIRS channels, sampling points]", where '"2" means two chromophores:
      HbO and HbR' (``model.py``, README). The rows of ``x`` are split into
      these two planes, from the stacked or MNE's interleaved order.
    - ``to_patch_embedding`` **Spatial level ("fNIRS Spatial
      Embedding").** :math:`\mathrm{Conv}_S` maps the two planes to
      :math:`C_S` = ``n_conv_filters`` maps with a :math:`k_s \times k_t`
      kernel and stride :math:`1 \times s_t` (the authors: 8 maps,
      :math:`5 \times 30`, stride :math:`1 \times 4`), so each token covers
      :math:`k_s` consecutive channels in the dataset's order. The
      :math:`C_S \times W_S` responses of a token are flattened, projected
      to :math:`D` and layer-normalized.
    - ``to_channel_embedding`` **Channel level ("fNIRS Channel
      Embedding").** The same with a :math:`1 \times k_t` kernel
      (:math:`\mathrm{Conv}_C`): one token per fNIRS channel.
    - ``cls_token_*``, ``pos_embedding_*``, ``transformer_*`` **Transformer
      Encoder.** Each level prepends a learned class token, adds a learned
      positional embedding (one row per token, the token's only identity),
      applies dropout and runs ``depth`` pre-norm blocks of multi-head
      self-attention and a GELU feed-forward layer of width ``mlp_dim``. The
      queries, keys and values come from bias-free linear maps, and each of
      the ``num_heads`` heads of width :math:`d_h` = ``head_dim`` computes,
      as in the authors' ``Attention``,

      .. math::

         \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(
         \frac{Q K^{\top}}{\sqrt{d_h}} \right) V .

    - ``final_norm``, ``final_layer`` **Head.** The class token of each
      level (or, with ``pool="mean"``, the mean of its tokens) is taken, the
      two vectors are concatenated ("Concat"), layer-normalized and
      classified by a linear layer (the authors' ``mlp_head``).

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* the convolutions slide a :math:`k_t`-sample kernel with
      stride :math:`s_t` along time, and the resulting values become
      features of a token. The README's segments are 140 samples (dataset
      A), 200 (B) and 256 (C). Kernels count samples, so ``sfreq`` is
      unused.
    - *Spatial:* channels or groups of :math:`k_s` consecutive channels,
      identified by their position in the token sequence; no montage or
      optode geometry. The channel order must be the one used in training.
    - *Chromophores:* HbO and HbR are the two input planes, mixed by both
      convolutions from the first layer.
    - *Spectral:* none in the model; fNIRS-T expects band-pass filtered,
      baseline-corrected segments, and :class:`FNIRSPreT` learns that step.

    .. rubric:: Pretrained Weights

    None: the authors release code, not weights.

    .. rubric:: Differences from the reference implementation

    ``x`` is ``(batch, n_chans, n_times)``, ``n_chans = 2 H``, HbO rows then
    HbR rows, or MNE's interleaved order read from ``chs_info``. The figure
    draws an "MLP Head" per level before "Concat"; the authors' code, which
    this module follows, concatenates first and applies one head, and the
    output is logits. The positional tables have one row per token (the
    authors allocate 101 rows and slice), so an authors' ``state_dict``
    loads after a conversion rather than a ``mapping`` (``to_qkv`` split
    into ``queries``/``keys``/``values``, tables cut, ``mlp_head.{0,1}`` to
    ``final_norm``/``final_layer``); the port then matches the authors'
    model to float32 rounding (max-abs below 1e-6 on their dataset A, B and
    C configurations). The attention always projects its output (the
    authors skip it when ``num_heads=1`` and ``head_dim=embed_dim``).
    ``mask`` is an addition of the port; ``ch_mask`` is refused, as the
    spatial kernel and the positional tables span the whole montage.

    Parameters
    ----------
    embed_dim : int, default=64
        :math:`D`, the authors' ``dim``, "last dimension of output tensor
        after linear transformation"; 64 for their datasets A and B, 128 for
        C.
    depth : int, default=6
        Authors' ``depth``, "number of Transformer blocks" per level.
    num_heads : int, default=8
        Authors' ``heads``, "number of the multi-head self-attention".
    mlp_dim : int, default=64
        Authors' ``mlp_dim``, "dimension of the MLP layer".
    pool : {"cls", "mean"}, default="cls"
        Read-out of each level (authors' ``pool``): ``"cls"``, its class
        token; ``"mean"``, the mean over its tokens (class token included).
    head_dim : int, default=64
        :math:`d_h`, the authors' ``dim_head``: width of each attention head.
    n_conv_filters : int, default=8
        :math:`C_S = C_C`, feature maps of both convolutions (authors: 8).
    spatial_kernel_size : int, default=5
        :math:`k_s`, height of :math:`\mathrm{Conv}_S` in HbO/HbR pairs
        (authors: 5); at most the number of pairs, ``n_chans // 2``.
    temporal_kernel_size : int, default=30
        :math:`k_t`, temporal kernel of both convolutions, in samples
        (authors: 30); ``n_times`` must be at least this.
    temporal_stride : int, default=4
        :math:`s_t`, temporal stride of both convolutions, in samples
        (authors: 4).
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_H, HbR_1..HbR_H]`` or MNE's ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info``.
    drop_prob : float, default=0.0
        Authors' ``dropout`` in the attention and feed-forward sub-layers.
    embed_drop_prob : float, default=0.0
        Authors' ``emb_dropout``, "dropout for patch embeddings".
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward sub-layers (authors: GELU).

    Notes
    -----
    The authors' code uses three datasets: A (mental arithmetic vs rest, 8
    subjects, 52 channels, 140 samples), B (Shin et al. 2017, same task, 29
    subjects, 36 channels, 200 samples) and C (Bak et al. 2019, finger and
    foot tapping, 3 classes, 30 subjects, 20 channels, 256 samples). The
    fNIRS-T inputs are band-passed (0.01-0.1 Hz on B and C),
    baseline-corrected and z-scored per window. Training is 5 x 5-fold
    cross-validation over pooled trials or leave-one-subject-out, 120
    epochs, batch 128, AdamW with cosine annealing (``T_max=30``) and label
    smoothing 0.1 in both loops, and the best test-fold accuracy is kept.
    On B and C the loss is "flooded" (``loss = (loss - b).abs() + b``) with
    a level :math:`b` the README decays "at the specified epoch to obtain
    continuous regularization": 0.45, 0.40 and 0.35 from the zero-based
    epochs 0, 30 and 50.

    Replication on dataset C with the authors' ``KFold_Train.py`` (mean of
    the 25 runs of each seed, averaged over seeds 0-2; macro-F1 at the kept
    epoch):

    ==========  =================  ================  ================
    Model       Paper (abstract)   Port (acc / F1)   Authors' code
    ==========  =================  ================  ================
    fNIRS-T     75.49 %            75.03 % / 0.7498  74.73 % / 0.7469
    fNIRS-PreT  "competitive"      72.50 % / 0.7244  72.50 % / 0.7245
    ==========  =================  ================  ================

    .. versionadded:: 2.1

    `License <https://github.com/wzhlearning/fNIRS-Transformer/blob/6ebe987faacc5587e11e8bd0d1177b40ec3ccd8d/LICENSE>`_

    References
    ----------
    .. footbibliography::
    """

    _with_preprocessing = False  # FNIRSPreT prepends the preprocessing module

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        embed_dim: int = 64,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 64,
        pool: Literal["cls", "mean"] = "cls",
        head_dim: int = 64,
        n_conv_filters: int = 8,
        spatial_kernel_size: int = 5,
        temporal_kernel_size: int = 30,
        temporal_stride: int = 4,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.0,
        embed_drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if self.n_times < temporal_kernel_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be >= temporal_kernel_size."
            )
        if self.n_chans % 2:  # the authors' input is [B, 2, pairs, samples]
            raise ValueError(
                f"{type(self).__name__} needs HbO/HbR pairs: n_chans must be even, "
                f"got {self.n_chans}."
            )
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}.")
        order = fnirs_channel_order(self._chs_info, channel_order)
        channel_axes = "(chans chrom)" if order == "interleaved" else "(chrom chans)"
        self.to_chromophore_planes = Rearrange(
            f"batch {channel_axes} time -> batch chrom chans time", chrom=2
        )
        n_pairs = self.n_chans // 2
        if spatial_kernel_size > n_pairs:  # the authors' Conv_S fails too
            raise ValueError(
                f"spatial_kernel_size ({spatial_kernel_size}) must be at most the "
                f"number of HbO/HbR pairs ({n_pairs})."
            )
        self.pool = pool
        self._class_name = type(self).__name__  # named errors in a scripted forward
        n_bins = (self.n_times - temporal_kernel_size) // temporal_stride + 1
        self.pre = (
            _PreprocessingModule(self.n_times) if self._with_preprocessing else None
        )

        def branch() -> nn.Sequential:
            return nn.Sequential(
                *[
                    PreNormTransformerBlock(
                        embed_dim,
                        MultiHeadAttention(
                            embed_dim, num_heads, head_dim=head_dim, qkv_bias=False
                        ),
                        FeedForwardBlock(
                            embed_dim, 1, drop_prob, activation, hidden_features=mlp_dim
                        ),
                        drop_prob,
                    )
                    for _ in range(depth)
                ]
            )

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(
                2,
                n_conv_filters,
                (spatial_kernel_size, temporal_kernel_size),
                (1, temporal_stride),
            ),
            Rearrange("batch maps tokens bins -> batch tokens (maps bins)"),
            nn.Linear(n_bins * n_conv_filters, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.cls_token_patch = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embedding_patch = nn.Parameter(
            torch.empty(1, n_pairs - spatial_kernel_size + 2, embed_dim)
        )
        self.dropout_patch = nn.Dropout(embed_drop_prob)
        self.transformer_patch = branch()

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(
                2,
                n_conv_filters,
                (1, temporal_kernel_size),
                (1, temporal_stride),
            ),
            Rearrange("batch maps tokens bins -> batch tokens (maps bins)"),
            nn.Linear(n_bins * n_conv_filters, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.cls_token_channel = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embedding_channel = nn.Parameter(
            torch.empty(1, n_pairs + 1, embed_dim)
        )
        self.dropout_channel = nn.Dropout(embed_drop_prob)
        self.transformer_channel = branch()

        self.token_mean = Reduce("batch tokens embed_dim -> batch embed_dim", "mean")
        self.final_norm = nn.LayerNorm(embed_dim * 2)
        self.final_layer = nn.Linear(embed_dim * 2, self.n_outputs)
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Drawn after the layers, so reset_parameters() repeats the constructor.
        with torch.no_grad():
            self.cls_token_patch.normal_()
            self.pos_embedding_patch.normal_()
            self.cls_token_channel.normal_()
            self.pos_embedding_channel.normal_()

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``x``: ``(batch, n_chans, n_times)``. ``mask`` (bool, same shape):
        True samples are zeroed first. ``ch_mask`` is refused: the spatial
        kernel and positional tables span every channel. ``return_features``
        gives the normalized pooled ``"features"`` and, as ``"cls_token"``, the
        two branches' class tokens ``(batch, 2 * embed_dim)``."""
        if ch_mask is not None:
            raise ValueError(
                self._class_name + " does not accept ch_mask (fixed channel layout)."
            )
        if mask is not None:
            x = x.masked_fill(mask, 0.0)  # (batch, n_chans, n_times)
        # (batch, n_chans, n_times) -> (batch, 2, n_pairs, n_times)
        x = self.to_chromophore_planes(x)
        if self.pre is not None:
            x = self.pre(x)  # (batch, 2, n_pairs, n_times)
        batch_size = x.shape[0]
        # spatial level, n_groups = n_pairs - spatial_kernel_size + 1 tokens
        patch = self.to_patch_embedding(x)  # (batch, n_groups, embed_dim)
        # (1, 1, embed_dim) -> (batch, 1, embed_dim)
        cls_patch = self.cls_token_patch.expand(batch_size, -1, -1)
        # -> (batch, n_groups + 1, embed_dim)
        patch = torch.cat((cls_patch, patch), dim=1)
        patch = patch + self.pos_embedding_patch  # (batch, n_groups + 1, embed_dim)
        patch = self.dropout_patch(patch)  # (batch, n_groups + 1, embed_dim)
        patch = self.transformer_patch(patch)  # (batch, n_groups + 1, embed_dim)
        # channel level, n_tokens = n_pairs + 1
        channel = self.to_channel_embedding(x)  # (batch, n_pairs, embed_dim)
        cls_channel = self.cls_token_channel.expand(batch_size, -1, -1)
        # -> (batch, n_tokens, embed_dim)
        channel = torch.cat((cls_channel, channel), dim=1)
        channel = channel + self.pos_embedding_channel  # (batch, n_tokens, embed_dim)
        channel = self.dropout_channel(channel)  # (batch, n_tokens, embed_dim)
        channel = self.transformer_channel(channel)  # (batch, n_tokens, embed_dim)
        patch_token = patch[:, 0]  # (batch, embed_dim)
        channel_token = channel[:, 0]  # (batch, embed_dim)
        # -> (batch, 2 * embed_dim)
        cls_tokens = torch.cat((patch_token, channel_token), dim=1)
        if self.pool == "mean":
            patch = self.token_mean(patch)  # (batch, embed_dim)
            channel = self.token_mean(channel)  # (batch, embed_dim)
            features = torch.cat((patch, channel), dim=1)  # (batch, 2 * embed_dim)
        else:
            features = cls_tokens
        features = self.final_norm(features)  # (batch, 2 * embed_dim)
        if return_features:
            return {"features": features, "cls_token": cls_tokens}
        return self.final_layer(features)  # (batch, n_outputs)


class FNIRSPreT(FNIRST, license="mit"):  # the kwarg is not inherited
    r"""fNIRS-PreT from Wang et al. (2022).

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. figure:: https://braindecode.org/dev/_static/model/fnirst.png
       :align: center
       :alt: fNIRS-T and fNIRS-PreT architecture

       The figure of the authors' repository (``fig/model.png`` of
       wzhlearning/fNIRS-Transformer at ``6ebe987f``, MIT). fNIRS-PreT is the
       right-hand input path: "Modified Beer-Lambert law", "Segmentation",
       then the "Preprocessing Module", then the two branches of fNIRS-T. The
       paper (IEEE JBHI 2022) is not open access; its caption is not
       reproduced.

    fNIRS-T expects signals that were band-pass filtered and
    baseline-corrected by hand. From the abstract :footcite:`wang2022fnirst`:
    "a preprocessing module, which consists of one-dimensional average
    pooling and layer normalization, is designed to replace filtering and
    baseline correction of data preprocessing. It makes fNIRS-T an
    end-to-end network, called fNIRS-PreT." The abstract adds that
    "fNIRS-PreT using raw data also achieves competitive performance to
    fNIRS-T", without a number.

    The authors' ``PreBlock`` (``model.py``) is that module: "Preprocessing
    module. It is designed to replace filtering and baseline correction."
    Only the abstract, the README, the figure and the code are public, and
    the description below comes from them.

    .. rubric:: Architecture Overview

    This is :class:`~braindecode.models.FNIRST` with the preprocessing module
    (``pre``) first, as the authors' ``fNIRS_PreT`` (``PreBlock``, then
    ``fNIRS_T``). With :math:`H` = ``n_chans // 2`` channels and :math:`W` =
    ``n_times`` samples::

        input: H HbO rows and H HbR rows             (batch, 2 H, W)
        to_chromophore_planes                        (batch, 2, H, W)
        pre: per plane, average_pooling, norms[i]    (batch, 2, H, W)
        fNIRS-T levels, Concat, head                 (batch, n_outputs)

    .. rubric:: Macro Components

    - ``pre.average_pooling`` **Average pooling.** Three moving averages
      along time, ``AvgPool1d`` with kernels :math:`k` = 5, 13 and 7, stride
      1 and "same" zero padding (the authors' ``pool1``, ``pool2``,
      ``pool3``), shared by HbO and HbR. With :math:`k = 2r + 1`, each is

      .. math::

         m_k(x)[t] = \frac{1}{k} \sum_{j=-r}^{r} x[t + j] .

    - ``pre.norms`` **Layer normalization.** One ``LayerNorm`` over the
      :math:`W` samples per chromophore (the authors' ``ln_0``, ``ln_1``):
      each channel row is centred and scaled over the window, then given a
      learned scale and shift per sample, so the weights are tied to
      ``n_times``.
    - Every other module is that of :class:`~braindecode.models.FNIRST`.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    The module keeps the :math:`W` samples and treats every channel row
    alike; HbO and HbR share the moving averages and have their own
    LayerNorm. It has no explicit frequency decomposition: it stands in for
    the filtering and baseline correction, so the inputs are raw segmented
    concentration changes. The rest is as in
    :class:`~braindecode.models.FNIRST`.

    .. rubric:: Pretrained Weights

    None: the authors release code, not weights.

    .. rubric:: Differences from the reference implementation

    ``pre.ln_i`` is ``pre.norms.i`` and the authors' ``fNIRS_T.`` prefix is
    removed: an authors' ``fNIRS_PreT`` ``state_dict`` loads after FNIRST's
    conversion with these two renames, and the port and the authors' model
    then agree to float32 rounding (max-abs below 1e-6 on the dataset A, B
    and C configurations). The port indexes each chromophore plane, where
    the authors' ``PreBlock`` calls ``squeeze()``, which also drops the
    batch axis for a batch of one. The input layout, the opt-in ``mask``,
    the logits and the parameters are those of
    :class:`~braindecode.models.FNIRST`.

    Notes
    -----
    The protocol of the authors' released code is that of
    :class:`~braindecode.models.FNIRST`, on raw data: their README asks to
    disable the filtering and baseline correction of ``B_mat2xls.m`` and
    ``C_mat2xls.m`` by hand for fNIRS-PreT, and ``Load_Dataset_A`` skips the
    dataset A filters for ``model='fNIRS-PreT'``. The flooding levels are
    those of fNIRS-T except on dataset B (0.40, 0.38 and 0.35 from the
    zero-based epochs 0, 30 and 50).

    The dataset C replication is the fNIRS-PreT row of the table of
    :class:`~braindecode.models.FNIRST`. Both sides received the band-pass
    filtered, baseline-corrected windows of fNIRS-T there, as the released
    ``Load_Dataset_C`` has no raw-data path.

    .. versionadded:: 2.1

    `License <https://github.com/wzhlearning/fNIRS-Transformer/blob/6ebe987faacc5587e11e8bd0d1177b40ec3ccd8d/LICENSE>`_

    References
    ----------
    .. footbibliography::
    """

    _with_preprocessing = True

    # Its own __init__: with docstring inheritance on (docs/conf.py), a subclass
    # sharing FNIRST.__init__ is given FNIRST's docstring.
    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        embed_dim: int = 64,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 64,
        pool: Literal["cls", "mean"] = "cls",
        head_dim: int = 64,
        n_conv_filters: int = 8,
        spatial_kernel_size: int = 5,
        temporal_kernel_size: int = 30,
        temporal_stride: int = 4,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.0,
        embed_drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            pool=pool,
            head_dim=head_dim,
            n_conv_filters=n_conv_filters,
            spatial_kernel_size=spatial_kernel_size,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            channel_order=channel_order,
            drop_prob=drop_prob,
            embed_drop_prob=embed_drop_prob,
            activation=activation,
        )


class _PreprocessingModule(nn.Module):
    """The authors' preprocessing module: moving averages, then a LayerNorm
    over time for HbO (``norms.0``) and one for HbR (``norms.1``)."""

    def __init__(self, n_times: int) -> None:
        super().__init__()
        self.average_pooling = nn.Sequential(  # "same" padding
            nn.AvgPool1d(5, stride=1, padding=2),
            nn.AvgPool1d(13, stride=1, padding=6),
            nn.AvgPool1d(7, stride=1, padding=3),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(n_times) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = []
        for chromophore, norm in enumerate(self.norms):
            plane = x[:, chromophore]  # (batch, n_pairs, n_times)
            plane = self.average_pooling(plane)  # (batch, n_pairs, n_times)
            plane = norm(plane)  # (batch, n_pairs, n_times)
            planes.append(plane)
        # (batch, 2, n_pairs, n_times)
        return torch.stack(planes, dim=1)
