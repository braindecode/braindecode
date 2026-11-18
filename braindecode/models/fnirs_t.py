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

    def __init__(
        self,
        # Braindecode parameters
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        # Model-specific parameters
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

        # Parameters
        self.pool = pool
        # H in the authors' figure: fNIRS channels, each an HbO/HbR pair of rows
        n_pairs = self.n_chans // 2
        # W_S = W_C: positions of the temporal kernel along the window; they
        # become features of a token, not tokens
        n_bins = (self.n_times - temporal_kernel_size) // temporal_stride + 1
        # H_S: groups of ``spatial_kernel_size`` consecutive channels
        n_groups = n_pairs - spatial_kernel_size + 1

        # Checkers
        # The temporal kernel needs at least one position in the window
        if self.n_times < temporal_kernel_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be >= temporal_kernel_size "
                f"({temporal_kernel_size})."
            )
        # The authors' input is [B, 2, channels, samples]: an odd number of
        # rows cannot be split into the HbO and the HbR plane
        if self.n_chans % 2:
            raise ValueError(
                f"{type(self).__name__} needs HbO/HbR pairs: n_chans must be even, "
                f"got {self.n_chans}."
            )
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}.")
        # HbO/HbR layout of the rows (the helper refuses a montage that is
        # not paired HbO/HbR)
        order = fnirs_channel_order(self._chs_info, channel_order)
        # Conv_S needs at least one group of channels (the authors' fails too)
        if spatial_kernel_size > n_pairs:
            raise ValueError(
                f"spatial_kernel_size ({spatial_kernel_size}) must be at most the "
                f"number of HbO/HbR pairs ({n_pairs})."
            )

        # Layers
        # Following the nomenclature of the authors' figure (fig/model.png)
        # Input: the rows are split into the HbO and the HbR plane, the
        # H x W x C input of the figure with the chromophores first
        channel_axes = "(chans chrom)" if order == "interleaved" else "(chrom chans)"
        self.to_chromophore_planes = Rearrange(
            f"batch {channel_axes} time -> batch chrom chans time", chrom=2
        )

        # fNIRS-T takes filtered, baseline-corrected segments as they are;
        # FNIRSPreT puts its Preprocessing Module here
        self.pre = nn.Identity()

        # Spatial level
        # fNIRS Spatial Embedding: Conv_S over k_s consecutive channels and
        # k_t samples; the responses of a group are flattened, projected to
        # the embedding size and layer-normalized: one token per group
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=n_conv_filters,
                kernel_size=(spatial_kernel_size, temporal_kernel_size),
                stride=(1, temporal_stride),
            ),
            Rearrange("batch maps tokens bins -> batch tokens (maps bins)"),
            nn.Linear(n_bins * n_conv_filters, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Learned class token and positional embedding, one row per token
        # (class token and H_S groups); the row is the token's only identity
        self.cls_token_patch = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embedding_patch = nn.Parameter(torch.empty(1, n_groups + 1, embed_dim))
        self.dropout_patch = nn.Dropout(embed_drop_prob)
        # Transformer Encoder: ``depth`` pre-norm blocks of bias-free
        # multi-head self-attention and a feed-forward layer of width mlp_dim
        self.transformer_patch = _TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            drop_prob=drop_prob,
            activation=activation,
        )

        # Channel level
        # fNIRS Channel Embedding: Conv_C, the same with a 1 x k_t kernel:
        # one token per fNIRS channel
        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=n_conv_filters,
                kernel_size=(1, temporal_kernel_size),
                stride=(1, temporal_stride),
            ),
            Rearrange("batch maps tokens bins -> batch tokens (maps bins)"),
            nn.Linear(n_bins * n_conv_filters, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Class token and positional embedding (class token and H channels)
        self.cls_token_channel = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embedding_channel = nn.Parameter(
            torch.empty(1, n_pairs + 1, embed_dim)
        )
        self.dropout_channel = nn.Dropout(embed_drop_prob)
        # Transformer Encoder of the channel level, same layout as above
        self.transformer_channel = _TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            drop_prob=drop_prob,
            activation=activation,
        )

        # Head (the authors' mlp_head): the two levels are concatenated
        # ("Concat"), layer-normalized and classified
        # Mean over the tokens of a level, the pool="mean" alternative to its
        # class token
        self.token_mean = Reduce("batch tokens embed_dim -> batch embed_dim", "mean")
        self.final_norm = nn.LayerNorm(embed_dim * 2)
        self.final_layer = nn.Linear(embed_dim * 2, self.n_outputs)

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Class tokens and positional tables are drawn from N(0, 1), after the
        # layers, so that reset_parameters() repeats the constructor.
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
        """Forward pass of the fNIRS-T (and fNIRS-PreT) model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, n_chans, n_times)``.
        return_features : bool, default=False
            If True, return the normalized pooled features and the class
            tokens of the two levels.
        ch_mask : torch.Tensor or None
            Not supported: the spatial kernel and the positional tables span
            every channel.
        mask : torch.Tensor or None
            Boolean tensor with the shape of ``x``; True samples are set to
            zero before the model reads them.

        Returns
        -------
        torch.Tensor or dict
            Logits with shape ``(batch_size, n_outputs)``, or with
            ``return_features=True`` the dict ``{"features", "cls_token"}``,
            both of shape ``(batch_size, 2 * embed_dim)``.
        """
        # A token is identified only by its row of the positional table, and
        # Conv_S groups neighbouring channels: a subset of them is another
        # layout
        if ch_mask is not None:
            raise ValueError(
                "ch_mask is not supported: the spatial kernel and the positional "
                "tables span every channel."
            )

        # Below, H = n_chans // 2 fNIRS channels, W = n_times, D = embed_dim,
        # H_S = H - spatial_kernel_size + 1 groups of channels

        # Hidden samples are set to zero (an addition of the port)
        # (batch, 2 H, W)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)

        # Input: the rows are split into the HbO and the HbR plane
        # (batch, 2 H, W) -> (batch, 2, H, W)
        x = self.to_chromophore_planes(x)
        # Preprocessing Module of fNIRS-PreT (identity in fNIRS-T, whose
        # segments are already filtered and baseline-corrected)
        # (batch, 2, H, W)
        x = self.pre(x)
        batch_size = x.shape[0]

        # Spatial level. fNIRS Spatial Embedding: Conv_S, flatten, linear
        # projection, LayerNorm; one token per group of consecutive channels
        # (batch, 2, H, W) -> (batch, H_S, D)
        patch = self.to_patch_embedding(x)
        # Class token of the level, one copy per sample
        # (1, 1, D) -> (batch, 1, D)
        cls_patch = self.cls_token_patch.expand(batch_size, -1, -1)
        # (batch, H_S, D) -> (batch, H_S + 1, D)
        patch = torch.cat((cls_patch, patch), dim=1)
        # Learned positional embedding, which tells the groups apart; shape
        # unchanged
        patch = patch + self.pos_embedding_patch
        patch = self.dropout_patch(patch)
        # Transformer Encoder: the groups attend to each other
        # (batch, H_S + 1, D)
        patch = self.transformer_patch(patch)

        # Channel level. fNIRS Channel Embedding: Conv_C, flatten, linear
        # projection, LayerNorm; one token per fNIRS channel
        # (batch, 2, H, W) -> (batch, H, D)
        channel = self.to_channel_embedding(x)
        # Class token of the level, one copy per sample
        # (1, 1, D) -> (batch, 1, D)
        cls_channel = self.cls_token_channel.expand(batch_size, -1, -1)
        # (batch, H, D) -> (batch, H + 1, D)
        channel = torch.cat((cls_channel, channel), dim=1)
        # Learned positional embedding, which tells the channels apart; shape
        # unchanged
        channel = channel + self.pos_embedding_channel
        channel = self.dropout_channel(channel)
        # Transformer Encoder: the channels attend to each other
        # (batch, H + 1, D)
        channel = self.transformer_channel(channel)

        # Class token of each level, the summary of its tokens
        # (batch, D)
        patch_token = patch[:, 0]
        channel_token = channel[:, 0]
        # "Concat" of the two levels
        # 2 x (batch, D) -> (batch, 2 D)
        cls_tokens = torch.cat((patch_token, channel_token), dim=1)

        if self.pool == "mean":
            # Alternative pooling: the mean of the tokens of each level
            # (batch, tokens, D) -> (batch, D)
            patch = self.token_mean(patch)
            channel = self.token_mean(channel)
            # 2 x (batch, D) -> (batch, 2 D)
            features = torch.cat((patch, channel), dim=1)
        else:
            features = cls_tokens

        # Head (the authors' mlp_head): LayerNorm of the concatenation
        # (batch, 2 D)
        features = self.final_norm(features)

        if return_features:
            return {"features": features, "cls_token": cls_tokens}

        # Linear classifier; the output is logits
        # (batch, 2 D) -> (batch, n_outputs)
        x = self.final_layer(features)
        return x


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

    # The signature repeats FNIRST's, so that both models list the same
    # parameters; with docstring inheritance on (docs/conf.py), a subclass
    # sharing FNIRST.__init__ would also be given FNIRST's docstring.
    def __init__(
        self,
        # Braindecode parameters
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        # Model-specific parameters
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

        # Layers
        # Preprocessing Module: learned in place of the band-pass filter and
        # baseline correction that fNIRS-T expects; it takes the place of the
        # identity between the chromophore planes and the two embeddings
        self.pre = _PreprocessingModule(n_times=self.n_times)


class _TransformerEncoder(nn.Sequential):
    """Transformer Encoder of one level (the authors' ``Transformer``).

    ``depth`` pre-norm blocks, each a multi-head self-attention with bias-free
    query, key and value maps and a feed-forward layer, both with a residual
    connection. Built only from the shared ``PreNormTransformerBlock``,
    ``MultiHeadAttention`` and ``FeedForwardBlock``; the class groups them
    under the authors' name for the two levels of the model.

    Parameters
    ----------
    embed_dim : int
        Size of the tokens.
    depth : int
        Number of blocks.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Width of each attention head.
    mlp_dim : int
        Hidden width of the feed-forward layer.
    drop_prob : float
        Dropout probability inside the blocks.
    activation : type[nn.Module]
        Activation of the feed-forward layer.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        drop_prob: float,
        activation: type[nn.Module],
    ) -> None:
        super().__init__(
            *[
                PreNormTransformerBlock(
                    embed_dim,
                    # Self-attention: the tokens (channels or groups of
                    # channels) attend to each other
                    MultiHeadAttention(
                        embed_dim, num_heads, head_dim=head_dim, qkv_bias=False
                    ),
                    # Feed-forward layer of hidden width mlp_dim
                    FeedForwardBlock(
                        embed_dim, 1, drop_prob, activation, hidden_features=mlp_dim
                    ),
                    drop_prob,
                )
                for _ in range(depth)
            ]
        )


class _PreprocessingModule(nn.Module):
    """Preprocessing Module of fNIRS-PreT (the authors' ``PreBlock``).

    Three moving averages in cascade, then a LayerNorm over time for HbO
    (``norms.0``) and one for HbR (``norms.1``).

    Parameters
    ----------
    n_times : int
        Number of samples of the window, the size of the LayerNorms.
    """

    def __init__(self, n_times: int) -> None:
        super().__init__()
        # Moving averages of 5, 13 and 7 samples (the authors' values), a
        # low-pass smoothing; the padding keeps the number of samples
        self.average_pooling = nn.Sequential(
            nn.AvgPool1d(kernel_size=5, stride=1, padding=2),
            nn.AvgPool1d(kernel_size=13, stride=1, padding=6),
            nn.AvgPool1d(kernel_size=7, stride=1, padding=3),
        )
        # One normalization over time per chromophore, with its own scale
        # and offset per sample
        self.norms = nn.ModuleList([nn.LayerNorm(n_times) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Smoothed and normalized planes, ``(batch, 2, n_pairs, n_times)``
        like ``x``."""
        planes = []
        # HbO (plane 0) and HbR (plane 1) are processed separately
        for chromophore, norm in enumerate(self.norms):
            # (batch, 2, n_pairs, n_times) -> (batch, n_pairs, n_times)
            plane = x[:, chromophore]
            # Cascade of moving averages along time; shape unchanged
            plane = self.average_pooling(plane)
            # LayerNorm over the samples of each channel; shape unchanged
            plane = norm(plane)
            planes.append(plane)
        # The two planes are stacked back
        # 2 x (batch, n_pairs, n_times) -> (batch, 2, n_pairs, n_times)
        x = torch.stack(planes, dim=1)
        return x
