# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: Apache-2.0
# Adapted from https://github.com/wzhlearning/fNIRS4D (Apache-2.0); S4D kernel after
# https://github.com/state-spaces/s4 (Apache-2.0).
"""fNIRS4D: Domain Knowledge Fused State Space Model for fNIRS-Based BCIs."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    _disable_batch_norm_training_if_batch_size_one,
    fnirs_channel_order,
    fnirs_split_hbo_hbr,
)


class FNIRS4D(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""fNIRS4D from Wang et al. (2026).

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. figure:: https://braindecode.org/dev/_static/model/fnirs4d.png
       :align: center
       :alt: fNIRS4D architecture

       ``fig/fig2.png`` of wzhlearning/fNIRS4D (``eb13e5e``, Apache-2.0),
       resized: preprocessing, "fNIRS4D for Signal Classification" with its
       "Selectable Branches" MAMHB (inter-subject) and MHB (intra-subject),
       and the three task paradigms. The paper (IEEE TIM 2026) is not open
       access; its caption is not reproduced.

    From the abstract :footcite:`wang2026fnirs4d`: "many studies focus
    solely on deep neural networks (DNNs) but neglect the importance of
    fNIRS domain knowledge. We refine domain knowledge into neuroscientific
    knowledge and expert prior. This article aims to integrate domain
    knowledge into DNNs, including activation patterns, hemodynamic
    responses, and feature engineering."

    The backbone is a state space model: "Diagonally structured state space
    sequence model (S4D), simple and efficient diagonal versions of SSMs,
    are introduced to the fNIRS domain. We incorporate domain knowledge into
    an S4D-based lightweight fNIRS classification model named fNIRS4D." The
    S4D layer is that of Gu et al. :footcite:`gu2022s4d`. The repository's
    ``fig/fig1.png`` ties each kind of knowledge to a block: activation
    patterns to the "Mixed Hemoglobin" input, hemodynamic responses to
    Knowledge Embedding (KE), feature engineering to the selectable mean
    hemoglobin branches, with Knowledge Aggregation (KA) over all of them.
    Only the abstract, README, figures and code are public, and the text
    below comes from them.

    .. rubric:: Architecture Overview

    With :math:`H` = ``n_chans`` rows (``d_model`` in the authors' S4D),
    :math:`T` = ``n_times`` samples, :math:`N` = ``n_layers`` and the
    figure's names::

        input: HbO and HbR rows                      (batch, H, T)
        Mixed Hemoglobin [HbO, HbT], to_fnirs_tensor (batch, 1, H, T)
        Selectable branch (mean_hemoglobin_branch):
          MHB or MAMHB                               (batch, H)
        Knowledge Embedding (fNIRS_knowledge_embedding):
          Conv_1 (1 x T), IN, Conv_1x1, Rearrange    (batch, H, H)
        Knowledge Aggregation:
          N x (S4D, dropout, residual, LN)           (batch, H, H)
          AP_4: mean over the sequence               (batch, H)
        concatenate with the branch, final_layer FC  (batch, n_outputs)

    .. rubric:: Macro Components

    - ``compute_hbt`` **Mixed Hemoglobin.** The rows become HbO and total
      hemoglobin, :math:`\mathrm{HbT} = \mathrm{HbO} + \mathrm{HbR}`, as in
      the authors' ``dataloader.py``; HbO/HbR pairs come from
      ``channel_order`` or ``chs_info``.
    - ``fNIRS_knowledge_embedding`` **Knowledge Embedding (KE).** A
      convolution with :math:`H` filters of :math:`1 \times T`, so each
      filter reads the whole window of one row (the kernel shape of
      :class:`~braindecode.models.FNIRSNet`'s DHRConv), instance
      normalization and a :math:`1 \times 1` convolution. The result is
      rearranged so that each row becomes a sequence of :math:`H` filter
      responses, the sequence the S4D layers read.
    - ``s4d_layers``, ``norms`` **Knowledge Aggregation (KA).** :math:`N`
      S4D blocks, each S4D, dropout, a residual sum and a LayerNorm over the
      rows (post-norm, or before the layer with ``prenorm=True``), then the
      mean over the sequence (:math:`\mathrm{AP}_4`). Each S4D layer runs one
      diagonal state space model per row. Gu et al. start from the
      continuous system (their Eq. (1)), discretize it with the zero-order
      hold (Section 3.1),

      .. math::

         x'(t) = A x(t) + B u(t), \quad y(t) = C x(t), \qquad
         \bar{A} = e^{\Delta A}, \quad
         \bar{B} = (\Delta A)^{-1} (e^{\Delta A} - I) \, \Delta B,

      and apply it as a convolution with a Vandermonde kernel
      (Eqs. (6)-(7)):

      .. math::

         y = u * \bar{K}, \qquad
         \bar{K}_\ell = \sum_{n} C_n \bar{A}_n^{\ell} \bar{B}_n .

      :math:`A` is diagonal and complex, initialized with S4D-Lin,
      :math:`A_n = -\tfrac{1}{2} + i \pi n` (Eq. (9)); the step
      :math:`\Delta` is learned per row from a log-uniform start in
      [0.001, 0.1]. As in the authors' ``S4DKernel``, :math:`B = 1`, the sum
      keeps ``d_state // 2`` entries and the kernel is twice its real part
      ("same as adding conjugate pairs", Listing 1 of Gu et al.). The layer
      adds a skip term :math:`D u`, then GELU, dropout, and a
      :math:`1 \times 1` convolution with a GLU.
    - ``mean_hemoglobin_branch`` **MHB and MAMHB.** The intra-subject Mean
      Hemoglobin Branch (``mean_branch="time_mean"``) is the mean of each
      row over time (:math:`\mathrm{AP}_1`). The inter-subject Multi-scale
      Aggregated Mean Hemoglobin Branch (``"multiscale"``) averages all rows
      into one signal (:math:`\mathrm{AP}_2`, ``avg_1``) and reads it twice,
      directly and after an ``avg_size``-sample moving average
      (:math:`\mathrm{AP}_3`, ``avg_2``), each time with a full-window
      convolution of :math:`H/2` filters, BN and ELU (``conv_1``/``bn_1``,
      ``conv_2``/``bn_2``); the two halves are concatenated.
    - ``final_layer`` **FC.** One linear layer on the :math:`H` KA features
      and the :math:`H` branch features.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* one 3 s window (40 samples on UFFT, 30 on MA and WG in
      the authors' scripts). KE reads all samples of a row at once, so the
      S4D layers run along the KE filters, not along time; the branches
      summarize time by means and full-window convolutions. Kernels count
      samples, so there is no sampling-rate argument.
    - *Spatial:* no montage or optode geometry. Each row is one S4D channel,
      LayerNorm normalizes across rows and MAMHB averages them; the row
      order must be the one used in training.
    - *Chromophores:* HbO and HbT; ``compute_hbt=False`` takes the rows as
      given.
    - *Spectral:* none in the model; the authors band-pass (UFFT, MA) or
      low-pass (WG) the signals first.

    .. rubric:: Pretrained Weights

    None: the authors release code, not weights.

    .. rubric:: Differences from the reference implementation

    ``x`` is ``(batch, n_chans, n_times)``; ``to_fnirs_tensor`` adds the
    authors' singleton axis, and HbT is formed inside the model (the authors
    sum before their z-score; to reproduce it, pass their [HbO, HbT] arrays
    with ``compute_hbt=False``). The S4D kernel is applied as a causal
    convolution in real arithmetic instead of a complex FFT, equal up to
    float32 rounding and free of complex tensors, so ``torch.jit.script``
    and ``torch.export`` run on it; ``nn.Dropout1d`` replaces the authors'
    ``DropoutNd``, and their unused ``_optim`` weight-decay hints are not
    set. Names are the authors'; ``mapping`` renames ``fc.0.*``, so their
    ``state_dict`` loads with ``strict=True``. ``mean_branch`` is their
    ``mode``, and ``avg_size``, ``d_state`` and the activations expose
    values their code fixes. A training batch of one sample runs the BN
    layers on their running statistics, and ``ch_mask`` raises a
    ``ValueError``.

    Parameters
    ----------
    n_layers : int, default=4
        Number of S4D blocks, the figure's :math:`N`; the authors use 4,
        and 1 for WG subject-independent.
    d_state : int, default=64
        State size of each S4D kernel (the authors' ``d_state``), even: the
        kernel keeps ``d_state // 2`` conjugate pairs.
    avg_size : int, default=10
        Width of the :math:`\mathrm{AP}_3` moving average of the
        ``"multiscale"`` branch (the authors' ``avg_size_2``), at most
        ``n_times``.
    drop_prob : float, default=0.5
        Dropout in and after each S4D layer (the authors' ``dropout``).
    prenorm : bool, default=False
        LayerNorm before each S4D layer instead of after the residual (the
        authors' default is post-norm).
    compute_hbt : bool, default=True
        Replace HbR by HbT = HbO + HbR (even ``n_chans``).
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO..., HbR...]`` (the authors') or ``[HbO_1, HbR_1, ...]`` (MNE);
        ``None`` reads ``chs_info``, else assumes ``"stacked"``. Ignored when
        ``compute_hbt=False``.
    mean_branch : {"multiscale", "time_mean"}, default="multiscale"
        The selectable branch (the authors' ``mode``): MAMHB for
        ``"LOSO-CV"`` (needs an even ``n_chans``) or MHB for ``"KFold-CV"``.
    activation : type[nn.Module], default=nn.GELU
        Activation inside each S4D layer.
    branch_activation : type[nn.Module], default=nn.ELU
        Activation of the ``"multiscale"`` branch.

    Notes
    -----
    The authors' code trains on UFFT :footcite:`bak2019ufft` (``n_chans`` =
    ``n_times`` = 40), MA :footcite:`shin2017openaccess` and WG
    :footcite:`shin2018simultaneous` (72 rows, 30 samples), with their BBCI
    preprocessing, 3 s windows with a 1 s step, HbT per window and a z-score
    from the training set. Subject-specific runs (MHB) use 5-fold CV over
    trial blocks, 120 epochs, AdamW (lr 0.001 on UFFT, 0.0005 on MA and WG)
    decayed by 0.2 at epochs 20, 40 and 60; subject-independent runs (MAMHB)
    leave one subject out, batch 512, cosine lr over 30 epochs. Last-epoch
    test accuracy.

    In the authors' training loops, on the same UFFT and WG arrays, splits
    and seeds (0-2), the port matches the authors' code: accuracy in %,
    paired difference in points with its 95 % interval, all including zero.

    ==========  =========  =============  ====================
    Cell        This port  Authors' code  Paired diff [95% CI]
    ==========  =========  =============  ====================
    UFFT KFold  71.98      72.00          -0.01 [-0.49, +0.46]
    UFFT LOSO   65.81      65.85          -0.04 [-0.32, +0.24]
    WG KFold    61.54      61.33          +0.21 [-0.50, +0.91]
    WG LOSO     61.92      62.00          -0.07 [-0.54, +0.39]
    ==========  =========  =============  ====================

    In the GeoNIRS benchmark :footcite:`li2026geonirs` (macro-F1 in %, 12 s
    windows, subject-level 5-fold CV, seeds 810 and 811, ``"multiscale"``),
    the port gives 67.89 on DS42 action mode and 49.86 on DS19 pain
    intensity, against 68.98 and 50.79 in GeoNIRS Table 14 and 69.03 and
    50.60 for the GeoNIRS copy of fNIRS4D in the same loop.

    .. versionadded:: 2.1

    `License <https://github.com/wzhlearning/fNIRS4D/blob/eb13e5e5b47043ad408d8a50b3ec9fe079926c12/LICENSE>`_

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
        n_layers: int = 4,
        d_state: int = 64,
        avg_size: int = 10,
        drop_prob: float = 0.5,
        prenorm: bool = False,
        compute_hbt: bool = True,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        mean_branch: Literal["multiscale", "time_mean"] = "multiscale",
        activation: type[nn.Module] = nn.GELU,
        branch_activation: type[nn.Module] = nn.ELU,
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

        # The authors' head is ``fc = Sequential(Linear)``: their state_dict
        # loads as is
        self.mapping = {f"fc.0.{k}": f"final_layer.{k}" for k in ("weight", "bias")}

        # Parameters
        self.prenorm = prenorm
        self.compute_hbt = compute_hbt

        # Checkers
        # HbT = HbO + HbR is a sum over pairs: an odd number of rows has an
        # unpaired chromophore
        if compute_hbt and self.n_chans % 2:
            raise ValueError(
                f"compute_hbt needs HbO/HbR pairs, got n_chans={self.n_chans}."
            )
        # Knowledge Embedding gives a (H, 1) map per filter, and the instance
        # normalization cannot normalize a single value
        if self.n_chans < 2:
            raise ValueError(f"FNIRS4D needs n_chans >= 2, got n_chans={self.n_chans}.")
        if mean_branch not in ("multiscale", "time_mean"):
            raise ValueError(
                f"mean_branch must be 'multiscale' or 'time_mean', got {mean_branch!r}."
            )
        if mean_branch == "multiscale":
            # MAMHB has H / 2 filters per scale, so that its two scales give
            # the H features the FC layer expects
            if self.n_chans % 2:
                raise ValueError(
                    "mean_branch='multiscale' needs an even n_chans, "
                    f"got {self.n_chans}."
                )
            # The moving average (AP_3) needs at least one full window
            if avg_size > self.n_times:
                raise ValueError(
                    f"avg_size must be at most n_times={self.n_times}, got {avg_size}."
                )
        # HbO/HbR layout of the rows, read from chs_info when HbT is formed
        # here (the helper refuses a montage that is not paired HbO/HbR)
        pairs_info = self._chs_info if compute_hbt else None
        self.channel_order = fnirs_channel_order(pairs_info, channel_order)

        # Layers
        # Following the nomenclature of the authors' figure (fig/fig2.png)
        # The authors' input has a singleton axis: (batch, 1, H, T)
        self.to_fnirs_tensor = Rearrange("batch chans time -> batch 1 chans time")

        # Selectable branch (feature engineering as expert prior)
        self.mean_hemoglobin_branch: nn.Module
        if mean_branch == "multiscale":
            # Multi-scale Aggregated Mean Hemoglobin Branch (MAMHB), the
            # inter-subject (LOSO-CV) branch
            self.mean_hemoglobin_branch = _MultiscaleMeanHemoglobinBranch(
                n_chans=self.n_chans,
                n_times=self.n_times,
                avg_size=avg_size,
                activation=branch_activation,
            )
        else:
            # Mean Hemoglobin Branch (MHB), the intra-subject (KFold-CV)
            # branch: AP_1, the mean of each row over the window
            self.mean_hemoglobin_branch = nn.Sequential(
                nn.AvgPool2d(kernel_size=(1, self.n_times), stride=1),
                Rearrange("batch 1 chans 1 -> batch chans"),
            )
        # Knowledge Embedding (KE): hemodynamic responses as domain knowledge
        self.fNIRS_knowledge_embedding = nn.Sequential(
            # Conv_1: H filters of 1 x T, each reads the whole response of
            # one row (the kernel shape of fNIRSNet's DHRConv)
            nn.Conv2d(
                in_channels=1,
                out_channels=self.n_chans,
                kernel_size=(1, self.n_times),
            ),
            # IN: each filter map is normalized over the rows
            nn.InstanceNorm2d(self.n_chans),
            # Conv_1x1: mixes the H filters
            nn.Conv2d(
                in_channels=self.n_chans,
                out_channels=self.n_chans,
                kernel_size=1,
            ),
            # Each row becomes one S4D channel (d_model = H) and the H filter
            # responses its sequence
            Rearrange("batch filters chans 1 -> batch chans filters"),
        )

        # Knowledge Aggregation (KA): N blocks of S4D and LayerNorm
        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.s4d_layers.append(
                _S4DLayer(
                    d_model=self.n_chans,
                    d_state=d_state,
                    drop_prob=drop_prob,
                    activation=activation,
                )
            )
            # LN over the rows (the S4D channels)
            self.norms.append(nn.LayerNorm(self.n_chans))
        # Dropout of whole S4D channels before the residual sum
        self.dropout = nn.Dropout1d(drop_prob)
        # LayerNorm and the sequence mean (AP_4) read the row axis last
        self.channels_last = Rearrange("batch chans filters -> batch filters chans")
        self.channels_first = Rearrange("batch filters chans -> batch chans filters")

        # Final fully connected layer, on the H features of KA and the H
        # features of the mean hemoglobin branch
        self.final_layer = nn.Linear(
            in_features=2 * self.n_chans,
            out_features=self.n_outputs,
        )

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass of the fNIRS4D model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, n_chans, n_times)``: HbO
            and HbR rows, or the rows as given with ``compute_hbt=False``.
        return_features : bool, default=False
            If True, return the features that feed the FC layer.
        ch_mask : torch.Tensor or None
            Not supported: every layer is sized by ``n_chans``.

        Returns
        -------
        torch.Tensor or dict
            Logits with shape ``(batch_size, n_outputs)``, or with
            ``return_features=True`` the dict ``{"features", "cls_token"}``,
            ``features`` of shape ``(batch_size, 2 * n_chans)`` and
            ``cls_token`` None.
        """
        # The rows are the S4D channels and the LayerNorm axis: a subset of
        # them is another model
        if ch_mask is not None:
            raise ValueError("FNIRS4D does not accept ch_mask: layers span n_chans.")

        # Below, H = n_chans and T = n_times

        # Mixed Hemoglobin (activation patterns as domain knowledge): the
        # rows become HbO and total hemoglobin, HbT = HbO + HbR
        if self.compute_hbt:
            # (batch, H, T) -> 2 x (batch, H / 2, T)
            hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
            # (batch, H / 2, T)
            hbt = hbo + hbr
            # [HbO, HbT]: (batch, H, T)
            x = torch.cat((hbo, hbt), dim=1)

        # Singleton axis of the authors' input
        # (batch, H, T) -> (batch, 1, H, T)
        x = self.to_fnirs_tensor(x)

        # Selectable branch, on the input: mean hemoglobin features (MHB:
        # time mean of each row; MAMHB: two scales of the row average)
        # (batch, 1, H, T) -> (batch, H)
        mean_features = self.mean_hemoglobin_branch(x)

        # Knowledge Embedding (KE): Conv_1 over the whole window of each
        # row, IN, Conv_1x1; each row is then a sequence of H filter responses
        # (batch, 1, H, T) -> (batch, H, H) = (batch, rows, filters)
        x = self.fNIRS_knowledge_embedding(x)

        # Knowledge Aggregation (KA): N x (S4D, dropout, residual, LN)
        for layer, norm in zip(self.s4d_layers, self.norms):
            # (batch, H, H)
            layer_input = x
            # Pre-norm variant: LN over the rows before the S4D layer
            if self.prenorm:
                # (batch, rows, filters) -> (batch, filters, rows)
                layer_input = self.channels_last(layer_input)
                layer_input = norm(layer_input)
                # (batch, filters, rows) -> (batch, rows, filters)
                layer_input = self.channels_first(layer_input)
            # S4D: one diagonal state space model per row, run along its
            # sequence of filter responses; shape unchanged
            layer_output = layer(layer_input)
            # Whole S4D channels are dropped; shape unchanged
            layer_output = self.dropout(layer_output)
            # Residual sum; shape unchanged
            x = x + layer_output
            # Post-norm (the authors' default): LN over the rows after the sum
            if not self.prenorm:
                # (batch, rows, filters) -> (batch, filters, rows)
                x = self.channels_last(x)
                x = norm(x)
                # (batch, filters, rows) -> (batch, rows, filters)
                x = self.channels_first(x)

        # (batch, rows, filters) -> (batch, filters, rows)
        x = self.channels_last(x)
        # AP_4: mean over the sequence. Tensor.mean and not a Reduce layer,
        # which permutes first and changes the float32 rounding of the
        # gradients
        # (batch, H, H) -> (batch, H)
        x = x.mean(dim=1)

        # KA features and mean hemoglobin features side by side
        # 2 x (batch, H) -> (batch, 2 H)
        features = torch.cat((x, mean_features), 1)

        if return_features:
            return {"features": features, "cls_token": None}

        # FC layer
        # (batch, 2 H) -> (batch, n_outputs)
        x = self.final_layer(features)
        return x


class _MultiscaleMeanHemoglobinBranch(nn.Module):
    """Multi-scale Aggregated Mean Hemoglobin Branch (MAMHB).

    The authors' inter-subject (LOSO-CV) branch: the average of all rows
    (AP_2) and its moving average (AP_3), each read by a full-window temporal
    convolution, batch normalization and ``activation``.

    Parameters
    ----------
    n_chans : int
        Number of rows of the input; each scale has ``n_chans // 2`` filters.
    n_times : int
        Number of samples of the window.
    avg_size : int
        Length of the moving average, in samples.
    activation : type[nn.Module]
        Activation after each batch normalization.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        avg_size: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        # Parameters
        # H / 2 filters per scale, as the authors': the two scales together
        # give as many features as KA
        n_filters = n_chans // 2
        # Samples left after the moving average, T' = T - avg_size + 1
        n_times_avg = n_times - avg_size + 1

        # Layers
        # AP_2: average over all rows, one mean hemoglobin signal
        self.avg_1 = nn.AvgPool2d(kernel_size=(n_chans, 1), stride=1)
        # AP_3: moving average over time, the second (smoother) scale
        self.avg_2 = nn.AvgPool2d(kernel_size=(1, avg_size), stride=1)
        # First scale: full-window convolution of the mean signal
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=n_filters, kernel_size=(1, n_times)
        )
        self.bn_1 = nn.BatchNorm2d(n_filters)
        # Second scale: full-window convolution of the smoothed signal
        self.conv_2 = nn.Conv2d(
            in_channels=1, out_channels=n_filters, kernel_size=(1, n_times_avg)
        )
        self.bn_2 = nn.BatchNorm2d(n_filters)
        self.activation = activation()
        self.flatten = Rearrange("batch filters 1 1 -> batch filters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean hemoglobin features ``(batch, n_chans)`` of ``x``
        ``(batch, 1, n_chans, n_times)``."""
        # Below, H = n_chans, T = n_times and T' = T - avg_size + 1

        # AP_2: the rows are averaged into one mean hemoglobin signal
        # (batch, 1, H, T) -> (batch, 1, 1, T)
        mean = self.avg_1(x)
        # AP_3: its moving average, the second scale
        # (batch, 1, 1, T) -> (batch, 1, 1, T')
        short = self.avg_2(mean)

        # First scale: H / 2 filters read the whole mean signal
        # (batch, 1, 1, T) -> (batch, H / 2, 1, 1)
        full = self.conv_1(mean)
        # Batch normalization and activation (ELU); shape unchanged
        full = self.bn_1(full)
        full = self.activation(full)

        # Second scale: H / 2 filters read the whole smoothed signal
        # (batch, 1, 1, T') -> (batch, H / 2, 1, 1)
        short = self.conv_2(short)
        # Batch normalization and activation (ELU); shape unchanged
        short = self.bn_2(short)
        short = self.activation(short)

        # Multi-scale aggregation: the two scales are concatenated
        # 2 x (batch, H / 2, 1, 1) -> (batch, H, 1, 1)
        features = torch.cat((full, short), dim=1)
        # (batch, H, 1, 1) -> (batch, H)
        features = self.flatten(features)
        return features


class _S4DKernel(nn.Module):
    """S4D-Lin kernel ``K = 2 Re(C (exp(dt A) - 1) / A exp(dt A l))``, in real
    arithmetic: ``A = -exp(log_A_real) + i A_imag``, ``dt = exp(log_dt)``."""

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.C = nn.Parameter(torch.empty(d_model, d_state // 2, 2))
        self.log_dt = nn.Parameter(torch.empty(d_model))
        self.log_A_real = nn.Parameter(torch.empty(d_model, d_state // 2))
        self.A_imag = nn.Parameter(torch.empty(d_model, d_state // 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.C, 0.0, math.sqrt(0.5))
        nn.init.uniform_(self.log_dt, math.log(0.001), math.log(0.1))  # reference dt
        nn.init.constant_(self.log_A_real, math.log(0.5))
        with torch.no_grad():
            self.A_imag.copy_(
                math.pi * torch.arange(self.A_imag.shape[1]).to(self.A_imag)
            )

    def forward(self, length: int) -> torch.Tensor:
        # (d_model, d_state // 2) unless noted; A = a + i b, one call per line
        dt = torch.exp(self.log_dt)  # (d_model,)
        dt = dt.unsqueeze(-1)  # (d_model, 1), broadcasts over the state
        a = -torch.exp(self.log_A_real)
        b = self.A_imag
        dt_a = a * dt
        dt_b = b * dt
        # exp(dt A) - 1 = e_real + i e_imag, with expm1 and 1 - cos = 2 sin^2
        exp_dt_a = torch.exp(dt_a)
        cos_dt_b = torch.cos(dt_b)
        sin_dt_b = torch.sin(dt_b)
        half_sin = torch.sin(0.5 * dt_b)
        e_real = torch.expm1(dt_a) * cos_dt_b - 2.0 * half_sin * half_sin
        e_imag = exp_dt_a * sin_dt_b
        # w = (exp(dt A) - 1) / A, then C~ = C w
        modulus = a * a + b * b  # > 0: a = -exp(log_A_real)
        w_real = (e_real * a + e_imag * b) / modulus
        w_imag = (e_imag * a - e_real * b) / modulus
        c_real, c_imag = self.C.unbind(-1)
        ct_real = c_real * w_real - c_imag * w_imag
        ct_imag = c_real * w_imag + c_imag * w_real
        # Vandermonde basis exp(dt A l), (d_model, d_state // 2, length)
        steps = torch.arange(length, device=dt.device, dtype=dt.dtype)
        dt_a_steps = dt_a.unsqueeze(-1) * steps
        decay = torch.exp(dt_a_steps)
        angles = dt_b.unsqueeze(-1) * steps
        basis_real = decay * torch.cos(angles)
        basis_imag = decay * torch.sin(angles)
        kernel = torch.einsum("hn,hnl->hl", ct_real, basis_real)
        kernel = kernel - torch.einsum("hn,hnl->hl", ct_imag, basis_imag)
        return 2.0 * kernel  # (d_model, length)


class _S4DLayer(nn.Module):
    """S4D layer: causal depthwise convolution with the S4D kernel, D skip, GLU."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        drop_prob: float,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.D = nn.Parameter(torch.empty(d_model))
        self.kernel = _S4DKernel(d_model, d_state)
        self.activation = activation()
        self.dropout = nn.Dropout1d(drop_prob)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=-2)
        )
        self.to_conv_weight = Rearrange("d_model length -> d_model 1 length")
        # draws D, then the kernel a second time (its __init__ drew it once):
        # the seeded weights of the replications in FNIRS4D's docstring follow
        # this order
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.D)
        self.kernel.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]  # x: (batch, d_model, length)
        kernel = self.kernel(length)  # (d_model, length)
        kernel = kernel.flip(-1)  # conv1d correlates: reverse for a convolution
        kernel = self.to_conv_weight(kernel)  # (d_model, 1, length)
        padded = F.pad(x, (length - 1, 0))  # causal: (batch, d_model, 2 length - 1)
        y = F.conv1d(padded, kernel, groups=x.shape[1])  # (batch, d_model, length)
        skip = self.D.unsqueeze(-1)  # (d_model, 1), broadcasts over time
        skip = x * skip  # (batch, d_model, length)
        y = y + skip  # (batch, d_model, length)
        y = self.activation(y)  # (batch, d_model, length)
        y = self.dropout(y)  # (batch, d_model, length)
        return self.output_linear(y)  # (batch, d_model, length)
