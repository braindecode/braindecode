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

    .. versionadded:: 2.1

    .. figure:: https://braindecode.org/dev/_static/model/fnirs4d.png
       :align: center
       :alt: fNIRS4D architecture

       ``fig/fig2.png`` of wzhlearning/fNIRS4D (``eb13e5e``, Apache-2.0),
       resized.

    fNIRS4D :footcite:`wang2026fnirs4d` builds fNIRS domain knowledge into a
    lightweight network of diagonal state space layers (S4D,
    :footcite:`gu2022s4d`): activation patterns give its mixed hemoglobin
    input, hemodynamic responses its knowledge embedding (KE) and feature
    engineering its selectable mean hemoglobin branch, and knowledge
    aggregation (KA) runs S4D blocks over them. The repository's
    ``fig/fig1.png`` ties each kind of knowledge to a block, and the layers
    follow the authors' ``fNIRS4D.py``. In the authors' training loops (UFFT
    and WG, same splits, seeds 0-2) the port matches the authors' accuracy,
    every paired difference with a 95 % interval including zero (:gh:`1582`);
    in the GeoNIRS loop it gives 67.89 and 49.86 macro-F1 on DS42 and DS19,
    against 68.98 and 50.79 in GeoNIRS Table 14 :footcite:`li2026geonirs` and
    69.03 and 50.60 for the GeoNIRS copy of fNIRS4D in the same loop.

    .. rubric:: Architecture Overview

    With :math:`H` = ``n_chans`` rows (``d_model`` in the authors' S4D),
    :math:`T` = ``n_times`` samples, :math:`N` = ``n_layers`` and the
    figure's names::

        input: HbO and HbR rows                      (batch, H, T)
        Mixed Hemoglobin [HbO, HbT], to_fnirs_tensor (batch, 1, H, T)
        Selectable branch (mean_hemoglobin_branch):
          MHB or MAMHB                               (batch, H)
        Knowledge Embedding (knowledge_embedding):
          Conv_1 (1 x T), IN, Conv_1x1, Rearrange    (batch, H, H)
        Knowledge Aggregation:
          N x (S4D, dropout, residual, LN)           (batch, H, H)
          AP_4: mean over the sequence               (batch, H)
        concatenate with the branch, final_layer FC  (batch, n_outputs)

    .. rubric:: Macro Components

    - ``compute_hbt`` **Mixed Hemoglobin.** The rows become HbO and total
      hemoglobin, :math:`\mathrm{HbT} = \mathrm{HbO} + \mathrm{HbR}`, as in
      the authors' ``dataloader.py``; the pairs come from ``channel_order``
      or ``chs_info``.
    - ``knowledge_embedding`` (``fNIRS_knowledge_embedding`` in the
      reference) **Knowledge Embedding (KE).** :math:`H` filters of
      :math:`1 \times T`, so each reads the whole window of one row (the
      kernel shape of :class:`~braindecode.models.FNIRSNet`'s DHRConv),
      instance normalization and a :math:`1 \times 1` convolution. Each row
      becomes a sequence of :math:`H` filter responses, which the S4D layers
      read.
    - ``s4d_layers``, ``norms`` **Knowledge Aggregation (KA).** :math:`N`
      blocks of S4D, dropout, a residual sum and a LayerNorm over the rows
      (post-norm, or before the layer with ``prenorm=True``), then the mean
      over the sequence (:math:`\mathrm{AP}_4`). An S4D layer runs one
      diagonal state space model per row: the continuous system (Gu et al.,
      Eq. (1)) is discretized with the zero-order hold (Section 3.1),

      .. math::

         x'(t) = A x(t) + B u(t), \quad y(t) = C x(t), \qquad
         \bar{A} = e^{\Delta A}, \quad
         \bar{B} = (\Delta A)^{-1} (e^{\Delta A} - I) \, \Delta B,

      and applied as a convolution with a Vandermonde kernel
      (Eqs. (6)-(7)):

      .. math::

         y = u * \bar{K}, \qquad
         \bar{K}_\ell = \sum_{n} C_n \bar{A}_n^{\ell} \bar{B}_n .

      :math:`A` is diagonal and complex, initialized with S4D-Lin,
      :math:`A_n = -a + i \pi n` with :math:`a` = ``a_real_init`` (1/2 in
      Eq. (9)); the step :math:`\Delta` is learned per row from a log-uniform
      start in [``dt_min``, ``dt_max``]. :math:`B = 1` as in the authors'
      ``S4DKernel``, the sum keeps ``d_state // 2`` entries and the kernel is
      twice its real part (conjugate pairs, Listing 1 of Gu et al.). The
      layer adds a skip term :math:`D u`, then GELU, dropout and a
      :math:`1 \times 1` convolution with a GLU.
    - ``mean_hemoglobin_branch`` **MHB and MAMHB.** The intra-subject mean
      hemoglobin branch (``mean_branch="time_mean"``) is the mean of each row
      over time (:math:`\mathrm{AP}_1`). The inter-subject multi-scale
      aggregated one (``"multiscale"``) averages all rows into one signal
      (:math:`\mathrm{AP}_2`, ``avg_1``) and reads it, and its
      ``avg_size``-sample moving average (:math:`\mathrm{AP}_3`, ``avg_2``),
      with full-window convolutions of :math:`H/2` filters, BN and ELU
      (``conv_1``/``bn_1``, ``conv_2``/``bn_2``); the two halves are
      concatenated.
    - ``final_layer`` **FC.** One linear layer on the :math:`H` KA features
      and the :math:`H` branch features.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal.** One 3 s window of :math:`T` samples (40 at 13.3 Hz, 30 at
      10 Hz, in the authors' scripts). KE reads all samples of a row at once,
      so the S4D layers run along the KE filters, not along time. Kernels
      count samples, so there is no sampling-rate argument.
    - **Spatial.** No montage or optode geometry. Each row is one S4D
      channel, LayerNorm normalizes across rows and MAMHB averages them, so
      the rows keep the order of training.
    - **Spectral.** None in the model; the signals are filtered beforehand.
      The authors band-pass (UFFT :footcite:`bak2019ufft`, MA
      :footcite:`shin2017openaccess`, as for :class:`~braindecode.models.FNIRSNet`)
      or low-pass at 0.2 Hz (WG :footcite:`shin2018simultaneous`), cut 3 s
      windows with a 1 s step, form HbT per window and z-score with the
      training-set statistics.
    - **Chromophores.** HbO and HbT; ``compute_hbt=False`` takes the rows as
      given.

    .. rubric:: Additional Mechanisms

    - **Hemoglobin input.** HbT is formed inside the model, as in the
      authors' ``dataloader.py``; the authors sum before their z-score, so to
      reproduce it pass their [HbO, HbT] arrays with ``compute_hbt=False``.
      There is no ``ch_mask``.
    - **S4D kernel.** Applied as a causal convolution in real arithmetic
      instead of the authors' complex FFT, equal up to float32 rounding and
      free of complex tensors, so ``torch.jit.script`` and ``torch.export``
      run on it; ``nn.Dropout1d`` replaces the authors' ``DropoutNd``.
    - **Pretrained weights.** None: the authors release code, not weights.

    `License <https://github.com/wzhlearning/fNIRS4D/blob/eb13e5e5b47043ad408d8a50b3ec9fe079926c12/LICENSE>`_

    Parameters
    ----------
    n_layers : int, default=4
        Number of S4D blocks, :math:`N`; the authors use 4, and 1 for WG
        subject-independent.
    d_state : int, default=64
        State size of each S4D kernel, even: it keeps ``d_state // 2``
        conjugate pairs.
    dt_min : float, default=0.001
        Lower end of the log-uniform initial range of the S4D step
        :math:`\Delta`, positive.
    dt_max : float, default=0.1
        Upper end of that range, at least ``dt_min``.
    a_real_init : float, default=0.5
        Initial decay rate of every state, :math:`-\mathrm{Re}(A_n)`, positive;
        0.5 is S4D-Lin (Eq. (9)).
    avg_size : int, default=10
        Width of the :math:`\mathrm{AP}_3` moving average of the
        ``"multiscale"`` branch (``avg_size_2`` in the reference), at most
        ``n_times``.
    drop_prob : float, default=0.5
        Dropout in and after each S4D layer (``dropout`` in the reference).
    prenorm : bool, default=False
        LayerNorm before each S4D layer instead of after the residual; the
        authors' default is post-norm.
    compute_hbt : bool, default=True
        Replace HbR by HbT = HbO + HbR (even ``n_chans``).
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO..., HbR...]`` (the authors') or ``[HbO_1, HbR_1, ...]`` (MNE);
        ``None`` reads ``chs_info``, else assumes ``"stacked"``. Ignored when
        ``compute_hbt=False``.
    mean_branch : {"multiscale", "time_mean"}, default="multiscale"
        The selectable branch (``mode`` in the reference): MAMHB,
        inter-subject (``"LOSO-CV"``, needs an even ``n_chans``), or MHB,
        intra-subject (``"KFold-CV"``).
    activation : type[nn.Module], default=nn.GELU
        Activation inside each S4D layer.
    branch_activation : type[nn.Module], default=nn.ELU
        Activation of the ``"multiscale"`` branch.

    References
    ----------
    .. footbibliography::
    """

    _time_axis = "fixed"  # fixed: knowledge_embedding.0.weight is sized by n_times
    _channel_axis = "fixed"  # fixed: final_layer.weight is sized by n_chans

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
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        a_real_init: float = 0.5,
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
        self.knowledge_embedding = nn.Sequential(
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
                    dt_min=dt_min,
                    dt_max=dt_max,
                    a_real_init=a_real_init,
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
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass of the fNIRS4D model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, n_chans, n_times)``: HbO
            and HbR rows, or the rows as given with ``compute_hbt=False``.
        return_features : bool, default=False
            If True, return the features that feed the FC layer.

        Returns
        -------
        torch.Tensor or dict
            Logits with shape ``(batch_size, n_outputs)``, or with
            ``return_features=True`` the dict ``{"features", "cls_token"}``,
            ``features`` of shape ``(batch_size, 2 * n_chans)`` and
            ``cls_token`` None.
        """
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
        x = self.knowledge_embedding(x)

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
            # sequence of filter responses
            layer_output = layer(layer_input)
            # Drops whole S4D channels, as a regularizer
            layer_output = self.dropout(layer_output)
            # Residual connection around the S4D layer
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
        # AP_4: mean over the sequence
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
        self.act_1 = activation()
        # Second scale: full-window convolution of the smoothed signal
        self.conv_2 = nn.Conv2d(
            in_channels=1, out_channels=n_filters, kernel_size=(1, n_times_avg)
        )
        self.bn_2 = nn.BatchNorm2d(n_filters)
        self.act_2 = activation()
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
        # Normalize, then the nonlinearity of the branch (ELU by default)
        full = self.bn_1(full)
        full = self.act_1(full)

        # Second scale: H / 2 filters read the whole smoothed signal
        # (batch, 1, 1, T') -> (batch, H / 2, 1, 1)
        short = self.conv_2(short)
        # Same normalization and nonlinearity as the first scale
        short = self.bn_2(short)
        short = self.act_2(short)

        # Multi-scale aggregation: the two scales are concatenated
        # 2 x (batch, H / 2, 1, 1) -> (batch, H, 1, 1)
        features = torch.cat((full, short), dim=1)
        # (batch, H, 1, 1) -> (batch, H)
        features = self.flatten(features)
        return features


class _S4DKernel(nn.Module):
    """Convolution kernel of a diagonal state space model (S4D-Lin).

    ``K_l = 2 Re(C (exp(dt A) - 1) / A exp(dt A l))`` for ``l = 0..length-1``,
    the Vandermonde kernel of Gu et al. (2022), Eqs. (6)-(7), with ``B = 1``
    and the zero-order hold. It is computed in real arithmetic, with
    ``A = -exp(log_A_real) + i A_imag`` and ``dt = exp(log_dt)``.

    Parameters
    ----------
    d_model : int
        Number of independent state space models (one per S4D channel).
    d_state : int
        State size; ``d_state // 2`` complex entries are kept, the kernel
        being twice the real part (conjugate pairs).
    dt_min : float
        Lower end of the log-uniform initial range of ``dt``.
    dt_max : float
        Upper end of that range.
    a_real_init : float
        Initial value of ``-Re(A)`` for every state; 1 / 2 in S4D-Lin.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dt_min: float,
        dt_max: float,
        a_real_init: float,
    ) -> None:
        super().__init__()
        # Initial values, as in Gu et al. (2022) and the S4D code of
        # state-spaces/s4, kept for reset_parameters(): the step dt is
        # log-uniform in [dt_min, dt_max] and the real part of A is -a_real_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.a_real_init = a_real_init
        # Output matrix C, complex: real and imaginary parts on the last axis
        self.C = nn.Parameter(torch.empty(d_model, d_state // 2, 2))
        # Step of the discretization, one per channel, learned in log scale
        self.log_dt = nn.Parameter(torch.empty(d_model))
        # Diagonal state matrix A: the real part is kept negative (stable)
        # through -exp(log_A_real)
        self.log_A_real = nn.Parameter(torch.empty(d_model, d_state // 2))
        self.A_imag = nn.Parameter(torch.empty(d_model, d_state // 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # C ~ complex N(0, 1): each part has variance 1 / 2
        nn.init.normal_(self.C, 0.0, math.sqrt(0.5))
        # dt log-uniform in [dt_min, dt_max]
        nn.init.uniform_(self.log_dt, math.log(self.dt_min), math.log(self.dt_max))
        # S4D-Lin (Gu et al., Eq. (9)): A_n = -a_real_init + i pi n, with
        # a_real_init = 1 / 2
        nn.init.constant_(self.log_A_real, math.log(self.a_real_init))
        with torch.no_grad():
            self.A_imag.copy_(
                math.pi * torch.arange(self.A_imag.shape[1]).to(self.A_imag)
            )

    def forward(self, length: int) -> torch.Tensor:
        """Kernel ``(d_model, length)`` for a sequence of ``length`` steps."""
        # Every tensor below is (d_model, d_state // 2) unless noted. The
        # complex numbers are carried as real and imaginary parts, A = a + i b

        # Step of the discretization (Delta in Gu et al.)
        # (d_model,) -> (d_model, 1), broadcasts over the state
        dt = torch.exp(self.log_dt)
        dt = dt.unsqueeze(-1)
        # Diagonal of A, with a negative real part
        a = -torch.exp(self.log_A_real)
        b = self.A_imag
        # dt A, the exponent of the zero-order hold
        dt_a = a * dt
        dt_b = b * dt

        # Zero-order hold, numerator of B_bar: exp(dt A) - 1 = e_real + i e_imag.
        # expm1 and 1 - cos = 2 sin^2 keep the precision for small dt
        exp_dt_a = torch.exp(dt_a)
        cos_dt_b = torch.cos(dt_b)
        sin_dt_b = torch.sin(dt_b)
        half_sin = torch.sin(0.5 * dt_b)
        e_real = torch.expm1(dt_a) * cos_dt_b - 2.0 * half_sin * half_sin
        e_imag = exp_dt_a * sin_dt_b

        # B_bar = (exp(dt A) - 1) / A with B = 1: a complex division, written
        # as a product with the conjugate of A over |A|^2 (> 0, since a < 0)
        modulus = a * a + b * b
        w_real = (e_real * a + e_imag * b) / modulus
        w_imag = (e_imag * a - e_real * b) / modulus
        # C B_bar, the coefficient of each state in the kernel (complex product)
        c_real, c_imag = self.C.unbind(-1)
        ct_real = c_real * w_real - c_imag * w_imag
        ct_imag = c_real * w_imag + c_imag * w_real

        # Vandermonde basis A_bar^l = exp(dt A l), l = 0..length - 1
        # (d_model, d_state // 2, length)
        steps = torch.arange(length, device=dt.device, dtype=dt.dtype)
        dt_a_steps = dt_a.unsqueeze(-1) * steps
        decay = torch.exp(dt_a_steps)
        angles = dt_b.unsqueeze(-1) * steps
        basis_real = decay * torch.cos(angles)
        basis_imag = decay * torch.sin(angles)

        # K_l = sum_n C_n B_bar_n A_bar_n^l: real part of the complex product,
        # summed over the states
        # (d_model, d_state // 2, length) -> (d_model, length)
        kernel = torch.einsum("hn,hnl->hl", ct_real, basis_real)
        kernel = kernel - torch.einsum("hn,hnl->hl", ct_imag, basis_imag)
        # Twice the real part: "same as adding conjugate pairs"
        kernel = 2.0 * kernel
        return kernel


class _S4DLayer(nn.Module):
    """S4D layer of Gu et al. (2022), as in the authors' ``S4D``.

    One diagonal state space model per channel, applied as a causal depthwise
    convolution with the :class:`_S4DKernel`, plus the skip term ``D u``,
    then ``activation``, dropout and a 1 x 1 convolution with a GLU.

    Parameters
    ----------
    d_model : int
        Number of S4D channels.
    d_state : int
        State size of each state space model.
    dt_min : float
        Lower end of the log-uniform initial range of the step ``dt``.
    dt_max : float
        Upper end of that range.
    a_real_init : float
        Initial value of ``-Re(A)`` for every state.
    drop_prob : float
        Dropout probability, over whole channels.
    activation : type[nn.Module]
        Activation after the state space model.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        dt_min: float,
        dt_max: float,
        a_real_init: float,
        drop_prob: float,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        # Skip term D of the state space model, y = C x + D u, one per channel
        self.D = nn.Parameter(torch.randn(d_model))
        # Impulse response of the state space model, as a convolution kernel
        self.kernel = _S4DKernel(
            d_model=d_model,
            d_state=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
            a_real_init=a_real_init,
        )
        self.activation = activation()
        self.dropout = nn.Dropout1d(drop_prob)
        # Output linear layer: a position-wise map to 2 d_model channels,
        # halved again by the gate of the GLU
        self.output_linear = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )
        # Weight layout of a depthwise conv1d: (out_channels, 1, length)
        self.to_conv_weight = Rearrange("d_model length -> d_model 1 length")

    def reset_parameters(self) -> None:
        # D only: the kernel is a submodule, and the model resets it itself
        nn.init.normal_(self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output ``(batch, d_model, length)`` of ``x`` of the same shape."""
        # x is the input u of the state space models: (batch, d_model, length)
        length = x.shape[-1]

        # Kernel K of each channel's state space model, as long as the sequence
        # (d_model, length)
        kernel = self.kernel(length)
        # conv1d computes a correlation: the kernel is reversed so that the
        # result is the convolution y = u * K
        kernel = kernel.flip(-1)
        # (d_model, length) -> (d_model, 1, length)
        kernel = self.to_conv_weight(kernel)

        # Causal: step l sees only the inputs up to l, so the sequence is
        # padded on the left
        # (batch, d_model, length) -> (batch, d_model, 2 * length - 1)
        padded = F.pad(x, (length - 1, 0))
        # y = u * K, one kernel per channel (depthwise)
        # (batch, d_model, 2 * length - 1) -> (batch, d_model, length)
        y = F.conv1d(padded, kernel, groups=x.shape[1])

        # Skip term D u of the state space model
        # (d_model,) -> (d_model, 1), broadcasts over the sequence
        skip = self.D.unsqueeze(-1)
        # (batch, d_model, length)
        skip = x * skip
        y = y + skip

        # Nonlinearity (GELU by default), then dropout of whole channels
        y = self.activation(y)
        y = self.dropout(y)

        # Output linear layer with its GLU gate
        # (batch, d_model, length) -> (batch, d_model, length)
        y = self.output_linear(y)
        return y
