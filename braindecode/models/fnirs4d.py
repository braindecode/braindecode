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
       resized: "fNIRS4D for Signal Classification", with the "Selectable
       Branches" MAMHB (inter-subject) and MHB (intra-subject).

    From the abstract :footcite:`wang2026fnirs4d`: "Diagonally structured
    state space sequence model (S4D), simple and efficient diagonal versions
    of SSMs, are introduced to the fNIRS domain. We incorporate domain
    knowledge into an S4D-based lightweight fNIRS classification model named
    fNIRS4D." The S4D layer is that of Gu et al. :footcite:`gu2022s4d`.

    The layers follow the authors' code (``fNIRS4D.py``); in the figure's terms:

    - "Mixed Hemoglobin" input: ``n_chans`` rows (HbO and HbR per fNIRS
      channel) by ``n_times``. ``compute_hbt=True`` makes the rows
      [HbO, HbT = HbO + HbR], as the authors' data loader does; the HbO/HbR
      pairing comes from ``channel_order`` or ``chs_info``. The authors sum
      before their z-score; to reproduce it on their arrays, pass
      [HbO, HbT] with ``compute_hbt=False``.
    - "Knowledge Embedding (KE)", ``fNIRS_knowledge_embedding``: a full-window
      convolution with ``n_chans`` filters, instance norm and a 1x1
      convolution; each row becomes a sequence over the filters.
    - "Knowledge Aggregation (KA)": ``n_layers`` S4D blocks (``s4d_layers``,
      ``norms``: S4D convolution, ``D`` skip, GELU, dropout, GLU, dropout,
      residual, ``LayerNorm``), then the mean over the sequence.
    - "Mean Hemoglobin Branch (MHB)" is ``mean_branch="time_mean"``; the
      "Multi-scale Aggregated" MAMHB is ``"multiscale"`` (``avg_1``,
      ``avg_2``, ``conv_1``, ``conv_2``, BN, ELU). ``final_layer`` (FC) reads both.

    The names are the authors'; ``mapping`` renames their ``fc.0.*``, so an
    authors' ``state_dict`` loads with ``strict=True``. No weights are
    released. There is no sampling-rate argument (the authors: 3 s windows,
    40 samples for UFFT, 30 for MA and WG). The S4D kernel is a causal
    convolution in real arithmetic (the authors: complex FFT), equal up to
    float32 rounding and free of complex tensors (``torch.jit.script`` and
    ``torch.export`` run on it).

    The authors' configurations (``KFold_Train.py``, ``LOSO_Train.py``):

    ===================  =======  =======  =======  ========  ================
    Protocol             Dataset  n_chans  n_times  n_layers  mean_branch
    ===================  =======  =======  =======  ========  ================
    Subject-specific     UFFT     40       40       4         ``"time_mean"``
    Subject-specific     MA, WG   72       30       4         ``"time_mean"``
    Subject-independent  UFFT     40       40       4         ``"multiscale"``
    Subject-independent  MA       72       30       4         ``"multiscale"``
    Subject-independent  WG       72       30       1         ``"multiscale"``
    ===================  =======  =======  =======  ========  ================

    Parameters
    ----------
    n_layers : int, default=4
        Number of S4D blocks (the figure's :math:`N`, ``n_layers``).
    d_state : int, default=64
        State size of each S4D kernel (``d_state``), even: the kernel keeps
        ``d_state // 2`` conjugate pairs.
    avg_size : int, default=10
        Moving-average width of the ``"multiscale"`` branch (``avg_size_2``),
        at most ``n_times``.
    drop_prob : float, default=0.5
        Dropout in and after each S4D layer (``dropout``).
    prenorm : bool, default=False
        ``LayerNorm`` before each S4D layer instead of after the residual
        (``prenorm``; the authors' default is post-norm).
    compute_hbt : bool, default=True
        Replace HbR by HbT = HbO + HbR (even ``n_chans``); ``False`` takes the
        rows as they are.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO..., HbR...]`` (the authors') or ``[HbO_1, HbR_1, ...]`` (MNE);
        ``None`` reads ``chs_info``, else assumes ``"stacked"``. Ignored when
        ``compute_hbt=False``.
    mean_branch : {"multiscale", "time_mean"}, default="multiscale"
        The authors' ``mode``: ``"multiscale"`` is MAMHB (``"LOSO-CV"``,
        needs an even ``n_chans``), ``"time_mean"`` is MHB (``"KFold-CV"``).
    activation : type[nn.Module], default=nn.GELU
        Activation inside each S4D layer.
    branch_activation : type[nn.Module], default=nn.ELU
        Activation of the ``"multiscale"`` branch.

    Notes
    -----
    Protocol of the authors' released code (the TIM paper is not open
    access): UFFT :footcite:`bak2019ufft`, MA :footcite:`shin2017openaccess`
    and WG :footcite:`shin2018simultaneous`; their BBCI scripts (UFFT and MA
    as for :class:`~braindecode.models.FNIRSNet`, WG low-pass 0.2 Hz), 3 s
    windows with a 1 s step, HbT per window, then a z-score with the
    training-set mean and std. Subject-specific: 5-fold CV over trial blocks,
    AdamW (lr 0.001 UFFT, 0.0005 MA/WG), cross-entropy, batch 64, 120 epochs,
    lr times 0.2 at epochs 20, 40, 60. Subject-independent:
    leave-one-subject-out, batch 512, cosine lr over 30 epochs. Last-epoch
    test accuracy.

    Replication in the authors' training loops, on the same UFFT and WG arrays
    (a Python re-implementation of their BBCI preprocessing), splits and seeds
    (0-2): accuracy in %, mean over subjects; paired difference (port minus
    authors') in points with its 95 % interval, all including zero (not
    resolved).

    ==========  =========  =============  ====================
    Cell        This port  Authors' code  Paired diff [95% CI]
    ==========  =========  =============  ====================
    UFFT KFold  71.98      72.00          -0.01 [-0.49, +0.46]
    UFFT LOSO   65.81      65.85          -0.04 [-0.32, +0.24]
    WG KFold    61.54      61.33          +0.21 [-0.50, +0.91]
    WG LOSO     61.92      62.00          -0.07 [-0.54, +0.39]
    ==========  =========  =============  ====================

    GeoNIRS benchmark :footcite:`li2026geonirs`, against its copy of fNIRS4D
    in the same loop (macro-F1 in %, 12 s windows, subject-level 5-fold CV,
    seeds 810 and 811, ``"multiscale"``):

    =====================  ================  =========  ============
    Task                   GeoNIRS Table 14  This port  GeoNIRS copy
    =====================  ================  =========  ============
    Action mode (DS42)     68.98             67.89      69.03
    Pain intensity (DS19)  50.79             49.86      50.60
    =====================  ================  =========  ============

    `License <https://github.com/wzhlearning/fNIRS4D/blob/eb13e5e5b47043ad408d8a50b3ec9fe079926c12/LICENSE>`_

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
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
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        n_chans = self.n_chans
        n_times = self.n_times
        if compute_hbt and n_chans % 2:
            raise ValueError(f"compute_hbt needs HbO/HbR pairs, got n_chans={n_chans}.")
        if n_chans < 2:  # the instance norm needs more than one row
            raise ValueError(f"FNIRS4D needs n_chans >= 2, got n_chans={n_chans}.")
        self.prenorm = prenorm
        self.compute_hbt = compute_hbt
        pairs_info = self._chs_info if compute_hbt else None
        self.channel_order = fnirs_channel_order(pairs_info, channel_order)
        self.to_fnirs_tensor = Rearrange("batch chans time -> batch 1 chans time")
        self.mean_hemoglobin_branch: nn.Module
        if mean_branch == "multiscale":
            if n_chans % 2:  # n_chans // 2 filters per scale
                raise ValueError(
                    f"mean_branch='multiscale' needs an even n_chans, got {n_chans}."
                )
            if avg_size > n_times:
                raise ValueError(
                    f"avg_size must be at most n_times={n_times}, got {avg_size}."
                )
            self.mean_hemoglobin_branch = _MultiscaleMeanHemoglobinBranch(
                n_chans, n_times, avg_size, branch_activation
            )
        elif mean_branch == "time_mean":
            self.mean_hemoglobin_branch = nn.Sequential(
                nn.AvgPool2d((1, n_times), 1),
                Rearrange("batch 1 chans 1 -> batch chans"),
            )
        else:
            raise ValueError(
                f"mean_branch must be 'multiscale' or 'time_mean', got {mean_branch!r}."
            )
        self.fNIRS_knowledge_embedding = nn.Sequential(
            nn.Conv2d(1, n_chans, (1, n_times)),
            nn.InstanceNorm2d(n_chans),
            nn.Conv2d(n_chans, n_chans, 1),
            # one S4D sequence per channel, along the filters
            Rearrange("batch filters chans 1 -> batch chans filters"),
        )
        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.s4d_layers.append(_S4DLayer(n_chans, d_state, drop_prob, activation))
            self.norms.append(nn.LayerNorm(n_chans))
        self.dropout = nn.Dropout1d(drop_prob)
        # LayerNorm and the mean read the channel axis last
        self.channels_last = Rearrange("batch chans filters -> batch filters chans")
        self.channels_first = Rearrange("batch filters chans -> batch chans filters")
        # n_chans S4D features and n_chans mean-hemoglobin features
        self.final_layer = nn.Linear(2 * n_chans, self.n_outputs)
        self.mapping = {f"fc.0.{k}": f"final_layer.{k}" for k in ("weight", "bias")}

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
        """Logits ``(batch, n_outputs)`` of ``x`` ``(batch, n_chans, n_times)``.

        ``return_features=True`` gives ``{"features", "cls_token": None}``; a
        ``ch_mask`` raises a ``ValueError`` (the layers span all channels).
        """
        if ch_mask is not None:
            raise ValueError("FNIRS4D does not accept ch_mask: layers span n_chans.")
        if self.compute_hbt:
            # 2 x (batch, n_chans // 2, n_times)
            hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
            hbt = hbo + hbr  # (batch, n_chans // 2, n_times)
            x = torch.cat((hbo, hbt), dim=1)  # [HbO, HbT]: (batch, n_chans, n_times)
        x = self.to_fnirs_tensor(x)  # (batch, 1, n_chans, n_times)
        mean_features = self.mean_hemoglobin_branch(x)  # (batch, n_mean)
        x = self.fNIRS_knowledge_embedding(x)  # (batch, n_chans, filters = n_chans)
        for layer, norm in zip(self.s4d_layers, self.norms):
            layer_input = x  # (batch, n_chans, filters)
            if self.prenorm:
                layer_input = self.channels_last(layer_input)
                layer_input = norm(layer_input)  # (batch, filters, n_chans)
                layer_input = self.channels_first(layer_input)
            layer_output = layer(layer_input)  # (batch, n_chans, filters)
            layer_output = self.dropout(layer_output)
            x = x + layer_output  # (batch, n_chans, filters)
            if not self.prenorm:
                x = self.channels_last(x)
                x = norm(x)  # (batch, filters, n_chans)
                x = self.channels_first(x)
        x = self.channels_last(x)  # (batch, filters, n_chans)
        # Tensor.mean: a Reduce layer would permute first and change the
        # float32 rounding of the gradients
        x = x.mean(dim=1)  # (batch, n_chans)
        features = torch.cat((x, mean_features), 1)  # (batch, 2 n_chans)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)  # (batch, n_outputs)


class _MultiscaleMeanHemoglobinBranch(nn.Module):
    """The authors' subject-independent (LOSO-CV) mean hemoglobin branch: the
    channel mean and its moving average, each through a full-window temporal
    ``Conv2d``, batch normalization and ``activation``."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        avg_size: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        n_filters = n_chans // 2  # per scale, as the authors'
        self.avg_1 = nn.AvgPool2d((n_chans, 1), 1)
        self.avg_2 = nn.AvgPool2d((1, avg_size), 1)
        self.conv_1 = nn.Conv2d(1, n_filters, (1, n_times))
        self.bn_1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(1, n_filters, (1, n_times - avg_size + 1))
        self.bn_2 = nn.BatchNorm2d(n_filters)
        self.activation = activation()
        self.flatten = Rearrange("batch filters 1 1 -> batch filters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # time' = n_times - avg_size + 1
        mean = self.avg_1(x)  # channel mean: (batch, 1, 1, n_times)
        short = self.avg_2(mean)  # moving average: (batch, 1, 1, time')
        full = self.conv_1(mean)  # (batch, n_chans // 2, 1, 1)
        full = self.bn_1(full)  # (batch, n_chans // 2, 1, 1)
        full = self.activation(full)  # (batch, n_chans // 2, 1, 1)
        short = self.conv_2(short)  # (batch, n_chans // 2, 1, 1)
        short = self.bn_2(short)  # (batch, n_chans // 2, 1, 1)
        short = self.activation(short)  # (batch, n_chans // 2, 1, 1)
        features = torch.cat((full, short), dim=1)  # (batch, n_chans, 1, 1)
        return self.flatten(features)  # (batch, n_chans)


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
