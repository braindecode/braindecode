# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: Apache-2.0
# Adapted from https://github.com/wzhlearning/fNIRS4D (Apache-2.0); S4D kernel after
# https://github.com/state-spaces/s4 (Apache-2.0).
"""fNIRS4D: diagonal state-space network for fNIRS classification."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    _disable_batch_norm_training_if_batch_size_one,
    fnirs_channel_order,
)


class FNIRS4D(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""fNIRS4D from Wang et al. (2026) [wang2026fnirs4d]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. versionadded:: 2.1

    With ``compute_hbt`` the HbR half of the montage becomes HbT = HbO + HbR. A
    full-window ``Conv2d`` maps the channels to a sequence processed by S4D-Lin
    layers (causal depthwise convolution, GLU, residual ``LayerNorm``); its mean
    joins a channel-average branch (full-window and smoothed temporal
    convolutions) before ``final_layer``. The S4D kernel is computed in real
    arithmetic (no complex tensors, no FFT); ``mapping`` loads ``fc.*`` weights.
    Adapted from the authors' code
    (`License <https://github.com/wzhlearning/fNIRS4D/blob/main/LICENSE>`_).

    Parameters
    ----------
    n_layers : int, default=4
        Number of S4D layers.
    d_state : int, default=64
        State dimension of each S4D kernel (even: conjugate pairs).
    avg_size : int, default=10
        Samples of the short moving average of the mean-hemoglobin branch,
        clamped to ``n_times``.
    drop_prob : float, default=0.5
        Dropout of the S4D layers and of their residual branch.
    prenorm : bool, default=False
        ``LayerNorm`` before each S4D layer instead of after the residual sum.
    compute_hbt : bool, default=True
        If ``n_chans`` is even, read the halves as (HbO, HbR) and use HbT.
    channel_order : {"stacked", "interleaved"} or None, default=None
        Pair order: ``[HbO..., HbR...]`` or ``[HbO_1, HbR_1, ...]`` (MNE);
        ``None`` reads it from ``chs_info``.
    activation, branch_activation : type[nn.Module], default=nn.GELU, nn.ELU
        Activations of the S4D layers and of the mean-hemoglobin branch.

    References
    ----------
    .. [wang2026fnirs4d] Wang, Z., Zhu, D., Zhang, J., Xiao, G., & Du, S.
       (2026). Domain Knowledge Fused State Space Model for fNIRS-Based
       Brain-Computer Interfaces. IEEE Transactions on Instrumentation and
       Measurement, 75, 4008016. doi:10.1109/TIM.2026.3693454
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
        n_layers: int = 4,
        d_state: int = 64,
        avg_size: int = 10,
        drop_prob: float = 0.5,
        prenorm: bool = False,
        compute_hbt: bool = True,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        activation: type[nn.Module] = nn.GELU,
        branch_activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        self.prenorm, self.compute_hbt = prenorm, compute_hbt
        c, t = self.n_chans, self.n_times
        pairs_info = self._chs_info if compute_hbt and c % 2 == 0 else None
        self.channel_order = fnirs_channel_order(pairs_info, channel_order)
        self.avg_size = min(avg_size, t)
        n_full = max(1, c // 2)
        n_short = max(1, c - n_full)

        def scale(n, k):  # temporal Conv2d over the channel mean, BN, activation
            return nn.Sequential(
                nn.Conv2d(1, n, (1, k)), nn.BatchNorm2d(n), branch_activation()
            )

        self.mean_hemoglobin_branch = nn.ModuleDict(
            {
                "full_scale": scale(n_full, t),
                "short_scale": scale(n_short, t - self.avg_size + 1),
            }
        )
        self.fNIRS_knowledge_embedding = nn.Sequential(
            nn.Conv2d(1, c, (1, t)),
            # a single channel has no instance statistics to normalize
            nn.InstanceNorm2d(c) if c > 1 else nn.Identity(),
            nn.Conv2d(c, c, 1),
        )
        self.s4d_layers, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.s4d_layers.append(_S4DLayer(c, d_state, drop_prob, activation))
            self.norms.append(nn.LayerNorm(c))
        self.dropout = nn.Dropout1d(drop_prob)
        self.final_layer = nn.Linear(c + n_full + n_short, self.n_outputs)
        self.mapping = {f"fc.{k}": f"final_layer.{k}" for k in ("weight", "bias")}

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
        if ch_mask is not None:
            raise ValueError("FNIRS4D does not accept ch_mask: layers span n_chans.")
        if self.compute_hbt and self.n_chans % 2 == 0:
            if self.channel_order == "interleaved":
                hbo, hbr = x[:, 0::2], x[:, 1::2]
            else:
                hbo, hbr = x.chunk(2, dim=1)
            x = torch.cat((hbo, hbo + hbr), dim=1)  # [HbO, HbT]
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        # nn.functional, not F: the decorator scripts forward with util's globals
        mean = nn.functional.avg_pool2d(x, (self.n_chans, 1), 1)  # (b, 1, 1, n_times)
        short = nn.functional.avg_pool2d(mean, (1, self.avg_size), 1)
        full = self.mean_hemoglobin_branch["full_scale"](mean).flatten(1)
        short = self.mean_hemoglobin_branch["short_scale"](short).flatten(1)
        mean_features = torch.cat((full, short), dim=1)  # (batch, n_full + n_short)
        x = self.fNIRS_knowledge_embedding(x)  # (batch, n_chans, n_chans, 1)
        # S4D runs along the input-channel axis: (batch, d_model=n_chans, L=n_chans)
        x = x.flatten(2).transpose(-1, -2)
        for layer, norm in zip(self.s4d_layers, self.norms):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            x = x + self.dropout(layer(z))
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        features = torch.cat((x.transpose(-1, -2).mean(dim=1), mean_features), 1)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)


class _S4DKernel(nn.Module):
    """S4D-Lin kernel ``2 Re(C (exp(dt A) - 1) / A exp(dt A l))`` in real arithmetic."""

    def __init__(self, d_model: int, N: int) -> None:
        super().__init__()
        self.C = nn.Parameter(torch.empty(d_model, N // 2, 2))
        self.log_dt = nn.Parameter(torch.empty(d_model))
        self.log_A_real = nn.Parameter(torch.empty(d_model, N // 2))
        self.A_imag = nn.Parameter(torch.empty(d_model, N // 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.C, 0.0, math.sqrt(0.5))
        nn.init.uniform_(self.log_dt, math.log(0.001), math.log(0.1))  # reference dt
        nn.init.constant_(self.log_A_real, math.log(0.5))
        with torch.no_grad():
            self.A_imag.copy_(
                math.pi * torch.arange(self.A_imag.shape[1]).to(self.A_imag)
            )

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (H, 1)
        a_r, a_i = -torch.exp(self.log_A_real), self.A_imag  # (H, N/2)
        dar, dai = a_r * dt, a_i * dt
        exp_dar, cos_dai, sin_dai = torch.exp(dar), torch.cos(dai), torch.sin(dai)
        half_sin = torch.sin(0.5 * dai)
        # exp(dt A) - 1, with expm1 for the real part
        e_r = torch.expm1(dar) * cos_dai - 2.0 * half_sin * half_sin
        e_i = exp_dar * sin_dai
        denom = (a_r * a_r + a_i * a_i).clamp_min(1e-12)
        w_r = (e_r * a_r + e_i * a_i) / denom  # (exp(dt A) - 1) / A
        w_i = (e_i * a_r - e_r * a_i) / denom
        c_r, c_i = self.C[..., 0], self.C[..., 1]
        ct_r, ct_i = c_r * w_r - c_i * w_i, c_r * w_i + c_i * w_r
        steps = torch.arange(L, device=dt.device, dtype=dt.dtype)
        exp_r = torch.exp(dar.unsqueeze(-1) * steps)  # (H, N/2, L)
        angles = dai.unsqueeze(-1) * steps
        basis_r, basis_i = exp_r * torch.cos(angles), exp_r * torch.sin(angles)
        return 2.0 * (
            torch.einsum("hn,hnl->hl", ct_r, basis_r)
            - torch.einsum("hn,hnl->hl", ct_i, basis_i)
        )  # (H, L)


class _S4DLayer(nn.Module):
    """S4D layer: causal depthwise convolution with the S4D kernel, D skip, GLU."""

    def __init__(
        self, d_model: int, d_state: int, drop_prob: float, activation: type
    ) -> None:
        super().__init__()
        self.D = nn.Parameter(torch.empty(d_model))
        self.kernel = _S4DKernel(d_model, d_state)
        self.activation = activation()
        self.dropout = nn.Dropout1d(drop_prob)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=-2)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.D)
        self.kernel.reset_parameters()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        L = u.shape[-1]  # u: (batch, H, L)
        kernel = self.kernel(L).flip(-1).unsqueeze(1)  # (H, 1, L)
        conv = F.conv1d(F.pad(u, (L - 1, 0)), kernel, groups=u.shape[1])
        y = conv + u * self.D.unsqueeze(-1)  # (batch, H, L), causal
        return self.output_linear(self.dropout(self.activation(y)))
