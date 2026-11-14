# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3-Clause
# Implemented from Naseer et al. (2016) and Glover (1999).
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin


class FNIRSStatNet(EEGModuleMixin, nn.Module):
    r"""FNIRSStatNet from Naseer et al. (2016) [Naseer2016]_.

    :bdg-warning:`Interpretability` :bdg-dark-line:`Channel`

    .. versionadded:: 2.1

    Differentiable classical fNIRS features, as the GeoNIRS ``MeanSlopeLDA`` and
    ``StatisticalRBFSVM`` baselines [Li2026]_: per channel and sub-window, mean and
    least-squares slope [Naseer2016]_, optionally std, min, max, RMS, skewness and
    excess kurtosis; projections of the mean-removed window on the unit-norm
    double-gamma HRF [Glover1999]_ and its time derivative; optional depthwise
    learned filters averaged over time. The features go through ``LayerNorm``, an
    optional MLP and a linear head. The HRF time axis uses ``sfreq``, or 10 Hz when
    it is unknown. NaN samples are read as zero.

    Parameters
    ----------
    n_windows : int, default=1
        Equal sub-windows for the statistics.
    include_higher_moments : bool, default=True
        Add std, min, max, RMS, skewness and kurtosis to the mean and slope.
    include_hrf : bool, default=True
        Add the projections on the HRF and its derivative.
    n_learned_filters : int, default=2
        Depthwise filters per channel (kernel ``min(n_times, 15)``, at least 3).
    hidden_dim : int or None, default=64
        Width of the readout MLP; ``None`` or ``0`` gives a linear readout.
    drop_prob : float, default=0.25
        Dropout probability before the head.
    activation : type[nn.Module], default=nn.GELU
        Activation after the filters and in the MLP.

    References
    ----------
    .. [Naseer2016] Naseer, N., et al. (2016). Determining optimal
       feature-combination for LDA classification of functional near-infrared
       spectroscopy signals in brain-computer interface application. Frontiers in
       Human Neuroscience, 10, 237.
    .. [Glover1999] Glover, G. H. (1999). Deconvolution of impulse response in
       event-related BOLD fMRI. NeuroImage, 9(4), 416-429.
    .. [Li2026] Li, Z., et al. (2026). GeoNIRS: Geometry-aware pretraining for
       heterogeneous fNIRS representation learning. arXiv:2610.17845.
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
        n_windows: int = 1,
        include_higher_moments: bool = True,
        include_hrf: bool = True,
        n_learned_filters: int = 2,
        hidden_dim: int | None = 64,
        drop_prob: float = 0.25,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if n_windows < 1:
            raise ValueError(f"n_windows must be positive, got {n_windows}.")
        self.n_windows = n_windows
        self.n_stats = 8 if include_higher_moments else 2
        self.include_hrf = include_hrf
        windowed = None not in (self._n_times, self._input_window_seconds)
        known = self._sfreq is not None or windowed
        self.hrf_sfreq = float(self.sfreq) if known else 10.0
        c = self.n_chans
        n_feat = c * (n_windows * self.n_stats + 2 * include_hrf + n_learned_filters)
        self.matched_filter = None
        if n_learned_filters > 0:
            k = max(3, min(self.n_times, 15))  # depthwise FIR length
            self.matched_filter = nn.Conv1d(
                c, c * n_learned_filters, k, padding=k // 2, groups=c
            )
            self.act = activation()
        self.feature_norm = nn.LayerNorm(n_feat)
        self.projector = nn.Dropout(drop_prob)
        if hidden_dim:
            self.projector = nn.Sequential(
                nn.Linear(n_feat, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation(),
                nn.Dropout(drop_prob),
            )
        self.final_layer = nn.Linear(hidden_dim or n_feat, self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def extract_channel_statistics(self, x: Tensor) -> Tensor:
        """Per-channel features: (batch, n_chans, n_times) -> (batch, n_feat)."""
        batch_size, n_chans, n_times = x.shape
        n_win, x_win = self.n_windows, x
        if n_win > 1 and n_times < 2 * n_win:  # at least two samples per window
            x_win = F.interpolate(x, size=2 * n_win, mode="linear", align_corners=False)
        bounds = [(w * x_win.shape[-1]) // n_win for w in range(1, n_win)]
        parts: list[Tensor] = []
        segments = x_win.tensor_split(bounds, dim=-1) if n_win > 1 else [x]
        for seg in segments:  # (batch, n_chans, length)
            mean = seg.mean(dim=-1)
            centered = seg - mean.unsqueeze(-1)
            t = torch.linspace(-1.0, 1.0, seg.shape[-1], device=x.device, dtype=x.dtype)
            slope = (centered * t).sum(dim=-1) / (t * t).sum()
            if seg.shape[-1] == 1:  # one sample has no slope
                slope = torch.zeros_like(mean)
            var = (centered * centered).mean(dim=-1)
            std = torch.sqrt(var.clamp_min(1e-24))
            rms = torch.sqrt((seg * seg).mean(dim=-1).clamp_min(1e-24))
            minimum, maximum = seg.amin(dim=-1), seg.amax(dim=-1)
            # flat channels: zero skewness and kurtosis; the 1e-24 floors keep the
            # moments of molar-scale (1e-6) signals exact
            active = var > 1e-20
            z = torch.where(
                active.unsqueeze(-1),
                centered / torch.where(active, std, 1.0).unsqueeze(-1),
                0.0,
            )
            skew = z.pow(3).mean(dim=-1)
            kurt = torch.where(active, z.pow(4).mean(dim=-1) - 3.0, 0.0)
            stats = [mean, slope, std, minimum, maximum, rms, skew, kurt]
            parts.append(torch.stack(stats[: self.n_stats], dim=-1))
        if self.include_hrf:
            # canonical double-gamma HRF (peak 6 s, undershoot 16 s), unit norm
            t = torch.arange(n_times, device=x.device, dtype=x.dtype) / self.hrf_sfreq
            t = t.clamp_min(1e-4)
            log_t = torch.log(t)
            hrf = (
                torch.exp(5.0 * log_t - t - math.lgamma(6.0))
                - torch.exp(15.0 * log_t - t - math.lgamma(16.0)) / 6.0
            )
            hrf = hrf / hrf.norm(p=2).clamp_min(1e-6)
            d_hrf = torch.gradient(hrf)[0] if n_times > 1 else torch.zeros_like(hrf)
            d_hrf = d_hrf / d_hrf.norm(p=2).clamp_min(1e-6)
            centered = x - x.mean(dim=-1, keepdim=True)
            # (batch, n_chans, n_times) @ (n_times,) -> (batch, n_chans, 2)
            parts.append(torch.stack((centered @ hrf, centered @ d_hrf), dim=-1))
        if self.matched_filter is not None:
            # (batch, n_chans * n_learned_filters, n_times) -> time mean
            mf = self.act(self.matched_filter(x)).mean(dim=-1)
            parts.append(mf.view(batch_size, n_chans, -1))
        # (batch, n_chans, n_feat / n_chans) -> (batch, n_feat)
        return torch.cat(parts, dim=-1).flatten(1)

    def forward(
        self, x: Tensor, return_features: bool = False, ch_mask: Tensor | None = None
    ) -> Tensor | dict[str, Tensor | None]:
        if ch_mask is not None:
            raise ValueError(
                "FNIRSStatNet does not accept ch_mask: its head spans all channels."
            )
        x = x.nan_to_num()  # NaN samples -> 0
        # (batch, n_feat) -> (batch, hidden_dim or n_feat)
        features = self.projector(self.feature_norm(self.extract_channel_statistics(x)))
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)  # (batch, n_outputs)
