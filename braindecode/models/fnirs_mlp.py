# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin


class FNIRSMLP(EEGModuleMixin, nn.Module):
    r"""FNIRSMLP from Hennrich et al. (2015) [Hennrich2015]_.

    :bdg-dark-line:`Channel`

    .. versionadded:: 2.1

    Fully connected baseline for fNIRS decoding [Hennrich2015]_ [Chiarelli2018]_:
    each channel is averaged into ``n_time_bins`` time bins (at most ``n_times``),
    the bins are flattened, projected to ``hidden_dim`` and passed through pre-norm
    residual dense stages, a ``LayerNorm`` and a linear head. NaN samples are read
    as zero.

    Parameters
    ----------
    n_time_bins : int, default=16
        Time bins per channel.
    hidden_dim : int, default=128
        Width of the dense layers.
    n_layers : int, default=2
        Dense layers: the input projection and ``n_layers - 1`` residual stages.
    drop_prob : float, default=0.3
        Dropout probability of each dense layer.
    activation : type[nn.Module], default=nn.GELU
        Activation of each dense layer.

    References
    ----------
    .. [Hennrich2015] Hennrich, J., Herff, C., Heger, D., & Schultz, T. (2015).
       Investigating deep learning for fNIRS based BCI. EMBC, 2844-2847.
    .. [Chiarelli2018] Chiarelli, A. M., et al. (2018). Deep learning for hybrid
       EEG-fNIRS brain-computer interface: application to motor imagery
       classification. Journal of Neural Engineering, 15(3), 036028.
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
        n_time_bins: int = 16,
        hidden_dim: int = 128,
        n_layers: int = 2,
        drop_prob: float = 0.3,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if min(n_time_bins, n_layers) < 1:
            raise ValueError("n_time_bins and n_layers must be positive.")
        n_time_bins = min(n_time_bins, self.n_times)
        n_in = self.n_chans * n_time_bins
        self.temporal_pool = nn.AdaptiveAvgPool1d(n_time_bins)
        self.input_proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(n_in),
            nn.Linear(n_in, hidden_dim),
            activation(),
            nn.Dropout(drop_prob),
        )
        self.stages = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Dropout(drop_prob),
            )
            for _ in range(n_layers - 1)
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(
        self, x: Tensor, return_features: bool = False, ch_mask: Tensor | None = None
    ) -> Tensor | dict[str, Tensor | None]:
        if ch_mask is not None:
            raise ValueError("FNIRSMLP does not accept ch_mask: it mixes all channels.")
        # (batch, n_chans, n_times) -> (batch, n_chans, n_time_bins); NaN samples -> 0
        h = self.input_proj(self.temporal_pool(x.nan_to_num()))  # (batch, hidden_dim)
        for stage in self.stages:
            h = h + stage(h)
        features = self.post_norm(h)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)  # (batch, n_outputs)
