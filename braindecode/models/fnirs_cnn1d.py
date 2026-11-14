# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3-Clause
# Adapted from https://github.com/GeoNIRS/GeoNIRS (DirectEventCNN1DClassifier;
# 81f8995; no licence file; paper arXiv:2610.17845).
from __future__ import annotations

from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin


class FNIRSCNN1D(EEGModuleMixin, nn.Module):
    r"""FNIRSCNN1D from Trakoolwilaiwan et al. (2017) [Trakoolwilaiwan2017]_.

    :bdg-success:`Convolution`

    .. versionadded:: 2.1

    Two-stage 1D temporal CNN for fNIRS classification, as the
    ``DirectEventCNN1DClassifier`` baseline of GeoNIRS [Li2026]_: ``Conv1d``,
    ``GroupNorm``, activation and ``MaxPool1d(2)``; ``Conv1d``, ``GroupNorm``,
    activation and a global time average; dropout and a linear head. NaN samples
    are read as zero.

    Parameters
    ----------
    conv_dim : int, default=64
        Channels of the first convolution.
    embed_dim : int, default=128
        Channels of the second convolution, the feature size.
    first_kernel_size, second_kernel_size : int, default=7, 5
        Kernel sizes of the two convolutions.
    n_groups : int, default=8
        Groups of both ``GroupNorm`` layers.
    drop_prob : float, default=0.5
        Dropout probability before the head.
    activation : type[nn.Module], default=nn.GELU
        Activation after each ``GroupNorm``.

    References
    ----------
    .. [Trakoolwilaiwan2017] Trakoolwilaiwan, T., et al. (2017). Convolutional
       neural network for high-accuracy functional near-infrared spectroscopy in a
       brain-computer interface. Neurophotonics, 5(1), 011008.
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
        conv_dim: int = 64,
        embed_dim: int = 128,
        first_kernel_size: int = 7,
        second_kernel_size: int = 5,
        n_groups: int = 8,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        k1, k2 = first_kernel_size, second_kernel_size
        self.temporal_features = nn.Sequential(
            nn.Conv1d(self.n_chans, conv_dim, k1, padding=k1 // 2),
            nn.GroupNorm(n_groups, conv_dim),
            activation(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            nn.Conv1d(conv_dim, embed_dim, k2, padding=k2 // 2),
            nn.GroupNorm(n_groups, embed_dim),
            activation(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(drop_prob),
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(
        self, x: Tensor, return_features: bool = False, ch_mask: Tensor | None = None
    ) -> Tensor | dict[str, Tensor | None]:
        if ch_mask is not None:
            raise ValueError(
                "FNIRSCNN1D does not accept ch_mask: it mixes all channels."
            )
        # (batch, n_chans, n_times) -> (batch, embed_dim); NaN samples -> 0
        features = self.temporal_features(x.nan_to_num())
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)  # (batch, n_outputs)
