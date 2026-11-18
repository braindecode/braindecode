# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: Apache-2.0
# Adapted from https://github.com/wzhlearning/fNIRSNet (Apache-2.0).
"""fNIRSNet: Rethinking Delayed Hemodynamic Responses for fNIRS Classification."""

from __future__ import annotations

from collections import OrderedDict

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one
from braindecode.modules import DepthwiseConv2d


class FNIRSNet(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""fNIRSNet from Wang et al. (2023).

    :bdg-success:`Convolution`

    .. versionadded:: 2.1

    .. figure:: https://braindecode.org/dev/_static/model/fnirsnet.png
       :align: center
       :alt: fNIRSNet architecture

       Fig. 2 of Wang et al. (2023), cropped
       (`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_): "(a)
       Overview architecture of fNIRSNet. (b) Schema of receptive fields.
       [...] (c) Depthwise separable convolution."

    From the paper :footcite:`wang2023fnirsnet`: "fNIRSNet consists of a
    delayed hemodynamic response module (DHR Module) and a global module.
    [...] Delayed hemodynamic response convolutions (DHRConv) extract the
    channel-level features of delayed hemodynamic responses. Depthwise
    separable convolutions consisting of depthwise convolutions (DWConv) and
    pointwise convolutions (PWConv) are used to reduce model parameters."

    The input is the paper's fNIRS tensor of :math:`C` rows (``n_chans``,
    "twice (two chromophores: HbO and HbR) the number of fNIRS channels") by
    :math:`S` sampling points (``n_times``). As in Fig. 2(a) and the authors'
    loaders, the HbO rows come first, then the HbR rows; ``chs_info`` is not
    read, so keep the row order used in training. The DHRConv kernel spans the
    window, so the model has no sampling-rate argument (the paper: 3 s at
    10 Hz, :math:`S = 30`, and at 13.3 Hz, :math:`S = 40`).

    Modules keep the authors' names: ``conv1`` (DHRConv, :math:`1 \times S`),
    ``bn1``, ``conv2.depth_conv`` (DWConv, :math:`C \times 1`, one filter per
    DHRConv map), ``conv2.point_conv`` (PWConv, :math:`F_1 \to F_2`), ``bn2``,
    sigmoid after each BN, and ``final_layer`` (FC). ``mapping`` renames their
    ``fc.*``, so an authors' ``state_dict`` loads with ``strict=True``; no
    weights are released. As in their code, the output is logits.

    The paper's two sizes (Section III-D), with the parameter counts of its
    Table III (this module gives the same counts):

    ===================  =======  ======  ======  ======  ======  ==========
    Protocol             Dataset  C       S       F1      F2      Parameters
    ===================  =======  ======  ======  ======  ======  ==========
    Subject-specific     MA       72      30      4       8       498
    Subject-specific     UFFT     40      40      4       8       419
    Subject-independent  MA       72      30      16      32      2,370
    Subject-independent  UFFT     40      40      16      32      2,051
    ===================  =======  ======  ======  ======  ======  ==========

    The authors' released ``LOSO_Train.py`` uses :math:`F_1 = 8`,
    :math:`F_2 = 16`.

    Parameters
    ----------
    n_dhr_filters : int, default=4
        DHRConv filters, :math:`F_1` (``num_DHRConv``). The paper's
        subject-specific value.
    n_pw_filters : int, default=8
        PWConv filters, the width of the global module, :math:`F_2`
        (``num_DWConv`` in the authors' code). The paper's subject-specific
        value.
    dhr_kernel_length : int or None, default=None
        DHRConv width :math:`k` (``DHRConv_width``), ``1 <= k <= S``. ``None``
        uses :math:`S`, the paper's Guideline 1. A shorter kernel, as in the
        paper's Table IV (:math:`k` = 10, 20, 30 on UFFT), gives
        :math:`F_2 (S - k + 1)` features to the FC layer.
    activation : type[nn.Module], default=nn.Sigmoid
        Activation after each BN (``act``). Guideline 3: activations "with
        saturated negative values" (sigmoid, tanh, ELU) work better.

    Notes
    -----
    Protocol (paper Section III; details from the authors' code): MA
    :footcite:`shin2017openaccess` (2 classes, 29 subjects, 10 Hz) and UFFT
    :footcite:`bak2019ufft` (3 classes, 30 subjects, 13.3 Hz); band-pass
    0.01-0.1 Hz (third-order Butterworth, zero-phase), baseline correction,
    3 s windows with a 1 s step, each window z-scored. Subject-specific:
    5-fold CV over contiguous trial blocks, AdamW (lr 0.001), label smoothing
    0.1, batch 64, 120 epochs, lr / 10 at epochs 60 and 90.
    Subject-independent: leave-one-subject-out, cosine lr over 30 epochs. Test
    metrics of the last epoch, averaged over folds and subjects.

    Replication on UFFT, accuracy / macro-F1 in %, the authors' training loops
    around this port and around their ``fNIRSNet`` (seeds 0-2). The LOSO runs
    use :math:`F_1 = 8`, :math:`F_2 = 16`, as the authors' ``LOSO_Train.py``;
    the paper's subject-independent row uses 16 and 32:

    ===================  ===============  =============  =============
    UFFT                 Paper, Table II  This port      Authors' code
    ===================  ===============  =============  =============
    Subject-specific     68.67 / 68.02    67.68 / 66.83  67.68 / 66.83
    Subject-independent  64.43 / 63.57    64.52 / 63.67  64.51 / 63.67
    ===================  ===============  =============  =============

    `License <https://github.com/wzhlearning/fNIRSNet/blob/11d9dad96308c65c039a0978b7eaf8a845065421/LICENSE>`_

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
        n_dhr_filters: int = 4,
        n_pw_filters: int = 8,
        dhr_kernel_length: int | None = None,
        activation: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        kernel_length = self.n_times if dhr_kernel_length is None else dhr_kernel_length
        if not 1 <= kernel_length <= self.n_times:
            raise ValueError(
                f"dhr_kernel_length must be in [1, n_times={self.n_times}], "
                f"got {dhr_kernel_length}."
            )
        self.to_fnirs_tensor = Rearrange("batch chans time -> batch 1 chans time")
        self.conv1 = nn.Conv2d(1, n_dhr_filters, (1, kernel_length))
        self.bn1 = nn.BatchNorm2d(n_dhr_filters)
        self.conv2 = nn.Sequential(
            OrderedDict(
                depth_conv=DepthwiseConv2d(n_dhr_filters, 1, (self.n_chans, 1)),
                point_conv=nn.Conv2d(n_dhr_filters, n_pw_filters, 1),
            )
        )
        self.bn2 = nn.BatchNorm2d(n_pw_filters)
        self.activation = activation()
        self.flatten = Rearrange("batch filters 1 time -> batch (filters time)")
        n_features = n_pw_filters * (self.n_times - kernel_length + 1)
        self.final_layer = nn.Linear(n_features, self.n_outputs)
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
        """Logits ``(batch, n_outputs)`` of ``x`` ``(batch, n_chans, n_times)``.

        ``return_features=True`` gives ``{"features", "cls_token": None}``; a
        ``ch_mask`` raises a ``ValueError`` (DWConv spans all channels).
        """
        if ch_mask is not None:
            raise ValueError("FNIRSNet does not accept ch_mask: DWConv spans n_chans.")
        # time' = n_times - dhr_kernel_length + 1 (1 with the default kernel)
        x = self.to_fnirs_tensor(x)  # (batch, 1, n_chans, n_times)
        x = self.conv1(x)  # DHRConv: (batch, n_dhr_filters, n_chans, time')
        x = self.bn1(x)  # (batch, n_dhr_filters, n_chans, time')
        x = self.activation(x)  # (batch, n_dhr_filters, n_chans, time')
        x = self.conv2.depth_conv(x)  # DWConv: (batch, n_dhr_filters, 1, time')
        x = self.conv2.point_conv(x)  # PWConv: (batch, n_pw_filters, 1, time')
        x = self.bn2(x)  # (batch, n_pw_filters, 1, time')
        x = self.activation(x)  # (batch, n_pw_filters, 1, time')
        features = self.flatten(x)  # (batch, n_pw_filters * time')
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)  # (batch, n_outputs)
