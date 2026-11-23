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
       (`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_).

    fNIRSNet :footcite:`wang2023fnirsnet` classifies HbO/HbR windows with
    three convolutions and one linear layer, without pooling or dropout.
    Delayed hemodynamic responses are slow, so the first convolution spans the
    whole window of one row and the second spans all rows. With 4 and 8
    filters it has 498 parameters on mental arithmetic (Table III).

    .. rubric:: Architecture Overview

    It has a delayed hemodynamic response module (DHR Module) and a global
    module (Section II-B, Table I). With :math:`C` = ``n_chans`` rows,
    :math:`S` = ``n_times`` samples, :math:`F_1` and :math:`F_2` filters, and
    :math:`S' = S - k + 1` (1 for the paper's kernel width :math:`k = S`)::

        x                                       (batch, C, S)
        - to_fnirs_tensor                       (batch, 1, C, S)
        - conv1 (1 x k), bn1, act1              (batch, F_1, C, S')
        - conv2.depth_conv (C x 1)              (batch, F_1, 1, S')
        - conv2.point_conv (1 x 1), bn2, act2   (batch, F_2, 1, S')
        - flatten, final_layer                  (batch, n_outputs)

    .. rubric:: Macro Components

    - ``conv1``, ``bn1``, ``act1`` **DHR Module.** The delayed hemodynamic
      response convolution (DHRConv) is a standard convolution with
      :math:`F_1` filters of :math:`1 \times k`: one filter reads the whole
      slow response of one row (Guideline 1), and it is not depthwise
      separable, which performs poorly in low dimension (Guideline 2). Batch
      normalization and a saturating activation follow (Guideline 3: sigmoid,
      tanh and ELU beat ReLU and leaky ReLU in Table VI; the paper uses the
      sigmoid).
    - ``conv2``, ``bn2``, ``act2`` **Global Module.** A depthwise separable
      convolution: ``depth_conv`` has one :math:`C \times 1` filter per DHR
      map, a global receptive field over the rows without stacking layers, and
      ``point_conv`` projects the :math:`F_1` maps onto :math:`F_2`. It costs
      :math:`F_1 (C + F_2)` multiply-adds per output position instead of the
      :math:`F_1 C F_2` of a standard convolution (Eqs. (2) and (3)), so a
      longer input only adds to the cost of DHRConv. Batch normalization and
      the activation follow. Table I lists :math:`F_2` filters for DWConv, but
      the authors' ``DWSConv`` has :math:`F_1` depthwise filters and an
      :math:`F_1 \to F_2` PWConv; this module follows the code, which gives
      Table III's parameter counts (498 and 419 subject-specific, 2,370 and
      2,051 subject-independent).
    - ``final_layer`` **Classifier.** One linear layer on the flattened maps.
      The paper's softmax is left to the loss, so the model returns logits.
      The modules keep the authors' names (``conv1``, ``bn1``,
      ``conv2.depth_conv``, ``conv2.point_conv``, ``bn2``) except ``fc``, now
      ``final_layer``.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal.** One window of :math:`S` samples, 3 s in the paper (30 at
      10 Hz, 40 at 13.3 Hz). DHRConv covers all of it, and the kernel counts
      samples, so the model takes no sampling rate.
    - **Spatial.** No montage or optode geometry: DWConv has one weight per
      row, so the rows keep the order of training. ``chs_info`` is not read.
    - **Spectral.** None in the model; the paper band-passes the concentration
      changes to 0.01-0.1 Hz beforehand (third-order Butterworth, zero-phase),
      corrects the baseline and z-scores each 3 s window, cut with a 1 s step
      (Section III-B).
    - **Chromophores.** :math:`C` is twice the number of fNIRS channels (HbO
      and HbR, Section II-B); as in Fig. 2(a) and the authors' loaders, all
      HbO rows come first, then all HbR rows.

    .. rubric:: Additional Mechanisms

    - **Reference.** In the authors' training loops on UFFT (seeds 0-2) the
      port and the authors' ``fNIRSNet`` agree within 0.01 point of accuracy
      and macro-F1 (:gh:`1579`); the subject-independent runs use the
      :math:`F_1 = 8`, :math:`F_2 = 16` of the authors' ``LOSO_Train.py``, not
      the paper's 16 and 32.
    - **Pretrained weights.** None: the authors release code, not weights, and
      the paper trains one model per subject or per left-out subject. There is
      no ``ch_mask``.

    `License <https://github.com/wzhlearning/fNIRSNet/blob/11d9dad96308c65c039a0978b7eaf8a845065421/LICENSE>`_

    Parameters
    ----------
    n_dhr_filters : int, default=4
        :math:`F_1`, the DHRConv filters (``num_DHRConv`` in the reference): 4
        for subject-specific and 16 for subject-independent training in the
        paper (Section III-D).
    n_pw_filters : int, default=8
        :math:`F_2`, the PWConv filters (``num_DWConv`` in the reference): 8
        and 32 in the paper, which recommends at least :math:`2 F_1` (Section
        IV-F).
    dhr_kernel_length : int or None, default=None
        DHRConv width :math:`k` (``DHRConv_width`` in the reference),
        ``1 <= k <= n_times``; ``None`` uses :math:`S` (Guideline 1). A shorter
        kernel is an addition to the paper's linear layer, which fits only
        :math:`k = S` (Table IV tries 10, 20 and 30 on UFFT): the layer then
        reads :math:`F_2 (S - k + 1)` values.
    activation : type[nn.Module], default=nn.Sigmoid
        Activation after each batch normalization (``act`` in the reference).

    References
    ----------
    .. footbibliography::
    """

    _time_axis = "fixed"  # fixed: conv1.weight
    _channel_axis = "fixed"  # fixed: conv2.depth_conv.weight

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
        n_dhr_filters: int = 4,
        n_pw_filters: int = 8,
        dhr_kernel_length: int | None = None,
        activation: type[nn.Module] = nn.Sigmoid,
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
        # Guideline 1: the DHRConv kernel is as wide as the window (k = S)
        kernel_length = self.n_times if dhr_kernel_length is None else dhr_kernel_length
        # Samples left after DHRConv, S' = S - k + 1 (1 with the paper's k = S)
        n_times_dhr = self.n_times - kernel_length + 1

        # Checkers
        # A kernel longer than the window has no valid position, and an empty
        # one reads nothing
        if not 1 <= kernel_length <= self.n_times:
            raise ValueError(
                f"dhr_kernel_length must be in [1, n_times={self.n_times}], "
                f"got {dhr_kernel_length}."
            )

        # Layers
        # Following paper nomenclature (Section II-B, Table I)
        # The authors' input has a depth axis: (batch, 1, C, S)
        self.to_fnirs_tensor = Rearrange("batch chans time -> batch 1 chans time")

        # Delayed hemodynamic response module (DHR Module)
        # DHRConv: F1 standard convolutions of 1 x k, so that one filter reads
        # the whole slow response of one channel (Guidelines 1 and 2)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_dhr_filters,
            kernel_size=(1, kernel_length),
        )
        self.bn1 = nn.BatchNorm2d(n_dhr_filters)
        # Guideline 3: an activation that saturates for negative inputs
        # (sigmoid in the paper)
        self.act1 = activation()

        # Global module: a depthwise separable convolution
        self.conv2 = nn.Sequential(
            OrderedDict(
                # DWConv: one C x 1 filter per DHRConv map, a global receptive
                # field over the channels without stacking layers
                depth_conv=DepthwiseConv2d(
                    in_channels=n_dhr_filters,
                    depth_multiplier=1,
                    kernel_size=(self.n_chans, 1),
                ),
                # PWConv: 1 x 1, projects the F1 maps into F2 new ones
                point_conv=nn.Conv2d(
                    in_channels=n_dhr_filters,
                    out_channels=n_pw_filters,
                    kernel_size=1,
                ),
            )
        )
        self.bn2 = nn.BatchNorm2d(n_pw_filters)
        self.act2 = activation()

        # Flatten layer
        self.flatten = Rearrange("batch filters 1 time -> batch (filters time)")

        # Final fully connected layer, without pooling or dropout
        self.final_layer = nn.Linear(
            in_features=n_pw_filters * n_times_dhr,
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
        """Forward pass of the fNIRSNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, n_chans, n_times)``.
        return_features : bool, default=False
            If True, return the flattened features that feed the FC layer.

        Returns
        -------
        torch.Tensor or dict
            Logits with shape ``(batch_size, n_outputs)``, or with
            ``return_features=True`` the dict ``{"features", "cls_token"}``,
            ``features`` of shape ``(batch_size, n_pw_filters * time')`` and
            ``cls_token`` None.
        """
        # Below, F1 = n_dhr_filters, F2 = n_pw_filters, C = n_chans, S = n_times
        # and S' = S - dhr_kernel_length + 1 (1 with the paper's kernel)

        # Depth axis of the authors' input
        # (batch, C, S) -> (batch, 1, C, S)
        x = self.to_fnirs_tensor(x)

        # DHR Module. Delayed hemodynamic response convolution (DHRConv,
        # Guideline 1): each of the F1 filters reads the whole window of one
        # channel, the channel-level feature of its slow response
        # (batch, 1, C, S) -> (batch, F1, C, S')
        x = self.conv1(x)
        # Normalize the maps, then saturate them: the activation alleviates the
        # information loss of the first layer (Guideline 3)
        # (batch, F1, C, S')
        x = self.bn1(x)
        x = self.act1(x)

        # Global Module. DWConv: one C x 1 filter per DHRConv map aggregates
        # all channels at once (global receptive field in space)
        # (batch, F1, C, S') -> (batch, F1, 1, S')
        x = self.conv2.depth_conv(x)
        # PWConv: 1 x 1 projection of the F1 maps into F2 new ones
        # (batch, F1, 1, S') -> (batch, F2, 1, S')
        x = self.conv2.point_conv(x)
        # Same normalization and saturation as after DHRConv
        # (batch, F2, 1, S')
        x = self.bn2(x)
        x = self.act2(x)

        # One feature vector per window: the F2 maps side by side
        # (batch, F2, 1, S') -> (batch, F2 * S')
        features = self.flatten(x)

        if return_features:
            return {"features": features, "cls_token": None}

        # FC layer; the paper's softmax is left to the loss
        # (batch, F2 * S') -> (batch, n_outputs)
        x = self.final_layer(features)
        return x
