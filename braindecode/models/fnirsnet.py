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

    .. figure:: https://braindecode.org/dev/_static/model/fnirsnet.png
       :align: center
       :alt: fNIRSNet architecture

       Fig. 2 of Wang et al. (2023), cropped
       (`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_): "(a)
       Overview architecture of fNIRSNet. (b) Schema of receptive fields. The
       green and yellow feature maps are the output of DHRConv and DWConv,
       respectively. [...] (c) Depthwise separable convolution."

    Changes of oxygenated and deoxygenated hemoglobin (HbO, HbR) are "a slow
    metabolic process manifested as delayed hemodynamic responses", and the
    paper :footcite:`wang2023fnirsnet` argues that deep networks for fNIRS
    ignore them: "A small kernel with a limited receptive field is
    challenging to extract the features of delayed hemodynamic responses
    because there is no significant change in HbO and HbR in small
    neighborhoods. However, stacking more convolutional layers to obtain
    larger receptive fields may cause overfitting on limited fNIRS data"
    (Section I).

    fNIRSNet answers with "two core ideas: 1) delayed hemodynamic responses
    as domain knowledge should be introduced into fNIRS classification
    models; 2) a simple and efficient model is beneficial to practical
    applications on limited fNIRS data". It has "three convolutional layers
    and one fully connected (FC) layer without pooling, dropout, and other
    complicated structures": a convolution whose kernel spans the whole
    window of one row, then one whose kernel spans all rows. With 4 and 8
    filters it has 498 parameters on mental arithmetic (Table III).

    .. rubric:: Architecture Overview

    "fNIRSNet consists of a delayed hemodynamic response module (DHR Module)
    and a global module" (Section II-B, Table I). With :math:`C` =
    ``n_chans`` rows, :math:`S` = ``n_times`` samples, :math:`F_1` and
    :math:`F_2` filters, and :math:`S' = S - k + 1` (1 for the paper's
    kernel width :math:`k = S`)::

        input                                      (batch, C, S)
        to_fnirs_tensor                            (batch, 1, C, S)
        DHR Module: conv1 (1 x k), bn1, sigmoid    (batch, F_1, C, S')
        Global Module:
          conv2.depth_conv (C x 1)                 (batch, F_1, 1, S')
          conv2.point_conv (1 x 1), bn2, sigmoid   (batch, F_2, 1, S')
        flatten, final_layer (FC)                  (batch, n_outputs)

    .. rubric:: Macro Components

    - ``conv1``, ``bn1`` **DHR Module.** "Delayed hemodynamic response
      convolutions (DHRConv) extract the channel-level features of delayed
      hemodynamic responses." DHRConv is a standard convolution with
      :math:`F_1` filters of :math:`1 \times S`: "the width of this kernel
      equals the number of input sampling points" (Guideline 1). It stays a
      standard convolution because "the depthwise separable convolution
      performs poorly in low-dimensional space (the first layer)"
      (Guideline 2). Batch normalization and a sigmoid follow.
    - ``conv2``, ``bn2`` **Global Module.** A depthwise separable
      convolution: DWConv, one :math:`C \times 1` filter per DHRConv map,
      "can directly obtain global receptive fields without stacking layers",
      and "the :math:`1 \times 1` PWConv projects the output of DWConv into
      a new channel space" with :math:`F_2` filters; then batch
      normalization and a sigmoid. Guideline 2 motivates the factorization
      by the cost of a standard convolution against a depthwise separable
      one, for a :math:`K_h \times K_w` kernel, :math:`F_{in}` input and
      :math:`F_{out}` output channels and an :math:`M_h \times M_w` map
      (Eqs. (2) and (3)):

      .. math::

         K_h K_w F_{in} F_{out} M_h M_w
         \quad \text{vs.} \quad
         K_h K_w F_{in} M_h M_w + F_{in} F_{out} M_h M_w ,

      so the global module costs :math:`F_1 (C + F_2)` instead of
      :math:`F_1 C F_2`, and "increasing the length of input signals only
      increases the computational cost of DHRConv".
    - ``activation`` **Saturating activations.** Guideline 3: "Activation
      functions with saturated negative values can alleviate information
      loss in the first layer"; sigmoid, tanh and ELU beat ReLU and leaky
      ReLU in the ablation of Table VI, and the paper uses the sigmoid.
    - ``final_layer`` **Classifier.** "The feature maps are flattened to 1D
      vectors and then fed into an FC layer", without dropout; the paper's
      softmax is left to the loss, and the module returns logits.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* one 3 s window of :math:`S = f \times T` samples (30 at
      10 Hz on MA, 40 at 13.3 Hz on UFFT). DHRConv covers all of it, because
      the response it targets is long: the paper models it with the
      canonical hemodynamic response function of three gamma functions
      (Section II-A, Eq. (1)), whose :math:`A`, :math:`\alpha` and
      :math:`\beta` "control the height and direction, shape, and scale of
      hemodynamic responses":

      .. math::

         h(t) = \sum_{i=1}^{3} A_i \frac{t^{\alpha_i - 1}
         \beta_i^{\alpha_i} e^{-\beta_i t}}{\Gamma(\alpha_i)} .

      The kernel counts samples, so the model takes no sampling rate.
    - *Spatial:* no montage or optode geometry. DWConv mixes all rows at
      once, since "DHRConv alone cannot aggregate spatial information from
      multi-channel fNIRS". ``chs_info`` is not read: the row order must be
      the one used in training.
    - *Chromophores:* :math:`C` "is twice (two chromophores: HbO and HbR) the
      number of fNIRS channels" (Section II-B); as in Fig. 2(a) and the
      authors' loaders, all HbO rows come first, then all HbR rows.
    - *Spectral:* none in the model; the paper band-passes the
      concentration changes to 0.01-0.1 Hz first (Section III-B).

    .. rubric:: Pretrained Weights

    None: the authors release code, not weights, and the paper trains one
    model per subject or per left-out subject.

    .. rubric:: Differences from the reference implementation

    ``x`` is ``(batch, n_chans, n_times)``; ``to_fnirs_tensor`` adds the
    depth axis of the authors' ``(batch, 1, C, S)`` input. The modules keep
    the authors' names except ``fc``, now ``final_layer``; ``mapping``
    renames ``fc.*``, so their ``state_dict`` loads with ``strict=True``.
    Table I lists :math:`F_2` filters for DWConv, but the authors'
    ``DWSConv`` and the cost :math:`F_1 (C + F_2)` above have :math:`F_1`
    depthwise filters and an :math:`F_1 \to F_2` PWConv; this module
    follows the code, which gives Table III's parameter counts. Two
    additions: ``dhr_kernel_length`` may be shorter than the window (the
    authors' FC layer only fits :math:`k = S`), and ``activation`` replaces
    the fixed sigmoid. A training batch of one sample runs the BN layers on
    their running statistics, and ``ch_mask`` raises a ``ValueError``.

    Parameters
    ----------
    n_dhr_filters : int, default=4
        :math:`F_1`, the DHRConv filters (the authors' ``num_DHRConv``): 4
        subject-specific, 16 subject-independent (Section III-D; the
        authors' ``LOSO_Train.py`` uses 8). "Except for the number of
        filters :math:`F_1` and :math:`F_2`, fNIRSNet has almost no other
        hyperparameters."
    n_pw_filters : int, default=8
        :math:`F_2`, the PWConv filters (the authors' ``num_DWConv``): 8
        subject-specific, 32 subject-independent (``LOSO_Train.py``: 16).
        The paper recommends "setting the value of :math:`F_2` to at least
        twice that of :math:`F_1`" (Section IV-F).
    dhr_kernel_length : int or None, default=None
        DHRConv width :math:`k`, ``1 <= k <= n_times``; ``None`` uses
        :math:`S` (Guideline 1). Table IV also tries :math:`k` = 10, 20 and
        30 on UFFT; the FC layer then reads :math:`F_2 (S - k + 1)` values.
    activation : type[nn.Module], default=nn.Sigmoid
        Activation after each BN (the authors' ``act``).

    Notes
    -----
    Protocol (Section III, details from the authors' code): MA
    :footcite:`shin2017openaccess` (2 classes, 29 subjects, 10 Hz) and UFFT
    :footcite:`bak2019ufft` (3 classes, 30 subjects, 13.3 Hz), band-passed
    0.01-0.1 Hz, baseline-corrected, cut into 3 s windows with a 1 s step
    and z-scored. Subject-specific runs use 5-fold CV over trial blocks for
    120 epochs, subject-independent runs leave one subject out for 30
    epochs; AdamW (lr 0.001), label smoothing 0.1, last-epoch test metrics.
    Table III gives 498 and 419 parameters for the subject-specific sizes
    (MA :math:`C = 72, S = 30`; UFFT :math:`C = S = 40`) and 2,370 and 2,051
    for the subject-independent ones; this module gives the same counts.

    On UFFT, in the authors' training loops (seeds 0-2), the port and the
    authors' ``fNIRSNet`` agree within 0.01 point (accuracy / macro-F1 in %;
    the LOSO runs use the authors' :math:`F_1 = 8`, :math:`F_2 = 16`):

    ===================  ===============  =============  =============
    UFFT                 Paper, Table II  This port      Authors' code
    ===================  ===============  =============  =============
    Subject-specific     68.67 / 68.02    67.68 / 66.83  67.68 / 66.83
    Subject-independent  64.43 / 63.57    64.52 / 63.67  64.51 / 63.67
    ===================  ===============  =============  =============

    .. versionadded:: 2.1

    `License <https://github.com/wzhlearning/fNIRSNet/blob/11d9dad96308c65c039a0978b7eaf8a845065421/LICENSE>`_

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

        # The authors' head is named ``fc``: their state_dict loads as is
        self.mapping = {f"fc.{k}": f"final_layer.{k}" for k in ("weight", "bias")}

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

        # Guideline 3: an activation that saturates for negative inputs
        # (sigmoid in the paper), used after both batch normalizations
        self.activation = activation()

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
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass of the fNIRSNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, n_chans, n_times)``.
        return_features : bool, default=False
            If True, return the flattened features that feed the FC layer.
        ch_mask : torch.Tensor or None
            Not supported: DWConv has one weight per channel.

        Returns
        -------
        torch.Tensor or dict
            Logits with shape ``(batch_size, n_outputs)``, or with
            ``return_features=True`` the dict ``{"features", "cls_token"}``,
            ``features`` of shape ``(batch_size, n_pw_filters * time')`` and
            ``cls_token`` None.
        """
        # DWConv has one weight per channel: a subset of the channels is
        # another model
        if ch_mask is not None:
            raise ValueError("FNIRSNet does not accept ch_mask: DWConv spans n_chans.")

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
        # Batch normalization of the F1 maps; shape unchanged
        x = self.bn1(x)
        # Saturating activation (Guideline 3: no information lost in the
        # first layer for negative inputs); shape unchanged
        x = self.activation(x)

        # Global Module. DWConv: one C x 1 filter per DHRConv map aggregates
        # all channels at once (global receptive field in space)
        # (batch, F1, C, S') -> (batch, F1, 1, S')
        x = self.conv2.depth_conv(x)
        # PWConv: 1 x 1 projection of the F1 maps into F2 new ones
        # (batch, F1, 1, S') -> (batch, F2, 1, S')
        x = self.conv2.point_conv(x)
        # Batch normalization of the F2 maps; shape unchanged
        x = self.bn2(x)
        # Same saturating activation as after DHRConv; shape unchanged
        x = self.activation(x)

        # "The feature maps are flattened to 1D vectors"
        # (batch, F2, 1, S') -> (batch, F2 * S')
        features = self.flatten(x)

        if return_features:
            return {"features": features, "cls_token": None}

        # FC layer; the paper's softmax is left to the loss
        # (batch, F2 * S') -> (batch, n_outputs)
        x = self.final_layer(features)
        return x
