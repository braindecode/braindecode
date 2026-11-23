# Authors: Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)
# Adapted from the 1D-CNN baseline of the GeoNIRS authors,
# https://github.com/GeoNIRS/GeoNIRS (``model/cnn1d.py`` at commit 81f8995,
# where the class is named ``DirectEventCNN1DClassifier``).
from __future__ import annotations

from collections import OrderedDict

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin


class FNIRSCNN1D(EEGModuleMixin, nn.Module):
    r"""FNIRSCNN1D, the 1D-CNN of Trakoolwilaiwan et al. (2018), as in GeoNIRS.

    :bdg-success:`Convolution`

    .. figure:: ../_static/model/fnirs_cnn1d_arch.png
       :align: center
       :alt: FNIRSCNN1D architecture
       :width: 500px

       Input of the CNN of Trakoolwilaiwan et al.: a filter spanning all HbO and
       HbR channels slides along time, reproduced from
       :footcite:`trakoolwilaiwan2018cnn` (CC BY 3.0).

    .. versionadded:: 2.1

    Two temporal convolutions, an average over time and a linear layer: the
    1D-CNN baseline of the GeoNIRS benchmark :footcite:`li2026geonirs`
    (Section 3.1; the "1D-CNN" rows of Tables 1, 14 and 15, and 88.4 K
    parameters in Table 10), after the fNIRS networks of Trakoolwilaiwan et al.
    :footcite:`trakoolwilaiwan2018cnn` for rest, right- and left-hand motor
    execution. This class ports the ``DirectEventCNN1DClassifier`` of the
    GeoNIRS repository (commit ``81f8995``) as its benchmark builds it, one
    classifier per event window. Its GroupNorm and GELU stages are the GeoNIRS
    authors' own; the networks of Trakoolwilaiwan et al. use one or three
    convolutions of kernel 3 with max pooling and dropout.

    .. rubric:: Architecture Overview

    With :math:`B` the batch size, :math:`C` = ``n_chans``, :math:`T` =
    ``n_times`` and :math:`F_1`, :math:`F_2` the two filter counts::

        x                                                     (B, C, T)
        - temporal_conv1: Conv1d, GroupNorm, GELU, MaxPool1d  (B, F_1, T // 2)
        - temporal_conv2: Conv1d, GroupNorm, GELU             (B, F_2, T // 2)
        - time_pool, flatten_layer, head_dropout              (B, F_2)
        - final_layer                                         (B, n_outputs)

    .. rubric:: Macro Components

    - ``FNIRSCNN1D.temporal_conv1`` **First temporal convolution.** A
      :class:`~torch.nn.Conv1d` whose filters read all the rows (7 samples,
      1.4 s at the 5 Hz of the GeoNIRS inputs), :class:`~torch.nn.GroupNorm`,
      GELU and a :class:`~torch.nn.MaxPool1d` that halves the time axis.
    - ``FNIRSCNN1D.temporal_conv2`` **Second temporal convolution.** A
      :class:`~torch.nn.Conv1d` (5 samples, 2 s of the input at half rate),
      :class:`~torch.nn.GroupNorm` and GELU.
    - ``FNIRSCNN1D.time_pool``, ``FNIRSCNN1D.flatten_layer``,
      ``FNIRSCNN1D.head_dropout``, ``FNIRSCNN1D.final_layer``
      **Classification head.** An average over time
      (:class:`~torch.nn.AdaptiveAvgPool1d`), which lets the model take a
      window of any length, then :class:`~torch.nn.Dropout` and one
      :class:`~torch.nn.Linear`.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal.** Two short kernels and one halving of the rate. The final
      average discards the position of a response, so the window is read as a
      bag of short patterns.
    - **Spatial.** None: the first convolution mixes every row and
      ``chs_info`` is not read.
    - **Spectral.** None.
    - **Chromophores.** HbO and HbR are ordinary rows, in the order used in
      training: the GeoNIRS benchmark stacks all HbO rows before the HbR rows,
      ``HbO_1, ..., HbO_C, HbR_1, ..., HbR_C``.

    .. rubric:: Additional Mechanisms

    - **Missing samples.** A NaN or inf marks a missing sample and is read as
      zero, as the reference does with its validity mask; that mask covers
      both chromophores of a pair, so a sample missing in one of them should
      be NaN in both. There is no ``ch_mask``: the first convolution spans
      every row.
    - **Reference.** With the same seed this class and the authors' module
      draw the same weights, and the same weights give the same logits
      (max-abs 0.0), in eval mode and in train mode. Through the authors'
      training loop it gives 55.85 and 50.22 macro-F1 on DS42 and DS19,
      against 55.53 and 50.05 in the paper, and AUROC within 1.1 points
      (:gh:`1592`). The reference's ``event_pair_mode`` (events of a pair
      aggregated before the head) is off in the benchmark and absent here.
    - **Inputs.** The benchmark feeds HbO and HbR in µmol/L, resampled to 5 Hz
      and z-scored per recording (GeoNIRS Appendix B.1.1; its baselines get
      the same inputs, Appendix B.5.3).
    - **Pretrained weights.** None: the baseline is trained from scratch on
      each task.

    Parameters
    ----------
    n_filters_conv1 : int, default=64
        Filters of the first temporal convolution (``conv_dim`` in the
        reference).
    n_filters_conv2 : int, default=128
        Filters of the second temporal convolution, and the feature size
        (``embedding_dim`` in the reference).
    kernel_size_conv1 : int, default=7
        Kernel length of the first temporal convolution, in samples; an odd
        length keeps the reference's symmetric padding.
    kernel_size_conv2 : int, default=5
        Kernel length of the second temporal convolution, in samples; an odd
        length keeps the reference's symmetric padding.
    pool_size : int, default=2
        Length and stride of the max pooling after the first convolution.
    n_norm_groups : int, default=8
        Groups of both :class:`~torch.nn.GroupNorm` layers. It must divide
        both filter counts.
    drop_prob : float, default=0.5
        Dropout probability before ``final_layer`` (``dropout`` in the
        reference).
    activation : type[nn.Module], default=nn.GELU
        Activation after each :class:`~torch.nn.GroupNorm`.

    References
    ----------
    .. footbibliography::
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: temporal_conv1.conv.weight

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
        n_filters_conv1: int = 64,
        n_filters_conv2: int = 128,
        kernel_size_conv1: int = 7,
        kernel_size_conv2: int = 5,
        pool_size: int = 2,
        n_norm_groups: int = 8,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.GELU,
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

        # Layers

        # First temporal convolution. This is where the channels are mixed:
        # every filter reads all the HbO and HbR rows at once
        self.temporal_conv1 = nn.Sequential(
            OrderedDict(
                # F1 filters over all the rows and kernel_size_conv1 samples
                # (1.4 s at the 5 Hz of the benchmark)
                conv=nn.Conv1d(
                    in_channels=self.n_chans,
                    out_channels=n_filters_conv1,
                    kernel_size=kernel_size_conv1,
                    padding="same",
                ),
                # Normalizes groups of filters within each window, so it needs
                # no batch statistics
                norm=nn.GroupNorm(n_norm_groups, n_filters_conv1),
                act=activation(),
                # Keeps the strongest response of every pool_size samples,
                # which halves the sampling rate. A last incomplete group is
                # dropped (no ceil_mode)
                pool=nn.MaxPool1d(kernel_size=pool_size),
            )
        )

        # Second temporal convolution, on the pooled feature maps
        self.temporal_conv2 = nn.Sequential(
            OrderedDict(
                # F2 filters over kernel_size_conv2 pooled samples, twice as
                # long in time as on the input (2 s at 5 Hz)
                conv=nn.Conv1d(
                    in_channels=n_filters_conv1,
                    out_channels=n_filters_conv2,
                    kernel_size=kernel_size_conv2,
                    padding="same",
                ),
                norm=nn.GroupNorm(n_norm_groups, n_filters_conv2),
                act=activation(),
            )
        )

        # Global average pooling: one value per filter, for a window of any
        # length. The position of a response in the window is not kept
        self.time_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # Drops the time axis, which the pooling leaves with one sample
        self.flatten_layer = Rearrange("batch filters 1 -> batch filters")

        # Classification head: dropout on the F2 features, then the linear
        # classifier that gives the logits
        self.head_dropout = nn.Dropout(drop_prob)
        self.final_layer = nn.Linear(
            in_features=n_filters_conv2,
            out_features=self.n_outputs,
        )
        self._min_n_times = max(1, pool_size)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass of the FNIRSCNN1D model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times). A
            non-finite value (NaN or inf) marks a missing sample and is read
            as zero.
        return_features : bool, default=False
            If True, return the input of ``final_layer`` instead of the
            logits.

        Returns
        -------
        torch.Tensor or dict
            Output tensor with shape (batch_size, n_outputs), or
            ``{"features": (batch_size, n_filters_conv2),
            "cls_token": None}`` with ``return_features=True``.
        """
        # Shapes use the symbols of the class docstring: C rows, T samples,
        # F1 and F2 filters.

        # Missing samples (NaN or inf) are read as zero
        # (batch, C, T), unchanged
        x = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # First temporal convolution: F1 filters over all the rows, GroupNorm
        # and GELU, then max pooling
        # (batch, C, T) -> (batch, F1, T // pool_size)
        x = self.temporal_conv1(x)

        # Second temporal convolution: F2 filters on the pooled maps, GroupNorm
        # and GELU
        # (batch, F1, T // pool_size) -> (batch, F2, T // pool_size)
        x = self.temporal_conv2(x)

        # Global average pooling: the mean of each feature map over time
        # (batch, F2, T // pool_size) -> (batch, F2, 1)
        x = self.time_pool(x)

        # The features of the window, one value per filter
        # (batch, F2, 1) -> (batch, F2)
        x = self.flatten_layer(x)

        # Dropout on the features, before the classifier
        # (batch, F2), unchanged
        x = self.head_dropout(x)
        if return_features:
            return {"features": x, "cls_token": None}

        # Linear classifier: the logits of the window
        # (batch, F2) -> (batch, n_outputs)
        x = self.final_layer(x)
        return x

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        # A new classifier over the same F2 features, with PyTorch's default
        # initialization, as in the constructor
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
