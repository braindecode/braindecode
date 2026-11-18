# Authors: Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# Adapted from the 1D-CNN baseline of the GeoNIRS authors,
# https://github.com/GeoNIRS/GeoNIRS (``model/cnn1d.py`` at commit 81f8995,
# where the class is named ``DirectEventCNN1DClassifier``).
from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin


class FNIRSCNN1D(EEGModuleMixin, nn.Module):
    r"""FNIRSCNN1D, the 1D-CNN of Trakoolwilaiwan et al. (2018), as in GeoNIRS.

    :bdg-success:`Convolution`

    .. versionadded:: 2.1

    Trakoolwilaiwan et al. :footcite:`trakoolwilaiwan2018cnn` proposed
    convolutional neural networks for fNIRS brain-computer interfaces, to
    classify rest, right- and left-hand motor execution from HbO/HbR signals.
    The GeoNIRS benchmark :footcite:`li2026geonirs` uses a "1D-CNN
    (Trakoolwilaiwan et al., 2018)" as its first supervised neural baseline
    (Section 3.1; the "1D-CNN" rows of Tables 1, 14 and 15, and 88.4 K
    parameters in Table 10). This class ports that baseline: the
    ``DirectEventCNN1DClassifier`` of the GeoNIRS repository (commit
    ``81f8995``) as its benchmark builds it, one classifier per event window.
    The two GroupNorm and GELU stages are the GeoNIRS authors' (the networks
    of Trakoolwilaiwan et al. use one or three convolutions of kernel 3 with
    max pooling and dropout), and the architecture is documented from their
    code.

    The model is small and direct. Two temporal convolutions read the window,
    the second of them at half the sampling rate; the result is averaged over
    time and classified. Nothing in it knows about channels, chromophores or
    optode geometry: the first convolution mixes every row. With :math:`B`
    the batch size, :math:`C` = ``n_chans``, :math:`T` = ``n_times`` and
    :math:`F_1`, :math:`F_2` the two filter counts, the steps are::

        x (B, C, T): HbO rows then HbR rows; NaN or inf = missing sample
        1. temporal_conv1: Conv1d, GroupNorm, GELU, MaxPool1d  -> (B, F_1, T // 2)
        2. temporal_conv2: Conv1d, GroupNorm, GELU             -> (B, F_2, T // 2)
        3. time_pool, flatten_layer, head_dropout              -> (B, F_2)
        4. final_layer                                         -> (B, n_outputs)

    .. rubric:: Macro Components

    - ``FNIRSCNN1D.temporal_conv1`` **First temporal convolution**

      :class:`~torch.nn.Conv1d` (64 filters, kernel 7, ``padding="same"``),
      :class:`~torch.nn.GroupNorm` (8 groups), GELU and
      :class:`~torch.nn.MaxPool1d` (kernel 2), which halves the time axis.
      The kernel spans 7 samples, 1.4 s at the 5 Hz of the GeoNIRS inputs.

    - ``FNIRSCNN1D.temporal_conv2`` **Second temporal convolution**

      :class:`~torch.nn.Conv1d` (128 filters, kernel 5, ``padding="same"``),
      :class:`~torch.nn.GroupNorm` (8 groups) and GELU, at half rate, so its
      5 samples span 2 s of the input.

    - ``FNIRSCNN1D.final_layer`` **Classification head**

      The feature maps are averaged over time
      (:class:`~torch.nn.AdaptiveAvgPool1d`), then
      :class:`~torch.nn.Dropout` (0.5) and one :class:`~torch.nn.Linear`.
      The average is what lets the model take a window of any length.

    .. rubric:: Temporal, Spatial, and Chromophore Encoding

    - **Temporal.** Two short kernels and one halving of the rate. The final
      average throws the position of a response away, so the model reads the
      window as a bag of short patterns.
    - **Spatial.** None. The first convolution mixes every row, and no
      optode position is read.
    - **Chromophores.** HbO and HbR are ordinary rows. Their order must be
      the one used in training: the GeoNIRS benchmark stacks the HbO rows
      before the HbR rows, ``HbO_1, ..., HbO_C, HbR_1, ..., HbR_C``.
      ``chs_info`` is not read.

    .. rubric:: Pretrained Weights

    None. The baseline is trained from scratch on each task.

    .. rubric:: Differences from the Reference Implementation

    - **Input.** One tensor ``(batch, n_chans, n_times)``. The reference
      takes a batch dictionary with ``hbo``, ``hbr`` and a validity mask,
      replaces the invalid samples by zero and stacks the HbO rows before
      the HbR rows. Here a non-finite value (NaN or inf) marks a missing
      sample and is read as zero. The reference's mask covers both
      chromophores of a pair, so a sample missing in one of them should be
      NaN in both.
    - **Parity.** With the same seed, this class and the authors' module
      draw the same weights and give the same logits (max-abs 0.0), in eval
      mode and in train mode. ``mapping`` renames their ``encoder.`` and
      ``linear.`` keys, so their state dict loads with ``strict=True``.
    - **Event pairs.** The reference can also aggregate the events of a pair
      before its head (``event_pair_mode``). The benchmark builds it with
      that mode off, and this class has only that mode.
    - **Additions.** The kernel lengths, the pooling length, the number of
      normalization groups and ``activation`` are arguments, fixed in the
      reference. ``ch_mask`` is refused, because the first convolution spans
      every row.

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

    Notes
    -----
    In the GeoNIRS benchmark (Section 3.1 and Appendices B.1 and B.5-B.6 of
    :footcite:`li2026geonirs`), "baseline models receive the same frozen
    downstream inputs as GeoNIRS" and "follow the unified protocol in Table
    8": HbO and HbR in µmol/L resampled to 5 Hz and z-scored per recording;
    eight held-out binary tasks with subject-level five-fold cross-validation
    and the seeds 810 and 811; 30 epochs, batch 16, AdamW (weight decay
    0.05; learning rate 1e-4, from the repository's ``downstream.yaml``),
    class-weighted cross-entropy, and model selection by validation balanced
    accuracy.

    Replication through the authors' training loop with this class in place
    of theirs (mean over the five test folds and the two seeds; the authors'
    module gives the same value in each of the 20 cells, max-abs 0.0; half
    the range across the two seeds is at most 0.29 points):

    =====================  ========  =================  ==============
    Task                   Metric    GeoNIRS paper (%)  This class (%)
    =====================  ========  =================  ==============
    DS42 (action mode)     macro-F1  55.53              55.85
    DS42 (action mode)     AUROC     59.67              59.32
    DS19 (pain intensity)  macro-F1  50.05              50.22
    DS19 (pain intensity)  AUROC     50.76              49.73
    =====================  ========  =================  ==============

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
        # Following the nomenclature of the reference implementation

        # First temporal convolution. This is where the channels are mixed:
        # every filter reads all the HbO and HbR rows at once
        self.temporal_conv1 = nn.Sequential(
            # F1 filters over all the rows and kernel_size_conv1 samples
            # (1.4 s at the 5 Hz of the benchmark)
            nn.Conv1d(
                in_channels=self.n_chans,
                out_channels=n_filters_conv1,
                kernel_size=kernel_size_conv1,
                padding="same",
            ),
            # Normalizes groups of filters within each window, so it needs no
            # batch statistics
            nn.GroupNorm(n_norm_groups, n_filters_conv1),
            activation(),
            # Keeps the strongest response of every pool_size samples, which
            # halves the sampling rate. A last incomplete group is dropped, as
            # in the reference (no ceil_mode)
            nn.MaxPool1d(kernel_size=pool_size),
        )

        # Second temporal convolution, on the pooled feature maps
        self.temporal_conv2 = nn.Sequential(
            # F2 filters over kernel_size_conv2 pooled samples, twice as long
            # in time as on the input (2 s at 5 Hz)
            nn.Conv1d(
                in_channels=n_filters_conv1,
                out_channels=n_filters_conv2,
                kernel_size=kernel_size_conv2,
                padding="same",
            ),
            nn.GroupNorm(n_norm_groups, n_filters_conv2),
            activation(),
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

        # Names of the reference, so that its state dict loads strictly. It
        # keeps both convolution stages in one "encoder" Sequential, where the
        # layers with parameters sit at positions 0, 1, 4 and 5, and names its
        # classifier "linear"
        self.mapping = {
            "encoder.0.weight": "temporal_conv1.0.weight",
            "encoder.0.bias": "temporal_conv1.0.bias",
            "encoder.1.weight": "temporal_conv1.1.weight",
            "encoder.1.bias": "temporal_conv1.1.bias",
            "encoder.4.weight": "temporal_conv2.0.weight",
            "encoder.4.bias": "temporal_conv2.0.bias",
            "encoder.5.weight": "temporal_conv2.1.weight",
            "encoder.5.bias": "temporal_conv2.1.bias",
            "linear.weight": "final_layer.weight",
            "linear.bias": "final_layer.bias",
        }

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
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
        ch_mask : None
            Not supported, because the first convolution spans every row.

        Returns
        -------
        torch.Tensor or dict
            Output tensor with shape (batch_size, n_outputs), or
            ``{"features": (batch_size, n_filters_conv2),
            "cls_token": None}`` with ``return_features=True``.
        """
        # Shapes use the symbols of the class docstring: C rows, T samples,
        # F1 and F2 filters.

        # The first convolution reads every row at once, so a padded channel
        # cannot be left out of it
        if ch_mask is not None:
            raise ValueError(
                "FNIRSCNN1D does not accept ch_mask: "
                "its first convolution spans every row."
            )

        # Missing samples (NaN or inf) are read as zero, as the reference does
        # with its validity mask
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
