# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint


class EEGNeX(EEGModuleMixin, nn.Module):
    """EEGNeX model from Chen et al. (2024) [eegnex]_.

    :bdg-success:`Convolution`

    .. figure:: https://braindecode.org/dev/_static/model/eegnex.jpg
        :align: center
        :alt: EEGNeX Architecture
        :width: 620px

    .. rubric:: Architectural Overview

    EEGNeX is a **purely convolutional** architecture that refines the EEGNet-style stem
    and deepens the temporal stack with **dilated temporal convolutions**. The end-to-end
    flow is:

    - (i) **Block-1/2**: two temporal convolutions ``(1 x L)`` with BN refine a
      learned FIR-like *temporal filter bank* (no pooling yet);
    - (ii) **Block-3**: depthwise **spatial** convolution across electrodes
      ``(n_chans x 1)`` with max-norm constraint, followed by ELU → AvgPool (time) → Dropout;
    - (iii) **Block-4/5**: two additional **temporal** convolutions with increasing **dilation**
      to expand the receptive field; the last block applies ELU → AvgPool → Dropout → Flatten;
    - (iv) **Classifier**: a max-norm–constrained linear layer.

    The published work positions EEGNeX as a compact, conv-only alternative that consistently
    outperforms prior baselines across MOABB-style benchmarks, with the popular
    “EEGNeX-8,32” shorthand denoting *8 temporal filters* and *kernel length 32*.


    .. rubric:: Macro Components

    - **Block-1 / Block-2 — Temporal filter (learned).**

       - *Operations.*
       - :class:`torch.nn.Conv2d` with kernels ``(1, L)``
       - :class:`torch.nn.BatchNorm2d` (no nonlinearity until Block-3, mirroring a linear FIR analysis stage).
         These layers set up frequency-selective detectors before spatial mixing.

       - *Interpretability.* Kernels can be inspected as FIR filters; two stacked temporal
         convs allow longer effective kernels without parameter blow-up.

    - **Block-3 — Spatial projection + condensation.**

        - *Operations.*
        - :class:`braindecode.modules.Conv2dWithConstraint` with kernel``(n_chans, 1)``
          and ``groups = filter_2`` (depthwise across filters)
        - :class:`torch.nn.BatchNorm2d`
        - :class:`torch.nn.ELU`
        - :class:`torch.nn.AvgPool2d` (time)
        - :class:`torch.nn.Dropout`.

    **Role**: Learns per-filter spatial patterns over the **full montage** while temporal
      pooling stabilizes and compresses features; max-norm encourages well-behaved spatial
      weights similar to EEGNet practice.

    - **Block-4 / Block-5 — Dilated temporal integration.**

        - *Operations.*
        - :class:`torch.nn.Conv2d` with kernels ``(1, k)`` and **dilations**
          (e.g., 2 then 4);
        - :class:`torch.nn.BatchNorm2d`
        - :class:`torch.nn.ELU`
        - :class:`torch.nn.AvgPool2d` (time)
        - :class:`torch.nn.Dropout`
        - :class:`torch.nn.Flatten`.

    **Role**: Expands the temporal receptive field efficiently to capture rhythms and
    long-range context after condensation.

    - **Final Classifier — Max-norm linear.**

        - *Operations.*
        - :class:`braindecode.modules.LinearWithConstraint` maps the flattened
          vector to the target classes; the max-norm constraint regularizes the readout.


    .. rubric:: Convolutional Details

    - **Temporal (where time-domain patterns are learned).**
      Blocks 1-2 learn the primary filter bank (oscillations/transients), while Blocks 4-5
      use **dilation** to integrate over longer horizons without extra pooling. The final
      AvgPool in Block-5 sets the output token rate and helps noise suppression.

    - **Spatial (how electrodes are processed).**
      A *single* depthwise spatial conv (Block-3) spans the entire electrode set
      (kernel ``(n_chans, 1)``), producing per-temporal-filter topographies; no cross-filter
      mixing occurs at this stage, aiding interpretability.

    - **Spectral (how frequency content is captured).**
      Frequency selectivity emerges from the learned temporal kernels; dilation broadens effective
      bandwidth coverage by composing multiple scales.

    .. rubric:: Additional Mechanisms

    - **EEGNeX-8,32 naming.** “8,32” indicates *8 temporal filters* and *kernel length 32*,
      reflecting the paper's ablation path from EEGNet-8,2 toward thicker temporal kernels
      and a deeper conv stack.
    - **Max-norm constraints.** Spatial (Block-3) and final linear layers use max-norm
      regularization—standard in EEG CNNs—to reduce overfitting and encourage stable spatial
      patterns.

    .. rubric:: Usage and Configuration

    - **Kernel schedule.** Start with the canonical **EEGNeX-8,32** (``filter_1=8``,
      ``kernel_block_1_2=32``) and keep **Block-3** depth multiplier modest (e.g., 2) to match
      the paper's “pure conv” profile.
    - **Pooling vs. dilation.** Use pooling in Blocks 3 and 5 to control compute and variance;
      increase dilations (Blocks 4-5) to widen temporal context when windows are short.
    - **Regularization.** Combine dropout (Blocks 3 & 5) with max-norm on spatial and
      classifier layers; prefer ELU activations for stable training on small EEG datasets.


    - The braindecode implementation follows the paper's conv-only design with five blocks
      and reproduces the depthwise spatial step and dilated temporal stack. See the class
      reference for exact kernel sizes, dilations, and pooling defaults. You can check the
      original implementation at [EEGNexCode]_.

    .. versionadded:: 1.1


    Parameters
    ----------
    activation : nn.Module, optional
        Activation function to use. Default is `nn.ELU`.
    depth_multiplier : int, optional
        Depth multiplier for the depthwise convolution. Default is 2.
    filter_1 : int, optional
        Number of filters in the first convolutional layer. Default is 8.
    filter_2 : int, optional
        Number of filters in the second convolutional layer. Default is 32.
    drop_prob: float, optional
        Dropout rate. Default is 0.5.
    kernel_block_4 : tuple[int, int], optional
        Kernel size for block 4. Default is (1, 16).
    dilation_block_4 : tuple[int, int], optional
        Dilation rate for block 4. Default is (1, 2).
    avg_pool_block4 : tuple[int, int], optional
        Pooling size for block 4. Default is (1, 4).
    kernel_block_5 : tuple[int, int], optional
        Kernel size for block 5. Default is (1, 16).
    dilation_block_5 : tuple[int, int], optional
        Dilation rate for block 5. Default is (1, 4).
    avg_pool_block5 : tuple[int, int], optional
        Pooling size for block 5. Default is (1, 8).

    References
    ----------
    .. [eegnex] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2024).
       Toward reliable signals decoding for electroencephalogram: A benchmark
       study to EEGNeX. Biomedical Signal Processing and Control, 87, 105475.
    .. [EEGNexCode] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2024).
       Toward reliable signals decoding for electroencephalogram: A benchmark
       study to EEGNeX. https://github.com/chenxiachan/EEGNeX
    """

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        activation: nn.Module = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
        filter_2: int = 32,
        drop_prob: float = 0.5,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = self.filter_2 * self.depth_multiplier
        self.drop_prob = drop_prob
        self.activation = activation
        self.kernel_block_1_2 = (1, kernel_block_1_2)
        self.kernel_block_4 = (1, kernel_block_4)
        self.dilation_block_4 = (1, dilation_block_4)
        self.avg_pool_block4 = (1, avg_pool_block4)
        self.kernel_block_5 = (1, kernel_block_5)
        self.dilation_block_5 = (1, dilation_block_5)
        self.avg_pool_block5 = (1, avg_pool_block5)

        # final layers output
        self.in_features = self._calculate_output_length()

        # Following paper nomenclature
        self.block_1 = nn.Sequential(
            Rearrange("batch ch time -> batch 1 ch time"),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_1_2,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_1_2,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_2),
        )

        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.filter_2,
                out_channels=self.filter_3,
                max_norm=max_norm_conv,
                kernel_size=(self.n_chans, 1),
                groups=self.filter_2,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_3),
            self.activation(),
            nn.AvgPool2d(
                kernel_size=self.avg_pool_block4,
                padding=(0, 1),
            ),
            nn.Dropout(p=self.drop_prob),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_3,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_4,
                dilation=self.dilation_block_4,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_2),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_2,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_5,
                dilation=self.dilation_block_5,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_1),
            self.activation(),
            nn.AvgPool2d(
                kernel_size=self.avg_pool_block5,
                padding=(0, 1),
            ),
            nn.Dropout(p=self.drop_prob),
            nn.Flatten(),
        )

        self.final_layer = LinearWithConstraint(
            in_features=self.in_features,
            out_features=self.n_outputs,
            max_norm=max_norm_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EEGNeX model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        # x shape: (batch_size, n_chans, n_times)
        x = self.block_1(x)
        # (batch_size, n_filter, n_chans, n_times)
        x = self.block_2(x)
        # (batch_size, n_filter*4, n_chans, n_times)
        x = self.block_3(x)
        # (batch_size, 1, n_filter*8, n_times//4)
        x = self.block_4(x)
        # (batch_size, 1, n_filter*8, n_times//4)
        x = self.block_5(x)
        # (batch_size, n_filter*(n_times//32))
        x = self.final_layer(x)

        return x

    def _calculate_output_length(self) -> int:
        # Pooling kernel sizes for the time dimension
        p4 = self.avg_pool_block4[1]
        p5 = self.avg_pool_block5[1]

        # Padding for the time dimension (assumed from padding=(0, 1))
        pad4 = 1
        pad5 = 1

        # Stride is assumed to be equal to kernel size (p4 and p5)

        # Calculate time dimension after block 3 pooling
        # Formula: floor((L_in + 2*padding - kernel_size) / stride) + 1
        T3 = math.floor((self.n_times + 2 * pad4 - p4) / p4) + 1

        # Calculate time dimension after block 5 pooling
        T5 = math.floor((T3 + 2 * pad5 - p5) / p5) + 1

        # Calculate final flattened features (channels * 1 * time_dim)
        # The spatial dimension is reduced to 1 after block 3's depthwise conv.
        final_in_features = (
            self.filter_1 * T5
        )  # filter_1 is the number of channels before flatten
        return final_in_features
