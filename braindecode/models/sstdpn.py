# Authors: Can Han <hancan@sjtu.edu.cn> (original paper and code,
#                                        first iteration of braindecode adaptation)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from braindecode.models.base import EEGModuleMixin


class SSTDPN(EEGModuleMixin, nn.Module):
    r"""SSTDPN from Can Han et al (2025) [Han2025]_.

    :bdg-info:`Small Attention` :bdg-success:`Convolution`

    .. figure:: https://raw.githubusercontent.com/hancan16/SST-DPN/refs/heads/main/figs/framework.png
        :align: center
        :alt: SSTDPN Architecture
        :width: 1000px

    The **Spatial-Spectral** and **Temporal - Dual Prototype Network** (SST-DPN)
    is an end-to-end 1D convolutional architecture designed for motor imagery (MI) EEG decoding,
    aiming to address challenges related to discriminative feature extraction and
    small-sample sizes [Han2025]_.

    The framework systematically addresses three key challenges: multi-channel spatial–spectral
    features and long-term temporal features [Han2025]_.

    .. rubric:: Architectural Overview

    SST-DPN consists of a feature extractor (_SSTEncoder, comprising Adaptive Spatial-Spectral
    Fusion and Multi-scale Variance Pooling) followed by Dual Prototype Learning classification [Han2025]_.

    1. **Adaptive Spatial-Spectral Fusion (ASSF)**: Uses :class:`_DepthwiseTemporalConv1d` to generate a
        multi-channel spatial-spectral representation, followed by :class:`_SpatSpectralAttn`
        (Spatial-Spectral Attention) to model relationships and highlight key spatial-spectral
        channels [Han2025]_.

    2. **Multi-scale Variance Pooling (MVP)**: Applies :class:`_MultiScaleVarPooler` with variance pooling
        at multiple temporal scales to capture long-range temporal dependencies, serving as an
        efficient alternative to transformers [Han2025]_.

    3. **Dual Prototype Learning (DPL)**: A training strategy that employs two sets of
        prototypes—Inter-class Separation Prototypes (proto_sep) and Intra-class Compact
        Prototypes (proto_cpt)—to optimize the feature space, enhancing generalization ability and
        preventing overfitting on small datasets [Han2025]_. During inference (forward pass),
        classification decisions are based on the distance (dot product) between the
        feature vector and proto_sep for each class [Han2025]_.

    .. rubric:: Macro Components

    - `SSTDPN.encoder` **(Feature Extractor)**

        - *Operations.* Combines Adaptive Spatial-Spectral Fusion and Multi-scale Variance Pooling
          via an internal :class:`_SSTEncoder`.
        - *Role.* Maps the raw MI-EEG trial :math:`X_i \in \mathbb{R}^{C \times T}` to the
          feature space :math:`z_i \in \mathbb{R}^d`.

    - `_SSTEncoder.temporal_conv` **(Depthwise Temporal Convolution for Spectral Extraction)**

        - *Operations.* Internal :class:`_DepthwiseTemporalConv1d` applying separate temporal
          convolution filters to each channel with kernel size `temporal_conv_kernel_size` and
          depth multiplier `n_spectral_filters_temporal` (equivalent to :math:`F_1` in the paper).
        - *Role.* Extracts multiple distinct spectral bands from each EEG channel independently.

    - `_SSTEncoder.spt_attn` **(Spatial-Spectral Attention for Channel Gating)**

        - *Operations.* Internal :class:`_SpatSpectralAttn` module using Global Context Embedding
          via variance-based pooling, followed by adaptive channel normalization and gating.
        - *Role.* Reweights channels in the spatial-spectral dimension to extract efficient and
          discriminative features by emphasizing task-relevant regions and frequency bands.

    - `_SSTEncoder.chan_conv` **(Pointwise Fusion across Channels)**

        - *Operations.* A 1D pointwise convolution with `n_fused_filters` output channels
          (equivalent to :math:`F_2` in the paper), followed by BatchNorm and the specified
          `activation` function (default: ELU).
        - *Role.* Fuses the weighted spatial-spectral features across all electrodes to produce
          a fused representation :math:`X_{fused} \in \mathbb{R}^{F_2 \times T}`.

    - `_SSTEncoder.mvp` **(Multi-scale Variance Pooling for Temporal Extraction)**

        - *Operations.* Internal :class:`_MultiScaleVarPooler` using :class:`_VariancePool1D`
          layers at multiple scales (`mvp_kernel_sizes`), followed by concatenation.
        - *Role.* Captures long-range temporal features at multiple time scales. The variance
          operation leverages the prior that variance represents EEG spectral power.

    - `SSTDPN.proto_sep` / `SSTDPN.proto_cpt` **(Dual Prototypes)**

        - *Operations.* Learnable vectors optimized during training using prototype learning losses.
          The `proto_sep` (Inter-class Separation Prototype) is constrained via L2 weight-normalization
          (:math:`\lVert s_i \rVert_2 \leq` `proto_sep_maxnorm`) during inference.
        - *Role.* `proto_sep` achieves inter-class separation; `proto_cpt` enhances intra-class compactness.

    .. rubric:: How the information is encoded temporally, spatially, and spectrally

    * **Temporal.**
       The initial :class:`_DepthwiseTemporalConv1d` uses a large kernel (e.g., 75). The MVP module employs pooling
       kernels that are much larger (e.g., 50, 100, 200 samples) to capture long-term temporal
       features effectively. Large kernel pooling layers are shown to be superior to transformer
       modules for this task in EEG decoding according to [Han2025]_.

    * **Spatial.**
       The initial convolution at the classes :class:`_DepthwiseTemporalConv1d` groups parameter :math:`h=1`,
       meaning :math:`F_1` temporal filters are shared across channels. The Spatial-Spectral Attention
       mechanism explicitly models the relationships among these channels in the spatial-spectral
       dimension, allowing for finer-grained spatial feature modeling compared to conventional
       GCNs according to the authors [Han2025]_.
       In other words, all electrode channels share :math:`F_1` temporal filters
       independently to produce the spatial-spectral representation.

    * **Spectral.**
       Spectral information is implicitly extracted via the :math:`F_1` filters in :class:`_DepthwiseTemporalConv1d`.
       Furthermore, the use of Variance Pooling (in MVP) explicitly leverages the neurophysiological
       prior that the **variance of EEG signals represents their spectral power**, which is an
       important feature for distinguishing different MI classes [Han2025]_.

    .. rubric:: Additional Mechanisms

    - **Attention.** A lightweight Spatial-Spectral Attention mechanism models spatial-spectral relationships
        at the channel level, distinct from applying attention to deep feature dimensions,
        which is common in comparison methods like :class:`ATCNet`.
    - **Regularization.** Dual Prototype Learning acts as a regularization technique
        by optimizing the feature space to be compact within classes and separated between
        classes. This enhances model generalization and classification performance, particularly
        useful for limited data typical of MI-EEG tasks, without requiring external transfer
        learning data, according to [Han2025]_.

    Notes
    ----------
    * The implementation of the DPL loss functions (:math:`\mathcal{L}_S`, :math:`\mathcal{L}_C`, :math:`\mathcal{L}_{EF}`)
      and the optimization of ICPs are typically handled outside the primary ``forward`` method, within the training strategy
      (see Ref. 52 in [Han2025]_).
    * The default parameters are configured based on the BCI Competition IV 2a dataset.
    * The use of Prototype Learning (PL) methods is novel in the field of EEG-MI decoding.
    * **Lowest FLOPs:** Achieves the lowest Floating Point Operations (FLOPs) (9.65 M) among competitive
      SOTA methods, including braindecode models like :class:`ATCNet` (29.81 M) and
      :class:`EEGConformer` (63.86 M), demonstrating computational efficiency [Han2025]_.
    * **Transformer Alternative:** Multi-scale Variance Pooling (MVP) provides a accuracy
      improvement over temporal attention transformer modules in ablation studies, offering a more
      efficient alternative to transformer-based approaches like :class:`EEGConformer` [Han2025]_.

    .. warning::

        **Important:** To utilize the full potential of SSTDPN with Dual Prototype Learning (DPL),
        users must implement the DPL optimization strategy outside the model's forward method.
        For implementation details and training strategies, please consult the official code at
        [Han2025Code]_:
        https://github.com/hancan16/SST-DPN/blob/main/train.py

    Parameters
    ----------
    n_spectral_filters_temporal : int, optional
        Number of spectral filters extracted per channel via temporal convolution.
        These represent the temporal spectral bands (equivalent to :math:`F_1` in the paper).
        Default is 9.

    n_fused_filters : int, optional
        Number of output filters after pointwise fusion convolution.
        These fuse the spectral filters across all channels (equivalent to :math:`F_2` in the paper).
        Default is 48.

    temporal_conv_kernel_size : int, optional
        Kernel size for the temporal convolution layer. Controls the receptive field for extracting
        spectral information. Default is 75 samples.

    mvp_kernel_sizes : list[int], optional
        Kernel sizes for Multi-scale Variance Pooling (MVP) module.
        Larger kernels capture long-term temporal dependencies .

    return_features : bool, optional
        If True, the forward pass returns (features, logits). If False, returns only logits.
        Default is False.

    proto_sep_maxnorm : float, optional
        Maximum L2 norm constraint for Inter-class Separation Prototypes during forward pass.
        This constraint acts as an implicit force to push features away from the origin. Default is 1.0.

    proto_cpt_std : float, optional
        Standard deviation for Intra-class Compactness Prototype initialization. Default is 0.01.

    spt_attn_global_context_kernel : int, optional
        Kernel size for global context embedding in Spatial-Spectral Attention module.
        Default is 250 samples.

    spt_attn_epsilon : float, optional
        Small epsilon value for numerical stability in Spatial-Spectral Attention. Default is 1e-5.

    spt_attn_mode : str, optional
        Embedding computation mode for Spatial-Spectral Attention ('var', 'l2', or 'l1').
        Default is 'var' (variance-based mean-var operation).

    activation : nn.Module, optional
        Activation function to apply after the pointwise fusion convolution in :class:`_SSTEncoder`.
        Should be a PyTorch activation module class. Default is nn.ELU.


    References
    ----------
    .. [Han2025] Han, C., Liu, C., Wang, J., Wang, Y., Cai, C.,
        & Qian, D. (2025). A spatial–spectral and temporal dual
        prototype network for motor imagery brain–computer
        interface. Knowledge-Based Systems, 315, 113315.
    .. [Han2025Code] Han, C., Liu, C., Wang, J., Wang, Y.,
        Cai, C., & Qian, D. (2025). A spatial–spectral and
        temporal dual prototype network for motor imagery
        brain–computer interface. Knowledge-Based Systems,
        315, 113315. GitHub repository.
        https://github.com/hancan16/SST-DPN.
    """

    def __init__(
        self,
        # Braindecode standard parameters
        n_chans=None,
        n_times=None,
        n_outputs=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        # models parameters
        n_spectral_filters_temporal: int = 9,
        n_fused_filters: int = 48,
        temporal_conv_kernel_size: int = 75,
        mvp_kernel_sizes: Optional[List[int]] = None,
        return_features: bool = False,
        proto_sep_maxnorm: float = 1.0,
        proto_cpt_std: float = 0.01,
        spt_attn_global_context_kernel: int = 250,
        spt_attn_epsilon: float = 1e-5,
        spt_attn_mode: str = "var",
        activation: Optional[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del input_window_seconds, sfreq, chs_info, n_chans, n_outputs, n_times

        # Set default activation if not provided
        if activation is None:
            activation = nn.ELU

        # Store hyperparameters
        self.n_spectral_filters_temporal = n_spectral_filters_temporal
        self.n_fused_filters = n_fused_filters
        self.temporal_conv_kernel_size = temporal_conv_kernel_size
        self.mvp_kernel_sizes = (
            mvp_kernel_sizes if mvp_kernel_sizes is not None else [50, 100, 200]
        )
        self.return_features = return_features
        self.proto_sep_maxnorm = proto_sep_maxnorm
        self.proto_cpt_std = proto_cpt_std
        self.spt_attn_global_context_kernel = spt_attn_global_context_kernel
        self.spt_attn_epsilon = spt_attn_epsilon
        self.spt_attn_mode = spt_attn_mode
        self.activation = activation

        # Encoder accepts (batch, n_chans, n_times)
        self.encoder = _SSTEncoder(
            n_times=self.n_times,
            n_chans=self.n_chans,
            n_spectral_filters_temporal=self.n_spectral_filters_temporal,
            n_fused_filters=self.n_fused_filters,
            temporal_conv_kernel_size=self.temporal_conv_kernel_size,
            mvp_kernel_sizes=self.mvp_kernel_sizes,
            spt_attn_global_context_kernel=self.spt_attn_global_context_kernel,
            spt_attn_epsilon=self.spt_attn_epsilon,
            spt_attn_mode=self.spt_attn_mode,
            activation=self.activation,
        )

        # Infer feature dimension analytically
        feat_dim = self._compute_feature_dim()

        # Prototypes: Inter-class Separation (ISP) and Intra-class Compactness (ICP)
        # ISP: provides inter-class separation via prototype learning
        # ICP: enhances intra-class compactness
        self.proto_sep = nn.Parameter(
            torch.empty(self.n_outputs, feat_dim), requires_grad=True
        )
        # This parameters is not used in the forward pass, only during training for the
        # prototype learning losses. You should implement the losses outside this class.
        self.proto_cpt = nn.Parameter(
            torch.empty(self.n_outputs, feat_dim), requires_grad=True
        )
        # just for braindecode compatibility
        self.final_layer = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize prototype parameters."""
        nn.init.kaiming_normal_(self.proto_sep)
        nn.init.normal_(self.proto_cpt, mean=0.0, std=self.proto_cpt_std)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Classification is based on the dot product similarity with
        Inter-class Separation Prototypes (:attr:`SSTDPN.proto_sep`).


        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Supported shapes:
              - (batch, n_chans, n_times)

        Returns
        -------
        logits : torch.Tensor
            If input was 3D: (batch, n_outputs)
        Or if self.return_features is True:
            (features, logits) where features shape is (batch, feat_dim)
        """

        features = self.encoder(x)  # (b, feat_dim)
        # Renormalize inter-class separation prototypes
        self.proto_sep.data = torch.renorm(
            self.proto_sep.data, p=2, dim=1, maxnorm=self.proto_sep_maxnorm
        )
        logits = torch.einsum("bd,cd->bc", features, self.proto_sep)  # (b, n_outputs)
        logits = self.final_layer(logits)

        if self.return_features:
            return features, logits

        return logits

    def _compute_feature_dim(self) -> int:
        """Compute encoder feature dimensionality without a forward pass."""
        if not self.mvp_kernel_sizes:
            raise ValueError(
                "`mvp_kernel_sizes` must contain at least one kernel size."
            )

        num_scales = len(self.mvp_kernel_sizes)
        channels_per_scale, rest = divmod(self.n_fused_filters, num_scales)
        if rest:
            raise ValueError(
                "Number of fused filters must be divisible by the number of MVP scales. "
                f"Got {self.n_fused_filters=} and {num_scales=}."
            )

        # Validate all kernel sizes at once (stride = k // 2 must be >= 1)
        invalid = [k for k in self.mvp_kernel_sizes if k // 2 == 0]
        if invalid:
            raise ValueError(
                "MVP kernel sizes too small to derive a valid stride (k//2 == 0): "
                f"{invalid}"
            )

        pooled_total = sum(
            self._pool1d_output_length(
                length=self.n_times, kernel_size=k, stride=k // 2, padding=0, dilation=1
            )
            for k in self.mvp_kernel_sizes
        )
        return channels_per_scale * pooled_total

    @staticmethod
    def _pool1d_output_length(
        length: int, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1
    ) -> int:
        """Temporal length after 1D pooling (PyTorch-style formula)."""
        return max(
            0,
            (length + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1,
        )


class _SSTEncoder(nn.Module):
    """Internal encoder combining Adaptive Spatial-Spectral Fusion and Multi-scale Variance Pooling.

    This class should not be instantiated directly. It is an internal component
    of :class:`SSTDPN`.

    Parameters
    ----------
    n_times : int
        Number of time samples in the input window.
    n_chans : int
        Number of EEG channels.
    n_spectral_filters_temporal : int
        Number of spectral filters extracted via temporal convolution (:math:`F_1`).
    n_fused_filters : int
        Number of output filters after pointwise fusion (:math:`F_2`).
    temporal_conv_kernel_size : int
        Kernel size for temporal convolution.
    mvp_kernel_sizes : list[int]
        Kernel sizes for Multi-scale Variance Pooling.
    spt_attn_global_context_kernel : int
        Kernel size for global context in Spatial-Spectral Attention.
    spt_attn_epsilon : float
        Epsilon for numerical stability in Spatial-Spectral Attention.
    spt_attn_mode : str
        Mode for Spatial-Spectral Attention computation ('var', 'l2', or 'l1').
    activation : nn.Module, optional
        Activation function class to use after pointwise convolution. Default is nn.ELU.
    """

    def __init__(
        self,
        n_times: int,
        n_chans: int,
        n_spectral_filters_temporal: int = 9,
        n_fused_filters: int = 48,
        temporal_conv_kernel_size: int = 75,
        mvp_kernel_sizes: Optional[List[int]] = None,
        spt_attn_global_context_kernel: int = 250,
        spt_attn_epsilon: float = 1e-5,
        spt_attn_mode: str = "var",
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if mvp_kernel_sizes is None:
            mvp_kernel_sizes = [50, 100, 200]

        if activation is None:
            activation = nn.ELU

        # Adaptive Spatial-Spectral Fusion (ASSF): Temporal convolution for spectral filtering
        self.temporal_conv = _DepthwiseTemporalConv1d(
            in_channels=n_chans,
            num_heads=1,
            n_spectral_filters_temporal=n_spectral_filters_temporal,
            kernel_size=temporal_conv_kernel_size,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        # Spatial-Spectral Attention: Gate mechanism for channel weighting
        self.spt_attn = _SpatSpectralAttn(
            T=n_times,
            num_channels=n_chans * n_spectral_filters_temporal,
            epsilon=spt_attn_epsilon,
            mode=spt_attn_mode,
            global_context_kernel=spt_attn_global_context_kernel,
        )

        # Pointwise convolution for fusing spectral filters across channels
        self.chan_conv = nn.Sequential(
            nn.Conv1d(
                n_chans * n_spectral_filters_temporal,
                n_fused_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm1d(n_fused_filters),
            activation(),
        )

        # Multi-scale Variance Pooling (MVP): Temporal feature extraction at multiple scales
        self.mvp = _MultiScaleVarPooler(kernel_sizes=mvp_kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Feature vector of shape (batch, feat_dim).
        """
        x = self.temporal_conv(x)  # (b, n_chans*n_spectral_filters_temporal, T)
        x, _ = self.spt_attn(x)  # (b, n_chans*n_spectral_filters_temporal, T)
        x_fused = self.chan_conv(x)  # (b, n_fused_filters, T)
        feature = self.mvp(x_fused)  # (b, feat_dim)
        return feature


class _DepthwiseTemporalConv1d(nn.Module):
    """Internal depthwise temporal convolution for spectral filtering.

    Applies separate temporal convolution filters to each channel independently
    to extract spectral information across multiple bands. This is used to generate
    the spatial-spectral representation in SSTDPN.

    Not intended for external use.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    num_heads : int, optional
        Number of filter groups (typically 1). Default is 1.
    n_spectral_filters_temporal : int, optional
        Number of spectral filters per channel (depth multiplier). Default is 1.
    kernel_size : int, optional
        Temporal convolution kernel size. Default is 1.
    stride : int, optional
        Convolution stride. Default is 1.
    padding : str or int, optional
        Padding mode. Default is 0.
    bias : bool, optional
        Whether to use bias. Default is True.
    weight_softmax : bool, optional
        Whether to apply softmax to weights. Default is False.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1,
        n_spectral_filters_temporal: int = 1,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Union[str, int] = 0,
        bias: bool = True,
        weight_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax

        self.weight = nn.Parameter(
            torch.Tensor(num_heads * n_spectral_filters_temporal, 1, kernel_size)
        )
        self.bias = (
            nn.Parameter(torch.Tensor(num_heads * n_spectral_filters_temporal))
            if bias
            else None
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inp : torch.Tensor
            Input of shape (batch, in_channels, time).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, num_heads * n_spectral_filters_temporal, time).
        """
        B, _, _ = inp.size()
        H = self.num_heads
        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        inp = rearrange(inp, "b (h c) t -> (b c) h t", h=H)
        if self.bias is None:
            output = F.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = F.conv1d(
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t -> b (h c) t", b=B)
        return output


class _GlobalContextVarPool1D(nn.Module):
    """Internal global context variance pooling module.

    Computes variance-based global context embeddings using specified kernel size.
    Used in the Spatial-Spectral Attention module.

    Not intended for external use.

    Parameters
    ----------
    T : int
        Sequence length.
    kernel_size : int
        Pooling kernel size.
    stride : int or None, optional
        Stride. If None, defaults to kernel_size. Default is None.
    padding : int, optional
        Padding. Default is 0.
    """

    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: int = 0
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing global context via variance pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Global context (variance-pooled) output.
        """
        mean_of_squares = F.avg_pool1d(
            x**2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        square_of_mean = (
            F.avg_pool1d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        ) ** 2
        variance = mean_of_squares - square_of_mean
        out = F.avg_pool1d(variance, variance.shape[-1])
        return out


class _VariancePool1D(nn.Module):
    """Internal variance pooling module for temporal feature extraction.

    Applies variance pooling at a specified kernel size to capture temporal dynamics.
    Used in the Multi-scale Variance Pooling (MVP) module.

    Not intended for external use.

    Parameters
    ----------
    kernel_size : int
        Pooling kernel size (receptive field width).
    stride : int or None, optional
        Stride. If None, defaults to kernel_size. Default is None.
    padding : int, optional
        Padding. Default is 0.
    """

    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: int = 0
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing variance pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Variance-pooled output.
        """
        mean_of_squares = F.avg_pool1d(
            x**2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        square_of_mean = (
            F.avg_pool1d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        ) ** 2
        variance = mean_of_squares - square_of_mean
        return variance


class _SpatSpectralAttn(nn.Module):
    """Internal Spatial-Spectral Attention module with global context gating.

    This attention mechanism computes channel-wise gates based on global context
    embedding and applies adaptive reweighting to highlight task-relevant
    spatial-spectral features.

    Not intended for external use. Used internally in :class:`_SSTEncoder`.

    Parameters
    ----------
    T : int
        Sequence (temporal) length.
    num_channels : int
        Number of channels in the spatial-spectral dimension.
    epsilon : float, optional
        Small value for numerical stability. Default is 1e-5.
    mode : str, optional
        Embedding computation mode: 'var' (variance-based), 'l2' (L2-norm),
        or 'l1' (L1-norm). Default is 'var'.
    after_relu : bool, optional
        Whether ReLU is applied before this module. Default is False.
    global_context_kernel : int, optional
        Kernel size for global context variance pooling. Default is 250.
    """

    def __init__(
        self,
        T: int,
        num_channels: int,
        epsilon: float = 1e-5,
        mode: str = "var",
        after_relu: bool = False,
        global_context_kernel: int = 250,
    ) -> None:
        super().__init__()
        # Learnable gating parameters: scale, normalize, and shift
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu
        # check mode validity
        if self.mode not in ["var", "l2", "l1"]:
            raise ValueError(
                f"Unsupported Spatial-Spectral Attention mode: {self.mode}"
            )

        # Global context module using variance pooling
        self.global_ctx = _GlobalContextVarPool1D(global_context_kernel)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing adaptive channel-wise gating.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, channels, time).

        Returns
        -------
        tuple of torch.Tensor
            (gated_output, gate) where both have the same shape as input.
        """

        if self.mode == "l2":
            # L2-norm based embedding
            embedding = (x.pow(2).sum(2, keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)
        elif self.mode == "l1":
            # L1-norm based embedding
            _x = torch.abs(x) if not self.after_relu else x
            embedding = _x.sum(2, keepdim=True)
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )
        elif self.mode == "var":
            # Variance-based embedding (global context)
            embedding = (self.global_ctx(x) + self.epsilon).pow(0.5) * self.alpha
            norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        # Compute adaptive gate: 1 + tanh(...)
        gate = 1 + torch.tanh(embedding * norm + self.beta)
        return x * gate, gate


class _MultiScaleVarPooler(nn.Module):
    """Internal Multi-scale Variance Pooling (MVP) module for temporal feature extraction.

    Applies variance pooling at multiple temporal scales in parallel, then concatenates
    the results to capture long-range temporal dependencies. Each scale processes a subset
    of channels independently, enabling efficient feature extraction.

    Not intended for external use. Used internally in :class:`_SSTEncoder`.

    Parameters
    ----------
    kernel_sizes : list[int] or None, optional
        Kernel sizes for variance pooling layers at each scale. If None,
        defaults to [50, 100, 200] (suitable for 1000-sample windows).
    """

    def __init__(self, kernel_sizes: Optional[List[int]] = None) -> None:
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [50, 100, 200]

        self.var_layers = nn.ModuleList()
        self.num_scales = len(kernel_sizes)

        # Create variance pooling layer for each scale
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    _VariancePool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying multi-scale variance pooling in parallel.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Concatenated multi-scale features of shape (batch, total_features).
        """
        _, num_channels, _ = x.shape
        # Split channels equally across scales
        assert num_channels % self.num_scales == 0, (
            f"Channel dimension ({num_channels}) must be divisible by "
            f"number of scales ({self.num_scales})"
        )
        channels_per_scale = num_channels // self.num_scales
        x_split = torch.split(x, channels_per_scale, dim=1)

        # Apply variance pooling at each scale
        multi_scale_features = []
        for scale_idx, x_scale in enumerate(x_split):
            multi_scale_features.append(self.var_layers[scale_idx](x_scale))

        # Concatenate features from all scales
        y = torch.concat(multi_scale_features, dim=1)
        return y
