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

__all__ = ["SSTDPN"]


class SSTDPN(EEGModuleMixin, nn.Module):
    r"""SSTDPN from Can Han et al (2025) [Han2025]_.

    :bdg-info:`Small Attention` :bdg-success:`Convolution`

    .. figure:: https://raw.githubusercontent.com/hancan16/SST-DPN/refs/heads/main/figs/framework.png
        :align: center
        :alt: SSTDPN Architecture
        :width: 1000px

    The **Spatial–Spectral** and **Temporal - Dual Prototype Network** (SST-DPN)
    is an end-to-end 1D convolutional architecture designed for motor imagery (MI) EEG decoding,
    aiming to address challenges related to discriminative feature extraction and
    small-sample sizes [Han2025]_.

    The framework systematically addresses three key challenges: multi-channel spatial–spectral
    features, long-term temporal features, and the small-sample dilemma [Han2025]_.

    .. rubric:: Architectural Overview

    SST-DPN consists of a feature extractor (_SSTEncoder, comprising Adaptive Spatial-Spectral
    Fusion and Multi-scale Variance Pooling) followed by Dual Prototype Learning classification [Han2025]_.

    1. **Adaptive Spatial–Spectral Fusion (ASSF)**: Uses _DepthwiseTemporalConv1d to generate a
        multi-channel spatial–spectral representation, followed by _SpatSpectralAttn
        (Spatial-Spectral Attention) to model relationships and highlight key spatial–spectral
        channels [Han2025]_.

    2. **Multi-scale Variance Pooling (MVP)**: Applies _MultiScaleVarPooler with variance pooling
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

        - *Operations.* Combines Adaptive Spatial–Spectral Fusion and Multi-scale Variance Pooling
          via an internal :class:`_SSTEncoder`.
        - *Role.* Maps the raw MI-EEG trial :math:`X_i \in \mathbb{R}^{C \times T}` to the
          feature space :math:`z_i \in \mathbb{R}^d`.

    - `_SSTEncoder.temporal_conv` **(Depthwise Temporal Convolution for Spectral Extraction)**

        - *Operations.* Internal :class:`_DepthwiseTemporalConv1d` applying separate temporal
          convolution filters to each channel with kernel size `temporal_conv_kernel_size` and
          depth multiplier `n_spectral_filters_temporal` (equivalent to :math:`F_1` in the paper).
        - *Role.* Extracts multiple distinct spectral bands from each EEG channel independently.

    - `_SSTEncoder.spt_attn` **(Spatial–Spectral Attention for Channel Gating)**

        - *Operations.* Internal :class:`_SpatSpectralAttn` module using Global Context Embedding
          via variance-based pooling, followed by adaptive channel normalization and gating.
        - *Role.* Reweights channels in the spatial–spectral dimension to extract efficient and
          discriminative features by emphasizing task-relevant regions and frequency bands.

    - `_SSTEncoder.chan_conv` **(Pointwise Fusion across Channels)**

        - *Operations.* A 1D pointwise convolution with `n_fused_filters` output channels
          (equivalent to :math:`F_2` in the paper), followed by BatchNorm and ELU activation.
        - *Role.* Fuses the weighted spatial–spectral features across all electrodes to produce
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

    .. rubric:: Convolutional Details

    * **Temporal (Long-term dependency).**
      The _DepthwiseTemporalConv1d uses a large kernel (e.g., 75). The MVP module employs pooling
      kernels that are much larger (e.g., 50, 100, 200 samples) to capture long-term temporal
      features effectively.

    * **Spatial (Fine-grained modeling).**
      The _DepthwiseTemporalConv1d uses :math:`h=1`, meaning all electrode channels share
      :math:`F_1` temporal filters independently to produce the spatial–spectral representation.
      The _SpatSpectralAttn mechanism explicitly models relationships among channels in the
      spatial–spectral dimension, enabling finer-grained spatial feature modeling.

    * **Spectral (Feature extraction).**
      Spectral information is implicitly extracted via the :math:`F_1` filters in
      _DepthwiseTemporalConv1d. The use of _VariancePool1D explicitly leverages the prior
      knowledge that variance of EEG signals represents their spectral power.

    .. rubric:: Additional Mechanisms

    - **Attention.** A lightweight attention mechanism (SSA) is used explicitly to model spatial–spectral relationships at the channel level, rather than applying attention to deep features.
    - **Regularization.** Dual Prototype Learning (DPL) acts as a regularization technique, enhancing model generalization and classification performance, particularly useful for limited data typical of MI-EEG tasks.

    Notes
    ----------
    * The implementation of the DPL loss functions (:math:`\mathcal{L}_S`, :math:`\mathcal{L}_C`, :math:`\mathcal{L}_{EF}`) and the optimization of ICPs are typically handled outside the primary `forward` method shown here.
    * The default parameters are configured based on the BCI Competition IV 2a dataset.
    * The model operates directly on raw MI-EEG signals without requiring traditional preprocessing steps like band-pass filtering.
    * The first iteration of the braindecode adaptation was done by Can Han [Han2025Code]_, the original author of the paper and code.

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
        Larger kernels capture long-term temporal dependencies. Default is [50, 100, 200] samples.
    return_features : bool, optional
        If True, the forward pass returns (features, logits). If False, returns only logits.
        Default is False.
    proto_sep_maxnorm : float, optional
        Maximum L2 norm constraint for Inter-class Separation Prototypes during forward pass.
        Ensures prototype regularization. Default is 1.0.
    proto_cpt_std : float, optional
        Standard deviation for Intra-class Compactness Prototype initialization. Default is 0.01.
    spt_attn_global_context_kernel : int, optional
        Kernel size for global context embedding in Spatial-Spectral Attention module.
        Default is 250 samples.
    spt_attn_epsilon : float, optional
        Small epsilon value for numerical stability in Spatial-Spectral Attention. Default is 1e-5.
    spt_attn_mode : str, optional
        Embedding computation mode for Spatial-Spectral Attention ('var', 'l2', or 'l1').
        Default is 'var' (variance-based).

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
        n_chans: int,
        n_outputs: int,
        n_times: int,
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
    ) -> None:
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del input_window_seconds, sfreq, chs_info

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
        )

        # Infer feature dimension with dry-run
        with torch.no_grad():
            dummy = torch.ones((1, self.n_chans, self.n_times))
            feat = self.encoder(dummy)
            feat_dim = int(feat.shape[-1])

        # Prototypes: Inter-class Separation (ISP) and Intra-class Compactness (ICP)
        # ISP: provides inter-class separation via prototype learning
        # ICP: enhances intra-class compactness
        self.proto_sep = nn.Parameter(
            torch.randn(self.n_outputs, feat_dim), requires_grad=True
        )
        self.proto_cpt = nn.Parameter(
            torch.randn(self.n_outputs, feat_dim), requires_grad=True
        )

        # Initialize prototypes
        nn.init.kaiming_normal_(self.proto_sep.data)
        nn.init.normal_(self.proto_cpt.data, mean=0.0, std=self.proto_cpt_std)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass supporting 3D EEG inputs.

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

        if self.return_features:
            return features, logits

        return logits


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
    ) -> None:
        super().__init__()

        if mvp_kernel_sizes is None:
            mvp_kernel_sizes = [50, 100, 200]

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
            nn.ELU(),
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
        B, C, T = x.shape

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
        else:
            raise ValueError(
                f"Unsupported Spatial-Spectral Attention mode: {self.mode}"
            )

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
