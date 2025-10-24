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
        :width: 620px

    The **Spatial–Spectral** and **Temporal - Dual Prototype Network** (SST-DPN)
    is an end-to-end 1D convolutional architecture designed for motor imagery (MI) EEG decoding,
    aiming to address challenges related to discriminative feature extraction and
    small-sample sizes [Han2025]_.

    The framework systematically addresses three key challenges: multi-channel spatial–spectral
    features, long-term temporal features, and the small-sample dilemma [Han2025]_.

    .. rubric:: Architectural Overview

    SST-DPN consists of a feature extractor (SSTEncoder, comprising ASSF and MVP) followed by the
    Dual Prototype Learning (DPL) classification module [Han2025]_.

    1. **Adaptive Spatial–Spectral Fusion (ASSF)**: Uses LightConv to generate a
        multi-channel spatial–spectral representation, followed by a lightweight
        attention mechanism (SSA) to model relationships and highlight key spatial–spectral
        channels [Han2025]_.

    2. **Multi-scale Variance Pooling (MVP)**: Applies a parameter-free variance pooling
        layer with large kernels across multiple scales to capture long-term temporal
        dependencies, serving as an efficient alternative to transformers [Han2025]_.

    3. **Dual Prototype Learning (DPL)**: A training strategy that employs two sets of
        prototypes—Inter-class Separation Prototypes (ISPs) and Intra-class Compact
        Prototypes (ICPs)—to optimize the feature space, enhancing generalization ability and
        preventing overfitting on small datasets [Han2025]_. During inference (forward pass),
        classification decisions are typically made based on the distance (dot product) between the
        feature vector and the ISP for each class [Han2025]_.

    .. rubric:: Macro Components

    - `SSTDPN.encoder` **(Feature Extractor)**

        - *Operations.* Combines the ASSF and MVP modules.
        - *Role.* Maps the raw MI-EEG trial $X_i \in R^{C \times T}$ to the feature space $z_i \in R^d$.

    - `SSTEncoder.time_conv` **(LightConv for Spatial–Spectral Representation)**

        - *Operations.* :class:`~LightweightConv1d` (1D depthwise convolution) with kernel size `lightconv_kernel_size` and a depth multiplier `depth_multiplier_F1` ($F_1$ filters).
        - *Role.* Extracts multiple distinct spectral bands from each EEG channel.

    - `SSTEncoder.ssa` **(Spatial–Spectral Attention)**

        - *Operations.* Global Context Embedding using a mean-var operation, followed by channel normalization and gating adaptation. The output scale is adjusted channel-wise.
        - *Role.* Reweights channels in the spatial–spectral dimension to extract more efficient and discriminative features by emphasizing task-relevant regions and bands.

    - `SSTEncoder.chan_conv` **(Pointwise Fusion)**

        - *Operations.* A simple 1D pointwise convolution with `n_pointwise_filters_F2` ($F_2$ filters), followed by BatchNorm and ELU activation.
        - *Role.* Fuses the weighted spatial–spectral features across all electrodes to produce $X_{assf} \in R^{F_2 \times T}$.

    - `SSTEncoder.mixer` **(Multi-scale Variance Pooling - MVP)**

        - *Operations.* :class:`~Mixer1D` uses :class:`~VarPool1D` layers with multi-scale large kernel sizes (`variance_pool_kernel_sizes`), followed by concatenation and flattening.
        - *Role.* Captures long-range temporal features. The variance operation is effective as it represents spectral power in EEG signals.

    - `SSTDPN.isp` / `SSTDPN.icp` **(Dual Prototypes)**

        - *Operations.* Learnable vectors initialized randomly and optimized during training using $\mathcal{L}_S$, $\mathcal{L}_C$, and $\mathcal{L}_{EF}$. Note that in the forward pass shown, only the ISP is used for classification via dot product, and the ISP is constrained using L2 weight-normalization ($\lVert s_i \rVert_2 \leq S=1$).
        - *Role.* ISP (Inter-class Separation Prototype) achieves inter-class separation, while ICP (Intra-class Compact Prototype) enhances intra-class compactness.

    .. rubric:: Convolutional Details

    * **Temporal (Long-term dependency).**
      The initial LightConv uses a large kernel (e.g., 75). The MVP module employs pooling kernels that are much larger (e.g., 50, 100, 200 samples) to capture long-term temporal features effectively.

    * **Spatial (Fine-grained modeling).**
      The LightConv uses $h=1$, meaning all electrode channels share $F_1$ temporal filters to produce the spatial–spectral representation. The SSA mechanism explicitly models relationships among multiple channels in the spatial–spectral dimension, allowing for finer-grained spatial feature modeling than standard GCNs.

    * **Spectral (Feature extraction).**
      Spectral information is implicitly extracted via the $F_1$ filters in the LightConv. The use of Variance Pooling explicitly leverages the prior knowledge that the variance of EEG signals represents their spectral power.

    .. rubric:: Additional Mechanisms

    - **Attention.** A lightweight attention mechanism (SSA) is used explicitly to model spatial–spectral relationships at the channel level, rather than applying attention to deep features.
    - **Regularization.** Dual Prototype Learning (DPL) acts as a regularization technique, enhancing model generalization and classification performance, particularly useful for limited data typical of MI-EEG tasks.

    Notes
    ----------
    * The implementation of the DPL loss functions ($\mathcal{L}_S, \mathcal{L}_C, \mathcal{L}_{EF}$) and the optimization of ICPs are typically handled outside the primary `forward` method shown here.
    * The default parameters are configured based on the BCI Competition IV 2a dataset.
    * The model operates directly on raw MI-EEG signals without requiring traditional preprocessing steps like band-pass filtering.
    * The first iteration of the braindecode adaptation was done by Can Han [Han2025Code], the original author of the paper and code.

    Parameters
    ----------
    depth_multiplier_F1 : int, optional
        Depth multiplier for LightConv ($F_1$), which corresponds to the number
        of spectral bands extracted per channel. Default is 9.

    n_pointwise_filters_F2 : int, optional
        Output channel dimension after the pointwise convolution ($F_2$).
        Default is 48.

    lightconv_kernel_size : int, optional
        Kernel size for the temporal LightweightConv1d layer ($k$). Default is 75.

    variance_pool_kernel_sizes : list[int], optional
        Kernel sizes used in the Multi-scale Variance Pooling (MVP) module. Default is [36, 37].

    return_features : bool, optional
        If True, the forward returns (features, logits). Default is False.

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
        n_chans: int,
        n_outputs: int,
        n_times: int,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        F1: int = 9,
        F2: int = 48,
        time_kernel1: int = 75,
        pool_kernels: List[int] = [50, 100, 200],
        return_features: bool = False,
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
        self.times_kernel1 = time_kernel1
        self.pool_kernels = pool_kernels
        self.return_features = return_features
        self.F1 = F1
        self.F2 = F2

        # Encoder accepts (batch, n_chans, n_times)
        self.encoder = SSTEncoder(
            n_times=self.n_times,
            n_chans=self.n_chans,
            F1=self.F1,
            F2=self.F2,
            time_kernel1=self.times_kernel1,
            pool_kernels=self.pool_kernels,
        )

        # infer feature dim with dry-run (safe - uses model init defaults)
        with torch.no_grad():
            dummy = torch.ones((1, n_chans, n_times))
            feat = self.encoder(dummy)
            feat_dim = int(feat.shape[-1])

        # prototypes: ISP (inter-class separation), ICP (intra-class compactness)
        self.isp = nn.Parameter(
            torch.randn(self.n_outputs, feat_dim), requires_grad=True
        )
        self.icp = nn.Parameter(
            torch.randn(self.n_outputs, feat_dim), requires_grad=True
        )

        nn.init.kaiming_normal_(self.isp.data)
        nn.init.normal_(self.icp.data, mean=0.0, std=0.01)

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
        # renorm prototypes
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=1, maxnorm=1.0)
        logits = torch.einsum("bd,cd->bc", features, self.isp)  # (b, n_outputs)

        if self.return_features:
            return features, logits

        return logits


class SSTEncoder(nn.Module):
    """Encoder inside SSTDPN

    Parameters
    ----------
    n_times : int
        Number of time samples in the input window.
    n_chans : int
        Number of EEG channels.
    F1, F2, time_kernel1, pool_kernels : see main model
    """

    def __init__(
        self,
        n_times: int,
        n_chans: int,
        F1: int = 16,
        F2: int = 36,
        time_kernel1: int = 75,
        pool_kernels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=n_chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(T=n_times, num_channels=n_chans * F1)

        self.chan_conv = nn.Sequential(
            nn.Conv1d(n_chans * F1, F2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )
        self.mixer = Mixer1D(kernel_sizes=pool_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_chans, n_times)
        returns: (batch, feat_dim)
        """
        x = self.time_conv(x)  # (b, n_chans*F1, T)
        x, _ = self.ssa(x)  # (b, n_chans*F1, T)
        x_chan = self.chan_conv(x)  # (b, F2, T')
        feature = self.mixer(x_chan)  # (b, feat_dim)
        return feature


class LightweightConv1d(nn.Module):
    """Lightweight grouped temporal conv."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1,
        depth_multiplier: int = 1,
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
            torch.Tensor(num_heads * depth_multiplier, 1, kernel_size)
        )
        self.bias = (
            nn.Parameter(torch.Tensor(num_heads * depth_multiplier)) if bias else None
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        B, C, T = inp.size()
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


class VarMaxPool1D(nn.Module):
    def __init__(
        self, T: int, kernel_size: int, stride: Optional[int] = None, padding: int = 0
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class VarPool1D(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: int = 0
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class SSA(nn.Module):
    def __init__(
        self,
        T: int,
        num_channels: int,
        epsilon: float = 1e-5,
        mode: str = "var",
        after_relu: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu
        self.GP = VarMaxPool1D(T, 250)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T = x.shape

        if self.mode == "l2":
            embedding = (x.pow(2).sum(2, keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)
        elif self.mode == "l1":
            _x = torch.abs(x) if not self.after_relu else x
            embedding = _x.sum(2, keepdim=True)
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )
        elif self.mode == "var":
            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)
        else:
            raise ValueError(f"Unsupported SSA mode: {self.mode}")

        gate = 1 + torch.tanh(embedding * norm + self.beta)
        return x * gate, gate


class Mixer1D(nn.Module):
    def __init__(self, kernel_sizes: Optional[List[int]] = None) -> None:
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [50, 100, 250]

        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)

        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)), nn.Flatten(start_dim=1)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, d, _ = x.shape
        assert d % self.L == 0, "Channel dim must be divisible by number of branches"
        x_split = torch.split(x, d // self.L, dim=1)
        out = []
        for i, part in enumerate(x_split):
            out.append(self.var_layers[i](part))
        y = torch.concat(out, dim=1)
        return y
