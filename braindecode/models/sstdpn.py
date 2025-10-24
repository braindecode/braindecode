from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from braindecode.models.base import EEGModuleMixin


class SSTDPN(EEGModuleMixin, nn.Module):
    """SSTDPN from Can Han et al. (2025) [Han2025].

    :bdg-info:`Small Attention` :bdg-success:`Convolution`

    .. figure:: https://raw.githubusercontent.com/hancan16/SST-DPN/refs/heads/main/figs/framework.png
        :align: center
        :alt: SSTDPN Architecture
        :width: 620px

    SST-DPN begins with a **Spatial-Spectral Attention (SSA)** module, which jointly models
    inter-channel (spatial) and inter-band (spectral) relationships. This mechanism adaptively
    emphasizes the most informative frequency bands and electrode channels for the current MI task,
    providing fine-grained spatial-spectral feature representations.

    Following SSA, a **Multi-Scale Variance Pooling (MVP)** module is applied to extract
    long-range temporal dependencies. Instead of using computationally expensive recurrent or
    transformer structures, MVP employs variance pooling over multiple temporal scales to capture
    signal dynamics at different temporal resolutions, enhancing robustness without adding
    significant parameters.

    On top of the spatial-spectral-temporal features, the model incorporates a **Dual Prototype
    Learning (DPL)** strategy. DPL constructs class prototypes in the feature space to enforce
    intra-class compactness and inter-class separability, improving discriminability and
    generalization under few-trial or cross-subject conditions.

    Overall, SST-DPN integrates lightweight attention, non-parametric temporal modeling, and
    prototype-based regularization into a unified framework for efficient and accurate MI-EEG
    decoding.

    The paper and original implementation with further methodological details are available at
    [Han2025] and [Han2025Code].

    Notes
    ----------
    [1] The following code only includes the forward inference process of SSTDPN.
    During training, SSTDPN employs multiple loss functions that are jointly optimized.
    [2] The default parameters below are configured for the BCI Competition IV 2a dataset.

    For detailed implementation and reproduction, please refer to the
    original paper [Han2025] and [Han2025Code].

    Parameters
    ----------
    F1 : int
        Depth multiplier for time conv, by default 9.
    F2 : int
        Channel mixing output dim, by default 48.
    time_kernel1 : int
        Kernel size for temporal lightweight conv, by default 75.
    pool_kernels : list[int]
        Mixer pooling kernel sizes, by default [50,100,200].
    return_features : bool
        If True, forward returns (features, logits).

    References
    ----------
    [Han2025] Han, C., Liu, C., Wang, J., Wang, Y., Cai, C.,
        & Qian, D. (2025). A spatial–spectral and temporal dual
        prototype network for motor imagery brain–computer
        interface. Knowledge-Based Systems, 315, 113315.
    [Han2025Code] Han, C., Liu, C., Wang, J., Wang, Y.,
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

        # 3D input
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
