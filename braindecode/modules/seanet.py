# Authors: Qian Xiao and OpenTSLab BrainOmni contributors
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Adapted from https://github.com/facebookresearch/encodec (MIT), through the
# BrainOmni port (https://github.com/OpenTSLab/BrainOmni, MIT). Shared by the
# models that run an EnCodec SEANet (BrainOmni, BBL).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class SEANetConv1d(nn.Module):
    """Weight-normed Conv1d with EnCodec's reflect padding.

    The output keeps ``ceil(n_times / stride)`` samples: the padding is split
    around the signal (``causal=False``) or put on the left (``causal=True``).
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, causal=False
    ):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        )
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = self.conv.stride[0]
        # Effective kernel size with dilations.
        kernel_size = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        padding_total = kernel_size - stride
        # Round the length up so the last strided window is full.
        extra = (kernel_size - padding_total - x.shape[-1]) % stride
        right = 0 if self.causal else padding_total // 2
        padding = (padding_total - right, right + extra)
        if max(padding) == 0:  # 1x1 convs: padding would only copy x
            return self.conv(x)
        # Zero-extend a signal too short to reflect, as EnCodec does.
        short = max(max(padding) - x.shape[-1] + 1, 0)
        if short > 0:  # F.pad with nothing to pad still copies x
            x = F.pad(x, (0, short))
        x = F.pad(x, padding, mode="reflect")
        return self.conv(x[..., : x.shape[-1] - short])


class SEANetConvTranspose1d(nn.Module):
    """Weight-normed ConvTranspose1d trimmed as EnCodec (all on the right if causal)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, causal=False):
        super().__init__()
        self.convtr = weight_norm(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        )
        self.padding_total = kernel_size - stride
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convtr(x)
        right = self.padding_total if self.causal else self.padding_total // 2
        return y[..., self.padding_total - right : y.shape[-1] - right]


class SEANetLSTM(nn.Module):
    """Two-layer LSTM with a skip connection over ``(batch, dim, time)``."""

    def __init__(self, dimension: int):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 4D-view permute: Gaudi lazy mode cannot compile the LSTM after a 3D one.
        x = x.unsqueeze(-1).permute(2, 0, 1, 3).squeeze(-1)
        y, _ = self.lstm(x)
        return (y + x).permute(1, 2, 0)


class SEANetResBlock(nn.Module):
    """``ELU -> conv3 -> ELU -> conv1`` plus a 1x1 conv (or identity) shortcut.

    Parameters
    ----------
    dim : int
        Input and output channels.
    dilation : int
        Dilation of the kernel-3 convolution.
    compress : int
        The hidden branch has ``dim // compress`` channels.
    true_skip : bool
        Identity shortcut instead of a 1x1 convolution.
    causal : bool
        Left padding only.
    """

    def __init__(self, dim, dilation=1, compress=2, true_skip=False, causal=False):
        super().__init__()
        hidden = dim // compress
        self.block = nn.Sequential(
            nn.ELU(),
            SEANetConv1d(dim, hidden, 3, dilation=dilation, causal=causal),
            nn.ELU(),
            SEANetConv1d(hidden, dim, 1, causal=causal),
        )
        self.shortcut = (
            nn.Identity() if true_skip else SEANetConv1d(dim, dim, 1, causal=causal)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Sequential):
    """EnCodec SEANet encoder: ``(N, in_channels, L) -> (N, dim, L / prod(ratios))``.

    Parameters
    ----------
    dim : int
        Output channels.
    n_filters : int
        Channels after the first convolution; each downsampling doubles them.
    ratios : sequence of int
        Downsampling ratios, applied in reverse order (EnCodec's convention).
    kernel_size : int
        Kernel size of the first convolution.
    last_kernel_size : int
        Kernel size of the last convolution.
    in_channels : int
        Input channels.
    n_residual_layers : int
        Residual blocks before each downsampling; block ``j`` has dilation
        ``dilation_base ** j``.
    dilation_base : int
        See ``n_residual_layers``.
    compress : int
        See :class:`SEANetResBlock`.
    true_skip : bool
        See :class:`SEANetResBlock`.
    causal : bool
        Left padding only, in every convolution.
    """

    def __init__(
        self,
        dim,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        in_channels=1,
        n_residual_layers=1,
        dilation_base=2,
        compress=2,
        true_skip=False,
        causal=False,
    ):
        layers = [SEANetConv1d(in_channels, n_filters, kernel_size, causal=causal)]
        mult = 1
        for ratio in reversed(ratios):
            layers += [
                SEANetResBlock(
                    mult * n_filters, dilation_base**j, compress, true_skip, causal
                )
                for j in range(n_residual_layers)
            ]
            layers += [
                nn.ELU(),
                SEANetConv1d(
                    mult * n_filters,
                    2 * mult * n_filters,
                    2 * ratio,
                    ratio,
                    causal=causal,
                ),
            ]
            mult *= 2
        layers += [
            SEANetLSTM(mult * n_filters),
            nn.ELU(),
            SEANetConv1d(mult * n_filters, dim, last_kernel_size, causal=causal),
        ]
        super().__init__(*layers)


class SEANetDecoder(nn.Sequential):
    """Mirror of :class:`SEANetEncoder` with transposed convolutions.

    Takes the same parameters, with ``out_channels`` in place of ``in_channels``.
    """

    def __init__(
        self,
        dim,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        out_channels=1,
        n_residual_layers=1,
        dilation_base=2,
        compress=2,
        true_skip=False,
        causal=False,
    ):
        mult = 2 ** len(ratios)
        layers = [SEANetConv1d(dim, mult * n_filters, kernel_size, causal=causal)]
        layers += [SEANetLSTM(mult * n_filters)]
        for ratio in ratios:
            out = mult * n_filters // 2
            layers += [
                nn.ELU(),
                SEANetConvTranspose1d(
                    mult * n_filters, out, 2 * ratio, ratio, causal=causal
                ),
            ]
            layers += [
                SEANetResBlock(out, dilation_base**j, compress, true_skip, causal)
                for j in range(n_residual_layers)
            ]
            mult //= 2
        layers += [
            nn.ELU(),
            SEANetConv1d(n_filters, out_channels, last_kernel_size, causal=causal),
        ]
        super().__init__(*layers)
