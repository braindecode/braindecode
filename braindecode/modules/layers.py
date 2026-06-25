# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)
#
# ``FourierEmb`` and ``ChannelMerger`` below are re-implemented from
# ``neuraltrain.models.common`` (Meta Platforms, Inc., MIT License) to match the
# upstream attention scores numerically (parity-gated); shared by BrainModule
# and DANCE.
from __future__ import annotations

import math
from collections import deque

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import drop_path


class Ensure4d(nn.Module):
    """Ensure the input tensor has 4 dimensions.

    This is a small utility layer that repeatedly adds a singleton dimension at
    the end until the input has shape ``(batch, channels, time, 1)``.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import Ensure4d
    >>> module = Ensure4d()
    >>> outputs = module(torch.randn(2, 3, 10))
    >>> outputs.shape
    torch.Size([2, 3, 10, 1])
    """

    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class Chomp1d(nn.Module):
    """Remove samples from the end of a sequence.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import Chomp1d
    >>> module = Chomp1d(chomp_size=5)
    >>> inputs = torch.randn(2, 3, 20)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 3, 15])
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def extra_repr(self):
        return "chomp_size={}".format(self.chomp_size)

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TimeDistributed(nn.Module):
    """Apply module on multiple windows.

    Apply the provided module on a sequence of windows and return their
    concatenation.
    Useful with sequence-to-prediction models (e.g. sleep stager which must map
    a sequence of consecutive windows to the label of the middle window in the
    sequence).

    Parameters
    ----------
    module : nn.Module
        Module to be applied to the input windows. Must accept an input of
        shape (batch_size, n_channels, n_times).

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from braindecode.modules import TimeDistributed
    >>> module = TimeDistributed(nn.Conv1d(3, 4, kernel_size=3, padding=1))
    >>> inputs = torch.randn(2, 5, 3, 20)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 5, 4])
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Sequence of windows, of shape (batch_size, seq_len, n_channels,
            n_times).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, seq_len, output_size).
        """
        b, s, c, t = x.shape
        out = self.module(x.view(b * s, c, t))
        return out.view(b, s, -1)


class DropPath(nn.Module):
    """Drop paths, also known as Stochastic Depth, per sample.

    When applied in main path of residual blocks.

    Parameters
    ----------
    drop_prob: float (default=None)
        Drop path probability (should be in range 0-1).

    Notes
    -----
    Code copied and modified from VISSL facebookresearch:
    https://github.com/facebookresearch/vissl/blob/0b5d6a94437bc00baed112ca90c9d78c6ccfbafb/vissl/models/model_helpers.py#L676

    All rights reserved.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import DropPath
    >>> module = DropPath(drop_prob=0.2)
    >>> module.train()
    >>> inputs = torch.randn(2, 3, 10)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 3, 10])
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    # Utility function to print DropPath module
    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class SqueezeFinalOutput(nn.Module):
    """

    Removes empty dimension at end and potentially removes empty time
    dimension. It does not just use squeeze as we never want to remove
    first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    def __init__(self):
        super().__init__()

        self.squeeze = Rearrange("b c t 1 -> b c t")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) drop feature dim
        x = self.squeeze(x)
        # 2) drop time dim if singleton
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x


class SubjectLayers(nn.Module):
    """Per-subject linear transformation layer.

    Applies subject-specific linear transformations to the input. Each subject
    owns an independent weight matrix, enabling personalized feature
    processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_id: bool = False,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            if in_channels != out_channels:
                raise AssertionError("init_id requires in_channels == out_channels")
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / (in_channels**0.5)

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        """Apply the subject-specific linear transforms."""
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self) -> str:
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class FourierEmb(nn.Module):
    """2D Fourier positional embedding over electrode ``(x, y)`` in ``[0, 1]``.

    Matches ``neuraltrain.models.common.FourierEmbModel`` (parity-gated) so
    :class:`ChannelMerger` produces identical attention scores.

    Parameters
    ----------
    dimension : int
        Output embedding dimension. ``dimension == (n_freqs ** 2) * 2`` with
        ``n_freqs = round((dimension / 2) ** 0.5)`` (must be an integer; ``32``
        for the default ``2048``).
    margin : float
        Position offset/width term from the upstream construction.
    """

    def __init__(self, dimension: int = 2048, margin: float = 0.2):
        super().__init__()
        n_freqs_f = (dimension / 2) ** (1 / 2)
        n_freqs = round(n_freqs_f)
        if abs(n_freqs_f - n_freqs) > 1e-6:
            raise ValueError("(dimension / 2) ** (1 / 2) must be an integer.")
        self.n_dims = 2
        self.margin = margin
        freqs = torch.arange(n_freqs)
        width = 1 + 2 * self.margin
        pos = 2 * math.pi * freqs / width
        self.register_buffer("pos", pos)

    @staticmethod
    def _outer_sum(x: torch.Tensor) -> torch.Tensor:
        # Outer sum over the last (n_dims) axes -> grid of size n_freqs**n_dims.
        inds = deque([slice(None)] + [None] * (x.shape[-1] - 1))
        out = x[..., 0][(...,) + tuple(inds)]
        for i in range(1, x.shape[-1]):
            inds.rotate()
            out = out + x[..., i][(...,) + tuple(inds)]
        return out

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, n_chans, 2) in [0, 1].
        *o, d = positions.shape
        if d != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} positions, got {d}.")
        positions = positions + self.margin
        locs = torch.einsum("bcd,f->bcfd", positions, self.pos)
        loc_grid = self._outer_sum(locs).view(*o, -1)  # (B, n_chans, n_freqs**2)
        emb = torch.cat([torch.cos(loc_grid), torch.sin(loc_grid)], dim=-1)
        return emb  # (B, n_chans, (n_freqs**2)*2) == (..., dimension)


class ChannelMerger(nn.Module):
    """Spatial Fourier-attention merge: ``n_chans`` -> ``out_channels``.

    Montage-agnostic spatial attention: each virtual channel softmax-attends
    over the input channels using a learned head weighted against a Fourier
    embedding of the electrode ``(x, y)`` positions. Matches
    ``neuraltrain.models.common.ChannelMergerModel`` (parity-gated); braindecode
    has no subjects, so ``per_subject=False`` and ``subject_ids`` is dropped.

    Parameters
    ----------
    out_channels : int
        Number of virtual output channels. The default is ``270``.
    pos_dim : int
        Fourier embedding dimension. The default is ``2048``.
    dropout : float
        Spatial-attention dropout *radius* (non-parametric, training only) --
        NOT a Bernoulli probability. Each training step draws a random center in
        normalized ``[0, 1]^2`` position space and bans every channel within
        this radius from the softmax. Values ``>= 1`` ban all channels and zero
        the output. The default is ``0.2``.
    invalid_value : float
        Position value marking padded/invalid channels (masked out of the
        softmax). The default is ``-0.1``.
    """

    def __init__(
        self,
        out_channels: int = 270,
        pos_dim: int = 2048,
        dropout: float = 0.2,
        invalid_value: float = -0.1,
    ):
        super().__init__()
        self.heads = nn.Parameter(torch.randn(out_channels, pos_dim))
        self.heads.data /= pos_dim**0.5
        self.embedding = FourierEmb(pos_dim)
        self.n_dims = self.embedding.n_dims
        self.dropout = dropout
        self.invalid_value = invalid_value

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chans, T); positions: (B, n_chans, 2). Transcribed from
        # neuraltrain ChannelMergerModel (per_subject=False, embed_ref=False,
        # unmerge=False; subject_ids unused -- braindecode has no subjects).
        B, C, _ = positions.shape
        embedding = self.embedding(positions)  # (B, n_chans, pos_dim)
        # score_offset lives on ``positions``' device (the invalid_mask is derived
        # from positions) and in ``x``'s dtype (so the final ``weights @ x`` works
        # under half precision); numerics-neutral in the usual fp32 same-device case.
        score_offset = torch.zeros(B, C, device=positions.device, dtype=x.dtype)
        invalid_mask = (positions == self.invalid_value).all(dim=-1)
        score_offset = score_offset.masked_fill(invalid_mask, float("-inf"))
        if self.training and self.dropout:
            center = torch.rand(self.n_dims, device=x.device)
            banned = (positions[:, :, : self.n_dims] - center).norm(
                dim=-1
            ) <= self.dropout
            score_offset = score_offset.masked_fill(banned, float("-inf"))
        heads = self.heads[None].expand(B, -1, -1)  # (B, out, pos_dim)
        scores = torch.einsum("bcd,bod->boc", embedding, heads)  # (B, out, n_chans)
        scores = scores + score_offset[:, None]
        weights = torch.softmax(scores, dim=2).nan_to_num()  # over n_chans
        out = weights @ x  # (B, out, n_chans) @ (B, n_chans, T) -> (B, out, T)
        return out  # (B, out_channels, T)
