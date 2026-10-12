# Authors: Tao Yang <sheeptao@outlook.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
from typing import Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin, _too_short_input_error


class MSVTNet(EEGModuleMixin, nn.Module):
    r"""MSVTNet model from Liu K et al (2024) from [msvt2024]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent` :bdg-info:`Attention/Transformer`

    This model implements a multi-scale convolutional transformer network
    for EEG signal classification, as described in [msvt2024]_.

    .. figure:: https://raw.githubusercontent.com/SheepTAO/MSVTNet/refs/heads/main/MSVTNet_Arch.png
       :align: center
       :alt: MSVTNet Architecture

    Parameters
    ----------
    n_filters_list : list[int], optional
        List of filter numbers for each TSConv block, by default (9, 9, 9, 9).
    conv1_kernels_size : list[int], optional
        List of kernel sizes for the first convolution in each TSConv block,
        by default (15, 31, 63, 125).
    conv2_kernel_size : int, optional
        Kernel size for the second convolution in TSConv blocks, by default 15.
    depth_multiplier : int, optional
        Depth multiplier for depthwise convolution, by default 2.
    pool1_size : int, optional
        Pooling size for the first pooling layer in TSConv blocks, by default 8.
    pool2_size : int, optional
        Pooling size for the second pooling layer in TSConv blocks, by default 7.
    drop_prob : float, optional
        Dropout probability for convolutional layers, by default 0.3.
    num_heads : int, optional
        Number of attention heads in the transformer encoder, by default 8.
    ffn_expansion_factor : float, optional
        Ratio to compute feedforward dimension in the transformer, by default 1.
    att_drop_prob : float, optional
        Dropout probability for the transformer, by default 0.5.
    num_layers : int, optional
        Number of transformer encoder layers, by default 2.
    activation : Type[nn.Module], optional
        Activation function class to use, by default nn.ELU.
    return_features : bool, optional
        Whether to return predictions from branch classifiers, by default False.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented based on the original code [msvt2024code]_.

    References
    ----------
    .. [msvt2024] Liu, K., et al. (2024). MSVTNet: Multi-Scale Vision
       Transformer Neural Network for EEG-Based Motor Imagery Decoding.
       IEEE Journal of Biomedical an Health Informatics.
    .. [msvt2024code] Liu, K., et al. (2024). MSVTNet: Multi-Scale Vision
       Transformer Neural Network for EEG-Based Motor Imagery Decoding.
       Source Code: https://github.com/SheepTAO/MSVTNet
    """

    def __init__(
        self,
        # braindecode parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        # Model's parameters
        n_filters_list: tuple[int, ...] = (9, 9, 9, 9),
        conv1_kernels_size: tuple[int, ...] = (15, 31, 63, 125),
        conv2_kernel_size: int = 15,
        depth_multiplier: int = 2,
        pool1_size: int = 8,
        pool2_size: int = 7,
        drop_prob: float = 0.3,
        num_heads: int = 8,
        ffn_expansion_factor: float = 1,
        att_drop_prob: float = 0.5,
        num_layers: int = 2,
        activation: Type[nn.Module] = nn.ELU,
        return_features: bool = False,
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

        self.return_features = return_features
        assert len(n_filters_list) == len(conv1_kernels_size), (
            "The length of n_filters_list and conv1_kernel_sizes should be equal."
        )

        self.ensure_dim = Rearrange("batch chans time -> batch 1 chans time")
        self.mstsconv = nn.ModuleList(
            [
                nn.Sequential(
                    _TSConv(
                        self.n_chans,
                        n_filters_list[b],
                        conv1_kernels_size[b],
                        conv2_kernel_size,
                        depth_multiplier,
                        pool1_size,
                        pool2_size,
                        drop_prob,
                        activation,
                    ),
                    Rearrange("batch channels 1 time -> batch time channels"),
                )
                for b in range(len(n_filters_list))
            ]
        )
        branch_linear_in = self._forward_flatten(cat=False)
        self.branch_head = nn.ModuleList(
            [
                _DenseLayers(branch_linear_in[b].shape[1], self.n_outputs)
                for b in range(len(n_filters_list))
            ]
        )

        seq_len, d_model = self._forward_mstsconv().shape[1:3]  # type: ignore
        self.transformer = _Transformer(
            seq_len,
            d_model,
            num_heads,
            ffn_expansion_factor,
            att_drop_prob,
            num_layers,
        )

        linear_in = self._forward_flatten().shape[1]  # type: ignore
        self.flatten_layer = nn.Flatten()
        self.final_layer = nn.Linear(linear_in, self.n_outputs)

    def _forward_mstsconv(
        self, cat: bool = True
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        x = torch.randn(1, 1, self.n_chans, self.n_times)
        try:
            x = [tsconv(x) for tsconv in self.mstsconv]
        except RuntimeError as exc:
            raise _too_short_input_error(exc, self.input_shape) or exc
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(
        self, cat: bool = True
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x with shape: (batch, n_chans, n_times)
        x = self.ensure_dim(x)
        # x with shape: (batch, 1, n_chans, n_times)
        x_list = [tsconv(x) for tsconv in self.mstsconv]
        # x_list contains 4 tensors, each of shape: [batch_size, seq_len, embed_dim]
        x = torch.stack(x_list, dim=2)
        x = x.view(x.size(0), x.size(1), -1)
        # x shape after concatenation: [batch_size, seq_len, total_embed_dim]
        x = self.transformer(x)
        # x shape after transformer: [batch_size, embed_dim]

        x = self.final_layer(x)
        if self.return_features:
            # Return BOTH the main-head logits and the stacked branch logits, as
            # in the authors' implementation (``return x, bx``). This is required
            # for the paper's joint deep-supervision loss
            # ``lambda * CE(main) + (1 - lambda) * sum_i CE(branch_i)`` and lets
            # callers score on the main head.
            # x: [batch_size, n_classes]; branches: [n_branches, batch_size, n_classes]
            branch_preds = [
                branch(x_list[idx]) for idx, branch in enumerate(self.branch_head)
            ]
            return x, torch.stack(branch_preds)
        return x


class _TSConv(nn.Sequential):
    r"""
    Time-Distributed Separable Convolution block.

    The architecture consists of:
    - **Temporal Convolution**
    - **Batch Normalization**
    - **Depthwise Spatial Convolution**
    - **Batch Normalization**
    - **Activation Function**
    - **First Pooling Layer**
    - **Dropout**
    - **Depthwise Temporal Convolution**
    - **Batch Normalization**
    - **Activation Function**
    - **Second Pooling Layer**
    - **Dropout**

    Parameters
    ----------
    n_channels : int
        Number of input channels (EEG channels).
    n_filters : int
        Number of filters for the convolution layers.
    conv1_kernel_size : int
        Kernel size for the first convolution layer.
    conv2_kernel_size : int
        Kernel size for the second convolution layer.
    depth_multiplier : int
        Depth multiplier for depthwise convolution.
    pool1_size : int
        Kernel size for the first pooling layer.
    pool2_size : int
        Kernel size for the second pooling layer.
    drop_prob : float
        Dropout probability.
    activation : Type[nn.Module], optional
        Activation function class to use, by default nn.ELU.
    """

    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        conv1_kernel_size: int,
        conv2_kernel_size: int,
        depth_multiplier: int,
        pool1_size: int,
        pool2_size: int,
        drop_prob: float,
        activation: Type[nn.Module] = nn.ELU,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(1, conv1_kernel_size),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(
                in_channels=n_filters,
                out_channels=n_filters * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=n_filters,
                bias=False,
            ),
            nn.BatchNorm2d(n_filters * depth_multiplier),
            activation(),
            nn.AvgPool2d(kernel_size=(1, pool1_size)),
            nn.Dropout(drop_prob),
            nn.Conv2d(
                in_channels=n_filters * depth_multiplier,
                out_channels=n_filters * depth_multiplier,
                kernel_size=(1, conv2_kernel_size),
                padding="same",
                groups=n_filters * depth_multiplier,
                bias=False,
            ),
            nn.BatchNorm2d(n_filters * depth_multiplier),
            activation(),
            nn.AvgPool2d(kernel_size=(1, pool2_size)),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() and x.device.type == "hpu":
            # On HPU without autograd the grouped convolution's forward is the
            # faster one.
            x = self[2](self[1](self[0](x)))
        elif x.numel() * self[0].weight.shape[0] >= 2**26:
            # ponytail: size heuristic. The spatial-first form adds ~50 small ops
            # per branch; they pay off once the (batch, filters, chans, times)
            # tensor it avoids reaches 2**26 elements (Gaudi lazy-mode crossover;
            # eager mode and CPU gain from smaller inputs already).
            x = self._spatial_first(x)
        else:
            # The depthwise (n_channels, 1) convolution as a matmul over the
            # channel axis: same values, a faster backward than the grouped
            # convolution's.
            x = self[1](self[0](x))
            b, f, c, t = x.shape
            x = torch.matmul(self[2].weight.view(f, -1, c), x).view(b, -1, 1, t)
        for i, layer in enumerate(self):
            if i > 2:
                x = layer(x)
        return x

    def _spatial_first(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal conv, batch norm and depthwise spatial conv (layers 0-2).

        The three are linear per filter, so the spatial weights project the
        input first and each filter's temporal kernel then runs on its
        projections; in training the batch norm's statistics come from
        :func:`_same_conv_moments`. The (batch, filters, chans, times) tensor
        is never built.
        """
        b, _, c, t = x.shape
        w = self[0].weight.flatten(1)
        f, k = w.shape
        w2 = self[2].weight.view(-1, c)
        d = w2.shape[0] // f
        if self[1].training or self[1].running_mean is None:
            mean, var = _same_conv_moments(x.reshape(b * c, t), w)
            if self[1].training and self[1].track_running_stats:
                self[1].num_batches_tracked.add_(1)
                momentum = self[1].momentum
                if momentum is None:
                    factor = 1.0 / float(self[1].num_batches_tracked)
                else:
                    factor = momentum
                n = b * c * t
                self[1].running_mean.mul_(1 - factor).add_(mean.detach(), alpha=factor)
                self[1].running_var.mul_(1 - factor).add_(
                    var.detach(), alpha=factor * n / (n - 1)
                )
        else:
            mean, var = self[1].running_mean, self[1].running_var
        scale = self[1].weight * torch.rsqrt(var + self[1].eps)
        shift = (self[1].bias - mean * scale).unsqueeze(1) * w2.sum(1).view(f, d)
        y = F.conv1d(
            torch.matmul(w2, x.reshape(b, c, t)),
            w.unsqueeze(1).expand(f, d, k).reshape(f * d, 1, k),
            padding="same",
            groups=f * d,
        )
        scale = scale.unsqueeze(1).expand(f, d).reshape(-1, 1)
        return torch.addcmul(shift.view(-1, 1), y, scale).unsqueeze(2)


def _same_conv_moments(
    x: torch.Tensor, w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and biased variance per filter of ``conv1d(x, w, padding="same")``.

    Over all rows and times, for ``x`` of shape (rows, times) and ``w`` of shape
    (filters, k), without building the (rows, filters, times) output: the sum of
    squares is the filter's autocorrelation against the input's (one matmul over
    blocks of k samples), minus the outputs that "same" crops at both ends. The
    input is centred first so the variance does not cancel against a large mean.
    """
    rows, t = x.shape
    f, k = w.shape
    n = rows * t
    m = x.mean()
    x = x - m
    # a[l] = sum over rows and s of x[s] x[s + l], for l < k
    nb = (t + k - 1) // k
    blk = F.pad(x, (0, (nb + 1) * k - t)).view(rows, nb + 1, k)
    nxt = torch.cat([blk[:, :nb], blk[:, 1:, : k - 1]], 2).reshape(-1, 2 * k - 1)
    a = blk[:, :nb].reshape(-1, k).t() @ nxt
    a = F.pad(a.reshape(-1), (0, k)).view(k, 2 * k)[:, :k].sum(0)
    # sum of squares over every output of the full (uncropped) correlation
    ww = F.conv1d(w.unsqueeze(0), w.unsqueeze(1), padding=k - 1, groups=f)[0]
    sq = torch.mv(ww, torch.cat([a[1:].flip([0]), a]))
    if k > 1:
        right = k - 1 - (k - 1) // 2
        pad = max(0, k - 1 - t)
        ends = torch.cat(
            [
                F.pad(x[:, : k - 1], (k - 1, pad)),
                F.pad(x[:, t - min(t, k - 1) :], (pad, k - 1)),
            ]
        )
        ends = F.conv1d(ends.unsqueeze(1), w.unsqueeze(1)).view(2, rows, f, k - 1)
        ends = ends.square().sum(1)
        sq = sq - ends[0, :, :right].sum(1) - ends[1, :, right:].sum(1)
    # undo the centring: conv(x + m) = conv(x) + m * v, v = conv(ones)
    u = F.conv1d(x.sum(0).view(1, 1, t), w.unsqueeze(1), padding="same")[0]
    v = F.conv1d(torch.ones_like(x[:1]).unsqueeze(0), w.unsqueeze(1), padding="same")[0]
    mean_c = u.sum(1) / n
    dv = v - v.mean(1, keepdim=True)
    var = (
        sq / n
        - mean_c.square()
        + m * (m * dv.square().mean(1) + 2 * (u * dv).sum(1) / n)
    )
    return mean_c + m * v.mean(1), var


class _PositionalEncoding(nn.Module):
    r"""
    Positional encoding module that adds learnable positional embeddings.

    Parameters
    ----------
    seq_length : int
        Sequence length.
    d_model : int
        Dimensionality of the model.
    """

    def __init__(self, seq_length: int, d_model: int) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_length, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        return x


class _Transformer(nn.Module):
    r"""
    Transformer encoder module with learnable class token and positional encoding.

    Parameters
    ----------
    seq_length : int
        Sequence length of the input.
    d_model : int
        Dimensionality of the model.
    num_heads : int
        Number of heads in the multihead attention.
    ffn_expansion_factor : float
        Ratio to compute the dimension of the feedforward network.
    drop_prob : float, optional
        Dropout probability, by default 0.5.
    num_layers : int, optional
        Number of transformer encoder layers, by default 4.
    """

    def __init__(
        self,
        seq_length: int,
        d_model: int,
        num_heads: int,
        ffn_expansion_factor: float,
        drop_prob: float = 0.5,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = _PositionalEncoding(seq_length + 1, d_model)

        dim_ff = int(d_model * ffn_expansion_factor)
        self.dropout = nn.Dropout(drop_prob)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                num_heads,
                dim_ff,
                drop_prob,
                batch_first=True,
                norm_first=True,
            ),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(batch_size, -1, -1), x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]


class _DenseLayers(nn.Sequential):
    r"""
    Final classification layers.

    Parameters
    ----------
    linear_in : int
        Input dimension to the linear layer.
    n_classes : int
        Number of output classes.
    """

    def __init__(self, linear_in: int, n_classes: int):
        super().__init__(
            nn.Flatten(),
            nn.Linear(linear_in, n_classes),
        )
