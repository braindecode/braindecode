# Authors: Tao Yang <sheeptao@outlook.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from typing import List, Type, Union, Tuple
from braindecode.models.base import EEGModuleMixin


class TSConv(nn.Sequential):
    """
    Time-Distributed Separable Convolution block.

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


class PositionalEncoding(nn.Module):
    """
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


class Transformer(nn.Module):
    """
    Transformer encoder module with learnable class token and positional encoding.

    Parameters
    ----------
    seq_length : int
        Sequence length of the input.
    d_model : int
        Dimensionality of the model.
    num_heads : int
        Number of heads in the multihead attention.
    feedforward_ratio : float
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
        feedforward_ratio: float,
        drop_prob: float = 0.5,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = PositionalEncoding(seq_length + 1, d_model)

        dim_ff = int(d_model * feedforward_ratio)
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


class FinalLayers(nn.Sequential):
    """
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


class MSVTNet(EEGModuleMixin, nn.Module):
    """MSVTNet model from Liu K et al (2024) from [mvtnet]_.

    This model implements a multi-scale convolutional transformer network
    for EEG signal classification, as described in [mvtnet]_.

    .. figure:: https://raw.githubusercontent.com/SheepTAO/MSVTNet/refs/heads/main/MSVTNet_Arch.png
       :align: center
       :alt: MSVTNet Architecture

    Parameters
    ----------
    n_filters_list : List[int], optional
        List of filter numbers for each TSConv block, by default [9, 9, 9, 9].
    conv1_kernel_sizes : List[int], optional
        List of kernel sizes for the first convolution in each TSConv block,
        by default [15, 31, 63, 125].
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
    feedforward_ratio : float, optional
        Ratio to compute feedforward dimension in the transformer, by default 1.
    drop_prob_trans : float, optional
        Dropout probability for the transformer, by default 0.5.
    num_layers : int, optional
        Number of transformer encoder layers, by default 2.
    activation : Type[nn.Module], optional
        Activation function class to use, by default nn.ELU.
    return_branch_preds : bool, optional
        Whether to return predictions from branch classifiers, by default True.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented bosed on the original code [msvtnetcode]_.


    References
    ----------
    .. [mvtnet] Liu, K., Yang, T., Yu, Z., Yi, W., Yu, H., Wang, G., & Wu, W. (2024).
       MSVTNet: Multi-Scale Vision Transformer Neural Network for EEG-Based
       Motor Imagery Decoding. *IEEE Journal of Biomedical and Health Informatics*.
    .. [msvtnetcode] Liu, K., et al. (2024). MSVTNet: Multi-Scale Vision
       Transformer Neural Network for EEG-Based Motor Imagery Decoding.
       Source Code: https://github.com/SheepTAO/MSVTNet
    """

    def __init__(
        self,
        # braindecode parameters
        n_times=None,
        n_chans=None,
        n_outputs=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        # Model's parameters
        n_filters_list: List[int] = [9, 9, 9, 9],
        conv1_kernel_sizes: List[int] = [15, 31, 63, 125],
        conv2_kernel_size: int = 15,
        depth_multiplier: int = 2,
        pool1_size: int = 8,
        pool2_size: int = 7,
        drop_prob: float = 0.3,
        num_heads: int = 8,
        feedforward_ratio: float = 1,
        drop_prob_trans: float = 0.5,
        num_layers: int = 2,
        activation: Type[nn.Module] = nn.ELU,
        return_branch_preds: bool = True,
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

        self.return_branch_preds = return_branch_preds
        assert len(n_filters_list) == len(
            conv1_kernel_sizes
        ), "The length of n_filters_list and conv1_kernel_sizes should be equal."

        self.ensure_dim = Rearrange("batch chans time -> batch 1 chans time")
        self.mstsconv = nn.ModuleList(
            [
                nn.Sequential(
                    TSConv(
                        self.n_chans,
                        n_filters_list[b],
                        conv1_kernel_sizes[b],
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
                FinalLayers(branch_linear_in[b].shape[1], self.n_outputs)
                for b in range(len(n_filters_list))
            ]
        )

        seq_len, d_model = self._forward_mstsconv().shape[1:3]  # type: ignore
        self.transformer = Transformer(
            seq_len,
            d_model,
            num_heads,
            feedforward_ratio,
            drop_prob_trans,
            num_layers,
        )

        linear_in = self._forward_flatten().shape[1]  # type: ignore
        self.flatten_layer = nn.Flatten()
        self.final_layers = nn.Linear(linear_in, self.n_outputs)

    def _forward_mstsconv(
        self, cat: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = torch.randn(1, 1, self.n_chans, self.n_times)
        x = [tsconv(x) for tsconv in self.mstsconv]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(
        self, cat: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        x = self.ensure_dim(x)
        x_list = [tsconv(x) for tsconv in self.mstsconv]
        branch_preds = [
            branch(x_list[idx]) for idx, branch in enumerate(self.branch_head)
        ]
        x = torch.cat(x_list, dim=2)
        x = self.transformer(x)
        x = self.final_layers(x)
        if self.return_branch_preds:
            return x, branch_preds
        else:
            return x
