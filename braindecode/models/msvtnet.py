# Authors: Tao Yang <sheeptao@outlook.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)
import torch
from torch import nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin


class TSConv(nn.Sequential):
    """
    Temporal-Spatial Convolution Block for feature extraction.
    """

    def __init__(
        self,
        n_chans: int,
        F: int,
        C1: int,
        C2: int,
        D: int,
        P1: int,
        P2: int,
        drop_prob: float,
    ):
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding="same", bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (n_chans, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(drop_prob),
            nn.Conv2d(F * D, F * D, (1, C2), padding="same", groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(drop_prob),
        )


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module for sequence processing.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        ff_ratio: int,
        dropout_prob: float,
        num_layers: int,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = PositionalEncoding(seq_len + 1, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        dim_feedforward = d_model * ff_ratio
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return x[:, 0]  # Return the CLS token representation


class ClassificationHead(nn.Sequential):
    """
    Classification head with a linear layer and log-softmax activation.
    """

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()


class MSVTNet(EEGModuleMixin, nn.Module):
    """
    MSVTNet model adapted for braindecode.

    work a little more tomorrow.

    Parameters
    ----------

    F : List[int], default=[9, 9, 9, 9]
        Filter sizes for each branch.
    C1 : List[int], default=[15, 31, 63, 125]
        Kernel sizes for the first convolution in each branch.
    C2 : int, default=15
        Kernel size for the second convolution.
    D : int, default=2
        Depth multiplier for depthwise convolution.
    P1 : int, default=8
        Pooling size after first convolution.
    P2 : int, default=7
        Pooling size after second convolution.
    drop_prob : float, default=0.3
        Dropout probability.
    nhead : int, default=8
        Number of attention heads in the transformer.
    ff_ratio : int, default=1
        Feedforward network expansion factor in the transformer.
    transformer_dropout : float, default=0.5
        Dropout probability in the transformer.
    num_layers : int, default=2
        Number of transformer encoder layers.
    branch_predictions : bool, default=True
        Whether to output predictions from each branch.
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans: int,
        n_times: int,
        n_outputs: int,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model-specific parameters
        F=None,
        C1=None,
        C2: int = 15,
        D: int = 2,
        P1: int = 8,
        P2: int = 7,
        drop_prob: float = 0.3,
        nhead: int = 8,
        ff_ratio: int = 1,
        transformer_dropout: float = 0.5,
        num_layers: int = 2,
        branch_predictions: bool = True,
    ):
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.branch_predictions = branch_predictions

        if F is None:
            F = [9, 9, 9, 9]
        if C1 is None:
            C1 = [15, 31, 63, 125]

        assert len(F) == len(C1), "The length of F and C1 should be equal."

        # Multi-Scale Temporal-Spatial Convolution Branches
        self.branches = nn.ModuleList()
        for f, c1 in zip(F, C1):
            branch = nn.Sequential(
                TSConv(
                    n_chans=self.n_chans,
                    F=f,
                    C1=c1,
                    C2=C2,
                    D=D,
                    P1=P1,
                    P2=P2,
                    drop_prob=drop_prob,
                ),
                Rearrange("batch channels 1 time -> batch time channels"),
            )
            self.branches.append(branch)

        # Determine the feature dimension after branches
        example_input = torch.randn(1, 1, self.n_chans, self.n_times)
        branch_outputs = [branch(example_input) for branch in self.branches]
        seq_len, d_model = branch_outputs[0].shape[1:]

        # Branch Classification Heads
        self.branch_heads = nn.ModuleList()
        for branch_output in branch_outputs:
            input_dim = branch_output.shape[1] * branch_output.shape[2]
            self.branch_heads.append(ClassificationHead(input_dim, self.n_outputs))

        # Transformer Encoder
        total_d_model = sum([output.shape[2] for output in branch_outputs])
        self.transformer = TransformerEncoder(
            seq_len=seq_len,
            d_model=total_d_model,
            nhead=nhead,
            ff_ratio=ff_ratio,
            dropout_prob=transformer_dropout,
            num_layers=num_layers,
        )

        # Final Classification Head
        self.final_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_d_model, self.n_outputs),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the MSVTNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, n_outputs).
        List[torch.Tensor], optional
            If branch_predictions is True, returns predictions from each branch.
        """
        # Reshape input to (batch_size, 1, n_chans, n_times)
        x = x.unsqueeze(1)

        # Process through branches
        branch_outputs = [branch(x) for branch in self.branches]

        # Branch predictions
        if self.branch_predictions:
            branch_preds = [
                head(output.flatten(1))
                for output, head in zip(branch_outputs, self.branch_heads)
            ]

        # Concatenate branch outputs along feature dimension
        x = torch.cat(branch_outputs, dim=2)

        # Transformer encoding
        x = self.transformer(x)

        # Final classification
        x = self.final_head(x)

        if self.branch_predictions:
            return x, branch_preds
        else:
            return x
