# @title ▶️ Run this first to define the model class

import torch
from torch import nn


class ConvSequence(nn.Module):
    """Sequence of residual, dilated convolutional layers with GLU activation."""

    def __init__(
        self,
        channels: list[int],
        kernel_size: int = 4,
        dilation_growth: int = 2,
        dilation_period: int = 5,
        glu: int = 2,
        glu_context: int = 1,
    ) -> None:
        super().__init__()

        if dilation_growth > 1:
            assert kernel_size % 2 != 0, (
                "Supports only odd kernel with dilation for now"
            )

        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()

        dilation = 1
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: list[nn.Module] = []

            # conv layer
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1

            layers.extend(
                [
                    nn.Conv1d(
                        chin,
                        chout,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2 * dilation,
                        dilation=dilation,
                        groups=1,
                    ),
                    nn.BatchNorm1d(num_features=chout),
                    nn.GELU(),
                ]
            )
            dilation *= dilation_growth

            self.sequence.append(nn.Sequential(*layers))
            if (k + 1) % glu == 0:
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(
                            chout, 2 * chout, 1 + 2 * glu_context, padding=glu_context
                        ),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.glus.append(None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ind, module in enumerate(self.sequence):
            x = x + module(x)
            if self.glus[ind] is not None:
                x = self.glus[ind](x)
        return x


class DilatedConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_hidden_dims: int = 320,
        n_conv_blocks: int = 2,
        kernel_size: int = 3,
        growth: float = 1.0,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "For padding to work, this must be verified"
        self.input_linear = nn.Conv1d(in_channels, n_hidden_dims, 1)

        # Build sequence of convolutional layers
        sizes = [n_hidden_dims] + [
            int(round(n_hidden_dims * growth**k)) for k in range(n_conv_blocks * 2)
        ]
        self.encoder = ConvSequence(
            sizes,
            kernel_size=kernel_size,
        )

        # Temporal aggregation
        self.time_aggregation = nn.LazyLinear(1)
        self.output_linear = nn.Linear(n_hidden_dims, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, T = x.shape
        x = self.input_linear(x)
        x = self.encoder(x)
        x = self.time_aggregation(x).squeeze(-1)  # (B, F, 1) -> (B, F)
        x = self.output_linear(x)
        return x
