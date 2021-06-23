import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu, upsample
from torch.nn.modules.batchnorm import BatchNorm1d


# TODO: check extra params


class USleep(nn.Module):
    def __init__(
        self,
        n_classes=5,
        depth=3,
        dilation=1,
        dense_classifier_activation="tanh",
        kernel_size=9,
        transition_window=1,
        filters_init=5,
        complexity_factor=2,
    ):
        """TODO: remove redundant arguments."""
        super().__init__()

        # Set attributes
        padding = (kernel_size - 1) // 2  # to preserve dimension (check)
        complexity_factor = np.sqrt(complexity_factor)

        # Instantiate encoder : input has shape (B, C, T)
        encoder = []
        filters = filters_init
        for _ in range(depth):
            # update nb of input / output channels
            in_channels = 2 if _ == 0 else out_channels
            out_channels = int(filters * complexity_factor)

            # add encoder block (down)
            encoder += [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    ),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
            ]

            # update nb of filters
            filters = int(filters * np.sqrt(2))
        self.encoder = nn.Sequential(*encoder)

        # Instantiate bottom
        in_channels = out_channels
        out_channels = int(filters * complexity_factor)
        self.bottom = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

        # Instantiate decoder
        decoder_preskip = []
        decoder_postskip = []

        for _ in range(depth):
            # update nb of filters
            filters = int(np.ceil(filters / np.sqrt(2)))

            # update nb of input / output channels
            in_channels = out_channels
            out_channels = int(filters * complexity_factor)

            in_channels = out_channels

            # add decoder blocks (up)
            decoder_preskip += [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    ),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
            ]

            # we will concatenate channels via a skip connection, so they multiply by 2
            in_channels = 2 * out_channels

            # add encoder block (down)
            decoder_postskip += [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    ),  # to preserve dimension (check)
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
            ]

        self.decoder_preskip = nn.Sequential(*decoder_preskip)
        self.decoder_postskip = nn.Sequential(*decoder_postskip)

        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(out_channels * 3000, n_classes)
        )

    def forward(self, x):
        """Input x has shape (B, C, T)."""

        # encoder
        residuals = []
        for down in self.encoder:
            x = down(x)
            print(x.shape)
            residuals.append(x)
            x = nn.MaxPool1d(kernel_size=2)(x)

        x = self.bottom(x)
        # decoder
        residuals = residuals[::-1]  # in order of up layers
        for (idx, (up_preskip, up_postskip)) in enumerate(
            zip(self.decoder_preskip, self.decoder_postskip)
        ):
            x = up_preskip(x)
            x = torch.cat([x, residuals[idx]], axis=1)  # (B, 2 * C, T)
            x = up_postskip(x)
            print(x.shape)
        # return self.fc(x.flatten(start_dim=1))
        return x


# Small testing script
batch_size, n_channels, n_times = 1024, 2, 3000

np.random.seed(0)
x = np.random.random((batch_size, n_times, 1, n_channels))
x = np.moveaxis(x, 1, 3)
x = torch.tensor(x, dtype=torch.float32)
x = x.squeeze()

model = USleep()
model(x)