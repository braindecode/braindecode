# This is the script of EEG-Deformer
# This is the network script
import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.eegdeformer import _Transformer as Transformer
from braindecode.models.eegnet import Conv2dWithConstraint


class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(
                1,
                out_chan,
                kernel_size,
                padding=self.get_padding(kernel_size[-1]),
                max_norm=2,
            ),
            Conv2dWithConstraint(
                out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2
            ),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
        )

    def __init__(
        self,
        num_chan,
        num_time,
        temporal_kernel,
        num_classes,
        num_kernel=64,
        depth=4,
        heads=16,
        mlp_dim=16,
        dim_head=16,
        dropout=0.0,
    ):
        super().__init__()

        self.cnn_encoder = self.cnn_block(
            out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan
        )

        dim = int(0.5 * num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange("b k c f -> b k (c f)")

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            in_chan=num_kernel,
            fine_grained_kernel=temporal_kernel,
        )

        out_size = int(num_kernel * int(dim * (0.5**depth))) + int(num_kernel * depth)

        self.mlp_head = nn.Sequential(nn.Linear(out_size, num_classes))

    def forward(self, eeg):
        # eeg: (b, chan, time)
        eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        x += self.pos_embedding
        x = self.transformer(x)
        return self.mlp_head(x)

    @staticmethod
    def get_padding(kernel):
        return 0, int(0.5 * (kernel - 1))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5**i)) for i in range(num_layer + 1)]


if __name__ == "__main__":
    data = torch.ones((16, 32, 1000))
    emt = Deformer(
        num_chan=32,
        num_time=1000,
        temporal_kernel=11,
        num_kernel=64,
        num_classes=2,
        depth=4,
        heads=16,
        mlp_dim=16,
        dim_head=16,
        dropout=0.5,
    )
    print(emt)

    out = emt(data)
