# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn


class CroppedLoss(nn.Module):
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        """Forward pass.

        Parameters
        ----------
        preds: torch.Tensor
            Model's prediction with shape (batch_size, n_classes, n_times).
        targets: torch.Tensor
            Target labels with shape (batch_size, n_classes, n_times).
        """
        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)
