# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
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


class TimeSeriesLoss(nn.Module):
    """Compute Loss between timeseries targets and predictions.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)
    Assumes targets are in shape:
    n_batch size x n_classes x window_len (in time)
    If the targets contain NaNs, the NaNs will be masked out and the loss will be only computed for
    predictions valid corresponding to valid target values."""

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
        n_preds = preds.shape[-1]
        # slice the targets to fit preds shape
        targets = targets[:, :, -n_preds:]
        # create valid targets mask
        mask = ~torch.isnan(targets)
        # select valid targets that have a matching predictions
        masked_targets = targets[mask]
        masked_preds = preds[mask]
        return self.loss_function(masked_preds, masked_targets)


def mixup_criterion(preds, target):
    """Implements loss for Mixup for EEG data. See [1]_.

    Implementation based on [2]_.

    Parameters
    ----------
    preds : torch.Tensor
        Predictions from the model.
    target : torch.Tensor | list of torch.Tensor
        For predictions without mixup, the targets as a tensor. If mixup has
        been applied, a list containing the targets of the two mixed
        samples and the mixing coefficients as tensors.

    Returns
    -------
    loss : float
        The loss value.

    References
    ----------
    .. [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
       mixup: Beyond Empirical Risk Minimization
       Online: https://arxiv.org/abs/1710.09412
    .. [2] https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
    if len(target) == 3:
        # unpack target
        y_a, y_b, lam = target
        # compute loss per sample
        loss_a = torch.nn.functional.nll_loss(preds,
                                              y_a,
                                              reduction='none')
        loss_b = torch.nn.functional.nll_loss(preds,
                                              y_b,
                                              reduction='none')
        # compute weighted mean
        ret = torch.mul(lam, loss_a) + torch.mul(1 - lam, loss_b)
        return ret.mean()
    else:
        return torch.nn.functional.nll_loss(preds,
                                            target)
