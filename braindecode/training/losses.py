import torch


class CroppedLoss:
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        """
        Initialize loss function.

        Args:
            self: (todo): write your description
            loss_function: (todo): write your description
        """
        self.loss_function = loss_function

    def __call__(self, preds, targets):
        """
        Calculate function.

        Args:
            self: (todo): write your description
            preds: (array): write your description
            targets: (list): write your description
        """
        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)
