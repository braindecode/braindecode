import torch


class CroppedNLLLoss:
    """Compute NLL Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __call__(self, preds, targets):
        return torch.nn.functional.nll_loss(torch.mean(preds, dim=2), targets)
