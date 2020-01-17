import numpy as np
from skorch.utils import to_numpy
from skorch.classifier import NeuralNet
from skorch.classifier import NeuralNetClassifier


class EEGClassifier(NeuralNetClassifier):
    """Classifier that does not assume softmax activation.
    Calls loss function directly without applying log or anything.
    """
    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Return the loss for this batch by calling NeuralNet get_loss.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether train mode should be used or not.

        """
        return NeuralNet.get_loss(self, y_pred, y_true, *args, **kwargs)
