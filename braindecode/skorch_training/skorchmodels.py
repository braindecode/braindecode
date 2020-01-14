from contextlib import contextmanager
import numpy as np
from skorch.utils import to_tensor
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks.scoring import EpochScoring


class BraindecodeClassifier(NeuralNetClassifier):
    """
    Modification of skorch classifier to support models with `torch.nn.LogSoftmax`,
    not `torch.nn.Softmax`.

    Note: May be removed if skorch supports this.
    """
    # pylint: disable=unused-argument
    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.

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
        y_true = to_tensor(y_true, device=self.device)
        return self.criterion_(y_pred, y_true)

    def evaluate(self, X, y=None):
        dataset = self.get_dataset(X,y)

        # Compute Predictions for caching
        y_preds = list(self.forward_iter(dataset.X))
        # Extract callbacks that compute metrics (EpochScoring)
        # Only use those on train
        cbs = self._default_callbacks + self.callbacks
        epoch_cbs = [cb for name, cb in cbs
                     if isinstance(cb, EpochScoring) and cb.on_train]

        results = {}
        for cb in epoch_cbs:
            with cache_net_forward_iter(self, use_caching=True,
                                        y_preds=y_preds) as cached_net:
                cb.initialize()
                score = cb._scoring(cached_net, dataset.X, dataset.y)
                results[cb.name[len('train_'):]] = score

        # Compute losses "by hand", not using callbacks
        losses = []
        n_examples_per_batch = []
        for batch_y_preds, (_, batch_y) in zip(
                y_preds, self.get_iterator(dataset, training=False)):
            loss = self.get_loss(
                batch_y_preds, batch_y, training=False).cpu().numpy()
            # Multiply with actual batch size to get mean per example later
            losses.append(loss * len(batch_y))
            n_examples_per_batch.append(len(batch_y))
        loss_per_example = np.sum(losses) / np.sum(n_examples_per_batch)
        results['loss'] = loss_per_example
        return results


@contextmanager
def cache_net_forward_iter(net, use_caching, y_preds):
    """Caching context for ``skorch.NeuralNet`` instance.
    Returns a modified version of the net whose ``forward_iter``
    method will subsequently return cached predictions. Leaving the
    context will undo the overwrite of the ``forward_iter`` method.
    """
    if not use_caching:
        yield net
        return
    y_preds = iter(y_preds)

    # pylint: disable=unused-argument
    def cached_forward_iter(*args, device=net.device, **kwargs):
        for yp in y_preds:
            yield yp.to(device)

    net.forward_iter = cached_forward_iter
    try:
        yield net
    finally:
        # By setting net.forward_iter we define an attribute
        # `forward_iter` that precedes the bound method
        # `forward_iter`. By deleting the entry from the attribute
        # dict we undo this.
        del net.__dict__['forward_iter']

