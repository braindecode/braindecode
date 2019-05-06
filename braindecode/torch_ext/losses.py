import torch as th


def log_categorical_crossentropy_1_hot(logpreds, targets, dims=None):
    """
    Returns log categorical crossentropy for given log-predictions and targets,
    targets should be one-hot-encoded.
    
    Computes :math:`-\mathrm{logpreds} \cdot \mathrm{targets}`
    
    Parameters
    ----------
    logpreds: `torch.autograd.Variable`
        Logarithm of softmax output.
    targets: `torch.autograd.Variable`
        One-hot encoded targets
    dims: int or iterable of int, optional.
        Compute sum across these dims

    Returns
    -------
    loss: `torch.autograd.Variable`
        :math:`-\mathrm{logpreds} \cdot \mathrm{targets}`
    """
    assert logpreds.size() == targets.size()
    result = -logpreds * targets
    # Sum across dims if axis given or more than 1 dim
    if dims is not None:
        if not hasattr(dims, "__len__"):
            dims = [dims]
        for dim in dims:
            result = th.sum(result, dim=int(dim))
    return result


def log_categorical_crossentropy(log_preds, targets, class_weights=None):
    """
    Returns log categorical crossentropy for given log-predictions and targets.
    
    Computes :math:`-\mathrm{logpreds} \cdot \mathrm{targets}` if you assume
    targets to be one-hot-encoded. Also works for targets that are not
    one-hot-encoded, in this case only uses targets that are in the range
    of the expected class labels, i.e., [0,log_preds.size()[1]-1].

    Parameters
    ----------
    log_preds: torch.autograd.Variable`
        Logarithm of softmax output.
    targets: `torch.autograd.Variable`
    class_weights: list of int, optional
        Weights given to loss of different classes

    Returns
    -------

    loss: `torch.autograd.Variable`
    """
    if log_preds.size() == targets.size():
        assert class_weights is None, "Class weights not implemented for one-hot"
        return log_categorical_crossentropy_1_hot(log_preds, targets)
    n_classes = log_preds.size()[1]
    n_elements = 0
    losses = []
    for i_class in range(n_classes):
        mask = targets == i_class
        mask = mask.type_as(log_preds)
        n_elements -= th.sum(mask)
        this_loss = th.sum(mask * log_preds[:, i_class])
        if class_weights is not None:
            this_loss = this_loss * class_weights[i_class]
        losses.append(this_loss)
    return th.sum(th.stack(losses)) / n_elements


def l2_loss(model):
    losses = [th.sum(p * p) for p in model.parameters()]
    return sum(losses)


def l1_loss(model):
    losses = [th.sum(th.abs(p)) for p in model.parameters()]
    return sum(losses)
