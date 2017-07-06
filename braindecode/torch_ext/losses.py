import torch as th


def log_categorical_crossentropy(logpreds, targets, dims=None):
    """
    Returns log categorical crossentropy for given log-predictions and targets.
    
    Computes :math:`-\mathrm{logpreds} \cdot \mathrm{targets}`
    
    Parameters
    ----------
    logpreds: `torch.autograd.Variable`
        Logarithm of softmax output.
    targets: `torch.autograd.Variable`
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
        if not hasattr(dims, '__len__'):
            dims = [dims]
        for dim in dims:
            result = th.sum(result, dim=int(dim))
    return result


def l2_loss(model):
    losses = [th.sum(p * p) for p in model.parameters()]
    return sum(losses)


def l1_loss(model):
    losses = [th.sum(th.abs(p)) for p in model.parameters()]
    return sum(losses)
