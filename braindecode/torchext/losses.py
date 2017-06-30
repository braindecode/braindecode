import torch as th

def log_categorical_crossentropy(preds, targets, dims=None):
    assert preds.size() == targets.size()
    result = -preds * targets
    # Sum across dims if axis given or more than 1 dim
    if dims is not None:
        if hasattr(dims, '__len__'):
            for dim in dims:
                result = th.sum(result, dim=int(dim))
    return result