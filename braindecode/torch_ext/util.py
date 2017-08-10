import torch as th
from torch.autograd import Variable
import numpy as np
import random


def np_to_var(X, requires_grad=False, dtype=None, pin_memory=False, **var_kwargs):
    """
    Convenience function to transform numpy array to `torch.autograd.Variable`.
        
    Converts `X` to ndarray using asarray if necessary.
    
    Parameters
    ----------
    X: ndarray or list
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor
    
    Returns
    -------
    var: `torch.autograd.Variable`
    """
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = th.from_numpy(X)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return Variable(X_tensor, requires_grad=requires_grad, **var_kwargs)


def var_to_np(var):
    """Convenience function to transform `torch.autograd.Variable` to numpy
    array.
    
    Should work both for CPU and GPU."""
    return var.cpu().data.numpy()


def set_random_seeds(seed, cuda):
    """
    Set seeds for python random module numpy.random and torch.
    
    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.

    """
    random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
