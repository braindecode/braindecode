import torch as th
from torch.autograd import Variable
import numpy as np
import random


def np_to_var(X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = th.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


def var_to_np(var):
    """Convenience function to transform `torch.Tensor` to numpy
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


def confirm_gpu_availability():
    """
    Should crash if gpu not available, attempts to create a FloatTensor on GPU.
    Returns
    -------
    success: bool
        Always returns true, should crash if gpu not available
    """
    a = th.FloatTensor(1).cuda()
    # Just make sure a is not somehow removed by any smart compiling,
    # probably not necessary.
    return a is not None
