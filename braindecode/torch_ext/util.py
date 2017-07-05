import torch as th
from torch.autograd import Variable
import numpy as np
import random


def np_to_var(X, requires_grad=False, **var_kwargs):
    """
    Convenience function to transform numpy array to `torch.autograd.Variable`.
        
    Returns
    -------
    var: `torch.autograd.Variable`
    """
    return Variable(th.from_numpy(X), requires_grad=requires_grad, **var_kwargs)


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
