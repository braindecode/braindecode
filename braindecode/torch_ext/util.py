import torch as th
from torch.autograd import Variable
import numpy as np
import random


def np_to_var(X, requires_grad=False, **var_kwargs):
    return Variable(th.from_numpy(X), requires_grad=requires_grad, **var_kwargs)


def var_to_np(var):
    return var.cpu().data.numpy()


def set_random_seeds(seed, cuda):
    random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def to_dense_prediction_model(model, axis=(2,3)):
    """
    
    Parameters
    ----------
    model
    axis

    Returns
    -------

    """
    if not hasattr(axis, '__len__'):
        axis = [axis]
    assert all([ax in [2,3] for ax in axis]), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, 'stride'):
            if not hasattr(module.stride, '__len__'):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)
        if hasattr(module, 'dilation'):
            assert module.dilation == 1 or (module.dilation == (1,1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?")
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
