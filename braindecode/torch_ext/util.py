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


def to_dense_prediction_model(model):
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, 'stride'):
            stride_so_far *= np.array(module.stride)
            module.stride = (1, 1)
        if hasattr(module, 'dilation'):
            assert module.dilation == 1 or (module.dilation == (1,1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?")
            module.dilation = (int(stride_so_far[0]), int(stride_so_far[1]))
