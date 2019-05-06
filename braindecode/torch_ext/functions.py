import torch as th


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return th.log(th.clamp(x, min=eps))


def identity(x):
    return x
