# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


def identity(x):
    return x


def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def transpose_time_to_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 3, 2, 1)


def _modify_eig_forward(X, function_applied):
    """Modified eigenvalue of symmetric matrix X."""
    s, U = torch.linalg.eigh(X)
    s_modified = function_applied(s)
    output = U @ torch.diag_embed(s_modified) @ U.transpose(-1, -2)
    return output, s, U, s_modified


def _modify_eig_backward(grad_output, s, U, s_modified, derivative, threshold=1e-4):
    """Backward pass of modified eigenvalue of symmetric matrix X."""
    # compute Loewner matrix
    denominator = s.unsqueeze(-1) - s.unsqueeze(-1).transpose(-1, -2)
    is_eq = denominator.abs() < threshold
    denominator[is_eq] = 1.0

    # case: sigma_i != sigma_j
    numerator = s_modified.unsqueeze(-1) - s_modified.unsqueeze(-1).transpose(-1, -2)

    # case: sigma_i == sigma_j
    s_derivativated = derivative(s)
    numerator[is_eq] = 0.5 * (
        s_derivativated.unsqueeze(-1) + s_derivativated.unsqueeze(-1).transpose(-1, -2)
    )[is_eq]
    L = numerator / denominator

    grad_input = U @  (L * (U.transpose(-1, -2) @ grad_output @ U)) @ U.transpose(-1, -2)

    return grad_input


class logm(torch.autograd.Function):
    """Matrix logarithm of a symmetric matrix.

    This class computes the matrix logarithm of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Matrix logarithm of X

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
            for SPD neural networks, NeurIPS
    """
    @staticmethod
    def forward(ctx, X):
        def function_applied(s): return s.log()
        output, s, U, s_modified = _modify_eig_forward(X, function_applied)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        def derivative(s): return s.reciprocal()
        return _modify_eig_backward(grad_output, s, U, s_modified, derivative)


class regm(torch.autograd.Function):
    """Regularized matrix logarithm of a symmetric matrix.

    This class computes the regularized matrix logarithm of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Regularized matrix logarithm of X

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
            for SPD neural networks, NeurIPS
    """
    @staticmethod
    def forward(ctx, X, threshold):
        def function_applied(s): return s.clamp(min=threshold)
        output, s, U, s_modified = _modify_eig_forward(X, function_applied)
        ctx.save_for_backward(s, U, s_modified)
        ctx.threshold_ = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        def derivative(s): return s > ctx.threshold_
        return _modify_eig_backward(grad_output, s, U, s_modified, derivative), None
