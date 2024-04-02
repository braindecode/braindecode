# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
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
    """Modified eigenvalue of symmetric matrix X.

    Steps:

    - Computes the eigenvalue decomposition of a real symmetric matrix.
    - Applies a function to the eigenvalues of the matrix.
    - Reconstructs the matrix with the modified eigenvalues.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (n_channels, n_channels)
    function_applied : callable function to be applied to the eigenvalues of X
        Function to apply to the eigenvalues of X

    Returns
    -------
    output : torch.Tensor
        Modified eigenvalue of X
    s : torch.Tensor
        Eigenvalues of X
    U : torch.Tensor
        Eigenvectors of X
    s_modified : torch.Tensor
        Modified eigenvalues of X
    """
    s, U = torch.linalg.eigh(X)
    s_modified = function_applied(s)
    output = U @ torch.diag_embed(s_modified) @ U.transpose(-1, -2)
    return output, s, U, s_modified


def _modify_eig_backward(grad_output, s, U, s_modified, derivative,
                         threshold=1e-4):
    """
    Backward pass of modified eigenvalue of symmetric matrix X.

    Parameters
    ----------
    grad_output : torch.Tensor
        Gradient of the loss with respect to the output of the forward pass
    s: torch.Tensor
        Eigenvalues of X
    U: torch.Tensor
        Eigenvectors of X
    s_modified: torch.Tensor
        Modified eigenvalues of X
    derivative: callable function to be applied to the eigenvalues of X
        Derivative of the function applied to the eigenvalues of X
    threshold: float
        Threshold to consider two eigenvalues as equal to each other

    Returns
    -------
    grad_input : torch.Tensor
        Gradient of the loss with respect to the input of the forward pass

    """
    # compute Loewner matrix
    # Compute the difference between each pair of eigenvalues,
    # expanding dimensions
    denominator = s.unsqueeze(-1) - s.unsqueeze(-2)

    # Create a boolean mask where the absolute difference between
    # eigenvalues is less than the threshold
    is_eq = denominator.abs() < threshold
    # For pairs of eigenvalues that are considered equal,
    # set the denominator to 1 to avoid division by zero
    denominator[is_eq] = 1.0

    # Compute the difference between each pair of modified eigenvalues,
    # expanding dimensions

    # case: sigma_i != sigma_j
    numerator = s_modified.unsqueeze(-1) - s_modified.unsqueeze(-2)

    # Compute the derivative of the function applied to the eigenvalues
    # case: sigma_i == sigma_j
    s_derivativated = derivative(s)
    # For pairs of eigenvalues that are considered equal,
    # set the numerator to the average of their derivatives

    numerator[is_eq] = (
        0.5
        * (
            s_derivativated.unsqueeze(-1)
            + s_derivativated.unsqueeze(-2)
        )[is_eq]
    )
    # Compute the Loewner matrix,
    # which is the element-wise division of the numerator and the denominator
    L = numerator / denominator

    # Compute the gradient of the loss input with
    #  grad_input = U * (L * (U^T * G * U)) * U^T (I think)
    grad_input = U @ (L * (U.transpose(-1, -2) @ grad_output @ U)) @ U.transpose(-1, -2)

    return grad_input


class logm(torch.autograd.Function):
    """Matrix logarithm of a symmetric matrix.

    This class computes the matrix logarithm of a symmetric matrix X.
    It also adapts the backpropagation according to the chain rule [1]_.

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
        def function_applied(s):
            return s.log()

        output, s, U, s_modified = _modify_eig_forward(X, function_applied)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors

        def derivative(s):
            return s.reciprocal()

        return _modify_eig_backward(grad_output, s, U, s_modified, derivative)


class regm(torch.autograd.Function):
    """Rank deficient matrix logarithm of a symmetric matrix.

    Also, known as Regulation in the SPD space.

    This class computes the regularized matrix logarithm of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Rank deficient matrix logarithm of X

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
            for SPD neural networks, NeurIPS
    """

    @staticmethod
    def forward(ctx, X, threshold):
        def function_applied(s):
            return s.clamp(min=threshold)

        output, s, U, s_modified = _modify_eig_forward(X, function_applied)
        ctx.save_for_backward(s, U, s_modified)
        ctx.threshold_ = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors

        def derivative(s):
            return s > ctx.threshold_

        return _modify_eig_backward(grad_output, s, U, s_modified, derivative), None
