# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)
import math
import torch


def rescale_parameter(param, layer_id):
    """ Recaling the l-th transformer layer.

    Rescales the parameter tensor by the inverse square root of the layer id.
    Made inplace. :math:`\frac{1}{\sqrt{2 \cdot \text{layer\_id}}}` [Beit2022]

    In the labram, this is used to rescale the output matrices
    (i.e., the last linear projection within each sub-layer) of the
    self-attention module.

    Parameters
    ----------
    param: :class:`torch.Tensor`
        tensor to be rescaled
    layer_id: int
        layer id in the neural network

    References
    ----------
    [Beit2022] Hangbo Bao, Li Dong, Songhao Piao, Furu We (2022). BEIT: BERT
    Pre-Training of Image Transformers.
    """
    param.div_(math.sqrt(2.0 * layer_id))


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


def drop_path(x,
              drop_prob: float = 0.0,
              training: bool = False,
              scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample.


    Notes: This implementation is taken from timm library.

    All credit goes to Ross Wightman.

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    drop_prob : float, optional
        survival rate (i.e. probability of being kept), by default 0.0
    training : bool, optional
        whether the model is in training mode, by default False
    scale_by_keep : bool, optional
        whether to scale output by (1/keep_prob) during training, by default True

    Returns
    -------
    torch.Tensor
        output tensor

    Notes from Ross Wightman:
    (when applied in main path of residual blocks)
    This is the same as the DropConnect impl I created for EfficientNet,
    etc. networks, however,
    the original name is misleading as 'Drop Connect' is a different form
    of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    ... I've opted for changing the layer and argument names to 'drop path'
    rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def modeig_forward(X, function_applied):
    """Modified eigenvalue of symmetric matrix X."""
    s, U = torch.linalg.eigh(X)
    s_modified = function_applied(s)
    output = U @ torch.diag_embed(s_modified) @ U.transpose(-1, -2)
    return output, s, U, s_modified


def modeig_backward(grad_output, s, U, s_modified, derivative, threshold=1e-4):
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
        def function_applied(s): return s.log()
        output, s, U, s_modified = modeig_forward(X, function_applied)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        def derivative(s): return s.reciprocal()
        return modeig_backward(grad_output, s, U, s_modified, derivative)


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
        def function_applied(s): return s.clamp(min=threshold)
        output, s, U, s_modified = modeig_forward(X, function_applied)
        ctx.save_for_backward(s, U, s_modified)
        ctx.threshold_ = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        def derivative(s): return s > ctx.threshold_
        return modeig_backward(grad_output, s, U, s_modified, derivative), None


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
