# Authors: Théo Gnassounou <theo.gnassounou@gmail.com>
# Code inspired from https://github.com/rkobler/TSMNet.git
#
# License: BSD (3-clause)
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel
import torch
import torch.nn as nn


class CovLayer(nn.Module):
    def __init__(self, estimator="cov"):
        super().__init__()
        self.estimator = estimator
        if self.estimator != "cov":
            raise NotImplementedError

    def forward(self, X):
        n_batch, n_channels, _ = X.size()
        torch_covs = torch.empty((n_batch, n_channels, n_channels)).to(X.device)
        for i, batch in enumerate(X):
            torch_covs[i] = batch.cov(correction=0)
        return torch_covs


class EigenvalueModificator(torch.autograd.Function):
    """Eigenvalue modificator following [1]_.

    This class modifies the eigenvalues of a symmetric matrix X
    according to a given function f. It also adapt the backpropagation
    according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix
    function_applied : callable
        Function applied to the eigenvalues of X
    derivative_function_applied : callable
        Derivative of the function applied to the
        eigenvalues of X during backpropagation

    Returns
    -------
    torch.Tensor
        Modified symmetric matrix

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
           for SPD neural networks, NeurIPS
    """

    @staticmethod
    def forward(ctx, X, function_applied, derivative_function_applied):
        s, U = torch.linalg.eigh(X)
        s_modified = function_applied(s)
        output = U @ torch.diag_embed(s_modified) @ U.transpose(-1, -2)
        ctx.save_for_backward(s, U, s_modified)
        ctx.derivative_function_applied_ = derivative_function_applied
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        # compute Loewner matrix
        denominator = s.unsqueeze(-1) - s.unsqueeze(-1).transpose(-1, -2)

        is_eq = denominator.abs() < 1e-4  # XXX maybe should be a parameter threshold
        denominator[is_eq] = 1.0

        # case: sigma_i != sigma_j
        numerator = s_modified.unsqueeze(-1) - s_modified.unsqueeze(-1).transpose(-1, -2)

        # case: sigma_i == sigma_j
        s_derivativated = ctx.derivative_function_applied_(s)
        numerator[is_eq] = 0.5 * (
            s_derivativated.unsqueeze(-1) + s_derivativated.unsqueeze(-1).transpose(-1, -2)
        )[is_eq]
        L = numerator / denominator

        grad_input = U @  (L * (U.transpose(-1, -2) @ grad_output @ U)) @ U.transpose(-1, -2)

        return grad_input, None, None


class BiMap(nn.Module):
    """BiMap layer from [1]_.

    This class generates more compact and discriminative
    representations of symmetric matrices by projecting them
    into a lower dimensional subspace Stiefel manifold.

    Parameters
    ----------
    input_shape : int
        Input shape
    output_shape : int
        Output shape

    Returns
    -------
    torch.Tensor
        Modified symmetric matrix

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van Gool, 2016,
           A Riemannian Network for SPD Matrix Learning
           AAAI
    """
    def __init__(self, input_shape, output_shape, threshold=1e-4):
        super(BiMap, self).__init__()
        self.input_shape_ = input_shape
        self.output_shape_ = output_shape
        self.threshold_ = threshold
        self.manifold_ = Stiefel()
        assert input_shape >= output_shape
        self.W = ManifoldParameter(
            torch.empty([1, input_shape, output_shape]), manifold=self.manifold_
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        W = torch.rand(self.W.shape, dtype=self.W.dtype, device=self.W.device)
        s, U = torch.linalg.eigh(W.transpose(-1, -2) @ W)
        smod = s.clamp(min=self.threshold_).rsqrt()
        A = U @ torch.diag_embed(smod) @ U.transpose(-1, -2)
        self.W.data = W @ A

    def forward(self, X):
        return self.W.transpose(-1, -2) @ X @ self.W


class ReEig(nn.Module):
    """ReEig layer from [1]_.

    This class add non-linearity to the network by
    applying a rectified linear unit to the eigenvalues
    of a symmetric matrix.

    Parameters
    ----------
    threshold : float
        Threshold for the rectified linear unit

    Returns
    -------
    torch.Tensor
        Modified symmetric matrix

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van Gool, 2016,
           A Riemannian Network for SPD Matrix Learning
           AAAI
    """
    def __init__(self, threshold=1e-4):
        super(ReEig, self).__init__()
        self.register_buffer("threshold_", torch.tensor(threshold))

    def function_applied(self, s):
        return s.clamp(min=self.threshold_)

    def derivative_function_applied(self, s):
        return s > self.threshold_

    def forward(self, X):
        return EigenvalueModificator.apply(
            X, self.function_applied, self.derivative_function_applied
        )


class LogEig(nn.Module):
    """LogEig layer from [1]_.

    This class perform Riemannian projection into a flat space
    by applying the logarithm to the eigenvalues of a symmetric matrix.

    Parameters
    ----------
    threshold : float
        Threshold for the rectified linear unit

    Returns
    -------
    torch.Tensor
        Modified symmetric matrix

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van Gool, 2016,
           A Riemannian Network for SPD Matrix Learning
           AAAI
    """

    def __init__(self, threshold=1e-4):
        super(LogEig, self).__init__()
        self.threshold_ = threshold

    def function_applied(self, s):
        return s.clamp(min=self.threshold_).log()

    def derivative_function_applied(self, s):
        s_derivated = s.reciprocal()
        s_derivated[s <= self.threshold_] = 0
        return s_derivated

    def forward(self, X):
        return EigenvalueModificator.apply(
            X, self.function_applied, self.derivative_function_applied
        )


class SPDNet(nn.Module):
    """SPDNet from [1]_.

    This class is a SPDNet implementation from [1]_.

    Parameters
    ----------
    input_shape : int
        Input shape
    subspacedim : int
        Subspace dimension
    threshold : float
        Threshold for the rectified linear unit
    out_features : int
        Output shape

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van Gool, 2016,
           A Riemannian Network for SPD Matrix Learning
           AAAI
    """
    def __init__(
        self,
        input_type,
        input_shape,
        subspacedim,
        threshold=1e-4,
        out_features=1,
        add_log_softmax=True
    ):
        super(SPDNet, self).__init__()
        if input_type == "raw":
            self.cov = CovLayer()
        elif input_type == "cov":
            self.cov = nn.Identity()
        self.bimap = BiMap(input_shape, subspacedim)
        self.reeig = ReEig(threshold)
        self.logeig = torch.nn.Sequential(
            LogEig(threshold),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(subspacedim**2, out_features),
        )
        if add_log_softmax:
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.logsoftmax = nn.Identity()

    def forward(self, X):
        X = self.cov(X)
        X = self.bimap(X)
        X = self.reeig(X)
        X = self.logeig(X)
        X = self.classifier(X)
        output = self.logsoftmax(X)
        return output
