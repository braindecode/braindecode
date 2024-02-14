# Authors: Théo Gnassounou <theo.gnassounou@gmail.com>
# Code inspired from https://github.com/rkobler/TSMNet.git
#
# License: BSD (3-clause)
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel
import torch
import torch.nn as nn
from .functions import logm, regm
from .base import EEGModuleMixin


class CovLayer(nn.Module):
    """Covariance layer.

    This class compute the covariance of a batch of
    symmetric matrices.
    """
    def forward(self, X):
        """
        Forward pass.

        Parameters
        ----------
        X: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).

        Returns
        -------
        torch.Tensor
            Batch of covariance matrices of shape (batch_size, n_channels, n_channels).
        """
        means = torch.mean(X, dim=2, keepdim=True)
        X_centered = X - means
        covariances = torch.einsum('bik,bjk->bij', X_centered, X_centered) / (X_centered.shape[2] - 0)
        return covariances


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
    of a symmetric matrix. If threshold > 0, the matrix
    is non-negative and positive.

    Parameters
    ----------
    threshold : float
        Threshold for the rectified linear unit

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van Gool, 2016,
           A Riemannian Network for SPD Matrix Learning
           AAAI
    """
    def __init__(self, threshold=1e-4):
        super(ReEig, self).__init__()
        self.register_buffer("threshold_", torch.tensor(threshold))

    def forward(self, X):
        return regm.apply(X, self.threshold_)


class LogEig(nn.Module):
    """LogEig layer from [1]_.

    This class perform Riemannian projection into a flat space
    by applying the logarithm to the eigenvalues of a symmetric matrix.
    The output is flattened to obtain a vector representation of the matrix.

    Parameters
    ----------
    dim : int
        Dimension of the symmetric matrix
    tril : bool
        If True, only the lower triangular part of the matrix is used

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van Gool, 2016,
           A Riemannian Network for SPD Matrix Learning
           AAAI
    """

    def __init__(self, dim, tril=True, ):
        super(LogEig, self).__init__()
        self.tril = tril
        if self.tril:
            idx_lower = torch.tril_indices(dim, dim, offset=-1)
            idx_diag = torch.arange(start=0, end=dim, dtype=torch.long)
            self.idx = torch.cat((idx_diag[None, :].tile((2, 1)), idx_lower), dim=1)
        self.dim = dim

    def forward(self, X):
        return self.embed(logm.apply(X))

    def embed(self, X):
        if self.tril:
            x_vec = X[:, self.idx[0], self.idx[1]]
            x_vec[:, self.dim:] *= 2 ** 0.5
        else:
            x_vec = X.flatten(start_dim=1)
        return x_vec


class SPDNet(EEGModuleMixin, nn.Module):
    """SPDNet from [1]_.

    This class is a SPDNet implementation from [1]_.

    Parameters
    ----------
    input_type : str
        If "raw", the input is a batch of raw EEG signals and
        a covariance matrix is computed
        If "cov", the input is a batch of covariance matrices and
        the covariance matrix is used as input
    subspacedim : int
        Subspace dimension
    threshold : float
        Threshold for the rectified linear unit
    tril : bool
        If True, only the lower triangular part of the matrix is used

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van Gool, 2016,
           A Riemannian Network for SPD Matrix Learning
           AAAI
    """
    def __init__(
        self,
        input_type,
        n_chans,
        subspacedim,
        threshold=1e-4,
        n_outputs=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        tril=True,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        if input_type == "raw":
            self.cov = CovLayer()
        elif input_type == "cov":
            self.cov = nn.Identity()
        self.bimap = BiMap(self.n_chans, subspacedim)
        self.reeig = ReEig(threshold)
        self.logeig = LogEig(subspacedim, tril=tril)
        self.len_last_layer = subspacedim * (subspacedim + 1) // 2 if tril else subspacedim**2
        self.classifier = torch.nn.Linear(self.len_last_layer, self.n_outputs)


    def forward(self, X):
        X = self.cov(X)
        X = self.bimap(X)
        X = self.reeig(X)
        X = self.logeig(X)
        X = self.classifier(X)
        
        return X
