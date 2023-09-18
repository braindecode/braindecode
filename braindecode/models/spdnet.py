# Authors: Théo Gnassounou <theo.gnassounou@gmail.com>
# Code inspired from https://github.com/rkobler/TSMNet.git
#
# License: BSD (3-clause)
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel
import torch
import torch.nn as nn
from .functions import Logm, Regm
from .base import EEGModuleMixin


class CovLayer(nn.Module):
    """Covariance layer.

    This class compute the covariance of a batch of
    symmetric matrices.

    Parameters
    ----------
    X : torch.Tensor
        Batch of symmetric matrices

    Returns
    -------
    torch.Tensor
        Batch of covariance matrices
    """
    def forward(self, X):
        n_batch, n_channels, _ = X.size()
        torch_covs = torch.empty((n_batch, n_channels, n_channels)).to(X.device)
        for i, batch in enumerate(X):
            torch_covs[i] = batch.cov(correction=0)
        return torch_covs


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

    def forward(self, X):
        return Regm.apply(X, self.threshold_)


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

    def forward(self, X):
        return Logm.apply(X)


class SPDNet(EEGModuleMixin, nn.Module):
    """SPDNet from [1]_.

    This class is a SPDNet implementation from [1]_.

    Parameters
    ----------
    input_type : str
        Input type
        If "raw", the input is a batch of raw EEG signals and
        a covariance matrix is computed
        If "cov", the input is a batch of covariance matrices and
        the covariance matrix is used as input
    n_chans : int
        Number of channels
    subspacedim : int
        Subspace dimension
    threshold : float
        Threshold for the rectified linear unit
    n_outputs : int
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
        n_chans,
        subspacedim,
        threshold=1e-4,
        n_outputs=1,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        add_log_softmax=False,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        if input_type == "raw":
            self.cov = CovLayer()
        elif input_type == "cov":
            self.cov = nn.Identity()
        self.bimap = BiMap(n_chans, subspacedim)
        self.reeig = ReEig(threshold)
        self.logeig = torch.nn.Sequential(
            LogEig(threshold),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(subspacedim**2, n_outputs),
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
