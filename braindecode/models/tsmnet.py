# Authors: Théo Gnassounou <theo.gnassounou@gmail.com>
# Code inspired from https://github.com/rkobler/TSMNet.git
#
# License: BSD (3-clause)
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import SymmetricPositiveDefinite
import torch
import torch.nn as nn

from .spdnet import CovLayer, EigenvalueModificator, BiMap, ReEig, LogEig


class Powm(torch.autograd.Function):
    """Power of a symmetric matrix.

    This class computes the power of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix
    p : float
        Power

    Returns
    -------
    torch.Tensor
        Power of X

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
            for SPD neural networks, NeurIPS
    """

    @staticmethod
    def forward(ctx, X, p):
        def function_applied(s):
            return s.pow(p)

        def derivative(s):
            return p * s.pow(p - 1)

        return EigenvalueModificator.apply(X, function_applied, derivative)


class Sqrtm(torch.autograd.Function):
    """Square root of a symmetric matrix.

    This class computes the square root of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix

    Returns
    -------
    torch.Tensor
        Square root of X

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
            for SPD neural networks, NeurIPS
    """

    @staticmethod
    def forward(ctx, X):
        def function_applied(s):
            return s.sqrt()

        def derivative(s):
            return 0.5 * s.pow(-0.5)

        return EigenvalueModificator.apply(X, function_applied, derivative)


class InvSqrtm(torch.autograd.Function):
    """Inverse square root of a symmetric matrix.

    This class computes the inverse square root of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix

    Returns
    -------
    torch.Tensor
        Inverse square root of X

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
            for SPD neural networks, NeurIPS
    """

    @staticmethod
    def forward(ctx, X):
        def function_applied(s):
            return s.rsqrt()

        def derivative(s):
            return -0.5 * s.pow(-1.5)

        return EigenvalueModificator.apply(X, function_applied, derivative)


class Expm(torch.autograd.Function):
    """Matrix exponential of a symmetric matrix.

    This class computes the matrix exponential of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix

    Returns
    -------
    torch.Tensor
        Matrix exponential of X

    References
    ----------
    .. [1] Brooks et al. 2019, Riemannian batch normalization
            for SPD neural networks, NeurIPS
    """

    @staticmethod
    def forward(ctx, X):
        def function_applied(s):
            return s.exp()

        def derivative(s):
            return s.exp()

        return EigenvalueModificator.apply(X, function_applied, derivative)


class Logm(torch.autograd.Function):
    """Matrix logarithm of a symmetric matrix.

    This class computes the matrix logarithm of a symmetric matrix X.
    It also adapt the backpropagation according to the chain rule [1]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix

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

        def derivative(s):
            return s.reciprocal()

        return EigenvalueModificator.apply(X, function_applied, derivative)


class SPDBatchNorm(nn.Module):
    def __init__(
        self,
        n_chans,
        dispersion,
        learn_mean=False,
        learn_std=False,
        karcher_steps=1,
        eta=1.0,
    ):
        super().__init__()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.karcher_steps = karcher_steps
        self.eta = eta

        init_mean = torch.diag_embed(torch.ones(n_chans))
        init_var = torch.ones((n_chans, 1))

        self.register_buffer(
            "running_mean",
            ManifoldTensor(init_mean, manifold=SymmetricPositiveDefinite()),
        )
        self.register_buffer("running_var", init_var)
        self.register_buffer(
            "running_mean_test",
            ManifoldTensor(init_mean, manifold=SymmetricPositiveDefinite()),
        )
        self.register_buffer("running_var_test", init_var)

        if self.learn_mean:
            self.mean = ManifoldParameter(
                init_mean.clone(), manifold=SymmetricPositiveDefinite()
            )
        else:
            self.mean = ManifoldTensor(
                init_mean.clone(), manifold=SymmetricPositiveDefinite()
            )

        if self.dispersion:
            if self.learn_std:
                self.std = nn.parameter.Parameter(init_var.clone())
            else:
                self.std = init_var.clone()

    def forward(self, X):
        manifold = self.running_mean.manifold
        if self.training:
            # compute the Karcher flow for the current batch
            batch_mean = X.mean(dim=self.batchdim, keepdim=True)
            for _ in range(self.karcher_steps):
                batch_mean_invsqrt = InvSqrtm.apply(batch_mean.detach())
                X_projected = Logm.apply(batch_mean_invsqrt @ X @ batch_mean_invsqrt)
                batch_mean_projected = X_projected.mean(dim=self.batchdim, keepdim=True)
                batch_mean_sqrt = Sqrtm.apply(batch_mean.detach())
                batch_mean = (
                    batch_mean_sqrt @ Expm.apply(batch_mean_projected) @ batch_mean_sqrt
                )

            # update the running mean
            running_mean_invsqrt = InvSqrtm.apply(self.running_mean)
            power = Powm.apply(
                running_mean_invsqrt @ batch_mean @ running_mean_invsqrt, self.eta
            )
            running_mean_sqrt = Sqrtm.apply(self.running_mean)
            running_mean = running_mean_sqrt @ power @ running_mean_sqrt

            if self.dispersion:
                projected_mean = Logm.apply(
                    batch_mean_invsqrt @ running_mean @ batch_mean_invsqrt
                )
                batch_var = (
                    torch.norm(
                        X_projected - projected_mean,
                        p="fro",
                        dim=(-2, -1),
                        keepdim=True,
                    )
                    .square()
                    .mean(dim=0, keepdim=True)
                    .squeeze(-1)
                )
                running_var = (1.0 - self.eta) * self.running_var + self.eta * batch_var

        else:
            running_mean = self.running_mean_test
            if self.dispersion:
                running_var = self.running_var_test

        # rescale to desired dispersion
        if self.dispersion:
            Xn = manifold.transp_identity_rescale_transp(
                X, running_mean, self.std / (running_var + self.eps).sqrt(), self.mean
            )
        else:
            Xn = manifold.transp_via_identity(X, running_mean, self.mean)

        if self.training:
            with torch.no_grad():
                self.running_mean.data = running_mean.clone()

                # update test running mean
                running_mean_invsqrt = InvSqrtm.apply(self.running_mean_test)
                power = Powm.apply(
                    running_mean_invsqrt @ batch_mean @ running_mean_invsqrt,
                    self.eta_test,
                )
                running_mean_sqrt = Sqrtm.apply(self.running_mean_test)
                self.running_mean_test.data = (
                    running_mean_sqrt @ power @ running_mean_sqrt
                )

                if self.dispersion:
                    self.running_var = running_var.clone()
                    batch_mean_test = Logm.apply(
                        batch_mean_invsqrt @ self.running_mean_test @ batch_mean_invsqrt
                    )
                    batch_var_test = (
                        torch.norm(
                            X_projected - batch_mean_test,
                            p="fro",
                            dim=(-2, -1),
                            keepdim=True,
                        )
                        .square()
                        .mean(dim=self.batchdim, keepdim=True)
                        .squeeze(-1)
                    )

                    self.running_var_test = (
                        1.0 - self.eta
                    ) * self.running_var_test + self.eta * batch_var_test

        return Xn


class TSMNet(nn.Module):
    """Tangent Space Mapping model from [1]_.

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
    dispersion : bool
        Dispersion
    input_type : str
        Input type
    add_log_softmax : bool
        Add log softmax
    learn_mean : bool
        Learn mean
    learn_std : bool
        Learn standard deviation

    References
    ----------
    .. [1] Kobler et al., 2022, SPD domain-specific batch normalization
           to crack interpretable unsupervised domain adaptation in EEG,
           Neurips

    """

    def __init__(
        self,
        n_chans,
        subspacedim,
        threshold=1e-4,
        out_features=1,
        dispersion=False,
        input_type="raw",
        add_log_softmax=True,
        learn_mean=False,
        learn_std=False,
    ):
        super(TSMNet, self).__init__()
        if input_type == "raw":
            self.cov = CovLayer()
        elif input_type == "cov":
            self.cov = nn.Identity()

        self.bimap = BiMap(n_chans, subspacedim)
        self.reeig = ReEig(threshold)
        self.batchnorm = SPDBatchNorm(
            n_chans, dispersion=dispersion, learn_mean=learn_mean, learn_std=learn_std
        )
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
        X = self.batchnorm(X)
        X = self.logeig(X)
        X = self.classifier(X)
        output = self.logsoftmax(X)
        return output
