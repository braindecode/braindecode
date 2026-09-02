# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)

"""Top-K sparse autoencoder over model activations."""

from numbers import Integral

import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    """Top-K sparse autoencoder over activation vectors.

    Re-expresses an activation as a sparse combination of
    ``n_inputs * expansion`` learned directions. Only the ``k`` largest
    encoder outputs are kept and the rest are zeroed [1]_, so sparsity is
    structural rather than traded against reconstruction error through an
    L1 coefficient.

    The encoder is rectified and its weights are untied from the decoder,
    whose columns are held at unit norm [2]_. Without the unit-norm
    constraint the reconstruction can be improved by shrinking the code and
    growing the decoder, which leaves feature magnitudes uninterpretable.
    The rectifier is applied before the mask, so ``k`` bounds the number of
    active features rather than fixing it.

    Following [2]_ the input and output biases are tied: the decoder bias is
    subtracted before encoding and added back on decode, so the module
    computes ``ReLU(W_enc (x - b_dec) + b_enc)`` and ``W_dec z + b_dec``.

    Parameters
    ----------
    n_inputs : int
        Width of the activation vectors, i.e. the embedding dimension of the
        layer being decomposed.
    expansion : int, default=8
        Dictionary size relative to ``n_inputs``. The autoencoder learns
        ``n_inputs * expansion`` features.
    k : int or None, default=None
        Number of features kept per input. Defaults to ``8 * expansion``,
        which holds the active fraction constant as the dictionary grows,
        capped at ``n_inputs * expansion`` so the default is always valid.

    Raises
    ------
    ValueError
        If ``n_inputs``, ``expansion`` or ``k`` is not a positive integer, or
        if ``k`` exceeds ``n_inputs * expansion``.

    Notes
    -----
    ``activation_mean`` and ``activation_std`` are registered buffers, so the
    standardisation used at fit time survives a ``state_dict`` round trip.
    They default to zero and one until
    :meth:`set_activation_normalization` is called.

    References
    ----------
    .. [1] Makhzani, A., & Frey, B. (2014). k-Sparse Autoencoders.
       International Conference on Learning Representations (ICLR).
       Online: https://arxiv.org/abs/1312.5663
    .. [2] Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards
       Monosemanticity: Decomposing Language Models With Dictionary
       Learning. Transformer Circuits Thread.
       Online: https://transformer-circuits.pub/2023/monosemantic-features
    """

    def __init__(self, n_inputs, expansion=8, k=None):
        super().__init__()
        for name, value in (("n_inputs", n_inputs), ("expansion", expansion)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.n_inputs = int(n_inputs)
        self.expansion = int(expansion)
        self.n_features = self.n_inputs * self.expansion
        k = min(self.n_features, 8 * self.expansion) if k is None else k
        if (
            isinstance(k, bool)
            or not isinstance(k, Integral)
            or not 1 <= k <= self.n_features
        ):
            raise ValueError(
                f"k must be an integer in [1, {self.n_features}], got {k!r}"
            )
        self.k = int(k)
        self.encoder = nn.Linear(self.n_inputs, self.n_features)
        self.decoder = nn.Linear(self.n_features, self.n_inputs)
        self.register_buffer("activation_mean", torch.zeros(self.n_inputs))
        self.register_buffer("activation_std", torch.ones(self.n_inputs))
        self.normalize_decoder_()

    def encode(self, x):
        """Encode standardised activations into a sparse feature code.

        Parameters
        ----------
        x : torch.Tensor
            Standardised activations of shape ``(..., n_inputs)``.

        Returns
        -------
        torch.Tensor
            Feature code of shape ``(..., n_features)`` with at most ``k``
            non-zero entries per activation. Fewer than ``k`` fire when
            fewer than ``k`` encoder outputs are positive.
        """
        if x.shape[-1] != self.n_inputs:
            raise ValueError(
                f"expected last dimension {self.n_inputs}, got {x.shape[-1]}"
            )
        hidden = torch.relu(self.encoder(x - self.decoder.bias))
        values, indices = torch.topk(hidden, self.k, dim=-1)
        return torch.zeros_like(hidden).scatter_(-1, indices, values)

    def decode(self, z):
        """Decode a sparse feature code back to standardised activations.

        Parameters
        ----------
        z : torch.Tensor
            Feature code of shape ``(..., n_features)``.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(..., n_inputs)``.
        """
        if z.shape[-1] != self.n_features:
            raise ValueError(
                f"expected last dimension {self.n_features}, got {z.shape[-1]}"
            )
        return self.decoder(z)

    def forward(self, x):
        """Encode and decode standardised activations.

        Parameters
        ----------
        x : torch.Tensor
            Standardised activations of shape ``(..., n_inputs)``.

        Returns
        -------
        recon : torch.Tensor
            Reconstruction of shape ``(..., n_inputs)``.
        latent : torch.Tensor
            Feature code of shape ``(..., n_features)``.
        """
        latent = self.encode(x)
        return self.decode(latent), latent

    def normalize_decoder_(self):
        """Rescale decoder columns to unit norm, in place."""
        with torch.no_grad():
            weight = self.decoder.weight
            weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    def set_activation_normalization(self, mean, std):
        """Store the per-dimension statistics used to standardise activations.

        Parameters
        ----------
        mean : array-like
            Per-dimension mean, of shape ``(n_inputs,)``.
        std : array-like
            Per-dimension standard deviation, of shape ``(n_inputs,)``. Must
            be finite and strictly positive.

        Raises
        ------
        ValueError
            If either argument has the wrong shape or holds a non-finite
            value, or if any standard deviation is not strictly positive.
        """
        mean = torch.as_tensor(mean, dtype=self.activation_mean.dtype)
        std = torch.as_tensor(std, dtype=self.activation_std.dtype)
        if mean.shape != (self.n_inputs,) or std.shape != (self.n_inputs,):
            raise ValueError(
                f"mean and std must both have shape ({self.n_inputs},), got "
                f"{tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError("mean and std must be finite")
        if not (std > 0).all():
            raise ValueError("std must be strictly positive")
        with torch.no_grad():
            self.activation_mean.copy_(mean.to(self.activation_mean.device))
            self.activation_std.copy_(std.to(self.activation_std.device))

    def reconstruct_activations(self, x):
        """Reconstruct raw activations, standardising and inverting internally.

        Parameters
        ----------
        x : torch.Tensor
            Raw activations of shape ``(..., n_inputs)``, on the scale the
            layer emits.

        Returns
        -------
        torch.Tensor
            Reconstruction of the same shape, on the same scale.
        """
        z = self.encode((x - self.activation_mean) / self.activation_std)
        return self.decode(z) * self.activation_std + self.activation_mean
