# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)

"""Sparse autoencoders over model activations.

Implements the Top-K sparse autoencoder of Gao et al. (2024) together with
the activation-capture and activation-substitution primitives needed to fit
one to a layer of a trained model and then measure whether the resulting
dictionary is faithful to what the model actually computes.

A Top-K autoencoder decomposes an activation vector into a sparse
combination of learned directions. Sparsity is structural rather than
penalised: the encoder keeps only its ``k`` largest rectified outputs, so
the number of active features per input is exactly known instead of being
traded off against reconstruction error through an L1 coefficient.

Applied to EEG foundation models by Lehn-Schioeler et al. (2026), who fit
one autoencoder per transformer block and read the resulting features as
candidate clinical concepts.

References
----------
.. [Gao2024] Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R.,
   Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and
   evaluating sparse autoencoders. arXiv:2406.04093.
.. [LehnSchioeler2026] Lehn-Schioeler, W., Kjaer, M. R., Thapa, R., et al.
   (2026). Mechanistic interpretability of EEG foundation models via sparse
   autoencoders. arXiv:2605.13930.
"""

from collections.abc import Mapping
from copy import deepcopy
from warnings import warn

import numpy as np
import torch
from torch import nn


def _flatten_features(x, n_inputs):
    """Collapse leading dimensions of ``x`` so the feature axis is last."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[-1] != n_inputs:
        raise ValueError(f"expected last dimension {n_inputs}, got {x.shape[-1]}")
    leading_shape = x.shape[:-1]
    return x.reshape(-1, n_inputs), leading_shape


def _reshape_features(x, leading_shape):
    """Restore the leading dimensions removed by :func:`_flatten_features`."""
    return x.reshape(*leading_shape, x.shape[-1])


def _detach_output(output):
    """Detach every tensor in a possibly nested module output."""
    if torch.is_tensor(output):
        return output.detach()
    if isinstance(output, tuple):
        return tuple(_detach_output(item) for item in output)
    if isinstance(output, list):
        return [_detach_output(item) for item in output]
    if isinstance(output, dict):
        return {key: _detach_output(value) for key, value in output.items()}
    return deepcopy(output)


def _as_module_items(layers):
    """Return ``(key, module)`` pairs from either a mapping or a sequence."""
    if isinstance(layers, Mapping):
        return list(layers.items())
    return list(enumerate(layers))


class SparseAutoencoder(nn.Module):
    """Top-K sparse autoencoder over activation vectors.

    Encodes an activation into ``n_inputs * expansion`` features, keeps only
    the ``k`` largest, and decodes back. Sparsity is imposed by the top-k
    mask rather than by an L1 penalty, so the number of active features per
    input is exactly ``k`` whenever at least ``k`` encoder outputs are
    positive [Gao2024]_.

    Decoder columns are held at unit norm. Without that constraint the
    reconstruction can be made arbitrarily good by shrinking the latent code
    and growing the decoder, which leaves the feature magnitudes
    uninterpretable.

    Parameters
    ----------
    n_inputs : int
        Width of the activation vectors, i.e. the model's embedding
        dimension at the layer being decomposed.
    expansion : int, default=8
        Dictionary size relative to ``n_inputs``. The autoencoder learns
        ``n_inputs * expansion`` features.
    k : int or None, default=None
        Number of active features per input. Defaults to ``8 * expansion``,
        which holds the active fraction ``k / n_features`` constant as the
        dictionary grows.

    Raises
    ------
    ValueError
        If ``k`` is outside ``[1, n_inputs * expansion]``.

    Notes
    -----
    ``activation_mean`` and ``activation_std`` are registered buffers, so
    the normalisation used at fit time survives a ``state_dict`` round trip.
    They default to zero and one, making normalisation a no-op unless
    :meth:`set_activation_normalization` is called.

    References
    ----------
    .. [Gao2024] Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R.,
       Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and
       evaluating sparse autoencoders. arXiv:2406.04093.
    """

    def __init__(self, n_inputs, expansion=8, k=None):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.expansion = int(expansion)
        self.n_features = self.n_inputs * self.expansion
        self.k = int(k) if k is not None else min(self.n_features, 8 * self.expansion)
        if not 1 <= self.k <= self.n_features:
            raise ValueError(
                "k must be between 1 and n_inputs * expansion "
                f"({self.n_features}), got {self.k}"
            )
        if self.k == self.n_features:
            warn(
                "k equals n_inputs * expansion, so the autoencoder is dense. "
                "Use a smaller k for sparse features.",
                UserWarning,
            )
        self.encoder = nn.Linear(self.n_inputs, self.n_features)
        self.decoder = nn.Linear(self.n_features, self.n_inputs)
        self.register_buffer("activation_mean", torch.zeros(self.n_inputs))
        self.register_buffer("activation_std", torch.ones(self.n_inputs))
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise encoder and decoder, then renormalise the decoder."""
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.normalize_decoder_()

    def get_config(self):
        """Return the constructor arguments needed to rebuild this module.

        Returns
        -------
        dict
            Keyword arguments accepted by :meth:`from_config`.
        """
        return dict(n_inputs=self.n_inputs, expansion=self.expansion, k=self.k)

    @classmethod
    def from_config(cls, config):
        """Rebuild an autoencoder from a :meth:`get_config` dictionary.

        Parameters
        ----------
        config : dict
            Configuration as returned by :meth:`get_config`.

        Returns
        -------
        SparseAutoencoder
            A new instance with freshly initialised weights. Load a
            ``state_dict`` into it to restore trained parameters.
        """
        return cls(**config)

    def encode(self, x):
        """Encode activations into a sparse feature code.

        Parameters
        ----------
        x : torch.Tensor
            Activations of shape ``(..., n_inputs)``.

        Returns
        -------
        torch.Tensor
            Feature code of shape ``(..., n_features)`` with at most ``k``
            non-zero entries per row.
        """
        flat, leading_shape = _flatten_features(x, self.n_inputs)
        # Rectify before the top-k mask, matching the reference implementation
        # of [LehnSchioeler2026]_. This binds only when fewer than k encoder
        # outputs are positive; above that the same k features are selected
        # either way.
        hidden = torch.relu(self.encoder(flat))
        if self.k < hidden.shape[-1]:
            values, indices = torch.topk(hidden, self.k, dim=-1)
            masked = torch.zeros_like(hidden)
            masked.scatter_(dim=-1, index=indices, src=values)
            hidden = masked
        return _reshape_features(hidden, leading_shape)

    def decode(self, z):
        """Decode a sparse feature code back to activation space.

        Parameters
        ----------
        z : torch.Tensor
            Feature code of shape ``(..., n_features)``.

        Returns
        -------
        torch.Tensor
            Reconstructed activations of shape ``(..., n_inputs)``.
        """
        flat, leading_shape = _flatten_features(z, self.n_features)
        recon = self.decoder(flat)
        return _reshape_features(recon, leading_shape)

    def forward(self, x):
        """Encode and decode ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Activations of shape ``(..., n_inputs)``.

        Returns
        -------
        recon : torch.Tensor
            Reconstruction of shape ``(..., n_inputs)``.
        latent : torch.Tensor
            Sparse feature code of shape ``(..., n_features)``.
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def normalize_decoder_(self, eps=1e-12):
        """Rescale decoder columns to unit norm, in place.

        Parameters
        ----------
        eps : float, default=1e-12
            Lower bound on a column norm before division.

        Returns
        -------
        SparseAutoencoder
            ``self``, so the call can be chained.
        """
        with torch.no_grad():
            weight = self.decoder.weight
            norms = weight.norm(dim=0, keepdim=True).clamp_min(eps)
            weight.div_(norms)
        return self

    def set_activation_normalization(self, mean, std, eps=1e-6):
        """Store the per-dimension statistics used to normalise activations.

        Parameters
        ----------
        mean : array-like
            Per-dimension mean, of shape ``(n_inputs,)``.
        std : array-like
            Per-dimension standard deviation, of shape ``(n_inputs,)``.
        eps : float, default=1e-6
            Lower bound applied to ``std`` before storing it.

        Returns
        -------
        SparseAutoencoder
            ``self``, so the call can be chained.

        Raises
        ------
        ValueError
            If ``mean`` or ``std`` does not have shape ``(n_inputs,)``.
        """
        mean = torch.as_tensor(mean, dtype=self.activation_mean.dtype)
        std = torch.as_tensor(std, dtype=self.activation_std.dtype)
        if mean.shape != (self.n_inputs,) or std.shape != (self.n_inputs,):
            raise ValueError(
                "mean and std must both have shape "
                f"({self.n_inputs},), got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        with torch.no_grad():
            self.activation_mean.copy_(mean.to(self.activation_mean.device))
            self.activation_std.copy_(std.clamp_min(eps).to(self.activation_std.device))
        return self

    def normalize_activations(self, x):
        """Standardise raw activations with the stored statistics.

        Parameters
        ----------
        x : torch.Tensor
            Raw activations of shape ``(..., n_inputs)``.

        Returns
        -------
        torch.Tensor
            Standardised activations of the same shape.
        """
        return (x - self.activation_mean) / self.activation_std

    def denormalize_activations(self, x):
        """Invert :meth:`normalize_activations`.

        Parameters
        ----------
        x : torch.Tensor
            Standardised activations of shape ``(..., n_inputs)``.

        Returns
        -------
        torch.Tensor
            Activations back on the original scale.
        """
        return x * self.activation_std + self.activation_mean

    def reconstruct_activations(self, x):
        """Reconstruct raw activations, normalising and inverting internally.

        This is the form needed by :func:`run_with_activation_substitution`,
        which sees a layer's output on its original scale.

        Parameters
        ----------
        x : torch.Tensor
            Raw activations of shape ``(..., n_inputs)``.

        Returns
        -------
        torch.Tensor
            Reconstruction on the original scale, same shape as ``x``.
        """
        normalized = self.normalize_activations(x)
        recon, _ = self(normalized)
        return self.denormalize_activations(recon)

    def resample_dead_features(
        self, activations, dead_features, optimizer=None, seed=None
    ):
        """Point dead features at inputs the dictionary reconstructs poorly.

        Features that stop firing contribute nothing and cannot recover on
        their own, since they receive no gradient. Following Anthropic's
        recipe, each is reassigned to a direction drawn from the activation
        set with probability proportional to squared reconstruction error,
        so a revived feature has something unexplained to claim.

        Parameters
        ----------
        activations : torch.Tensor
            Activations to draw replacement directions from, shape
            ``(..., n_inputs)``. These should already be normalised if the
            autoencoder was fitted on normalised activations.
        dead_features : torch.Tensor
            Either a boolean mask of length ``n_features`` or an integer
            tensor of feature indices.
        optimizer : torch.optim.Optimizer or None, default=None
            If given, the optimizer state for the resampled rows is cleared.
            Only those rows are touched; a global reset discards momentum
            belonging to features that are training normally.
        seed : int or None, default=None
            Seed for the sampling, for reproducibility.

        Returns
        -------
        int
            Number of features resampled.

        Raises
        ------
        ValueError
            If the boolean mask has the wrong length, or if every feature is
            marked dead so there is no live encoder norm to scale against.
        """
        flat, _ = _flatten_features(torch.as_tensor(activations), self.n_inputs)
        dead = torch.as_tensor(dead_features, device=flat.device)
        if dead.dtype == torch.bool:
            if dead.numel() != self.n_features:
                raise ValueError("boolean dead_features mask has wrong length")
            dead_idx = dead.nonzero(as_tuple=False).flatten()
        else:
            dead_idx = dead.to(dtype=torch.long).flatten()
        if dead_idx.numel() == 0:
            return 0

        alive_idx = torch.ones(self.n_features, dtype=torch.bool, device=flat.device)
        alive_idx[dead_idx] = False
        if not bool(alive_idx.any()):
            raise ValueError("cannot resample when all features are dead")
        alive_norm = self.encoder.weight[alive_idx].norm(dim=1).mean()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=flat.device)
            generator.manual_seed(int(seed))

        # Draw replacement directions in proportion to squared reconstruction
        # error, not uniformly. A revived feature only survives if it has
        # something unexplained to claim: pointed at an input the live features
        # already reconstruct, it loses the top-k competition immediately, goes
        # dead, and is resampled again, so the dictionary churns indefinitely
        # instead of settling.
        n_candidates = min(flat.shape[0], 8192)
        candidate_idx = torch.randint(
            0,
            flat.shape[0],
            (n_candidates,),
            generator=generator,
            device=flat.device,
        )
        candidates = flat[candidate_idx]
        with torch.no_grad():
            recon, _ = self(candidates)
        err = (candidates - recon).pow(2).sum(dim=-1)
        if float(err.sum()) <= 0.0:
            # Nothing left to explain; fall back to a uniform draw.
            err = torch.ones_like(err)
        picks = torch.multinomial(
            err / err.sum(), dead_idx.numel(), replacement=True, generator=generator
        )
        samples = candidates[picks]
        sample_norm = samples.norm(dim=1, keepdim=True).clamp_min(1e-12)
        directions = samples / sample_norm

        # Break ties between features that drew the same row. Sampling is with
        # replacement and concentrated on a few high-error rows, so duplicates
        # are common; identical directions plus a zeroed bias and optimizer
        # state make the gradients identical too, and the pair stays welded
        # together for the rest of training as one feature in two slots.
        noise = torch.randn(
            directions.shape, generator=generator, device=directions.device
        )
        directions = directions + 1e-3 * noise
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # Revived encoder rows are scaled well below the live norm. Given the
        # full norm they would win the top-k competition immediately on the
        # strength of their initialisation, seizing budget from trained
        # features before having learned anything.
        with torch.no_grad():
            for slot, feature_idx in enumerate(dead_idx.tolist()):
                direction = directions[slot]
                self.decoder.weight[:, feature_idx].copy_(direction)
                self.encoder.weight[feature_idx].copy_(0.2 * alive_norm * direction)
                if self.encoder.bias is not None:
                    self.encoder.bias[feature_idx].zero_()

        if optimizer is not None:
            self._reset_optimizer_state(optimizer, dead_idx)

        return int(dead_idx.numel())

    def _reset_optimizer_state(self, optimizer, dead_idx):
        """Zero the optimizer state belonging to resampled features only."""
        # The decoder bias is deliberately absent: it is the single shared
        # reconstruction offset, not a per-feature parameter, so it has no
        # slice belonging to a resampled feature. Clearing it would throw away
        # momentum accumulated on behalf of every feature at once, on every
        # resampling interval.
        params = (
            self.encoder.weight,
            self.encoder.bias,
            self.decoder.weight,
        )
        for param in params:
            if param is None or param not in optimizer.state:
                continue
            state = optimizer.state[param]
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.shape == param.shape:
                    if value.ndim == 2:
                        if param is self.encoder.weight:
                            value[dead_idx, :] = 0
                        elif param is self.decoder.weight:
                            value[:, dead_idx] = 0
                    elif value.ndim == 1:
                        if param is self.encoder.bias:
                            value[dead_idx] = 0
                elif torch.is_tensor(value) and value.ndim == 0:
                    continue


def fit_sparse_autoencoder(
    activations,
    n_inputs=None,
    expansion=8,
    k=None,
    epochs=10,
    batch_size=256,
    lr=1e-3,
    weight_decay=0.0,
    seed=None,
    normalize_activations=True,
    resample_steps=500,
    resample_until=0.8,
    dead_rate=1e-5,
    verbose=0,
):
    """Fit a Top-K sparse autoencoder to a tensor of activations.

    Trains with mean squared reconstruction error only; sparsity comes from
    the top-k mask. Decoder columns are renormalised after every step, and
    features that stop firing are periodically resampled.

    Parameters
    ----------
    activations : torch.Tensor
        Activations of shape ``(..., n_inputs)``. Leading dimensions are
        flattened, so a ``(n_windows, n_tokens, n_inputs)`` tensor is
        treated as one row per token. Training runs on whichever device the
        tensor already lives on.
    n_inputs : int or None, default=None
        Width of the activation vectors. Inferred from the last dimension
        of ``activations`` when ``None``.
    expansion : int, default=8
        Dictionary size relative to ``n_inputs``.
    k : int or None, default=None
        Active features per input. Defaults to ``8 * expansion``.
    epochs : int, default=10
        Number of passes over the activation set.
    batch_size : int, default=256
        Rows per optimisation step.
    lr : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=0.0
        Adam weight decay.
    seed : int or None, default=None
        Seed for weight initialisation and resampling.
    normalize_activations : bool, default=True
        Standardise activations per dimension before fitting, and store the
        statistics on the returned autoencoder.
    resample_steps : int, default=500
        Optimisation steps between dead-feature resampling. Set to ``0`` to
        disable resampling.
    resample_until : float, default=0.8
        Fraction of training after which dead features are left alone.
        Resampling to the final step leaves the last batch of revived
        features untrained, so any diagnostic run afterwards scores them
        cold and reports as dead what is merely newly seeded.
    dead_rate : float, default=1e-5
        Firing rate below which a feature counts as dead in the reported
        history. A plain never-fired test is unusable at this scale, since
        winning top-k once in two million rows is not evidence a feature is
        alive, and the test then reads zero whatever the dictionary is doing.
    verbose : int, default=0
        Print interval in epochs. ``0`` keeps the fit silent.

    Returns
    -------
    sae : SparseAutoencoder
        The fitted autoencoder, carrying the normalisation statistics.
    history : dict
        ``loss`` and ``dead`` per epoch, and ``resampled`` per resampling
        interval. The dead fraction is the one to watch: reconstruction
        error keeps falling while features are still being revived and lost,
        so a flat loss curve does not by itself mean the dictionary has
        settled.

    See Also
    --------
    sae_diagnostics : Reconstruction and sparsity statistics for a fitted model.
    """
    activations = torch.as_tensor(activations, dtype=torch.float32)
    if activations.ndim == 1:
        activations = activations.unsqueeze(0)
    if n_inputs is None:
        n_inputs = activations.shape[-1]
    flat, _ = _flatten_features(activations, n_inputs)

    if seed is not None:
        torch.manual_seed(int(seed))

    sae = SparseAutoencoder(n_inputs=n_inputs, expansion=expansion, k=k)
    sae = sae.to(flat.device)
    if normalize_activations:
        act_mean = flat.mean(dim=0)
        act_std = flat.std(dim=0).clamp_min(1e-6)
        sae.set_activation_normalization(act_mean, act_std)
        flat = sae.normalize_activations(flat)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"loss": [], "dead": [], "resampled": []}
    active_counts = torch.zeros(sae.n_features, device=flat.device)
    steps_since_resample = 0
    epochs = int(epochs)
    steps_per_epoch = max(1, -(-flat.shape[0] // batch_size))
    last_resample_step = int(resample_until * epochs * steps_per_epoch)
    global_step = 0
    for epoch in range(epochs):
        perm = torch.randperm(flat.shape[0], device=flat.device)
        losses = []
        # Counted per epoch as well as per resampling interval: the interval
        # counter is cleared on every resample, so on its own it cannot show
        # whether the dictionary is still churning.
        epoch_fires = torch.zeros(sae.n_features, device=flat.device)
        n_resampled_epoch = 0
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[perm[start : start + batch_size]]
            recon, latent = sae(batch)
            loss = torch.mean((recon - batch) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder_()
            fired = (latent.detach() != 0).sum(dim=0)
            active_counts += fired
            epoch_fires += fired
            steps_since_resample += 1
            global_step += 1
            if (
                resample_steps
                and steps_since_resample >= resample_steps
                and global_step <= last_resample_step
            ):
                # Offset the seed per call, otherwise every resample draws the
                # same rows and the revived features point the same way.
                resample_seed = (
                    None if seed is None else int(seed) + len(history["resampled"])
                )
                n_resampled = sae.resample_dead_features(
                    flat, active_counts == 0, optimizer=optimizer, seed=resample_seed
                )
                history["resampled"].append(n_resampled)
                n_resampled_epoch += n_resampled
                active_counts.zero_()
                steps_since_resample = 0
            losses.append(float(loss.detach()))

        epoch_loss = float(np.mean(losses))
        dead = float((epoch_fires / flat.shape[0] < dead_rate).float().mean())
        history["loss"].append(epoch_loss)
        history["dead"].append(dead)
        if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
            print(
                f"  sae epoch {epoch + 1:3d}/{epochs}  loss={epoch_loss:.4f}  "
                f"dead={dead:.2%}  resampled={n_resampled_epoch}",
                flush=True,
            )
    return sae, history


def sae_diagnostics(sae, activations, chunk_size=16384, dead_rate=1e-5):
    """Return reconstruction and sparsity diagnostics for ``sae``.

    Parameters
    ----------
    sae : SparseAutoencoder
        A fitted autoencoder.
    activations : torch.Tensor
        Raw layer outputs of shape ``(..., n_inputs)``. They are normalised
        with the statistics stored on ``sae``, which default to zero mean
        and unit standard deviation, so passing already-normalised
        activations to an autoencoder fitted with
        ``normalize_activations=False`` is also correct.
    chunk_size : int, default=16384
        Rows per forward pass. The latent is ``expansion`` times wider than
        the input, so a single pass over a large activation set needs many
        times its own size in memory: two million rows against a
        1600-feature dictionary is roughly 13 GB for the latent alone,
        before the top-k mask doubles it.
    dead_rate : float, default=1e-5
        Firing rate below which a feature counts as dead. This is a rate
        rather than a never-fired check because over a few million rows
        almost every feature wins top-k at least once, so the stricter test
        reports zero whatever the dictionary is doing.

    Returns
    -------
    dict
        ``l0``, the mean number of active features per row; ``dead``, the
        fraction of features firing on less than ``dead_rate`` of rows; and
        ``r2``, the fraction of activation variance reconstructed.

    Notes
    -----
    ``r2`` measures reconstructed variance, which is not the same as
    task relevance. A dictionary can reconstruct almost all the variance
    while discarding the directions a downstream head relies on, and the
    converse also occurs. Pair it with a substitution check built on
    :func:`run_with_activation_substitution`.
    """
    activations = torch.as_tensor(activations, dtype=torch.float32)
    flat, _ = _flatten_features(activations, sae.n_inputs)
    device = sae.activation_mean.device
    n_rows = flat.shape[0]

    # r2 needs deviations from the global mean, so the mean is accumulated
    # first rather than approximated per chunk.
    with torch.no_grad():
        mean = torch.zeros(sae.n_inputs, device=device)
        for start in range(0, n_rows, chunk_size):
            batch = flat[start : start + chunk_size].to(device)
            mean += sae.normalize_activations(batch).sum(dim=0)
        mean /= max(n_rows, 1)

        fires = torch.zeros(sae.n_features, device=device)
        active_total = 0.0
        sse = torch.zeros((), device=device)
        sst = torch.zeros((), device=device)
        for start in range(0, n_rows, chunk_size):
            batch = sae.normalize_activations(
                flat[start : start + chunk_size].to(device)
            )
            recon, latent = sae(batch)
            active = latent != 0
            active_total += float(active.sum())
            fires += active.sum(dim=0)
            sse += torch.sum((recon - batch) ** 2)
            sst += torch.sum((batch - mean) ** 2)

    return {
        "l0": active_total / max(n_rows, 1),
        "dead": float((fires / max(n_rows, 1) < dead_rate).float().mean()),
        "r2": float(1.0 - sse / sst.clamp_min(1e-12)),
    }


def capture_activations(model, x, layers, forward_fn=None, detach=True):
    """Capture the outputs of one or more modules during a forward pass.

    Hooks are registered before the forward pass and removed in a ``finally``
    block, so they do not leak if the forward pass raises.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run.
    x : torch.Tensor
        Input passed to ``forward_fn``.
    layers : torch.nn.Module or list of torch.nn.Module or dict
        Modules whose outputs to capture. A single module is keyed ``0``, a
        sequence is keyed by position, and a mapping keeps its own keys.
    forward_fn : callable or None, default=None
        Callable invoked as ``forward_fn(x)``. Defaults to ``model``. Use
        this to reach a submodule's forward, for instance a backbone's
        feature extractor rather than the full classifier.
    detach : bool, default=True
        Detach captured tensors from the autograd graph. Set to ``False``
        only when gradients through the captured activations are needed.

    Returns
    -------
    dict
        One entry per requested module, keyed as described above.

    See Also
    --------
    run_with_activation_substitution : Replace a layer's output instead.
    """
    forward_fn = model if forward_fn is None else forward_fn
    if isinstance(layers, nn.Module):
        items = [(0, layers)]
    else:
        items = _as_module_items(layers)
    captured = {}
    handles = []

    def make_hook(key):
        """Build a forward hook that stores its module's output under ``key``."""

        def hook(_module, _inputs, output):
            """Store this module's output, detaching it when asked."""
            captured[key] = _detach_output(output) if detach else output

        return hook

    for key, module in items:
        handles.append(module.register_forward_hook(make_hook(key)))

    try:
        forward_fn(x)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def run_with_activation_substitution(model, x, layer, substitute_fn, forward_fn=None):
    """Run ``model`` with the output of ``layer`` replaced.

    The basis of a faithfulness check: substitute a sparse autoencoder's
    reconstruction for a layer's output and re-measure the task with the
    model's own head. A small change in the metric means the dictionary
    captured what the model actually uses, whereas reconstruction error
    alone only shows it captured the variance.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run.
    x : torch.Tensor
        Input passed to ``forward_fn``.
    layer : torch.nn.Module
        Module whose output is replaced.
    substitute_fn : callable
        Called with the layer's output; its return value is used instead.
        For an autoencoder this is usually
        :meth:`SparseAutoencoder.reconstruct_activations`, which handles
        normalisation internally.
    forward_fn : callable or None, default=None
        Callable invoked as ``forward_fn(x)``. Defaults to ``model``.

    Returns
    -------
    Any
        Whatever ``forward_fn`` returns, computed with the substitution in
        place.

    See Also
    --------
    capture_activations : Record a layer's output instead of replacing it.
    """
    forward_fn = model if forward_fn is None else forward_fn

    def hook(_module, _inputs, output):
        """Return the substituted output in place of the module's own."""
        return substitute_fn(output)

    handle = layer.register_forward_hook(hook)
    try:
        return forward_fn(x)
    finally:
        handle.remove()


def grouped_train_valid_test_split(
    groups, train_size=0.7, valid_size=0.15, test_size=0.15, seed=0
):
    """Split sample indices by group without leaking a group across splits.

    Windows from one recording or subject are far from independent, so a
    random split over samples inflates every downstream score. This splits
    over unique groups instead.

    Parameters
    ----------
    groups : array-like
        Group label per sample, for instance a recording or subject
        identifier. Length equals the number of samples.
    train_size : float, default=0.7
        Fraction of groups assigned to the training split.
    valid_size : float, default=0.15
        Fraction of groups assigned to the validation split.
    test_size : float, default=0.15
        Fraction of groups assigned to the test split.
    seed : int, default=0
        Seed for the group shuffle.

    Returns
    -------
    train_idx : numpy.ndarray
        Sample indices of the training split.
    valid_idx : numpy.ndarray
        Sample indices of the validation split.
    test_idx : numpy.ndarray
        Sample indices of the test split.

    Raises
    ------
    ValueError
        If the three fractions do not sum to one, or if there are fewer
        than three distinct groups.
    """
    fractions = np.array([train_size, valid_size, test_size], dtype=float)
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train_size + valid_size + test_size must equal 1")
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        raise ValueError("need at least three groups for a train/valid/test split")
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n_groups = shuffled.size
    counts = np.maximum(1, np.round(fractions * n_groups).astype(int))
    while counts.sum() > n_groups:
        for idx in np.argsort(-counts):
            if counts.sum() == n_groups:
                break
            if counts[idx] > 1:
                counts[idx] -= 1
    while counts.sum() < n_groups:
        counts[int(np.argmax(fractions))] += 1
    n_train, n_valid, _ = counts.tolist()
    train_groups = shuffled[:n_train]
    valid_groups = shuffled[n_train : n_train + n_valid]
    test_groups = shuffled[n_train + n_valid :]
    train_idx = np.flatnonzero(np.isin(groups, train_groups))
    valid_idx = np.flatnonzero(np.isin(groups, valid_groups))
    test_idx = np.flatnonzero(np.isin(groups, test_groups))
    return train_idx, valid_idx, test_idx
