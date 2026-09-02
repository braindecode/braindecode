import pytest
import torch

from braindecode.visualization import SparseAutoencoder

N_INPUTS = 4
EXPANSION = 2
K = 3


def _sae():
    return SparseAutoencoder(n_inputs=N_INPUTS, expansion=EXPANSION, k=K).eval()


def test_encode_keeps_at_most_k_and_preserves_leading_dims():
    sae = _sae()
    z = sae.encode(torch.randn(2, 5, N_INPUTS))

    assert z.shape == (2, 5, N_INPUTS * EXPANSION)
    assert ((z != 0).sum(-1) <= K).all()


def test_relu_can_leave_fewer_than_k_active():
    """The rectifier binds before the mask, so ``k`` is a bound, not a count."""
    sae = _sae()
    with torch.no_grad():
        sae.encoder.bias.fill_(-50.0)

    z = sae.encode(torch.randn(6, N_INPUTS))

    assert (z == 0).all()


def test_the_decoder_bias_is_subtracted_before_encoding():
    """The input and output biases are tied, so shifting one cancels it."""
    sae = _sae()
    x = torch.randn(6, N_INPUTS)
    with torch.no_grad():
        sae.decoder.bias.zero_()
    baseline = sae.encode(x)

    shift = torch.arange(1.0, N_INPUTS + 1.0)
    with torch.no_grad():
        sae.decoder.bias.copy_(shift)

    torch.testing.assert_close(sae.encode(x + shift), baseline)


def test_gradients_reach_the_selected_features_and_no_others():
    """A mask that dropped the gradient would leave the encoder untrainable."""
    sae = _sae()
    with torch.no_grad():
        sae.encoder.bias.fill_(10.0)
        sae.decoder.weight.fill_(1.0)
        sae.normalize_decoder_()

    recon, latent = sae(torch.ones(1, N_INPUTS))
    recon.sum().backward()

    fired = latent[0] != 0
    assert fired.sum() == K
    assert torch.equal(sae.encoder.weight.grad.abs().sum(dim=1) != 0, fired)


def test_decoder_columns_are_unit_norm():
    sae = _sae()
    norms = sae.decoder.weight.norm(dim=0)

    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_reconstruct_activations_applies_the_stored_statistics():
    """Feeding the mean itself standardises to zero, pinning both directions.

    The expected code comes from encoding zeros, so it is wrong if the input
    is not standardised on the way in, and the expected scale is wrong if the
    reconstruction is not de-standardised on the way out.
    """
    sae = _sae()
    mean = torch.arange(1.0, N_INPUTS + 1.0)
    std = torch.full((N_INPUTS,), 2.0)
    sae.set_activation_normalization(mean, std)

    out = sae.reconstruct_activations(mean.expand(6, N_INPUTS))

    expected = sae.decode(sae.encode(torch.zeros(6, N_INPUTS))) * std + mean
    torch.testing.assert_close(out, expected)
    # Buffers, not plain attributes: a saved autoencoder that lost these would
    # reload with mean 0 and std 1 and reconstruct wrongly without erroring.
    assert {"activation_mean", "activation_std"} <= set(sae.state_dict())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_inputs=True), "n_inputs must be a positive integer"),
        (dict(n_inputs=0), "n_inputs must be a positive integer"),
        (dict(k=True), "k must be an integer"),
        (dict(k=2.0), "k must be an integer"),
        (dict(k=N_INPUTS * EXPANSION + 1), "k must be an integer"),
    ],
)
def test_constructor_rejects_unusable_dimensions(kwargs, match):
    """``int(True)`` is 1, so a bool would otherwise be silently accepted."""
    with pytest.raises(ValueError, match=match):
        SparseAutoencoder(**{"n_inputs": N_INPUTS, "expansion": EXPANSION, **kwargs})


@pytest.mark.parametrize(
    "method, width", [("encode", N_INPUTS), ("decode", N_INPUTS * EXPANSION)]
)
def test_wrong_width_names_the_expected_dimension(method, width):
    sae = _sae()
    with pytest.raises(ValueError, match=f"expected last dimension {width}"):
        getattr(sae, method)(torch.randn(6, width + 1))


@pytest.mark.parametrize(
    "mean, std, match",
    [
        (torch.zeros(N_INPUTS + 1), torch.ones(N_INPUTS), "must both have shape"),
        (torch.full((N_INPUTS,), float("nan")), torch.ones(N_INPUTS), "must be finite"),
        (torch.zeros(N_INPUTS), torch.zeros(N_INPUTS), "strictly positive"),
    ],
)
def test_activation_normalization_rejects_unusable_statistics(mean, std, match):
    """A zero standard deviation would divide a constant dimension to infinity."""
    with pytest.raises(ValueError, match=match):
        _sae().set_activation_normalization(mean, std)
