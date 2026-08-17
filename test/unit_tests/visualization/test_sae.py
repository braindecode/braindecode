import io

import numpy as np
import pytest
import torch
from torch import nn

from braindecode.visualization import (
    SparseAutoencoder,
    capture_activations,
    fit_sparse_autoencoder,
    grouped_train_valid_test_split,
    run_with_activation_substitution,
    sae_diagnostics,
)


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, x):
        x = self.block(x)
        return self.head(x)


class ExplodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, x):
        x = self.block(x)
        raise RuntimeError("boom")


def _make_positive_sae():
    sae = SparseAutoencoder(n_inputs=4, expansion=2, k=3)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
        sae.encoder.bias.fill_(0.5)
        sae.decoder.weight.copy_(torch.randn_like(sae.decoder.weight))
        sae.decoder.bias.zero_()
    return sae


def test_encode_returns_exact_k_nonzeros():
    sae = _make_positive_sae()
    x = torch.ones(2, 4)
    latent = sae.encode(x)
    assert latent.shape == (2, 8)
    assert torch.equal((latent != 0).sum(dim=1), torch.full((2,), 3, dtype=torch.long))


def test_decoder_normalization_and_diagnostics():
    sae = SparseAutoencoder(n_inputs=4, expansion=2, k=2)
    with torch.no_grad():
        sae.decoder.weight.copy_(torch.randn_like(sae.decoder.weight))
    sae.normalize_decoder_()
    norms = sae.decoder.weight.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms))

    x = torch.randn(6, 4)
    stats = sae_diagnostics(sae, x)
    assert set(stats) == {"l0", "dead", "r2"}
    assert stats["l0"] >= 0
    assert 0.0 <= stats["dead"] <= 1.0


def test_config_and_state_dict_roundtrip():
    sae = SparseAutoencoder(n_inputs=4, expansion=2, k=3)
    with torch.no_grad():
        sae.encoder.weight.uniform_(-0.5, 0.5)
    sae.set_activation_normalization(torch.arange(4.0), torch.arange(1.0, 5.0))
    clone = SparseAutoencoder.from_config(sae.get_config())
    clone.load_state_dict(sae.state_dict())

    buf = io.BytesIO()
    torch.save(sae.state_dict(), buf)
    buf.seek(0)
    reloaded = SparseAutoencoder.from_config(sae.get_config())
    reloaded.load_state_dict(torch.load(buf))

    torch.testing.assert_close(clone.encoder.weight, sae.encoder.weight)
    torch.testing.assert_close(clone.activation_mean, sae.activation_mean)
    torch.testing.assert_close(clone.activation_std, sae.activation_std)
    torch.testing.assert_close(reloaded.decoder.bias, sae.decoder.bias)


def test_capture_activations_returns_one_entry_and_cleans_hooks():
    model = ToyNet().eval()
    x = torch.randn(3, 4)
    captured = capture_activations(model, x, {"block": model.block})
    assert set(captured) == {"block"}
    assert captured["block"].shape == (3, 6)
    assert len(model.block._forward_hooks) == 0


def test_capture_activations_cleans_up_on_exception():
    model = ExplodingNet().eval()
    x = torch.randn(2, 4)
    with pytest.raises(RuntimeError, match="boom"):
        capture_activations(model, x, [model.block])
    assert len(model.block._forward_hooks) == 0


def test_activation_substitution_identity_and_change():
    model = ToyNet().eval()
    x = torch.randn(4, 4)

    baseline = model(x)
    identity = run_with_activation_substitution(model, x, model.block, lambda out: out)
    torch.testing.assert_close(identity, baseline)
    assert len(model.block._forward_hooks) == 0

    shifted = run_with_activation_substitution(
        model, x, model.block, lambda out: out + 1.0
    )
    assert not torch.allclose(shifted, baseline)
    assert len(model.block._forward_hooks) == 0


def test_resample_dead_features_changes_only_dead_columns_and_resets_state():
    sae = SparseAutoencoder(n_inputs=4, expansion=2, k=2)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
    x = torch.randn(12, 4)
    recon, _ = sae(x)
    loss = torch.mean((recon - x) ** 2)
    loss.backward()
    optimizer.step()

    enc_before = sae.encoder.weight.detach().clone()
    dec_before = sae.decoder.weight.detach().clone()
    bias_before = sae.decoder.bias.detach().clone()
    alive_norm = enc_before[2:].norm(dim=1).mean()
    dead = torch.tensor([True, True, False, False, False, False, False, False])
    changed = sae.resample_dead_features(x, dead, optimizer=optimizer, seed=0)

    assert changed == 2
    assert not torch.allclose(sae.encoder.weight[:2], enc_before[:2])
    assert not torch.allclose(sae.decoder.weight[:, :2], dec_before[:, :2])
    torch.testing.assert_close(sae.encoder.weight[2:], enc_before[2:])
    torch.testing.assert_close(sae.decoder.weight[:, 2:], dec_before[:, 2:])
    torch.testing.assert_close(sae.decoder.bias, bias_before)
    torch.testing.assert_close(
        sae.encoder.weight[:2].norm(dim=1), (0.2 * alive_norm).expand(2)
    )
    enc_state = optimizer.state[sae.encoder.weight]["exp_avg"]
    dec_state = optimizer.state[sae.decoder.weight]["exp_avg"]
    assert torch.count_nonzero(enc_state[:2]) == 0
    assert torch.count_nonzero(dec_state[:, :2]) == 0
    # Only the resampled slots are cleared; the rest keep their momentum.
    assert torch.count_nonzero(enc_state[2:]) > 0


def test_diagnostics_are_chunk_size_invariant():
    # A single pass over a large activation set needs many times its own size,
    # since the latent is `expansion` times wider than the input. Chunking must
    # not change the answer.
    sae = SparseAutoencoder(n_inputs=4, expansion=2, k=2)
    x = torch.randn(200, 4)
    whole = sae_diagnostics(sae, x, chunk_size=1000)
    chunked = sae_diagnostics(sae, x, chunk_size=16)
    for key in ("l0", "dead", "r2"):
        assert abs(whole[key] - chunked[key]) < 1e-4, key


def test_fit_sparse_autoencoder_resamples_dead_features():
    x = torch.randn(8, 4)
    sae, history = fit_sparse_autoencoder(
        x,
        expansion=2,
        k=2,
        epochs=1,
        batch_size=4,
        resample_steps=1,
        seed=0,
    )
    # Two steps per epoch, but resample_until=0.8 leaves the last one alone so
    # revived features are not scored before they have trained.
    assert len(history["resampled"]) == 1
    assert sae.activation_mean.shape == (4,)
    assert sae.activation_std.shape == (4,)


def test_resampling_stops_before_the_end_of_training():
    x = torch.randn(8, 4)
    _, cooled = fit_sparse_autoencoder(
        x, expansion=2, k=2, epochs=1, batch_size=4, resample_steps=1, seed=0
    )
    _, full = fit_sparse_autoencoder(
        x,
        expansion=2,
        k=2,
        epochs=1,
        batch_size=4,
        resample_steps=1,
        seed=0,
        resample_until=1.0,
    )
    assert len(cooled["resampled"]) < len(full["resampled"])


def test_grouped_split_keeps_groups_together():
    groups = np.repeat(np.arange(6), 3)
    train_idx, valid_idx, test_idx = grouped_train_valid_test_split(groups, seed=1)

    train_groups = set(groups[train_idx])
    valid_groups = set(groups[valid_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)
    assert sorted(train_idx.tolist() + valid_idx.tolist() + test_idx.tolist()) == list(
        range(groups.size)
    )
