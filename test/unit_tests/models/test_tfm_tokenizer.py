import json

import pytest
import torch

from braindecode.models import TFMTokenizer


def _small_tfm_tokenizer(**kwargs):
    params = {
        "sfreq": 200,
        "embed_dim": 16,
        "codebook_size": 32,
        "freq_patch_size": 5,
        "freq_encoder_depth": 1,
        "temporal_encoder_depth": 1,
        "decoder_depth": 1,
        "max_seq_len": 32,
    }
    params.update(kwargs)
    return TFMTokenizer(**params)


def test_tfm_tokenizer_shapes_and_codebook_range():
    model = _small_tfm_tokenizer().eval()
    x = torch.randn(2, 3, 500)

    output = model.tokenize(x)

    assert output.reconstruction.shape == (2, 3, 100, 4)
    assert output.token_ids.shape == (2, 3, 4)
    assert output.quantized.shape == output.embeddings.shape == (6, 4, 16)
    assert output.target_spectrogram.shape == (2, 3, 100, 4)
    assert output.token_ids.dtype == torch.long
    assert output.token_ids.min() >= 0
    assert output.token_ids.max() < model.codebook_size
    assert torch.isfinite(output.quantization_loss)


def test_tfm_tokenizer_uses_complementary_masks_and_keeps_full_target():
    model = _small_tfm_tokenizer().eval()
    x = torch.randn(2, 3, 500)
    target = model.compute_spectrogram(x)
    mask_a, mask_b = model.make_complementary_masks(target)

    assert mask_a.shape == mask_b.shape == target.shape
    assert torch.equal(mask_a ^ mask_b, torch.ones_like(mask_a))
    output = model.tokenize(x, spectrogram_mask=mask_a)
    torch.testing.assert_close(output.target_spectrogram, target)


def test_tfm_tokenizer_masks_are_reference_shared_and_complementary():
    model = _small_tfm_tokenizer().eval()
    target = model.compute_spectrogram(torch.randn(3, 2, 500))

    mask_a, mask_b = model.make_complementary_masks(target)

    # The released training code samples one set of frequency/time groups and
    # applies it to every flattened EEG channel in the batch.
    torch.testing.assert_close(mask_a[0, 0], mask_a[-1, -1])
    torch.testing.assert_close(mask_b[0, 0], mask_b[-1, -1])
    assert torch.equal(mask_b, ~mask_a)


def test_tfm_tokenizer_ema_codebook_updates_only_in_training_mode():
    model = _small_tfm_tokenizer()
    x = torch.randn(2, 2, 500)

    before_cluster = model.quantizer.cluster_size.clone()
    before_ema = model.quantizer.ema_weight.clone()
    model.tokenize(x)

    assert not torch.equal(model.quantizer.cluster_size, before_cluster)
    assert not torch.equal(model.quantizer.ema_weight, before_ema)

    model.eval()
    frozen_cluster = model.quantizer.cluster_size.clone()
    frozen_ema = model.quantizer.ema_weight.clone()
    model.tokenize(x)

    torch.testing.assert_close(model.quantizer.cluster_size, frozen_cluster)
    torch.testing.assert_close(model.quantizer.ema_weight, frozen_ema)


def test_tfm_tokenizer_ema_does_not_inflate_unseen_codebook_entries():
    torch.manual_seed(7)
    model = _small_tfm_tokenizer(codebook_size=64)
    before = model.quantizer.embedding.weight.detach().clone()

    output = model.tokenize(torch.randn(1, 1, 200))

    used = torch.zeros(model.codebook_size, dtype=torch.bool)
    used[output.token_ids.unique()] = True
    assert used.any()
    assert (~used).any()

    # EMA centroids should move for selected codes, but never-selected codes
    # must stay available for future batches instead of being divided by eps.
    after = model.quantizer.embedding.weight.detach()
    torch.testing.assert_close(after[~used], before[~used])
    assert torch.isfinite(after).all()
    assert not torch.equal(after[used], before[used])


def test_tfm_tokenizer_reference_defaults_match_released_2x2x8_variant():
    model = TFMTokenizer(
        n_chans=3,
        n_outputs=2,
        n_times=1000,
        sfreq=200,
    )

    assert model.window_size == 200
    assert model.n_freqs == 100
    assert model.embed_dim == 64
    assert model.codebook_size == 8192
    assert model.freq_patch_size == 5
    assert model.commitment_cost == 1.0


def test_tfm_tokenizer_reconstruction_backpropagates_to_both_paths():
    model = _small_tfm_tokenizer()
    x = torch.randn(2, 2, 500)

    output = model.tokenize(x)
    (output.reconstruction.square().mean() + output.quantization_loss).backward()

    assert model.frequency_patch_embedding[0].weight.grad is not None
    assert model.temporal_patch_embedding[0].weight.grad is not None
    assert model.quantizer.embedding.weight.grad is None
    assert model.quantizer.embedding.weight.requires_grad is False
    assert torch.isfinite(model.frequency_patch_embedding[0].weight.grad).all()
    assert torch.isfinite(model.temporal_patch_embedding[0].weight.grad).all()


def test_tfm_tokenizer_vq_loss_keeps_nonzero_encoder_gradient_with_ema_codebook():
    model = _small_tfm_tokenizer(commitment_cost=1.0)
    x = torch.randn(2, 2, 500)

    output = model.tokenize(x)
    output.quantization_loss.backward()

    freq_grad = model.frequency_patch_embedding[0].weight.grad
    temporal_grad = model.temporal_patch_embedding[0].weight.grad
    assert freq_grad is not None
    assert temporal_grad is not None
    assert freq_grad.norm() > 0
    assert temporal_grad.norm() > 0
    assert model.quantizer.embedding.weight.grad is None


def test_tfm_tokenizer_optimizer_step_does_not_update_post_ema_codebook():
    model = _small_tfm_tokenizer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 2, 500)

    output = model.tokenize(x)
    post_ema = model.quantizer.embedding.weight.detach().clone()
    optimizer.zero_grad()
    output.quantization_loss.backward()
    optimizer.step()

    assert model.quantizer.embedding.weight.grad is None
    torch.testing.assert_close(model.quantizer.embedding.weight.detach(), post_ema)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sfreq": 201}, "positive even integer"),
        ({"sfreq": 200.5}, "positive even integer"),
        ({"embed_dim": 12}, "multiple of 8"),
        ({"codebook_size": 1}, "at least 2"),
        ({"freq_patch_size": 6}, "divisible"),
        ({"drop_prob": 1.0}, "drop_prob must be in \\[0, 1\\)"),
    ],
)
def test_tfm_tokenizer_rejects_invalid_architecture(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _small_tfm_tokenizer(**kwargs)


def test_tfm_tokenizer_rejects_invalid_input():
    model = _small_tfm_tokenizer()

    with pytest.raises(ValueError, match=r"shape \(batch, channels, samples\)"):
        model(torch.randn(2, 500))
    with pytest.raises(ValueError, match="at least one second"):
        model(torch.randn(2, 3, 199))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(2, 3, 500, dtype=torch.long))


def test_tfm_tokenizer_rejects_a_mask_with_the_wrong_shape():
    model = _small_tfm_tokenizer()
    with pytest.raises(ValueError, match="spectrogram_mask must match"):
        model(torch.randn(2, 3, 500), spectrogram_mask=torch.ones(2, 3, 100, 3))


def test_tfm_tokenizer_rejects_sequences_longer_than_max_seq_len():
    model = _small_tfm_tokenizer(max_seq_len=32)
    with pytest.raises(ValueError, match="exceeding max_seq_len=32"):
        model(torch.randn(1, 1, 200 + 32 * 100))


def test_tfm_tokenizer_temporal_limit_is_independent_of_frequency_tokens():
    # sfreq=200 and freq_patch_size=5 produce 20 frequency tokens, but the
    # frequency encoder has its own sequence limit. A four-frame temporal input
    # is therefore valid with max_seq_len=4.
    model = _small_tfm_tokenizer(max_seq_len=4).eval()
    output = model(torch.randn(1, 1, 500))

    assert output.shape == (1, 1, 100, 4)


def test_tfm_tokenizer_config_round_trip_is_json_serializable():
    model = TFMTokenizer(
        n_chans=3,
        n_outputs=2,
        n_times=1000,
        sfreq=200,
        codebook_size=256,
        activation=torch.nn.ReLU,
    )

    config = model.get_config()
    json.dumps(config)
    restored = TFMTokenizer.from_config(config)

    assert restored.n_chans == model.n_chans
    assert restored.n_times == model.n_times
    assert restored.sfreq == model.sfreq
    assert restored.codebook_size == model.codebook_size
    assert restored.activation is torch.nn.ReLU


def test_tfm_tokenizer_state_dict_preserves_ema_codebook_state():
    model = _small_tfm_tokenizer()
    model.tokenize(torch.randn(2, 2, 500))

    restored = _small_tfm_tokenizer()
    restored.load_state_dict(model.state_dict())

    torch.testing.assert_close(
        restored.quantizer.embedding.weight,
        model.quantizer.embedding.weight,
    )
    torch.testing.assert_close(
        restored.quantizer.cluster_size,
        model.quantizer.cluster_size,
    )
    torch.testing.assert_close(
        restored.quantizer.ema_weight,
        model.quantizer.ema_weight,
    )
    assert "stft_window" not in model.state_dict()


def test_tfm_tokenizer_default_forward_is_tensor_valued():
    model = _small_tfm_tokenizer().eval()
    x = torch.randn(2, 3, 500)

    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, 100, 4)
