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

    output = model(x)

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
    output = model(x, spectrogram_mask=mask_a)
    torch.testing.assert_close(output.target_spectrogram, target)


def test_tfm_tokenizer_reconstruction_backpropagates_to_both_paths():
    model = _small_tfm_tokenizer()
    x = torch.randn(2, 2, 500)

    output = model(x)
    (output.reconstruction.square().mean() + output.quantization_loss).backward()

    assert model.frequency_patch_embedding[0].weight.grad is not None
    assert model.temporal_patch_embedding[0].weight.grad is not None
    assert model.quantizer.embedding.weight.grad is not None
    assert torch.isfinite(model.frequency_patch_embedding[0].weight.grad).all()
    assert torch.isfinite(model.temporal_patch_embedding[0].weight.grad).all()
    assert torch.isfinite(model.quantizer.embedding.weight.grad).all()


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
