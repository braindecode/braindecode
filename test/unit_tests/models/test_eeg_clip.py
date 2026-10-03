# Authors: lindicaphxag-tech
#
# License: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from braindecode.models import EEGCLIP


class _MeanEEGEncoder(nn.Module):
    def forward(self, X):
        return X.mean(dim=-1)


class _TemporalEEGEncoder(nn.Module):
    def forward(self, X):
        return X


class _TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size=16, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _make_model(**kwargs):
    kwargs.setdefault("drop_prob", 0)
    return EEGCLIP(
        n_chans=3,
        n_times=20,
        n_outputs=4,
        eeg_encoder=_MeanEEGEncoder(),
        eeg_embedding_dim=3,
        text_embedding_dim=8,
        text_encoder=_TinyTextEncoder(),
        **kwargs,
    )


def test_eeg_clip_standard_forward_is_tensor_valued():
    model = _make_model().eval()
    X = torch.randn(3, 3, 20)

    output = model(X)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 4)
    torch.testing.assert_close(output, model.encode_eeg(X))


def test_eeg_clip_encodes_paired_batches_and_returns_symmetric_logits():
    model = _make_model()
    X = torch.randn(5, 3, 20)
    input_ids = torch.randint(0, 16, (5, 6))
    attention_mask = torch.ones_like(input_ids)

    output = model.forward_paired(X, input_ids, attention_mask=attention_mask)

    assert output["eeg_embeds"].shape == (5, 4)
    assert output["text_embeds"].shape == (5, 4)
    assert output["logits_per_eeg"].shape == (5, 5)
    torch.testing.assert_close(output["logits_per_text"], output["logits_per_eeg"].T)
    scale = model.logit_scale.clamp(max=torch.tensor(100.0).log()).exp()
    torch.testing.assert_close(
        output["logits_per_eeg"],
        scale * output["eeg_embeds"] @ output["text_embeds"].T,
    )


def test_eeg_clip_projects_temporal_predictions_before_pooling():
    model = EEGCLIP(
        n_chans=2,
        n_times=2,
        n_outputs=2,
        eeg_encoder=_TemporalEEGEncoder(),
        eeg_embedding_dim=2,
        text_embedding_dim=2,
        projection_layers=2,
        drop_prob=0,
    ).eval()

    with torch.no_grad():
        first_linear = model.final_layer[0]
        batch_norm = model.final_layer[1]
        last_linear = model.final_layer[-1]
        first_linear.weight.copy_(torch.eye(2))
        first_linear.bias.zero_()
        batch_norm.weight.fill_(1)
        batch_norm.bias.zero_()
        batch_norm.running_mean.zero_()
        batch_norm.running_var.fill_(1)
        last_linear.weight.copy_(torch.eye(2))
        last_linear.bias.zero_()

    X = torch.tensor([[[-1.0, 1.0], [0.0, 0.0]]])
    actual = model.encode_eeg(X)
    temporal = X.transpose(1, 2).reshape(-1, 2)
    expected = model.final_layer(temporal).reshape(1, 2, 2).mean(dim=1)
    pool_first = model.final_layer(X.mean(dim=-1))

    # The released model projects each dense Deep4 prediction before averaging.
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, pool_first)


def test_eeg_clip_masked_mean_pooling_ignores_padding():
    model = _make_model(text_pooling="mean")
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    encoded = model.text_encoder(tokens).last_hidden_state

    actual = model.encode_text(tokens, attention_mask=mask)
    expected_features = torch.stack(
        [encoded[0, :2].mean(dim=0), encoded[1, :1].mean(dim=0)]
    )
    expected = model.text_projection(expected_features)
    torch.testing.assert_close(actual, expected)


def test_eeg_clip_projection_matches_published_architecture():
    model = _make_model()
    for head in (model.text_projection, model.final_layer):
        assert sum(isinstance(module, nn.Linear) for module in head) == 3
        assert sum(isinstance(module, nn.BatchNorm1d) for module in head) == 2
        assert sum(isinstance(module, nn.ReLU) for module in head) == 2
        assert sum(isinstance(module, nn.Dropout) for module in head) == 2
        assert head[-1].out_features == model.n_outputs


def test_eeg_clip_projection_depth_is_configurable():
    model = _make_model(projection_layers=2)
    assert sum(isinstance(module, nn.Linear) for module in model.text_projection) == 2
    assert sum(isinstance(module, nn.Linear) for module in model.final_layer) == 2

    with pytest.raises(ValueError, match="projection_layers"):
        _make_model(projection_layers=0)


def test_eeg_clip_logits_match_released_scaled_dot_product():
    model = _make_model().eval()
    with torch.no_grad():
        model.logit_scale.fill_(torch.log(torch.tensor(2.0)))

    eeg = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    text = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    logits_eeg, logits_text = model.compute_logits(eeg, text)
    expected = 2.0 * eeg @ text.T

    torch.testing.assert_close(logits_eeg, expected)
    torch.testing.assert_close(logits_text, expected.T)


def test_eeg_clip_contrastive_loss_is_differentiable():
    model = _make_model()
    X = torch.randn(4, 3, 20)
    input_ids = torch.randint(0, 16, (4, 6))
    output = model.forward_paired(X, input_ids)

    loss = model.contrastive_loss(output["eeg_embeds"], output["text_embeds"])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.logit_scale.grad is not None
    assert model.final_layer[0].weight.grad is not None
    assert model.text_projection[0].weight.grad is not None


def test_eeg_clip_requires_paired_non_empty_batches():
    model = _make_model()
    eeg = torch.randn(2, 4)
    text = torch.randn(3, 4)

    with pytest.raises(ValueError, match="paired examples"):
        model.contrastive_loss(eeg, text)
    with pytest.raises(ValueError, match="must not be empty"):
        model.contrastive_loss(eeg[:0], text[:0])


def test_eeg_clip_rejects_mismatched_attention_mask():
    model = _make_model(text_pooling="mean")
    input_ids = torch.randint(0, 16, (2, 5))

    with pytest.raises(ValueError, match="attention_mask must match"):
        model.encode_text(input_ids, attention_mask=torch.ones(2, 4))


def test_eeg_clip_default_encoder_uses_dense_temporal_predictions():
    model = EEGCLIP(
        n_chans=3,
        n_times=1000,
        n_outputs=4,
        eeg_embedding_dim=8,
        text_embedding_dim=8,
        drop_prob=0,
    ).eval()

    with torch.no_grad():
        features = model.eeg_encoder(torch.randn(2, 3, 1000))
        embeds = model.encode_eeg(torch.randn(2, 3, 1000))

    assert features.ndim == 3
    assert features.shape[:2] == (2, 8)
    assert features.shape[-1] > 1
    assert embeds.shape == (2, 4)


def test_eeg_clip_reset_head_updates_both_projection_dimensions():
    model = EEGCLIP(
        n_chans=3,
        n_times=1000,
        n_outputs=4,
        eeg_embedding_dim=8,
        text_embedding_dim=8,
        drop_prob=0,
    )
    model.reset_head(6)

    output = model.forward_paired(torch.randn(2, 3, 1000), torch.randn(2, 8))
    assert model.n_outputs == 6
    assert model.get_config()["n_outputs"] == 6
    assert output["eeg_embeds"].shape == (2, 6)
    assert output["text_embeds"].shape == (2, 6)


def test_eeg_clip_reset_head_preserves_mixed_projection_training_modes():
    model = _make_model(drop_prob=0.5).eval()
    text_dropout = next(
        module for module in model.text_projection.modules() if isinstance(module, nn.Dropout)
    )
    text_dropout.train()  # Keep text-side MC dropout enabled.
    text_modes = [module.training for module in model.text_projection.modules()]
    eeg_modes = [module.training for module in model.final_layer.modules()]

    model.reset_head(6)

    assert [module.training for module in model.text_projection.modules()] == text_modes
    assert [module.training for module in model.final_layer.modules()] == eeg_modes


def test_eeg_clip_reset_head_preserves_projection_device_and_dtype():
    model = _make_model().to(dtype=torch.float64)
    text_device = next(model.text_projection.parameters()).device
    eeg_device = next(model.final_layer.parameters()).device

    model.reset_head(6)

    assert {parameter.dtype for parameter in model.text_projection.parameters()} == {
        torch.float64
    }
    assert {parameter.dtype for parameter in model.final_layer.parameters()} == {
        torch.float64
    }
    assert next(model.text_projection.parameters()).device == text_device
    assert next(model.final_layer.parameters()).device == eeg_device

    output = model.forward_paired(
        torch.randn(3, 3, 20, dtype=torch.float64),
        torch.randint(0, 16, (3, 6)),
    )
    assert output["eeg_embeds"].dtype == torch.float64
    assert output["text_embeds"].dtype == torch.float64


def test_eeg_clip_custom_encoders_require_manual_config_round_trip():
    model = _make_model()

    with pytest.raises(ValueError, match="cannot serialize custom encoder"):
        model.get_config()
