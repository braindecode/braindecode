# Authors: Fashad Ahmed <Fashad-Ahmed@users.noreply.github.com>
#
# License: BSD-3

import pytest
import torch

from braindecode.models.sleepfm import (
    SleepFM,
    SleepFMStager,
    _SleepFMAttentionPooling,
    _SleepFMTokenizer,
)
from braindecode.models.util import models_dict


def _small_sleepfm(**kwargs):
    defaults = dict(
        n_chans=3,
        n_times=640,
        n_outputs=4,
        sfreq=128.0,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        pooling_heads=4,
        drop_prob=0.0,
        max_seq_length=4,
    )
    return SleepFM(**(defaults | kwargs))


def _small_stager(**kwargs):
    defaults = dict(
        n_chans=3,
        n_times=1280,
        n_outputs=5,
        sfreq=128.0,
        embed_dim=16,
        staging_num_heads=4,
        staging_num_layers=1,
        staging_pooling_heads=4,
        drop_prob=0.0,
        max_seq_length=4,
    )
    return SleepFMStager(**(defaults | kwargs))


@pytest.mark.parametrize("n_times,n_patches", [(640, 1), (1280, 2), (1287, 2)])
def test_sleepfm_tokenizer_shape(n_times, n_patches):
    tokenizer = _SleepFMTokenizer(patch_size=640, embed_dim=128)
    out = tokenizer(torch.randn(2, 3, n_times))
    assert out.shape == (2, 3, n_patches, 128)


def test_sleepfm_tokenizer_rejects_short_input():
    tokenizer = _SleepFMTokenizer()
    with pytest.raises(ValueError, match="at least one complete patch"):
        tokenizer(torch.randn(2, 3, 639))


def test_attention_pooling_shape():
    pooling = _SleepFMAttentionPooling(16, num_heads=4, drop_prob=0.0)
    out = pooling(torch.randn(6, 3, 16))
    assert out.shape == (6, 16)


def test_attention_pooling_ignores_masked_values():
    torch.manual_seed(7)
    pooling = _SleepFMAttentionPooling(16, num_heads=4, drop_prob=0.0).eval()
    x = torch.randn(2, 3, 16)
    mask = torch.tensor([[False, False, True], [False, True, True]])
    changed = x.clone()
    changed[mask] = 1e6
    with torch.no_grad():
        actual = pooling(x, mask)
        expected = pooling(changed, mask)
    torch.testing.assert_close(actual, expected)


def test_attention_pooling_rejects_all_masked_sample():
    pooling = _SleepFMAttentionPooling(16, num_heads=4)
    with pytest.raises(ValueError, match="at least one valid channel"):
        pooling(torch.randn(2, 3, 16), torch.ones(2, 3, dtype=torch.bool))


def test_compiled_attention_pooling_rejects_all_masked_sample():
    pooling = torch.compile(
        _SleepFMAttentionPooling(16, num_heads=4), backend="eager", dynamic=False
    )
    with pytest.raises(AssertionError, match="at least one valid channel"):
        pooling(torch.randn(2, 3, 16), torch.ones(2, 3, dtype=torch.bool))


def test_sleepfm_forward_and_features():
    model = _small_sleepfm().eval()
    x = torch.randn(2, 3, 640)
    with torch.no_grad():
        logits = model(x)
        result = model(x, return_features=True)
    assert logits.shape == (2, 4)
    assert result["features"].shape == (2, 16)
    assert result["cls_token"] is None


def test_sleepfm_masked_channels_do_not_change_output():
    model = _small_sleepfm().eval()
    x = torch.randn(2, 3, 640)
    mask = torch.tensor([[False, False, True], [False, True, False]])
    changed = x.masked_fill(mask.unsqueeze(-1), 1e6)
    with torch.no_grad():
        torch.testing.assert_close(model(x, mask), model(changed, mask))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"sfreq": 100.0}, "128 Hz"),
        ({"n_times": 639}, "complete patch"),
        ({"n_times": 3200, "max_seq_length": 4}, "max_seq_length"),
        ({"embed_dim": 15, "num_heads": 4}, "divisible"),
    ],
)
def test_sleepfm_constructor_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _small_sleepfm(**kwargs)


def test_sleepfm_reset_head():
    model = _small_sleepfm()
    model.reset_head(7)
    assert model(torch.randn(2, 3, 640)).shape == (2, 7)


def test_sleepfm_rejects_wrong_channel_mask_shape():
    model = _small_sleepfm()
    with pytest.raises(ValueError, match="channel_mask"):
        model(torch.randn(2, 3, 640), torch.zeros(2, 2, dtype=torch.bool))


@pytest.mark.parametrize(
    "channel_mask,error_type",
    [
        (torch.full((2, 3), 2), ValueError),
        (torch.zeros(2, 3, dtype=torch.complex64), TypeError),
    ],
)
def test_sleepfm_channel_mask_errors_use_public_argument_name(
    channel_mask, error_type
):
    model = _small_sleepfm()
    with pytest.raises(error_type, match="channel_mask") as exc_info:
        model(torch.randn(2, 3, 640), channel_mask)
    assert "key_padding_mask" not in str(exc_info.value)


def test_sleepfm_stager_output_shape():
    model = _small_stager().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 1280))
    assert out.shape == (2, 5, 2)


def test_sleepfm_stager_return_features():
    model = _small_stager().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 1280), return_features=True)
    assert out["features"].shape == (2, 2, 16)
    assert out["cls_token"] is None


def test_sleepfm_stager_reset_head():
    model = _small_stager()
    model.reset_head(7)
    assert model(torch.randn(2, 3, 1280)).shape == (2, 7, 2)


def test_sleepfm_stager_masked_channels_do_not_change_output():
    model = _small_stager().eval()
    x = torch.randn(2, 3, 1280)
    mask = torch.tensor([[False, False, True], [False, True, False]])
    changed = x.masked_fill(mask.unsqueeze(-1), 1e6)
    with torch.no_grad():
        torch.testing.assert_close(model(x, mask), model(changed, mask))


def test_sleepfm_stager_backpropagates_to_tokenizer():
    model = _small_stager()
    output = model(torch.randn(2, 3, 1280))
    output.mean().backward()
    assert model.patch_embedding.tokenizer[0].weight.grad is not None


def test_sleepfm_loads_reference_backbone_state_dict():
    source = _small_sleepfm()
    reference = {
        f"module.{key}": value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("final_layer.")
    }
    reference["module.positional_encoding.pe"] = reference.pop(
        "module.positional_encoding"
    )
    target = _small_sleepfm()
    target.load_pretrained_backbone(reference)
    for key, value in source.state_dict().items():
        if not key.startswith("final_layer."):
            torch.testing.assert_close(target.state_dict()[key], value)


def test_sleepfm_stager_loads_reference_head_state_dict():
    source = _small_stager()
    reference = {}
    for key, value in source.staging_head.state_dict().items():
        reference_key = key
        if key == "positional_encoding":
            reference_key = "positional_encoding.pe"
        reference[f"module.{reference_key}"] = value.clone()
    for key, value in source.final_layer.state_dict().items():
        reference[f"module.fc.{key}"] = value.clone()

    target = _small_stager()
    target.load_pretrained_staging_head(reference)
    for key, value in source.staging_head.state_dict().items():
        torch.testing.assert_close(target.staging_head.state_dict()[key], value)
    for key, value in source.final_layer.state_dict().items():
        torch.testing.assert_close(target.final_layer.state_dict()[key], value)


def test_sleepfm_stager_loads_tokenizer_from_reference_backbone():
    source = _small_stager()
    reference = {
        f"module.patch_embedding.{key}": value.clone()
        for key, value in source.patch_embedding.state_dict().items()
    }
    target = _small_stager()
    target.load_pretrained_backbone(reference)
    for key, value in source.patch_embedding.state_dict().items():
        torch.testing.assert_close(target.patch_embedding.state_dict()[key], value)


def test_sleepfm_reference_loader_rejects_unexpected_keys():
    model = _small_sleepfm()
    with pytest.raises(RuntimeError, match="unexpected"):
        model.load_pretrained_backbone({"module.unknown.weight": torch.ones(1)})


def test_sleepfm_models_are_publicly_registered():
    from braindecode.models import SleepFM as PublicSleepFM
    from braindecode.models import SleepFMStager as PublicSleepFMStager

    assert PublicSleepFM is SleepFM
    assert PublicSleepFMStager is SleepFMStager
    assert models_dict["SleepFM"] is SleepFM
    assert models_dict["SleepFMStager"] is SleepFMStager


@pytest.mark.parametrize("model_factory", [_small_sleepfm, _small_stager])
def test_sleepfm_models_compile(model_factory):
    model = model_factory().eval()
    x = torch.randn(1, 3, model.n_times)
    with torch.no_grad():
        expected = model(x)
        # The eager backend exercises Dynamo graph capture without the
        # platform-dependent, multi-minute Inductor code-generation cost.
        actual = torch.compile(model, backend="eager", dynamic=False)(x)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
