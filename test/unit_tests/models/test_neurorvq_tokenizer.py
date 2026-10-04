# Authors: Braindecode contributors
#
# License: BSD (3-clause)

import pytest
import torch

import braindecode.models.neurorvq_tokenizer as neurorvq_tokenizer
from braindecode.models import NeuroRVQTokenizer
from braindecode.models.neurorvq_tokenizer import _EMAVectorQuantizer


def _small_tokenizer():
    return NeuroRVQTokenizer(
        n_chans=3,
        n_times=400,
        sfreq=200,
        channel_names=("f3", "f4", "cz"),
        max_patches=4,
        out_chans=4,
        num_heads=4,
        encoder_depth=1,
        decoder_depth=1,
        n_code=16,
        code_dim=16,
        num_quantizers=2,
    )


def test_neurorvq_tokenizer_reconstructs_and_emits_discrete_codes():
    model = _small_tokenizer().eval()
    assert model.quantize_1.layers[0].embedding.initted.dtype == torch.float32
    signal = torch.randn(2, 3, 400)

    target, reconstruction = model(signal)
    codes = model.tokenize(signal)

    assert target.shape == reconstruction.shape == (2, 6, 200)
    assert codes.shape == (4, 2, 2, 6)
    assert codes.dtype == torch.long

    codebook_state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.startswith("quantize_")
    }
    model.tokenize(signal)
    for name, value in codebook_state.items():
        torch.testing.assert_close(model.state_dict()[name], value)

    model.train()
    _, reconstruction = model(signal)
    assert model.quantize_1.layers[0].cluster_size.sum() > 0
    reconstruction.square().mean().backward()
    assert model.encode_task_layer_1[0].weight.grad is not None


def test_neurorvq_tokenizer_loads_a_local_state_dict(tmp_path):
    model = _small_tokenizer()
    checkpoint = tmp_path / "tokenizer.pt"
    torch.save(model.state_dict(), checkpoint)

    loaded = _small_tokenizer().load_pretrained_weights(str(checkpoint))

    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)

def test_neurorvq_tokenizer_pretrained_loading_requires_channel_metadata():
    model = NeuroRVQTokenizer(
        n_chans=3,
        n_times=400,
        sfreq=200,
        channel_names=None,
        out_chans=4,
        num_heads=4,
        encoder_depth=1,
        decoder_depth=1,
        n_code=16,
        code_dim=16,
        num_quantizers=2,
    )

    with pytest.raises(ValueError, match="requires channel_names or chs_info"):
        model.load_pretrained_weights("checkpoint-is-not-read-before-validation.pt")


def test_neurorvq_tokenizer_initializes_cold_codebooks_once():
    model = _small_tokenizer().eval()
    signal = torch.randn(1, 3, 400)

    assert not model.quantize_1.layers[0].embedding.initted.item()
    codes = model.tokenize(signal)
    assert all(
        layer.embedding.initted.item()
        for scale in range(1, 5)
        for layer in getattr(model, f"quantize_{scale}").layers
    )
    state = {name: value.clone() for name, value in model.state_dict().items()}

    torch.testing.assert_close(model.tokenize(signal), codes)
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_ema_quantizer_matches_normalized_ema_update():
    quantizer = _EMAVectorQuantizer(
        n_codes=2, code_dim=2, decay=0.5, kmeans_init=False
    ).train()
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.eye(2))
        quantizer.embedding.initted.fill_(True)
    vectors = torch.tensor([[[[0.8, -0.6]], [[0.6, 0.8]]]])

    _, _, indices = quantizer(vectors)

    expected = torch.tensor([[0.9, 0.3], [-0.3, 0.9]])
    expected = torch.nn.functional.normalize(expected, dim=-1)
    assert indices.tolist() == [0, 1]
    torch.testing.assert_close(quantizer.embedding.weight, expected)
    torch.testing.assert_close(quantizer.cluster_size, torch.tensor([0.5, 0.5]))

def test_ema_quantizer_syncs_training_statistics_across_distributed_ranks(
    monkeypatch,
):
    quantizer = _EMAVectorQuantizer(
        n_codes=2, code_dim=2, decay=0.5, kmeans_init=False
    ).train()
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.eye(2))
        quantizer.embedding.initted.fill_(True)

    calls = []
    monkeypatch.setattr(neurorvq_tokenizer.distributed, "is_available", lambda: True)
    monkeypatch.setattr(neurorvq_tokenizer.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(value):
        calls.append(tuple(value.shape))
        value.mul_(2)

    monkeypatch.setattr(
        neurorvq_tokenizer.distributed, "all_reduce", fake_all_reduce
    )
    vectors = torch.tensor([[[[0.8, -0.6]], [[0.6, 0.8]]]])

    _, _, indices = quantizer(vectors)

    expected = torch.tensor([[0.9, 0.3], [-0.3, 0.9]])
    expected = torch.nn.functional.normalize(expected, dim=-1)
    assert indices.tolist() == [0, 1]
    assert calls == [(2,), (2, 2)]
    torch.testing.assert_close(quantizer.cluster_size, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(quantizer.embedding.weight, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sfreq": 250}, "200 Hz"),
        ({"n_times": 401}, "divisible by patch_size"),
        ({"channel_names": ("not-an-electrode",)}, "Unsupported NeuroRVQ channel"),
    ],
)
def test_neurorvq_tokenizer_rejects_incompatible_signal_metadata(kwargs, message):
    params = {
        "n_chans": 1,
        "n_times": 400,
        "sfreq": 200,
        "channel_names": ("f3",),
        "out_chans": 4,
        "num_heads": 4,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "n_code": 16,
        "code_dim": 16,
        "num_quantizers": 1,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        NeuroRVQTokenizer(**params)
