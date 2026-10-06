import pytest
import torch

from braindecode.models import EEGDINO
from braindecode.models.eegdino import EEGDINO_CONFIGS

# EEG-DINO-specific behaviour only. Instantiation, output shape, ``final_layer``,
# ``activation``/``drop_prob``, ``return_features`` and ``reset_head`` are already
# covered by the auto-parametrized suites (``test_integration.py`` and
# ``test_return_features.py``). The forward pass is checked for numerical equality
# against the original EEG-DINO implementation offline (not in CI).


def test_forward_and_presets():
    # Small (default) and Medium presets.
    model = EEGDINO(n_chans=16, n_outputs=4, n_times=1000)
    assert model(torch.randn(2, 16, 1000)).shape == (2, 4)

    medium = EEGDINO(n_chans=16, n_outputs=4, n_times=1000, **EEGDINO_CONFIGS["medium"])
    assert medium.emb_dim == 512
    assert len(medium.encoder_layers) == 16
    assert medium.encoder_layers[0].attn.nhead == 8  # matches the released checkpoint


def test_from_pretrained_local_roundtrip(tmp_path):
    pytest.importorskip("huggingface_hub")
    model = EEGDINO(n_chans=19, n_outputs=2, n_times=1000).eval()
    save_dir = tmp_path / "eegdino-small"
    model.save_pretrained(save_dir)
    reloaded = EEGDINO.from_pretrained(save_dir).eval()
    x = torch.randn(1, 19, 1000)
    assert torch.allclose(model(x), reloaded(x), atol=1e-5)
    assert EEGDINO.from_pretrained(save_dir, n_outputs=6)(x).shape == (1, 6)


def test_from_pretrained_takes_geometry_from_the_caller_not_the_config(tmp_path):
    """Passing ``chs_info`` (or ``n_chans``, or ``n_times``/``sfreq``) to
    ``from_pretrained`` must not collide with the geometry saved in config.json."""
    pytest.importorskip("huggingface_hub")
    model = EEGDINO(n_chans=19, n_times=400, sfreq=200, n_outputs=2, n_layer=1)
    model.save_pretrained(tmp_path)
    eight = [{"ch_name": f"E{i}"} for i in range(8)]
    loaded = EEGDINO.from_pretrained(tmp_path, chs_info=eight)
    assert loaded.n_chans == 8
    loaded = EEGDINO.from_pretrained(tmp_path, n_chans=8)
    assert loaded.n_chans == 8 and loaded._chs_info is None
    loaded = EEGDINO.from_pretrained(tmp_path, n_times=600)
    assert loaded.n_times == 600 and loaded.sfreq == 200
    loaded = EEGDINO.from_pretrained(tmp_path, sfreq=100)
    assert loaded.sfreq == 100 and loaded.n_times == 400


def test_attention_calls_qkv_module():
    # Adapters such as LoRA hook or replace ``attn.qkv``; the attention has to call
    # the module, not only read its weight, or they have no effect.
    model = EEGDINO(n_chans=19, n_outputs=2, n_times=1000).eval()
    for layer in model.encoder_layers:  # non-zero biases, as in the checkpoint
        torch.nn.init.normal_(layer.attn.q_bias)
        torch.nn.init.normal_(layer.attn.v_bias)
    x = torch.randn(2, 19, 1000)
    reference = model(x)

    calls = []
    handles = [
        layer.attn.qkv.register_forward_hook(lambda *_: calls.append(None))
        for layer in model.encoder_layers
    ]
    assert torch.equal(model(x), reference)
    assert len(calls) == len(model.encoder_layers)
    for handle in handles:
        handle.remove()

    for layer in model.encoder_layers:  # an adapter that changes the projection
        layer.attn.qkv.register_forward_hook(lambda _m, _i, out: 2 * out)
    assert not torch.allclose(model(x), reference)
