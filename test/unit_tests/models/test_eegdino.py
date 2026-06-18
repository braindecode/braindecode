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
