# Authors: Braindecode contributors
#
# License: BSD (3-clause)

import pytest
import torch

from braindecode.models import NeuroRVQTokenizer


def test_neurorvq_tokenizer_reconstructs_and_emits_discrete_codes():
    model = NeuroRVQTokenizer(
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
    ).eval()
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
