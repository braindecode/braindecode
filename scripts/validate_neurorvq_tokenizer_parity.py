"""Compare the Braindecode NeuroRVQ tokenizer with its released reference.

This optional validation downloads or reads the 304 MB released checkpoint,
then checks inference tokens, reconstruction, one training update, EMA buffers,
and every trainable parameter and input gradient on a deterministic CPU input.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from braindecode.models import NeuroRVQTokenizer

_REPO_ID = "ntinosbarmpas/NeuroRVQ"
_REVISION = "d944b87f44ae0ba2923b2f10d0518f23f6803b76"
_FILENAME = "pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt"
_CHANNELS = ("f3", "f4", "cz")
_PATCH_SIZE = 200
_N_PATCHES = 2


def _checkpoint_path(path: Path | None) -> Path:
    if path is not None:
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Pass --checkpoint or install huggingface_hub to download the "
            "released weights."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=_REPO_ID,
            filename=_FILENAME,
            revision=_REVISION,
        )
    )


def _reference_components(source_root: Path):
    eeg_source = source_root / "NeuroRVQ_EEG"
    if not (eeg_source / "NeuroRVQ.py").is_file():
        raise SystemExit(
            "--neurorvq-source must point to the root of the official NeuroRVQ "
            "repository."
        )
    sys.path.insert(0, str(eeg_source))
    sys.path.insert(0, str(source_root))
    from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import (
        ch_names_global,
        create_embedding_ix,
    )
    from NeuroRVQ import NeuroRVQTokenizer as ReferenceTokenizer
    from NeuroRVQ_modules import get_encoder_decoder_params

    return (
        ReferenceTokenizer,
        get_encoder_decoder_params,
        ch_names_global,
        create_embedding_ix,
    )


def _load_pair(checkpoint: Path, source_root: Path):
    (
        ReferenceTokenizer,
        get_encoder_decoder_params,
        ch_names_global,
        create_embedding_ix,
    ) = _reference_components(source_root)
    config = {
        "patch_size": 200,
        "n_patches": 256,
        "n_global_electrodes": len(ch_names_global),
        "embed_dim": 200,
        "num_heads_tokenizer": 10,
        "mlp_ratio_tokenizer": 4,
        "qkv_bias_tokenizer": True,
        "drop_rate_tokenizer": 0.0,
        "attn_drop_rate_tokenizer": 0.0,
        "drop_path_rate_tokenizer": 0.0,
        "init_values_tokenizer": 0.0,
        "init_scale_tokenizer": 0.001,
        "in_chans_encoder": 1,
        "depth_encoder": 12,
        "num_classes": 0,
        "out_chans_encoder": 8,
        "depth_decoder": 3,
        "code_dim": 128,
        "n_code": 8192,
        "decoder_out_dim": 200,
    }
    encoder_config, decoder_config = get_encoder_decoder_params(config)
    reference = ReferenceTokenizer(
        encoder_config,
        decoder_config,
        n_code=config["n_code"],
        code_dim=config["code_dim"],
        decoder_out_dim=config["decoder_out_dim"],
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference.load_state_dict(state, strict=True)
    del state

    port = NeuroRVQTokenizer(
        n_chans=len(_CHANNELS),
        n_times=_PATCH_SIZE * _N_PATCHES,
        sfreq=200,
        channel_names=_CHANNELS,
    )
    port.load_pretrained_weights(str(checkpoint))
    names = np.asarray([name.encode() for name in _CHANNELS])
    temporal, spatial = create_embedding_ix(
        _N_PATCHES, config["n_patches"], names, ch_names_global
    )
    return reference, port, temporal, spatial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neurorvq-source",
        type=Path,
        required=True,
        help="Path to a local clone of github.com/KonstantinosBarmpas/NeuroRVQ",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local tokenizer checkpoint; omitted means download the pinned HF revision.",
    )
    args = parser.parse_args()
    checkpoint = _checkpoint_path(args.checkpoint)
    torch.set_num_threads(1)
    signal = torch.from_numpy(
        np.random.default_rng(43)
        .standard_normal((2, len(_CHANNELS), _PATCH_SIZE * _N_PATCHES))
        .astype(np.float32)
    )

    reference, port, temporal, spatial = _load_pair(checkpoint, args.neurorvq_source)
    reference.eval()
    port.eval()
    with torch.no_grad():
        reference_target, reference_reconstruction = reference(
            signal, temporal, spatial
        )
        port_target, port_reconstruction = port(signal)
        reference_codes = reference.get_codebook_indices(
            signal.reshape(2, len(_CHANNELS), _N_PATCHES, _PATCH_SIZE),
            temporal,
            spatial,
        )
        port_codes = port.tokenize(signal)
    eval_target_error = (reference_target - port_target).abs().max().item()
    eval_reconstruction_error = (
        (reference_reconstruction - port_reconstruction).abs().max().item()
    )
    codes_match = torch.equal(reference_codes, port_codes)
    del reference, port
    gc.collect()

    reference, port, temporal, spatial = _load_pair(checkpoint, args.neurorvq_source)
    reference.train()
    port.train()
    signal_reference = signal.detach().clone().requires_grad_()
    signal_port = signal.detach().clone().requires_grad_()
    reference_target, reference_reconstruction = reference(
        signal_reference, temporal, spatial
    )
    port_target, port_reconstruction = port(signal_port)
    train_target_error = (reference_target - port_target).abs().max().item()
    train_reconstruction_error = (
        (reference_reconstruction - port_reconstruction).abs().max().item()
    )
    (
        reference_target.square().mean() + reference_reconstruction.square().mean()
    ).backward()
    (port_target.square().mean() + port_reconstruction.square().mean()).backward()
    reference_parameters = dict(reference.named_parameters())
    port_parameters = dict(port.named_parameters())
    assert reference_parameters.keys() == port_parameters.keys()
    gradient_errors = {}
    for name, reference_parameter in reference_parameters.items():
        port_parameter = port_parameters[name]
        assert reference_parameter.requires_grad == port_parameter.requires_grad
        if reference_parameter.grad is None or port_parameter.grad is None:
            assert reference_parameter.grad is port_parameter.grad is None, name
            continue
        gradient_errors[name] = (
            (reference_parameter.grad - port_parameter.grad).abs().max().item()
        )
    input_gradient_error = (signal_reference.grad - signal_port.grad).abs().max().item()
    gradient_error = max(gradient_errors.values(), default=0.0)
    reference_state = reference.state_dict()
    port_state = port.state_dict()
    quantizer_keys = [key for key in reference_state if key.startswith("quantize_")]
    ema_state_matches = all(
        reference_state[key].dtype == port_state[key].dtype
        and torch.equal(reference_state[key], port_state[key])
        for key in quantizer_keys
    )

    results = {
        "checkpoint_revision": _REVISION,
        "input_shape": list(signal.shape),
        "eval_target_max_abs": eval_target_error,
        "eval_reconstruction_max_abs": eval_reconstruction_error,
        "eval_codes_equal": codes_match,
        "train_target_max_abs": train_target_error,
        "train_reconstruction_max_abs": train_reconstruction_error,
        "trainable_parameter_gradient_max_abs": gradient_error,
        "trainable_parameter_gradient_count": len(gradient_errors),
        "input_gradient_max_abs": input_gradient_error,
        "ema_state_equal_after_one_step": ema_state_matches,
    }
    print(json.dumps(results, indent=2))

    assert codes_match, "Discrete token indices differ from the reference."
    assert ema_state_matches, "EMA codebook state differs from the reference."
    assert (
        max(
            eval_target_error,
            eval_reconstruction_error,
            train_target_error,
            train_reconstruction_error,
            gradient_error,
            input_gradient_error,
        )
        < 1e-6
    ), "Numerical parity exceeded the 1e-6 tolerance."


if __name__ == "__main__":
    main()
