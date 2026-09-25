# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD (3-clause)
"""Convert, check and publish the two released DIVER-1 encoders on CPU.

Run from an editable Braindecode checkout with the ``hub`` extra installed::

    python scripts/convert_diver1_weights.py --weights-dir DIVER-1/weights \\
        --reference-dir DIVER-1 --output-dir /tmp/diver1-converted

``--weights-dir`` holds the release files of the official repository,
``ieeg_pretrained_weights.pt`` (0.1 s Tiny, iEEG) and
``i_eeg_pretrained_weights.pt`` (1 s Small, iEEG and EEG). Their GitHub LFS
objects are no longer served, so download them from the authors' Google Drive
folder linked in their README; the SHA-256 values below must match.
``--reference-dir`` is a clone of the official repository at ``REVISION``: its
model runs on the same input as the converted port, for intracranial and scalp
channels, and the encoder features must agree within ``TOLERANCE``.
``--variant`` converts one checkpoint only.

Add ``--push-to braindecode`` to upload each converted encoder with
:meth:`~braindecode.models.base.EEGModuleMixin.push_to_hub`, together with this
script, the MIT licence of the weights and a model card. The release holds
pretraining heads but no classification head, so the published files hold the
encoder alone and Braindecode initializes the head on load. Fine-tune it before
prediction.
"""

import argparse
import hashlib
import json
import shutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn

from braindecode.models import DIVER1

REPOSITORY = "https://github.com/DIVER-Project/DIVER-1"
REVISION = "25638eb38ef297b582ab79ae1c96260f57c155b3"
DRIVE = "https://drive.google.com/drive/folders/1Wmv36jifjE0Jj6noGOFFsqRzK-4xnj1b"
# Width, depth and patch size of each release file, as the official scripts load
# them (all at 500 Hz, with muP attention): scripts/finetune_neuroprobe.sh for the
# iEEG encoder, scripts/finetune_{faced,physionet,mentalarithmetic}.sh for the
# joint iEEG and EEG one.
VARIANTS = {
    "tiny": dict(
        filename="ieeg_pretrained_weights.pt",
        sha256="b812093779fb5cf1b76e18f8307df4136b67b008b6cd94c6be75fa286a87864c",
        repo="DIVER-1-0.1s-tiny",
        title="DIVER-1, 0.1 s Tiny (iEEG)",
        summary="the iEEG encoder DIVER-1-0.1s Tiny (patches of 0.1 s at 500 Hz, "
        "pretrained on intracranial recordings)",
        script="scripts/finetune_neuroprobe.sh",
        n_times=500,
        config=dict(patch_size=50, d_model=256, n_layers=12),
    ),
    "small": dict(
        filename="i_eeg_pretrained_weights.pt",
        sha256="dbfa48289989475a52719b1bcb868e62a82877120ac8272ca7dab772e407b891",
        repo="DIVER-1-1s-small",
        title="DIVER-1, 1 s Small (iEEG and EEG)",
        summary="the joint encoder DIVER-1-1s Small (patches of 1 s at 500 Hz, "
        "pretrained on intracranial and scalp recordings; described in versions 1 "
        "and 2 of the paper)",
        script="scripts/finetune_faced.sh",
        n_times=1000,
        config=dict(patch_size=500, d_model=512, n_layers=12),
    ),
}
COMMON = dict(sfreq=500.0, mup_attention=True)
TOLERANCE = 1e-4
RENAMES = {
    "token_manager.special_tokens.N_token.param": "patch_register",
    "token_manager.special_tokens.C_token.param": "chan_register",
    "token_manager.special_tokens.NC_token.param": "global_register",
    "embedding.2.embedding.embedding2.models.0.embedding.weight": "chan_emb.type_emb.weight",
    "embedding.2.embedding.embedding2.models.1.embedding.weight": "chan_emb.subtype_emb.weight",
}
PREFIXES = (
    ("var_attn_bias.emb.weight", "channel_bias.weight"),
    ("embedding.0.proj_in.", "patch_cnn.proj_in."),
    ("embedding.1.embedding.spectral_proj.", "spectral_emb.spectral_proj."),
    ("embedding.3.embedding.0.module.", "stcpe.down."),
    ("embedding.3.embedding.2.module.", "stcpe.up."),
    ("embedding.3.embedding.1.model.encoder.", "stcpe.encoder."),
    ("encoder.encoder.", "encoder."),
)
# Pretraining-only tensors: the mask token and the reconstruction heads.
DROPPED = ("mask_encoding", "heads.")
LICENSE = """MIT License

Copyright (c) 2026 DIVER Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rename(key):
    """Braindecode name of a released tensor, or ``None`` if it is dropped."""
    if key in RENAMES:
        return RENAMES[key]
    if key.startswith(DROPPED):
        return None
    for old, new in PREFIXES:
        key = key.replace(old, new)
    return key


def convert_weights(state, model):
    """Load the released encoder into ``model``; its head keeps its own init."""
    mapped = {rename(k): v.float() for k, v in state.items() if rename(k) is not None}
    head = {k: v for k, v in model.state_dict().items() if k.startswith("final_layer.")}
    model.load_state_dict({**mapped, **head}, strict=True)
    return mapped


def load_reference(reference_dir):
    """The official DIVER class, with stubs for its annotation-only imports."""
    for name, attrs in (("jaxtyping", ("Float", "Int", "Bool")), ("mup", ())):
        if name not in sys.modules:
            module = types.ModuleType(name)
            for attr in attrs:
                setattr(
                    module,
                    attr,
                    type(
                        "_Any",
                        (),
                        {"__class_getitem__": classmethod(lambda c, i: object)},
                    ),
                )
            sys.modules[name] = module
    # MuReadout is only used by the pretraining heads, which are dropped.
    sys.modules["mup"].MuReadout = type(
        "MuReadout",
        (nn.Linear,),
        {
            "__init__": lambda self, i, o, output_mult=1.0, **kw: nn.Linear.__init__(
                self, i, o
            )
        },
    )
    sys.path.insert(0, str(reference_dir))
    from models.diver import DIVER

    return DIVER


def placeholder_montage(n_chans, kind="seeg"):
    """Channels without positions (zeros, which DIVER1 reads as unknown)."""
    chs = mne.create_info([f"C{i}" for i in range(n_chans)], COMMON["sfreq"], kind)[
        "chs"
    ]
    for ch in chs:
        ch["loc"][:] = 0.0  # not NaN: config.json must stay valid JSON
    return chs


def check_against_reference(
    state, config, reference_dir, n_times, kind, n_chans=6, batch=3
):
    """Largest encoder-feature difference between the port and the reference."""
    reference = load_reference(reference_dir)(
        d_model=config["d_model"],
        e_layer=config["n_layers"],
        mup=True,
        patch_size=config["patch_size"],
    )
    reference.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    for module in reference.modules():
        # The reference keeps attention dropout active in eval mode.
        if hasattr(module, "attn_dropout_p"):
            module.attn_dropout_p = 0.0
    reference.eval()
    rng = np.random.default_rng(0)
    xyz_mm = torch.tensor(rng.uniform(-60, 60, size=(n_chans, 3)), dtype=torch.float32)
    chs = placeholder_montage(n_chans, kind)
    for ch, xyz in zip(chs, xyz_mm):
        ch["loc"][:3] = (xyz / 1e3).numpy()  # MNE metres; the reference takes mm
    model = DIVER1(
        chs_info=chs, n_times=n_times, n_outputs=2, **config, **COMMON
    ).eval()
    convert_weights(state, model)
    x = torch.randn(batch, n_chans, n_times)
    # The port reads modality and sub-modality from the channel kind.
    modality, subtype = ("iEEG", "depth") if kind == "seeg" else ("EEG", "Unknown")
    info = [
        {"xyz_id": xyz_mm, "modality": modality, "coord_subtype": [subtype] * n_chans}
    ] * batch
    with torch.no_grad():
        expected = reference(x, data_info_list=info, use_mask=False)[
            "token_manager_output"
        ]["org_x_position_features"]
    features = {}
    handle = model.final_layer.register_forward_hook(
        lambda m, args, out: features.setdefault("x", args[0])
    )
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    got = features["x"].reshape(expected.shape)
    return (got - expected).abs().max().item()


@contextmanager
def without_head(model):
    """Hide the untrained classifier so only encoder tensors get serialized."""
    head, model.final_layer = model.final_layer, nn.Identity()
    try:
        yield
    finally:
        model.final_layer = head


def model_card(variant, report):
    v = VARIANTS[variant]
    patch = v["config"]["patch_size"]
    return f"""---
library_name: braindecode
license: mit
pipeline_tag: feature-extraction
tags:
- braindecode
- DIVER-1
- ieeg
- seeg
- eeg
- foundation-model
- pytorch_model_hub_mixin
---

# {v["title"]}

The released encoder of [Han et al. (2025)](https://arxiv.org/abs/2512.19097),
{v["summary"]}, {report["n_params"] / 1e6:.1f}M parameters, converted to
[`braindecode.models.DIVER1`](https://braindecode.org/stable/generated/braindecode.models.DIVER1.html).

Source: `weights/{v["filename"]}` of [DIVER-Project/DIVER-1]({REPOSITORY}/tree/{REVISION}),
sha256 `{v["sha256"]}` (the GitHub LFS object is no longer served; the same file is in
the authors' [Google Drive folder]({DRIVE})). Conversion casts the bfloat16
tensors to float32, renames them and drops the pretraining-only mask token and
reconstruction heads; `convert_diver1_weights.py` in this repository reproduces
it. Encoder features match the official model to {report["max_abs_diff"]:.1e} in
float32 on CPU, for intracranial and scalp channels, with muP attention scaling
(`mup_attention=True`).

## Usage

Resample to 500 Hz and pass the electrode positions in `chs_info` (metres, as in
MNE). The channel kinds set the modality: SEEG, ECoG and DBS are intracranial,
EEG is scalp.

```python
import torch
from braindecode.models import DIVER1

model = DIVER1.from_pretrained(
    "braindecode/{v["repo"]}", chs_info=raw.info["chs"], n_times={v["n_times"]}, n_outputs=2
)
logits = model(torch.randn(8, len(raw.ch_names), {v["n_times"]}))
```

`n_times` must be a multiple of {patch}. The saved geometry is {report["n_chans"]}
channels and {report["n_times"]} samples; the encoder does not depend on it, so
pass your own montage, `n_times` and `n_outputs` when loading.

## Limitations

These files hold the encoder only; the release has no classification head.
Braindecode initializes the head on load, so it needs fine-tuning, as in the
paper's protocol (`{v["script"]}`). The check above covers
float32 CPU encoder features, not downstream accuracy, GPU kernels or mixed
precision.

## Citation

```bibtex
@article{{han2025diver,
    title={{DIVER-1: Scaling intracranial EEG foundation models for transferable representations}},
    author={{Han, Danny Dongyeop and Gwon, Yonghyeon and Lee, Ahhyun Lucy and Lee, Taeyang and Lee, Seong Jin and Choi, Jubin and Lee, Sebin and Bang, Jihyun and Lee, Seungju and Park, David Keetae and Yoo, Shinjae and Chung, Chun Kee and Cha, Jiook}},
    journal={{arXiv preprint arXiv:2512.19097}},
    year={{2025}}
}}
```

## License

The weights are released by the DIVER Project under the MIT licence (`LICENSE`).
Braindecode's code is BSD-3-Clause.
"""


def convert(variant, args):
    """Convert, check and write one release file; push it if asked."""
    v = VARIANTS[variant]
    path = args.weights_dir / v["filename"]
    if sha256(path) != v["sha256"]:
        raise SystemExit(f"{path} is not the released checkpoint (sha256 mismatch)")
    state = torch.load(path, map_location="cpu", weights_only=True)["module"]
    state = {k: t for k, t in state.items() if torch.is_tensor(t)}
    n_times = args.n_times or v["n_times"]
    max_abs_diff = max(
        check_against_reference(
            state, v["config"], args.reference_dir, n_times, kind, n_chans=args.n_chans
        )
        for kind in ("seeg", "eeg")
    )
    if max_abs_diff > TOLERANCE:
        raise SystemExit(f"encoder features differ by {max_abs_diff:.2e} > {TOLERANCE}")
    # Users pass their own chs_info when loading; the weights do not depend on it.
    model = DIVER1(
        chs_info=placeholder_montage(args.n_chans),
        n_times=n_times,
        n_outputs=args.n_outputs,
        **v["config"],
        **COMMON,
    ).eval()
    mapped = convert_weights(state, model)
    report = {
        "source": f"{REPOSITORY}@{REVISION}:weights/{v['filename']}",
        "sha256": v["sha256"],
        "tensors_loaded": len(mapped),
        "tensors_dropped": sorted(k for k in state if rename(k) is None),
        "n_params": sum(t.numel() for t in mapped.values()),
        "max_abs_diff": max_abs_diff,
        "n_chans": args.n_chans,
        "n_times": n_times,
        "torch_version": torch.__version__,
    }
    print(json.dumps(report, indent=2))
    destination = args.output_dir / v["repo"]
    destination.mkdir(parents=True, exist_ok=True)
    with without_head(model):
        model.save_pretrained(destination)
    # The Hub rejects NaN and Infinity, which Python's json writes by default.
    json.loads(
        (destination / "config.json").read_text(),
        parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError(f"{c} in config.json")
        ),
    )
    shutil.copyfile(Path(__file__).resolve(), destination / Path(__file__).name)
    (destination / "LICENSE").write_text(LICENSE)
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    # Written last: save_pretrained leaves a placeholder card behind.
    (destination / "README.md").write_text(model_card(variant, report))
    if args.push_to:
        from huggingface_hub import HfApi

        repo_id = f"{args.push_to}/{v['repo']}"
        with without_head(model):
            model.push_to_hub(repo_id)
        api = HfApi()
        for name in ("README.md", "LICENSE", "report.json", Path(__file__).name):
            api.upload_file(
                path_or_fileobj=str(destination / name),
                path_in_repo=name,
                repo_id=repo_id,
            )
        print(f"published https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append")
    parser.add_argument("--n-chans", type=int, default=6)
    parser.add_argument("--n-times", type=int, help="default: 1 s (tiny), 2 s (small)")
    parser.add_argument("--n-outputs", type=int, default=2)
    parser.add_argument("--push-to", help="Hub namespace to publish the models to")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for variant in args.variant or VARIANTS:
        torch.manual_seed(0)
        convert(variant, args)


if __name__ == "__main__":
    main()
