"""Convert OpenTSLab/BrainOmni upstream checkpoints to braindecode format.

This script downloads, remaps, and validates BrainOmni/BrainTokenizer pretrained
weights from the HuggingFace Hub (``OpenTSLab/BrainOmni``) and saves them in the
braindecode hub format (config.json + safetensors).

Upstream checkpoint format
--------------------------
Plain ``torch.save(state_dict)`` files, loadable with
``torch.load(path, map_location="cpu", weights_only=True)``.  Upstream loads
with ``strict=False``.

Key remapping
-------------
- **BrainTokenizer**: upstream ``decoder.*`` -> braindecode ``final_layer.*``.
  Everything else (``sensor_embed.*``, ``encoder.*``, ``quantizer.*``) is 1:1.
- **BrainOmni**: ``tokenizer.decoder.*`` -> ``tokenizer.final_layer.*``.
  DROP ``mask_token`` and any ``predict_head.*`` (pretraining-only).
  Pass through ``tokenizer.*``, ``projection.*``, ``blocks.*``.
  The braindecode head ``final_layer.*`` is FRESH (not in checkpoint) — this
  causes expected "missing key" entries on load.

Coverage assertions
-------------------
After loading, ``unexpected_keys`` MUST be empty (every remapped upstream key
corresponds to a real braindecode parameter/buffer).  ``missing_keys`` is
allowed ONLY for the BrainOmni fresh head (``final_layer.*``).  Violations
raise ``AssertionError`` — a real signal of architecture divergence.

Numerical parity (optional)
----------------------------
Parity against the upstream ``encode()`` requires deepspeed/einx (not
installable here) and must be done in the upstream Linux environment.

**To generate a reference output** (run in the upstream env)::

    import torch
    from brainomni.model import BrainOmni as UpstreamBrainOmni
    from factory.model import build_model

    # Load upstream model with the relevant config
    model = build_model(cfg)   # upstream factory
    model.load_state_dict(torch.load("BrainOmni.pt", weights_only=True), strict=False)
    model.eval()

    example = torch.load("assets/example_data.pt", weights_only=True)
    x = example["x"].unsqueeze(0)            # (1, C, T)
    pos = example["pos"].unsqueeze(0)        # (1, C, 6)
    sensor_type = example["sensor_type"].unsqueeze(0).long()  # (1, C)
    with torch.no_grad():
        ref = model.encode(x, pos, sensor_type)
    torch.save(ref, "reference_encode_output.pt")

Then pass ``--reference-output reference_encode_output.pt`` to this script.

Override note: braindecode ``BrainOmni.encode(x)`` uses self.pos/sensor_type
buffers.  To match the upstream call signature, set::

    model.tokenizer.pos = example["pos"][0]        # (C, 6)
    model.tokenizer.sensor_type = example["sensor_type"][0].long()  # (C,)

before calling ``encode(x)``.

Usage
-----
::

    # Download, convert, and validate tokenizer:
    python scripts/convert_brainomni_weights.py --variant tokenizer --out /tmp/bt_out

    # Download, convert, and validate BrainOmni tiny:
    python scripts/convert_brainomni_weights.py --variant tiny --out /tmp/bo_out

    # With local files (skip auto-discovery):
    python scripts/convert_brainomni_weights.py \\
        --variant tokenizer --ckpt /path/BrainTokenizer.pt --cfg /path/model_cfg.json \\
        --out /tmp/bt_out

    # Numerical parity (after generating reference):
    python scripts/convert_brainomni_weights.py \\
        --variant tiny --out /tmp/bo_out \\
        --reference-output reference_encode_output.pt

    # Offline self-test (always works without network):
    python scripts/convert_brainomni_weights.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration mapping
# ---------------------------------------------------------------------------


def _cfg_to_kwargs(cfg: dict[str, Any], *, is_omni: bool) -> dict[str, Any]:
    """Map upstream ``model_cfg.json`` keys to braindecode constructor kwargs.

    Parameters
    ----------
    cfg : dict
        Parsed ``model_cfg.json`` as a Python dict.
    is_omni : bool
        ``True`` for BrainOmni, ``False`` for BrainTokenizer.

    Returns
    -------
    dict
        Keyword arguments suitable for the braindecode constructor.
    """
    # Keys that pass through unchanged
    pass_through = [
        "window_length",
        "n_filters",
        "ratios",
        "kernel_size",
        "last_kernel_size",
        "n_neuro",
        "codebook_dim",
        "codebook_size",
        "num_quantizers",
        "rotation_trick",
        "quantize_optimize_method",
    ]
    kwargs: dict[str, Any] = {}
    for k in pass_through:
        if k in cfg:
            kwargs[k] = cfg[k]

    # Renamed keys (shared by both)
    kwargs["emb_dim"] = cfg["n_dim"]
    kwargs["tokenizer_num_heads"] = cfg["n_head"]
    kwargs["drop_prob"] = cfg["dropout"]  # tokenizer drop_prob

    if is_omni:
        # BrainOmni-specific pass-through
        if "overlap_ratio" in cfg:
            kwargs["overlap_ratio"] = cfg["overlap_ratio"]
        # BrainOmni-specific renames
        kwargs["lm_dim"] = cfg["lm_dim"]
        kwargs["num_heads"] = cfg["lm_head"]
        kwargs["depth"] = cfg["lm_depth"]
        # Warn when tokenizer dropout and LM dropout differ.  braindecode
        # BrainOmni uses a single drop_prob (set to lm_dropout) for both
        # components.  Dropout has no learned parameters, so this does NOT
        # affect the loaded weights, but the tokenizer's stored drop_prob will
        # differ from the upstream tokenizer dropout value.
        if cfg.get("dropout") != cfg.get("lm_dropout"):
            warnings.warn(
                f"drop_prob mismatch: upstream tokenizer dropout={cfg.get('dropout')!r} "
                f"!= lm_dropout={cfg.get('lm_dropout')!r}.  "
                "braindecode BrainOmni uses a single drop_prob parameter (set to "
                "lm_dropout).  This does NOT affect loaded weights — dropout layers "
                "have no trainable parameters — but the tokenizer's stored drop_prob "
                "will differ from the upstream tokenizer dropout value.",
                UserWarning,
                stacklevel=2,
            )
        kwargs["drop_prob"] = cfg["lm_dropout"]  # overwrite with LM dropout
        # DROP: mask_ratio, num_quantizers_used (pretraining-only)

    return kwargs


# ---------------------------------------------------------------------------
# State-dict key remapping
# ---------------------------------------------------------------------------


def _remap_state_dict(
    sd: dict[str, torch.Tensor],
    *,
    is_omni: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Remap upstream state-dict keys to braindecode conventions.

    Parameters
    ----------
    sd : dict
        Upstream state dict.
    is_omni : bool
        ``True`` for BrainOmni, ``False`` for BrainTokenizer.

    Returns
    -------
    new_sd : dict
        Remapped state dict (ready for ``model.load_state_dict``).
    dropped_keys : list of str
        Keys that were dropped (pretraining-only tensors).
    """
    new_sd: dict[str, torch.Tensor] = {}
    dropped_keys: list[str] = []

    for key, val in sd.items():
        if is_omni:
            # Drop pretraining-only tensors
            if key == "mask_token" or key.startswith("predict_head."):
                dropped_keys.append(key)
                continue
            # Rename tokenizer decoder
            if key.startswith("tokenizer.decoder."):
                new_key = "tokenizer.final_layer." + key[len("tokenizer.decoder."):]
                new_sd[new_key] = val
            else:
                # Pass through: tokenizer.*, projection.*, blocks.*
                new_sd[key] = val
        else:
            # BrainTokenizer: rename decoder -> final_layer
            if key.startswith("decoder."):
                new_key = "final_layer." + key[len("decoder."):]
                new_sd[new_key] = val
            else:
                new_sd[key] = val

    return new_sd, dropped_keys


# ---------------------------------------------------------------------------
# Coverage assertion
# ---------------------------------------------------------------------------


def _assert_coverage(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    is_omni: bool,
    label: str = "",
) -> None:
    """Assert exhaustive key coverage after ``load_state_dict``.

    Parameters
    ----------
    missing_keys : list of str
        Keys present in the braindecode model but absent in the checkpoint.
    unexpected_keys : list of str
        Keys present in the checkpoint but absent in the braindecode model.
    is_omni : bool
        ``True`` for BrainOmni (head keys are expected missing).
    label : str
        Descriptive label for error messages.
    """
    if unexpected_keys:
        msg = (
            f"[{label}] COVERAGE FAILURE: {len(unexpected_keys)} unexpected key(s) — "
            "checkpoint contains weights that have no matching braindecode parameter.\n"
            "This means remap logic or architecture diverged.  Keys:\n"
        )
        for k in sorted(unexpected_keys):
            msg += f"  {k}\n"
        raise AssertionError(msg)

    # Geometry buffers (pos, sensor_type) are braindecode-specific: they are
    # computed from chs_info at model init and never present in upstream
    # checkpoints.  They are always valid missing keys for both model types.
    _geometry_buffers = frozenset(
        [
            "pos",
            "sensor_type",
            "tokenizer.pos",
            "tokenizer.sensor_type",
        ]
    )

    if is_omni:
        bad_missing = [
            k
            for k in missing_keys
            if not k.startswith("final_layer.") and k not in _geometry_buffers
        ]
        if bad_missing:
            msg = (
                f"[{label}] COVERAGE FAILURE: {len(bad_missing)} missing key(s) not in "
                "the classification head (final_layer.*) or geometry buffers.  Keys:\n"
            )
            for k in sorted(bad_missing):
                msg += f"  {k}\n"
            raise AssertionError(msg)
    else:
        bad_missing = [k for k in missing_keys if k not in _geometry_buffers]
        if bad_missing:
            msg = (
                f"[{label}] COVERAGE FAILURE: {len(bad_missing)} missing key(s) — "
                "BrainTokenizer should have no unexpected missing keys after "
                "conversion (only geometry buffers pos/sensor_type are allowed).  "
                "Keys:\n"
            )
            for k in sorted(bad_missing):
                msg += f"  {k}\n"
            raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Minimal chs_info helper (avoids MNE dependency for scripting)
# ---------------------------------------------------------------------------


def _make_eeg_chs_info(n_chans: int) -> list[dict]:
    """Build a minimal list of EEG channel dicts for braindecode constructors.

    Uses a roughly spherical arrangement of sensor positions to populate the
    ``loc`` field required by ``_geometry_from_chs_info``.
    """
    chs = []
    for i in range(n_chans):
        angle = 2 * np.pi * i / n_chans
        x = np.cos(angle) * 0.09
        y = np.sin(angle) * 0.09
        z = 0.12
        loc = np.zeros(12, dtype=np.float64)
        loc[:3] = [x, y, z]
        chs.append(
            {
                "ch_name": f"EEG{i+1:03d}",
                "kind": "eeg",
                "loc": loc,
            }
        )
    return chs


# ---------------------------------------------------------------------------
# HuggingFace file discovery
# ---------------------------------------------------------------------------


def _discover_hf_files(variant: str) -> tuple[str, str]:
    """Discover cfg and checkpoint paths on ``OpenTSLab/BrainOmni`` HF repo.

    Parameters
    ----------
    variant : str
        One of ``"tiny"``, ``"base"``, ``"tokenizer"``.

    Returns
    -------
    cfg_path : str
        Remote path to ``model_cfg.json``.
    ckpt_path : str
        Remote path to ``.pt`` checkpoint.
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub not installed.  Run:\n"
            "    pip install huggingface_hub\n"
            "Or download files manually and pass --ckpt / --cfg."
        ) from exc

    repo_id = "OpenTSLab/BrainOmni"
    try:
        all_files = list(list_repo_files(repo_id))
    except Exception as exc:
        raise RuntimeError(
            f"Could not list files from {repo_id!r}.  Network unavailable?\n"
            f"Error: {exc}\n"
            "Download files manually and pass --ckpt /path/to/file.pt "
            "--cfg /path/to/model_cfg.json"
        ) from exc

    cfg_files = [
        f for f in all_files
        if f.endswith("model_cfg.json") and variant.lower() in f.lower()
    ]
    pt_files = [
        f for f in all_files
        if f.endswith(".pt") and variant.lower() in f.lower()
    ]

    if not cfg_files:
        raise FileNotFoundError(
            f"No model_cfg.json found for variant={variant!r} in {repo_id}.\n"
            f"Available files: {all_files}"
        )
    if not pt_files:
        raise FileNotFoundError(
            f"No .pt file found for variant={variant!r} in {repo_id}.\n"
            f"Available files: {all_files}"
        )

    return cfg_files[0], pt_files[0]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _hf_download(repo_id: str, filename: str) -> str:
    """Download a file from HuggingFace Hub and return local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub not installed.  Run: pip install huggingface_hub"
        ) from exc
    return hf_hub_download(repo_id=repo_id, filename=filename)


# ---------------------------------------------------------------------------
# Build braindecode model from cfg
# ---------------------------------------------------------------------------


def _build_model(
    cfg: dict[str, Any],
    *,
    is_omni: bool,
    n_outputs: int = 2,
    chs_info: list[dict] | None = None,
) -> Any:
    """Instantiate a braindecode BrainTokenizer or BrainOmni from upstream cfg.

    Parameters
    ----------
    cfg : dict
        Upstream ``model_cfg.json`` as a Python dict.
    is_omni : bool
        ``True`` for BrainOmni, ``False`` for BrainTokenizer.
    n_outputs : int
        Number of classification outputs (BrainOmni only).
    chs_info : list of dict or None
        Channel info for geometry.  If ``None``, a synthetic 4-channel EEG
        arrangement is used.
    """
    from braindecode.models.brainomni import BrainOmni, BrainTokenizer

    if chs_info is None:
        chs_info = _make_eeg_chs_info(4)

    kwargs = _cfg_to_kwargs(cfg, is_omni=is_omni)

    if is_omni:
        return BrainOmni(
            chs_info=chs_info,
            n_times=cfg.get("window_length", 512),
            n_outputs=n_outputs,
            **kwargs,
        )
    else:
        return BrainTokenizer(
            chs_info=chs_info,
            n_times=cfg.get("window_length", 512),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------


def _validate_round_trip(model: Any, out_dir: Path, label: str) -> None:
    """Save and reload model; assert state-dicts match."""
    model.save_pretrained(str(out_dir))
    reloaded = type(model).from_pretrained(str(out_dir))

    sd_orig = model.state_dict()
    sd_rt = reloaded.state_dict()

    keys_orig = set(sd_orig.keys())
    keys_rt = set(sd_rt.keys())
    assert keys_orig == keys_rt, (
        f"[{label}] Round-trip key mismatch: "
        f"orig={keys_orig - keys_rt} extra={keys_rt - keys_orig}"
    )
    for k in sd_orig:
        if not torch.allclose(sd_orig[k].float(), sd_rt[k].float(), atol=1e-6):
            raise AssertionError(
                f"[{label}] Round-trip tensor mismatch at key {k!r}"
            )
    print(f"[{label}] Round-trip OK — {len(sd_orig)} tensors verified.")


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------


_DEFAULT_EXAMPLE_DATA = (
    Path(__file__).parent.parent / "BrainOmni" / "assets" / "example_data.pt"
)


def _parity_check(
    model: Any,
    reference_path: str,
    *,
    label: str,
    example_data_path: Path | None = None,
) -> None:
    """Numerical parity check against upstream ``encode()`` output.

    Parameters
    ----------
    model : BrainOmni
        Already-converted braindecode BrainOmni model.
    reference_path : str
        Path to ``reference_encode_output.pt`` saved in the upstream env.
    label : str
        Descriptive label for messages.
    example_data_path : Path or None
        Path to ``example_data.pt``.  Defaults to the bundled asset at
        ``BrainOmni/assets/example_data.pt`` relative to the repo root.
        Pass ``--example-data`` on the CLI to override.
    """
    if example_data_path is None:
        example_path = _DEFAULT_EXAMPLE_DATA
    else:
        example_path = Path(example_data_path)

    if not example_path.exists():
        raise FileNotFoundError(
            f"[{label}] example_data.pt not found at {example_path}.\n"
            "Pass the correct path via --example-data /path/to/example_data.pt."
        )

    example = torch.load(str(example_path), map_location="cpu", weights_only=True)
    ref = torch.load(reference_path, map_location="cpu", weights_only=True)

    # Override model's geometry buffers with example's pos/sensor_type
    # (braindecode derives geometry from chs_info at init; the example provides
    # them directly).
    model.tokenizer.pos = example["pos"][0].float()         # (C, 6)
    model.tokenizer.sensor_type = example["sensor_type"][0].long()  # (C,)

    x = example["x"].unsqueeze(0)  # (1, C, T)
    model.eval()
    with torch.no_grad():
        out = model.encode(x)

    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"[{label}] Parity max_abs_diff = {max_diff:.2e}")
    assert max_diff <= 1e-5, (
        f"[{label}] Parity FAILED: max_abs_diff={max_diff:.2e} > 1e-5"
    )
    print(f"[{label}] Parity PASSED.")


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------


def convert(
    *,
    variant: str,
    ckpt_path: str | None = None,
    cfg_path: str | None = None,
    out_dir: str,
    reference_output: str | None = None,
    example_data: str | None = None,
) -> None:
    """Download, remap, validate, and save a BrainOmni checkpoint.

    Parameters
    ----------
    variant : str
        ``"tiny"``, ``"base"``, or ``"tokenizer"``.  Always required — when
        using local ``--ckpt``/``--cfg`` it disambiguates BrainTokenizer from
        BrainOmni (BrainTokenizer has no learned dropout parameters so they are
        architecturally indistinguishable from ``ckpt_path`` alone).
    ckpt_path : str or None
        Local path to ``.pt`` checkpoint (bypasses HF discovery).
    cfg_path : str or None
        Local path to ``model_cfg.json`` (bypasses HF discovery).
    out_dir : str
        Output directory for the converted checkpoint.
    reference_output : str or None
        Path to upstream reference output for parity check (optional).
    example_data : str or None
        Path to ``example_data.pt`` for parity check (optional, overrides
        default bundled asset).
    """
    # is_omni is always derived from variant, never from ckpt_path presence.
    is_omni = variant in ("tiny", "base")

    # Determine source files
    if ckpt_path is None or cfg_path is None:
        print(f"Discovering files for variant={variant!r} on OpenTSLab/BrainOmni...")
        remote_cfg, remote_ckpt = _discover_hf_files(variant)
        print(f"  cfg : {remote_cfg}")
        print(f"  ckpt: {remote_ckpt}")

        repo_id = "OpenTSLab/BrainOmni"
        print("Downloading...")
        cfg_path = _hf_download(repo_id, remote_cfg)
        ckpt_path = _hf_download(repo_id, remote_ckpt)

    print(f"\nLoading cfg from: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    print(f"Loading upstream checkpoint from: {ckpt_path}")
    upstream_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    n_upstream = len(upstream_sd)
    print(f"  Upstream keys: {n_upstream}")

    # Remap
    label = f"BrainOmni/{variant}" if variant else "converted"
    new_sd, dropped_keys = _remap_state_dict(upstream_sd, is_omni=is_omni)
    print(f"  Dropped (pretraining-only): {len(dropped_keys)}")
    for k in dropped_keys:
        print(f"    {k}")

    # Build model
    print("\nBuilding braindecode model...")
    model = _build_model(cfg, is_omni=is_omni)
    model.eval()

    # Load with strict=False
    result = model.load_state_dict(new_sd, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    print(f"\n=== Coverage summary [{label}] ===")
    print(f"  Upstream keys total    : {n_upstream}")
    print(f"  Dropped (pretraining)  : {len(dropped_keys)}")
    print(f"  Remapped keys loaded   : {len(new_sd)}")
    print(f"  Missing keys (model)   : {len(missing_keys)}")
    for k in sorted(missing_keys):
        print(f"    {k}")
    print(f"  Unexpected keys (ckpt) : {len(unexpected_keys)}")
    for k in sorted(unexpected_keys):
        print(f"    {k}")

    _assert_coverage(missing_keys, unexpected_keys, is_omni=is_omni, label=label)
    print(f"\n[{label}] Coverage assertion PASSED.")

    # Save
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_path} ...")
    _validate_round_trip(model, out_path, label)

    # Optional parity
    if reference_output is not None:
        print(f"\nRunning parity check against: {reference_output}")
        _parity_check(
            model,
            reference_output,
            label=label,
            example_data_path=Path(example_data) if example_data else None,
        )
    else:
        print(
            "\nParity check skipped (no --reference-output provided).\n"
            "To generate a reference in the upstream env, see the module docstring."
        )

    print(f"\nConversion complete.  Saved to: {out_path}")


# ---------------------------------------------------------------------------
# Self-test (offline, no network)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Offline self-test validating remap logic without network access."""
    import torch

    from braindecode.models.brainomni import BrainOmni, BrainTokenizer

    print("Running SELF-TEST...")

    # Tiny test config
    n_chans = 4
    chs_info = _make_eeg_chs_info(n_chans)
    base_kwargs = dict(
        chs_info=chs_info,
        n_times=512,
        emb_dim=16,
        n_neuro=3,
        n_filters=8,
        codebook_dim=16,
        codebook_size=32,
        num_quantizers=2,
        tokenizer_num_heads=4,
        window_length=512,
        ratios=[4, 2],
        kernel_size=3,
        last_kernel_size=3,
        rotation_trick=False,
        quantize_optimize_method="ema",
        drop_prob=0.0,
    )

    # ---------- BrainTokenizer self-test ----------
    print("\n[SELF-TEST] BrainTokenizer...")
    bt = BrainTokenizer(**base_kwargs)
    real_sd = bt.state_dict()

    # Synthesize upstream-style SD: rename final_layer -> decoder
    upstream_bt_sd: dict[str, torch.Tensor] = {}
    for k, v in real_sd.items():
        if k.startswith("final_layer."):
            upstream_bt_sd["decoder." + k[len("final_layer."):]] = v
        else:
            upstream_bt_sd[k] = v

    # Remap back
    remapped_bt, dropped_bt = _remap_state_dict(upstream_bt_sd, is_omni=False)
    assert len(dropped_bt) == 0, f"BrainTokenizer self-test: unexpected drops: {dropped_bt}"

    fresh_bt = BrainTokenizer(**base_kwargs)
    result_bt = fresh_bt.load_state_dict(remapped_bt, strict=False)
    _assert_coverage(
        list(result_bt.missing_keys),
        list(result_bt.unexpected_keys),
        is_omni=False,
        label="BrainTokenizer-self-test",
    )
    # Verify all tensors actually match
    loaded_sd = fresh_bt.state_dict()
    for k in real_sd:
        assert torch.allclose(real_sd[k].float(), loaded_sd[k].float()), (
            f"BrainTokenizer self-test: tensor mismatch at {k!r}"
        )
    print("[SELF-TEST] BrainTokenizer: PASSED")

    # ---------- BrainOmni self-test ----------
    print("\n[SELF-TEST] BrainOmni...")
    omni_kwargs = dict(
        **base_kwargs,
        lm_dim=16,
        num_heads=4,
        depth=2,
        n_outputs=2,
    )
    bo = BrainOmni(**omni_kwargs)
    real_omni_sd = bo.state_dict()

    # Synthesize upstream-style SD:
    # 1. Rename tokenizer.final_layer -> tokenizer.decoder
    # 2. Remove top-level final_layer.* (fresh head, not in upstream)
    # 3. Add fake mask_token and predict_head.weight/bias
    upstream_omni_sd: dict[str, torch.Tensor] = {}
    for k, v in real_omni_sd.items():
        if k.startswith("tokenizer.final_layer."):
            upstream_omni_sd["tokenizer.decoder." + k[len("tokenizer.final_layer."):]] = v
        elif k.startswith("final_layer."):
            pass  # dropped (not in upstream)
        else:
            upstream_omni_sd[k] = v
    # Add fake pretraining-only keys
    upstream_omni_sd["mask_token"] = torch.randn(16)
    upstream_omni_sd["predict_head.weight"] = torch.randn(64, 16)
    upstream_omni_sd["predict_head.bias"] = torch.randn(64)

    # Remap back
    remapped_omni, dropped_omni = _remap_state_dict(upstream_omni_sd, is_omni=True)

    # Verify fake keys are in dropped_keys
    assert "mask_token" in dropped_omni, "mask_token should be in dropped_keys"
    assert "predict_head.weight" in dropped_omni, (
        "predict_head.weight should be in dropped_keys"
    )
    assert "predict_head.bias" in dropped_omni, (
        "predict_head.bias should be in dropped_keys"
    )

    fresh_bo = BrainOmni(**omni_kwargs)
    result_bo = fresh_bo.load_state_dict(remapped_omni, strict=False)

    _assert_coverage(
        list(result_bo.missing_keys),
        list(result_bo.unexpected_keys),
        is_omni=True,
        label="BrainOmni-self-test",
    )

    # Verify that missing_keys are only final_layer.*
    for k in result_bo.missing_keys:
        assert k.startswith("final_layer."), (
            f"BrainOmni self-test: unexpected missing key: {k!r}"
        )

    # Verify that all non-head tensors match
    loaded_omni_sd = fresh_bo.state_dict()
    for k in real_omni_sd:
        if k.startswith("final_layer."):
            continue  # head is fresh, allowed to differ
        assert torch.allclose(real_omni_sd[k].float(), loaded_omni_sd[k].float()), (
            f"BrainOmni self-test: tensor mismatch at {k!r}"
        )
    print("[SELF-TEST] BrainOmni: PASSED")

    print("\nSELF-TEST PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenTSLab/BrainOmni checkpoints to braindecode format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant",
        choices=["tiny", "base", "tokenizer"],
        help="Checkpoint variant to convert (auto-discovers HF paths).",
    )
    parser.add_argument(
        "--ckpt",
        metavar="PATH",
        help="Local path to upstream .pt checkpoint (bypasses HF discovery).",
    )
    parser.add_argument(
        "--cfg",
        metavar="PATH",
        help="Local path to upstream model_cfg.json (bypasses HF discovery).",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        help="Output directory for converted checkpoint.",
    )
    parser.add_argument(
        "--reference-output",
        metavar="PATH",
        help=(
            "Path to upstream reference encode() output (optional). "
            "See module docstring for instructions to generate it."
        ),
    )
    parser.add_argument(
        "--example-data",
        metavar="PATH",
        default=None,
        help=(
            "Path to example_data.pt used in parity check.  "
            "Defaults to BrainOmni/assets/example_data.pt relative to repo root.  "
            "Required when --reference-output is given and the default path does "
            "not exist."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline self-test (no network required) and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.self_test:
        _self_test()
        return

    if args.variant is None and (args.ckpt is None or args.cfg is None):
        print(
            "ERROR: Either --variant or both --ckpt and --cfg must be provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Require --variant when local --ckpt/--cfg are given so that BrainTokenizer
    # is never misidentified as BrainOmni (they are architecturally
    # indistinguishable from ckpt_path presence alone).
    if (args.ckpt is not None or args.cfg is not None) and args.variant is None:
        print(
            "ERROR: --variant {tiny,base,tokenizer} must be provided alongside "
            "--ckpt/--cfg to distinguish BrainTokenizer from BrainOmni.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.out is None:
        print("ERROR: --out is required.", file=sys.stderr)
        sys.exit(1)

    convert(
        variant=args.variant,
        ckpt_path=args.ckpt,
        cfg_path=args.cfg,
        out_dir=args.out,
        reference_output=args.reference_output,
        example_data=args.example_data,
    )


if __name__ == "__main__":
    main()
