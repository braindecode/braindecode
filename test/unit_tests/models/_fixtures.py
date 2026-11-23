"""Model-building fixtures for the model tests: default signal parameters and
the keyword combinations each model is built with."""

import csv
import functools
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from braindecode.models import util as _models_util
from braindecode.models._summary import _TOKENIZERS, SUMMARY_CSV_PATH  # noqa: F401
from braindecode.models.util import (
    SigArgName,
    _draw_chs_info,
    models_mandatory_parameters,
)

PARAMS = pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
    ids=[m[0] for m in models_mandatory_parameters],
)

non_classification_models = [
    # Returns reconstructed patches and discrete codes, not class logits.
    "NeuroRVQTokenizer",
    "SignalJEPA",
    # Emits token-wise logits (batch, n_outputs, n_patches).
    "SleepFMStager",
    # Emits one prediction per sleep epoch (batch, n_outputs, n_epochs).
    "CareSleepNet",
    # Emits a (batch, T_out, vocab) sequence for CTC, not class logits.
    "MetaNeuromotorHand",
    "EMG2QwertyNet",
    # Returns a reconstruction tensor (VQ-VAE output), not class logits.
    "BrainTokenizer",
    "TFMTokenizer",
    # Dense per-frame pose sequences (batch, T, n_joints), not logits.
    "VEMG2Pose",
    "NeuroPose",
    "SensingDynamics",
    # forward returns (batch, num_latents, n_outputs) dense per-token logits,
    # not class logits.
    "DANCE",
    # forward returns (batch, n_outputs, n_times) per-sample logits, not
    # window logits.
    "SeizureTransformer",
    # forward returns (batch, n_outputs, n_patches), one logit per second.
    "DeepSOZ",
]

# Head-frame positions in metres for the default synthetic EEG input, inlined
# to avoid loading the montage; test_chs_info_3ch_matches_the_montage checks them.
_chs_info_3ch = [
    {"ch_name": name, "kind": "eeg", "loc": np.array([x, y, z, *[0.0] * 9])}
    for name, (x, y, z) in {
        "C1": (-0.03793783, 0.02633745, 0.12977061),
        "C2": (0.03589271, 0.02635814, 0.12841234),
        "C3": (-0.06714873, 0.02335823, 0.10451068),
    }.items()
]

default_signal_params: dict[SigArgName, Any] = {
    "n_times": 1000,
    "sfreq": 250.0,
    "n_outputs": 2,
    "chs_info": _chs_info_3ch,
    "n_chans": len(_chs_info_3ch),
    "input_window_seconds": 4.0,
}


def _get_signal_params(
    signal_params: dict[SigArgName, Any] | None | Any,
    required_params: list[SigArgName] | None = None,
) -> dict[SigArgName, Any]:
    """Get signal parameters for model initialization in tests.

    ``signal_params`` may also be a zero-argument callable that returns a
    ``dict``; this defers work that should not run at module-load time:
    imports that would cause circular dependencies (e.g. the Labram canonical
    channel list), or loading the fixture montage (e.g. DANCE, whose channels
    are drawn from it).

    When only ``n_chans`` is given, ``chs_info`` is made of ``n_chans`` distinct
    real channels of the fixture montage, drawn with a seed that depends only on
    ``n_chans`` (deterministic, independent of the test order).
    """
    if callable(signal_params):
        signal_params = signal_params()
    sp = deepcopy(default_signal_params)
    if signal_params is not None:
        sp.update(signal_params)
        if "chs_info" in signal_params and "n_chans" not in signal_params:
            sp["n_chans"] = len(signal_params["chs_info"])
        if "n_chans" in signal_params and "chs_info" not in signal_params:
            n = signal_params["n_chans"]
            # Fixed seed per n_chans: same names and positions whatever the
            # order of the tests.
            sp["chs_info"] = _draw_chs_info(n, np.random.default_rng(n))
        assert isinstance(sp["n_times"], int)
        assert isinstance(sp["sfreq"], float)
        assert isinstance(sp["input_window_seconds"], float)
        if "input_window_seconds" not in signal_params:
            sp["input_window_seconds"] = sp["n_times"] / sp["sfreq"]
        if "sfreq" not in signal_params:
            sp["sfreq"] = sp["n_times"] / sp["input_window_seconds"]
        if "n_times" not in signal_params:
            sp["n_times"] = int(sp["input_window_seconds"] * sp["sfreq"])
    if required_params is not None:
        sp = {
            k: sp[k] for k in set((signal_params or {}).keys()).union(required_params)
        }
    return sp


def _get_all_possible_signal_params(
    signal_params: dict[SigArgName, Any], required_params: list[SigArgName]
) -> list[dict[str, Any]]:
    """Return the full Cartesian product (up to 42 combinations) of valid signal kwargs.

    Used on lightweight dummy modules in ``test_base.py`` and ``test_config.py`` to
    exhaustively test ``EEGModuleMixin`` and Pydantic signal-parameter inference.
    """
    sp = signal_params

    output_kwargs: list[dict[str, Any]] = [dict(n_outputs=sp["n_outputs"])]
    if "n_outputs" not in required_params:
        output_kwargs.append(dict(n_outputs=None))

    channel_kwargs: list[dict[str, Any]] = [dict(chs_info=sp["chs_info"], n_chans=None)]
    if "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=sp["n_chans"], chs_info=None))
    if "n_chans" not in required_params and "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=None, chs_info=None))

    time_kwargs: list[dict[str, Any]] = [
        dict(n_times=sp["n_times"], sfreq=sp["sfreq"], input_window_seconds=None),
        dict(
            n_times=None,
            sfreq=sp["sfreq"],
            input_window_seconds=sp["input_window_seconds"],
        ),
        dict(
            n_times=sp["n_times"],
            sfreq=None,
            input_window_seconds=sp["input_window_seconds"],
        ),
    ]
    if "n_times" not in required_params and "sfreq" not in required_params:
        time_kwargs.append(
            dict(
                n_times=None,
                sfreq=None,
                input_window_seconds=sp["input_window_seconds"],
            )
        )
    if (
        "n_times" not in required_params
        and "input_window_seconds" not in required_params
    ):
        time_kwargs.append(
            dict(n_times=None, sfreq=sp["sfreq"], input_window_seconds=None)
        )
    if "sfreq" not in required_params and "input_window_seconds" not in required_params:
        time_kwargs.append(
            dict(n_times=sp["n_times"], sfreq=None, input_window_seconds=None)
        )
    if (
        "n_times" not in required_params
        and "sfreq" not in required_params
        and "input_window_seconds" not in required_params
    ):
        time_kwargs.append(dict(n_times=None, sfreq=None, input_window_seconds=None))

    extra: dict[str, Any] = {
        str(k): v for k, v in sp.items() if k not in default_signal_params
    }
    return [
        {**o, **c, **t, **extra}
        for o in output_kwargs
        for c in channel_kwargs
        for t in time_kwargs
    ]


def _get_possible_signal_params(
    signal_params: dict[SigArgName, Any], required_params: list[SigArgName]
) -> list[dict[str, Any]]:
    """Return 2-3 minimal 1-wise covering combinations of valid signal kwargs.

    Covers:
    1. Index ``[0]``: ``chs_info`` given (``n_chans=None`` inferred),
       ``(n_times, sfreq)`` given (``input_window_seconds=None`` inferred),
       and ``n_outputs`` given.
    2. Minimal required-only combination: sets every non-required signal
       parameter to ``None`` simultaneously so any undeclared dependency in
       ``required_params`` raises ``ValueError``.
    3. Alternative inference combination: exercises ``n_chans`` with
       ``chs_info=None`` (when ``chs_info`` is not required) and infers a
       required ``n_times`` or ``sfreq`` from the complementary pair so
       models reading private ``self._n_times`` / ``self._sfreq`` fail.
    """
    sp = signal_params
    req = set(required_params)
    extra: dict[str, Any] = {
        str(k): v for k, v in sp.items() if k not in default_signal_params
    }

    c0_out: dict[str, Any] = {"n_outputs": sp["n_outputs"]}
    c0_ch: dict[str, Any] = {"chs_info": sp["chs_info"], "n_chans": None}
    c0_time: dict[str, Any] = {
        "n_times": sp["n_times"],
        "sfreq": sp["sfreq"],
        "input_window_seconds": None,
    }
    combo_0: dict[str, Any] = {**c0_out, **c0_ch, **c0_time, **extra}

    min_out: dict[str, Any] = {
        "n_outputs": sp["n_outputs"] if "n_outputs" in req else None
    }
    if "chs_info" in req:
        min_ch: dict[str, Any] = {"chs_info": sp["chs_info"], "n_chans": None}
        scalar_ch: dict[str, Any] = min_ch
    elif "n_chans" in req:
        min_ch = {"n_chans": sp["n_chans"], "chs_info": None}
        scalar_ch = min_ch
    else:
        min_ch = {"n_chans": None, "chs_info": None}
        scalar_ch = {"n_chans": sp["n_chans"], "chs_info": None}

    req_time = req & {"n_times", "sfreq", "input_window_seconds"}
    t_infer_ntimes: dict[str, Any] = {
        "n_times": None,
        "sfreq": sp["sfreq"],
        "input_window_seconds": sp["input_window_seconds"],
    }
    t_infer_sfreq: dict[str, Any] = {
        "n_times": sp["n_times"],
        "sfreq": None,
        "input_window_seconds": sp["input_window_seconds"],
    }

    combos = [combo_0]
    if len(req_time) >= 2:
        combos.append({**c0_out, **scalar_ch, **t_infer_ntimes, **extra})
        combos.append({**min_out, **min_ch, **t_infer_sfreq, **extra})
    elif req_time == {"n_times"}:
        combos.append({**c0_out, **scalar_ch, **t_infer_ntimes, **extra})
        min_time: dict[str, Any] = {
            "n_times": sp["n_times"],
            "sfreq": None,
            "input_window_seconds": None,
        }
        combos.append({**min_out, **min_ch, **min_time, **extra})
    elif req_time == {"sfreq"}:
        combos.append({**c0_out, **scalar_ch, **t_infer_sfreq, **extra})
        min_time = {"n_times": None, "sfreq": sp["sfreq"], "input_window_seconds": None}
        combos.append({**min_out, **min_ch, **min_time, **extra})
    elif req_time == {"input_window_seconds"}:
        min_time = {
            "n_times": None,
            "sfreq": None,
            "input_window_seconds": sp["input_window_seconds"],
        }
        combos.append({**min_out, **min_ch, **min_time, **extra})
    else:
        min_time = {"n_times": None, "sfreq": None, "input_window_seconds": None}
        if scalar_ch != min_ch:
            combos.append({**c0_out, **scalar_ch, **t_infer_ntimes, **extra})
        combos.append({**min_out, **min_ch, **min_time, **extra})

    unique: list[dict[str, Any]] = []
    for c in combos:
        if not any(
            all(
                (a is b if isinstance(a, list) else a == b)
                for k in c
                for a, b in [(c[k], u[k])]
            )
            for u in unique
        ):
            unique.append(c)
    return unique


def _architecture_kwargs(model_name: str) -> dict[str, Any]:
    """Architecture arguments of a model's registry entry, without its signal
    arguments: the small instance of the largest models that every generic
    test builds (see ``models_mandatory_parameters``)."""
    signal_params = next(
        sp for name, _, sp in models_mandatory_parameters if name == model_name
    )
    sp = _get_signal_params(signal_params)
    return {k: v for k, v in sp.items() if k not in default_signal_params}


def _tensor_leaves(value: Any) -> Iterator[Tensor]:
    """Yield tensor leaves from nested model outputs."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _tensor_leaves(child)


def _shapes(out: Any) -> list[tuple[int, ...]]:
    """Return shapes of tensor leaves in a model output."""
    return [tuple(leaf.shape) for leaf in _tensor_leaves(out)]


def _build_case(
    model_name: str,
    required_params: list[SigArgName],
    signal_params: Any,
    headless: bool = False,
) -> tuple[nn.Module, Tensor]:
    """Build an eval-mode model and synthetic batch ``(2, n_chans, n_times)``,
    with ``n_outputs=None`` when ``headless``."""
    signal = _get_signal_params(signal_params)
    model_kwargs = _get_signal_params(signal_params, required_params)
    if headless:
        model_kwargs["n_outputs"] = None
    model = _models_util.models_dict[model_name](**model_kwargs).eval()
    x = torch.randn(2, len(signal["chs_info"]), signal["n_times"])
    return model, x


def _materialize(model: nn.Module, x: Tensor, return_features: bool = False) -> None:
    """Run one no-grad forward if lazy parameters still need their shapes."""
    if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
        with torch.no_grad():
            model(x, return_features=True) if return_features else model(x)


@functools.cache
def _summary_rows() -> dict[str, dict[str, str]]:
    with SUMMARY_CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return {row["Model"]: row for row in csv.DictReader(handle)}


def _summary_row(model_name: str) -> dict[str, str]:
    """Return a model's row from ``braindecode/models/summary.csv``."""
    return _summary_rows()[model_name]
