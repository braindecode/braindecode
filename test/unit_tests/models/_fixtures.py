"""Model-building fixtures for the model tests: default signal parameters and
the keyword combinations each model is built with."""

from copy import deepcopy
from typing import Any

import numpy as np

from braindecode.models.util import SigArgName, _chs_info_3ch, _draw_chs_info

non_classification_models = [
    # Returns reconstructed patches and discrete codes, not class logits.
    "NeuroRVQTokenizer",
    "SignalJEPA",
    # Emits token-wise logits (batch, n_outputs, n_patches).
    "SleepFMStager",
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


def _get_possible_signal_params(
    signal_params: dict[SigArgName, Any], required_params: list[SigArgName]
):
    sp = signal_params

    # List possible model kwargs:
    output_kwargs = []
    output_kwargs.append(dict(n_outputs=sp["n_outputs"]))

    if "n_outputs" not in required_params:
        output_kwargs.append(dict(n_outputs=None))

    channel_kwargs = []
    channel_kwargs.append(dict(chs_info=sp["chs_info"], n_chans=None))
    if "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=sp["n_chans"], chs_info=None))
    if "n_chans" not in required_params and "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=None, chs_info=None))

    time_kwargs = []
    time_kwargs.append(
        dict(n_times=sp["n_times"], sfreq=sp["sfreq"], input_window_seconds=None)
    )
    time_kwargs.append(
        dict(
            n_times=None,
            sfreq=sp["sfreq"],
            input_window_seconds=sp["input_window_seconds"],
        )
    )
    time_kwargs.append(
        dict(
            n_times=sp["n_times"],
            sfreq=None,
            input_window_seconds=sp["input_window_seconds"],
        )
    )
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

    return [
        dict(**o, **c, **t)
        for o in output_kwargs
        for c in channel_kwargs
        for t in time_kwargs
    ]
