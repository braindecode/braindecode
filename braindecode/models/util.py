# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#          Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: BSD (3-clause)
import inspect
import math
import sys
import warnings
from collections.abc import Callable as CallableABC
from copy import deepcopy
from functools import lru_cache, wraps
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import mne
import numpy as np
import torch  # noqa: F401  # exposed so TorchScript can resolve ``torch`` through the BatchNorm-guard wrapper's __globals__
from mne.io.constants import FIFF
from torch import nn
from torchaudio.transforms import Resample

from braindecode import modules as bd_modules
from braindecode.functional import check_n_chans  # noqa: F401
from braindecode.util import resolve_montage_name

# Canonical MNE channel-type name for the electrode kinds braindecode models
# tell apart.
_KIND_TO_CH_TYPE = {
    FIFF.FIFFV_EEG_CH: "eeg",
    FIFF.FIFFV_SEEG_CH: "seeg",
    FIFF.FIFFV_DBS_CH: "dbs",
    FIFF.FIFFV_ECOG_CH: "ecog",
}

#: MNE channel types recorded from inside the skull.
INTRACRANIAL_CH_TYPES = frozenset({"seeg", "dbs", "ecog"})


# Read through ``models_dict`` (module ``__getattr__`` below), which fills it on
# first access so importing braindecode does not import every model.
_models_dict: dict[str, Any] = {}
# Annotations only, for type checkers; the values come from ``__getattr__``.
models_dict: dict[str, Any]
_summary_table: "pd.DataFrame"

if TYPE_CHECKING:
    import pandas as pd
else:  # hidden from checkers so misspelled names stay errors

    def __getattr__(name):
        if name == "models_dict":
            if not _models_dict:
                _init_models_dict()
            return _models_dict
        if name == "_summary_table":  # read summary.csv once, on first access
            table = globals()[name] = get_summary_table()
            return table
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Resolve serialized classes by exact lookup, never by importing a config path.
# Include public aliases and the defining paths emitted by get_config().
_CONFIG_MODULE_TYPES = {}
for _module in (nn, bd_modules):
    for _name, _type in vars(_module).items():
        if (
            not _name.startswith("_")
            and isinstance(_type, type)
            and issubclass(_type, nn.Module)
        ):
            _CONFIG_MODULE_TYPES[f"{_module.__name__}.{_name}"] = _type
            _CONFIG_MODULE_TYPES[f"{_type.__module__}.{_type.__qualname__}"] = _type


def resolve_config_type(value):
    """Decode a shipped module class without importing user-selected modules.

    Python class/callable objects supplied by application code are left intact;
    serialized strings must name a public torch.nn or braindecode.modules class.
    """
    if not isinstance(value, str):
        return value
    if value not in _CONFIG_MODULE_TYPES:
        raise ValueError(f"Unsupported serialized module type: {value!r}")
    return _CONFIG_MODULE_TYPES[value]


def _has_type_annotation(annotation):
    """Whether an annotation includes a class or callable argument."""
    return (
        annotation in (type, CallableABC)
        or get_origin(annotation) in (type, CallableABC)
        or any(_has_type_annotation(arg) for arg in get_args(annotation))
    )


SigArgName = Literal[
    "n_outputs",
    "n_chans",
    "chs_info",
    "n_times",
    "input_window_seconds",
    "sfreq",
]
_EEG_PARAMS = frozenset(get_args(SigArgName))

_JSON_SAFE = (int, float, str, bool, type(None))
_BATCH_NORM_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def _disable_batch_norm_training_if_batch_size_one(forward):
    """Temporarily set BatchNorm layers to train(False) for batch size one.

    TorchScript resolves the names of a wrapped forward in this module, hence
    the imports of ``torch`` and ``check_n_chans`` at the top.
    """
    forward_signature = inspect.signature(forward)

    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        bound = forward_signature.bind(self, *args, **kwargs)
        x = next(value for name, value in bound.arguments.items() if name != "self")
        batch_norms = []
        if self.training and x.shape[0] == 1:
            batch_norms = [
                layer
                for layer in self.modules()
                if isinstance(layer, _BATCH_NORM_MODULES) and layer.training
            ]
            for batch_norm in batch_norms:
                batch_norm.train(False)
        try:
            return forward(self, *args, **kwargs)
        finally:
            for batch_norm in batch_norms:
                batch_norm.train(True)

    return wrapped


def _is_jsonable(val):
    """Return True if *val* is directly JSON-serializable."""
    if isinstance(val, _JSON_SAFE):
        return True
    if isinstance(val, (list, tuple)):
        return all(_is_jsonable(v) for v in val)
    if isinstance(val, dict):
        return all(isinstance(k, str) and _is_jsonable(v) for k, v in val.items())
    return False


def track_model_init_kwargs(cls) -> None:
    """Instrument a model class so constructor kwargs are tracked."""
    init = cls.__init__
    if getattr(init, "_braindecode_tracks_init_kwargs", False):
        return

    init_signature = inspect.signature(init)
    accepts_channel_strategy = "channel_strategy" in init_signature.parameters or any(
        p.kind is p.VAR_KEYWORD for p in init_signature.parameters.values()
    )

    @wraps(init)
    def wrapped(self, *args, **kwargs):
        # Reached through super(): the subclass already captured its own kwargs,
        # and this parent sees only the ones the subclass forwards.
        if type(self).__init__ is not wrapped:
            return init(self, *args, **kwargs)
        if "channel_strategy" in kwargs and not accepts_channel_strategy:
            name = type(self).__name__
            raise TypeError(
                f"{name} takes no channel_strategy: it reads its input "
                "natively (see the channel strategies page of the models documentation)."
            )
        bound = init_signature.bind(self, *args, **kwargs)
        explicit_keys = frozenset(bound.arguments.keys() - {"self"})
        bound.apply_defaults()

        captured_kwargs = {}
        for name, param in init_signature.parameters.items():
            if name == "self" or name not in bound.arguments:
                continue
            value = bound.arguments[name]
            if param.kind == param.VAR_KEYWORD:
                captured_kwargs.update(value)
            elif param.kind != param.VAR_POSITIONAL:
                captured_kwargs[name] = value
        self._braindecode_explicit_init_keys = explicit_keys
        self._braindecode_init_kwargs = captured_kwargs
        init(self, *args, **kwargs)
        self._drop_headless_head()
        self._apply_deprecated_init_flags()

    setattr(wrapped, "_braindecode_tracks_init_kwargs", True)
    cls.__init__ = wrapped


def build_model_config(model) -> dict:
    """Build a JSON-serializable config dict from a model instance.

    Prefers ``_hub_mixin_config`` when available (it uses Hub coders to
    encode custom types like ``functools.partial``). Otherwise reuses the
    ``__init__`` keyword arguments captured at construction time.

    Parameters
    ----------
    model : EEGModuleMixin
        The model instance.

    Returns
    -------
    dict
        All ``__init__`` parameters, JSON-serializable.
    """
    hub_config = getattr(model, "_hub_mixin_config", None)
    if hub_config is not None:
        config = dict(hub_config)
    else:
        init_kwargs = getattr(model, "_braindecode_init_kwargs", None)
        if init_kwargs is None:
            raise RuntimeError(
                f"{type(model).__name__} was instantiated without captured init kwargs; "
                "cannot build a reliable config."
            )

        config = {}
        for name, val in deepcopy(init_kwargs).items():
            if name == "chs_info":
                continue
            if isinstance(val, type):
                val = f"{val.__module__}.{val.__qualname__}"
            elif not _is_jsonable(val):
                continue
            config[name] = val
    adaptation = deepcopy(getattr(model, "_adapt_config", None) or {})
    # Save the input montage, not the target; an adapted layer is saved apart.
    layer = (
        None
        if "channel_strategy" in adaptation
        else getattr(model, "channel_layer", None)
    )
    chs_info = (
        layer.chs_info if layer is not None else getattr(model, "_chs_info", None)
    )
    if chs_info is not None:
        config["chs_info"] = model._serialize_chs_info(chs_info)
    if adaptation.get("chs_info") is not None:
        adaptation["chs_info"] = model._serialize_chs_info(adaptation["chs_info"])
    if adaptation:
        config["input_adaptation"] = adaptation
    return config


def resolve_type_kwargs(cls, kwargs):
    """Resolve type-encoded strings in *kwargs* back to Python types.

    When a model config is saved to JSON, ``type[nn.Module]`` parameters
    (e.g. ``activation=nn.ELU``) are stored as dotted import paths like
    ``"torch.nn.modules.activation.ELU"``.  This function resolves them
    back to known shipped module classes without importing config-selected modules.

    Parameters
    ----------
    cls : type
        The model class whose ``__init__`` signature is inspected.
    kwargs : dict
        Keyword arguments to resolve in-place.

    Returns
    -------
    dict
        The same *kwargs* dict, with type strings resolved.
    """
    init = inspect.unwrap(cls.__init__)
    for name, param in inspect.signature(init).parameters.items():
        val = kwargs.get(name)
        if not isinstance(val, str):
            continue
        is_type = param.default is not inspect.Parameter.empty and isinstance(
            param.default, type
        )
        if not is_type and param.annotation is not inspect.Parameter.empty:
            # Resolve only this effective argument, not unrelated metadata or
            # return annotations (which may use TYPE_CHECKING-only imports).
            annotation = get_type_hints(
                SimpleNamespace(__annotations__={name: param.annotation}),
                globalns=init.__globals__,
                localns=dict(vars(cls)),
            )[name]
            is_type = _has_type_annotation(annotation)
        if is_type:
            kwargs[name] = resolve_config_type(val)
    return kwargs


# For the models inside the init model, go through all the models
# check those have the EEGMixin class inherited. If they are, add them to the
# list.


def _init_models_dict():
    from braindecode import models
    from braindecode.models.base import EEGModuleMixin

    # Plain getattr over every public name: a model that fails to import raises here.
    found = {}
    for name in sorted(models._LAZY):
        obj = getattr(models, name)
        is_model = inspect.isclass(obj) and issubclass(obj, EEGModuleMixin)
        if is_model and obj is not EEGModuleMixin:
            found[name] = obj
    # One update() publishes every entry at once, so no thread sees a partial dict.
    _models_dict.update(found)


def _get_model_class(model_name: str):
    """Return the model class registered under ``model_name``.

    Looks the name up in :data:`models_dict` (e.g. for skorch wrappers and
    pydantic configs).

    Parameters
    ----------
    model_name : str
        Name of the model class to retrieve.

    Returns
    -------
    type
        The model class registered under ``model_name``.

    Raises
    ------
    ValueError
        If ``model_name`` is not a registered model.
    """
    registry = sys.modules[__name__].models_dict  # also honours a patched models_dict
    if model_name in registry:
        return registry[model_name]
    raise ValueError(f"Unknown model name {model_name!r}.")


@lru_cache(maxsize=1)
def _fixture_montage_chs() -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Read the distinct EEG positions of the 10-05 montage for model examples.

    Each entry pairs a channel name with a 12-value MNE location. The first
    three values are head coordinates in metres, and the remaining values
    are zero. The first name at each position takes precedence over its
    legacy aliases (T3, T4, T5 and T6).

    The montage is ``colin27_1005``, named ``standard_1005`` before MNE 1.13
    (see :func:`braindecode.util.resolve_montage_name`). It is loaded on the
    first call and cached, so that importing ``braindecode`` never reads the
    montage file nor risks a montage-name deprecation warning.

    Returns
    -------
    tuple of (str, tuple of float)
        ``(ch_name, loc)`` for the 339 distinct channels of the montage.
    """
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    info = mne.create_info(montage.ch_names, 1.0, "eeg")
    info.set_montage(montage)
    chs = []
    seen = set()
    for ch in info["chs"]:
        position = tuple(np.round(ch["loc"][:3], 6).tolist())
        if position in seen:  # legacy alias of a channel listed before
            continue
        seen.add(position)
        loc = np.zeros(12)
        loc[:3] = ch["loc"][:3]
        chs.append((ch["ch_name"], tuple(loc.tolist())))
    return tuple(chs)


def _draw_chs_info(n_chans: int, rng: np.random.Generator) -> list[dict]:
    """Sample EEG channel names and their head positions without replacement.

    Parameters
    ----------
    n_chans : int
        Number of distinct channels to draw.
    rng : numpy.random.Generator
        Source of the draw. Same state, same channels.

    Returns
    -------
    list of dict
        ``{"ch_name": str, "kind": "eeg", "loc": ndarray of 12 floats}``.

    Raises
    ------
    ValueError
        If the montage has fewer than ``n_chans`` channels.
    """
    montage_chs = _fixture_montage_chs()
    if n_chans > len(montage_chs):
        raise ValueError(
            f"Cannot draw {n_chans} distinct channels from a montage of "
            f"{len(montage_chs)} channels; pass chs_info explicitly."
        )
    return [
        {
            "ch_name": montage_chs[i][0],
            "kind": "eeg",
            "loc": np.array(montage_chs[i][1]),
        }
        for i in rng.choice(len(montage_chs), size=n_chans, replace=False)
    ]


def _canonical_example_channels(channels) -> list[dict]:
    """Give canonical model channels the 12-value location used in the public examples."""
    result = []
    for ch in channels:
        loc = np.zeros(12)
        loc[:3] = ch["loc"][:3]
        result.append(
            {"ch_name": ch["ch_name"], "kind": ch["kind"], "loc": loc.tolist()}
        )
    return result


def _get_labram_chs_info() -> list[dict]:
    # Import when the example is requested to avoid the model/base import cycle.
    from braindecode.models.labram import _LABRAM_TARGET_CHS_INFO

    return _canonical_example_channels(_LABRAM_TARGET_CHS_INFO)


def _get_bendr_chs_info() -> list[dict]:
    # Lazy import for the same model/base import cycle as LaBraM.
    from braindecode.models.bendr import _BENDR_TARGET_CHS_INFO

    return _canonical_example_channels(_BENDR_TARGET_CHS_INFO)


# FreeSurfer's left-hemisphere Desikan-Killiany cortical gyri occupy the
# contiguous id block 1000-1035 of its bundled colour table (the right
# hemisphere mirrors it 1000 ids higher). DKT drops five of these 36 ids:
# 1000 (``unknown``) and 1004 (``corpuscallosum``) are not cortical parcels,
# and 1001 (``bankssts``), 1032 (``frontalpole``), 1033 (``temporalpole``) are
# the three gyri whose boundaries the DKT protocol could not define reliably,
# so they were folded into their neighbours. These are FreeSurfer *ids*, not
# parcel names, so the exclusion needs no name vocabulary.
_DK_LH_CORTICAL_IDS = range(1000, 1036)
_DKT_EXCLUDED_IDS = frozenset({1000, 1001, 1004, 1032, 1033})

# Aseg subcortical structures get a ``Left-``/``Right-`` id pair for every
# lateralized entry; DKT keeps the structures whose left id falls in 10-18
# and that have a ``Right-`` counterpart elsewhere in the table. The 3rd/4th
# ventricles and the brain stem fall in that id range too but have no
# ``Right-`` counterpart, so this test excludes them by structure, not by
# typing their names.
_ASEG_LH_IDS = range(10, 19)

# MAPA's released region embedding orders the six kept aseg structures as
# Hippocampus, Amygdala, Caudate, Putamen, Pallidum, Thalamus-Proper. That is
# neither ascending-id order (Thalamus, Caudate, Putamen, Pallidum,
# Hippocampus, Amygdala) nor alphabetical order (Amygdala, Caudate,
# Hippocampus, Pallidum, Putamen, Thalamus-Proper): no table MNE ships
# encodes it. It is the one piece of metadata this helper cannot derive --
# confirmed only by the released checkpoint, via the state-dict transplant in
# ``test_mapa_released_checkpoint_reproduces_the_reference_features`` -- so it
# is kept as a permutation of *positions* in the alphabetically-sorted,
# MNE-derived set below, never as a list of structure names.
_ASEG_SLOT_PERMUTATION: tuple[int, ...] = (2, 0, 1, 4, 3, 5)


def dkt_region_slots() -> tuple[str, ...]:
    """FreeSurfer DKT region names in MAPA's released embedding slot order.

    The 62 hemisphere-qualified cortical parcels come first (left hemisphere,
    then right, each alphabetically sorted), then the 12 subcortical
    structures (left, then right, in the released, non-alphabetical order --
    see ``_ASEG_SLOT_PERMUTATION``). Every name comes straight out of
    :func:`mne.read_freesurfer_lut`, MNE's bundled FreeSurfer colour table
    (no download); braindecode contributes only the id-based selection rule
    and the slot order, never a hand-typed name list.

    Returns
    -------
    tuple of str
        The 74 region names, index ``i`` being embedding slot ``i``.
    """
    lut, _ = mne.read_freesurfer_lut()
    cortical = sorted(
        name.removeprefix("ctx-lh-")
        for name, id_ in lut.items()
        if id_ in _DK_LH_CORTICAL_IDS and id_ not in _DKT_EXCLUDED_IDS
    )
    aseg_lh_alpha = sorted(
        name.removeprefix("Left-")
        for name, id_ in lut.items()
        if id_ in _ASEG_LH_IDS
        and name.startswith("Left-")
        and f"Right-{name.removeprefix('Left-')}" in lut
    )
    aseg = tuple(aseg_lh_alpha[i] for i in _ASEG_SLOT_PERMUTATION)
    return (
        tuple(f"ctx-lh-{parcel}" for parcel in cortical)
        + tuple(f"ctx-rh-{parcel}" for parcel in cortical)
        + tuple(f"Left-{structure}" for structure in aseg)
        + tuple(f"Right-{structure}" for structure in aseg)
    )


def _dance_signal_params() -> dict[str, Any]:
    """DANCE signal parameters, deferred as their channels come from a montage.

    19 distinct real channels, drawn with a fixed seed: the same channels on
    every call, whatever the order of the tests.
    """
    return {
        "n_chans": 19,
        # 1 s @ 200 Hz: CPU float16/bfloat16 dilated convolutions take a
        # scalar path, so 32 s windows took 25 min per dtype case.
        "n_times": 200,
        "sfreq": 200.0,
        "input_window_seconds": 1.0,
        "n_outputs": 4,
        "chs_info": _draw_chs_info(19, np.random.default_rng(19)),
        # 6 of 10 convolution blocks (dilation 97 is the first to skip taps
        # that only read padding), 1 of 6 Perceiver and 1 of 4 decoder blocks
        "conv_depth": 6,
        "perceiver_depth": 1,
        "decoder_depth": 1,
    }


def _fnirs_chs_info(n_pairs: int, interleaved: bool = False) -> list[dict]:
    """Stacked HbO then HbR channels of ``n_pairs`` source-detector pairs 3 cm
    apart on an 8 cm ring (metres), named as MNE names them (``"S1_D1 hbo"``),
    with the source and detector in ``loc[3:9]`` as MNE stores them;
    ``interleaved=True`` gives MNE's ``[HbO_1, HbR_1, ...]`` order."""
    chs = []
    for kind in ("hbo", "hbr"):
        for i in range(n_pairs):
            src = np.array([0.08 * np.cos(i), 0.08 * np.sin(i), 0.05])
            det = src + np.array([0.03, 0.0, 0.0])
            loc = np.concatenate([(src + det) / 2, src, det, np.zeros(3)])
            chs.append({"ch_name": f"S{i + 1}_D{i + 1} {kind}", "loc": loc})
    if interleaved:
        chs = [ch for pair in zip(chs[:n_pairs], chs[n_pairs:]) for ch in pair]
    return chs


_barista_rng = np.random.default_rng(12)

# Braindecode exports these ordered triples. Config generation reads the name
# and required signal arguments. The third field supplies example constructor
# arguments for models that use different settings from the default example
# (``default_signal_params`` in the test package); for the largest models it
# also shrinks the architecture to keep CI small, so it is not the released
# configuration (covered by ``test_pretrained_compat``).
models_mandatory_parameters: list[
    tuple[str, list[SigArgName], dict[str, Any] | Callable[[], dict[str, Any]] | None]
] = [
    ("ATCNet", ["n_chans", "n_outputs", "n_times"], None),
    # 2 of the 22 released blocks (118.6M -> 10.9M parameters), here and below:
    # fewer identical blocks, every tensor keeps its released shape.
    (
        "AXON",
        ["chs_info", "n_outputs"],
        {"sfreq": 200.0, "n_times": 1000, "depth": 2},
    ),
    (
        "Pratibha2",
        ["chs_info", "n_outputs"],
        {"sfreq": 200.0, "n_times": 1000, "depth": 2},  # 2 of 22 blocks
    ),
    (
        "BaRISTA",
        ["chs_info", "n_outputs", "n_times"],
        {
            # 3 s at the paper's 2048 Hz, i.e. 12 patches of 512 samples.
            "n_times": 6144,
            "sfreq": 2048.0,
            # Intracranial positions, in metres as MNE stores them, so that the
            # default coordinate-scale spatial encoding gets distinct bins.
            "chs_info": [
                {
                    "ch_name": f"E{i + 1}",
                    "kind": "seeg",
                    "loc": np.concatenate(
                        [0.06 * (2 * _barista_rng.random(3) - 1), np.zeros(9)]
                    ),
                }
                for i in range(8)
            ],
        },
    ),
    ("BDTCN", ["n_chans", "n_outputs"], None),
    ("Deep4Net", ["n_chans", "n_outputs", "n_times"], None),
    ("DeepSleepNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGConformer", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGCLIP", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGInceptionERP", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGInceptionMI", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGITNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGPT", ["n_chans", "n_outputs", "n_times", "chs_info"], None),
    ("ShallowFBCSPNet", ["n_chans", "n_outputs", "n_times"], None),
    (
        "SleepStagerBlanco2020",
        ["n_chans", "n_outputs", "n_times"],
        {"n_chans": 4},  # n_chans dividable by n_groups=2
    ),
    ("SleepStagerChambon2018", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    (
        "SleepFM",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {"n_chans": 3, "n_times": 640, "sfreq": 128.0},
    ),
    (
        "SleepFMStager",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {"n_chans": 3, "n_times": 640, "sfreq": 128.0},
    ),
    (
        "AttnSleep",
        ["n_outputs", "n_times", "sfreq"],
        {
            "sfreq": 100.0,
            "n_times": 3000,
            "chs_info": [{"ch_name": "C1", "kind": "eeg"}],
        },
    ),  # 1 channel
    ("TIDNet", ["n_chans", "n_outputs", "n_times"], None),
    ("USleep", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 128.0}),
    ("BIOT", ["n_chans", "n_outputs", "sfreq", "n_times"], None),
    # The default 1000 samples leave 1 STFT frame; the centre pooling needs 10.
    (
        "BrainBERT",
        ["n_chans", "n_outputs", "n_times"],
        {"n_times": 2048, "sfreq": 2048.0},  # the pretraining rate, 1 s windows
    ),
    ("PopulationTransformer", ["n_chans", "n_outputs", "n_times"], None),
    (
        "DuIN",
        ["n_chans", "n_outputs", "n_times"],
        {"n_times": 3000, "sfreq": 1000.0},  # the downstream 3 s windows at 1 kHz
    ),
    ("AttentionBaseNet", ["n_chans", "n_outputs", "n_times"], None),
    (
        "Labram",
        ["chs_info", "n_outputs", "n_times"],
        # Callable: Labram requires the exact 128-ch canonical order; deferred to
        # avoid a circular import (util.py is loaded by base.py before labram.py).
        lambda: {"chs_info": _get_labram_chs_info()},
    ),
    ("EEGSimpleConv", ["n_chans", "n_outputs", "sfreq"], None),
    ("SPARCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ContraWR", ["n_chans", "n_outputs", "sfreq", "n_times"], {"sfreq": 200.0}),
    ("EEGNeX", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGSym", ["chs_info", "n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("TSception", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 200.0}),
    ("EEGTCNet", ["n_chans", "n_outputs", "n_times"], None),
    (
        "TCCNet",
        ["n_chans", "n_outputs"],
        # NinaPro DB1: 10 channels at 100 Hz. TCCNet takes whole gestures of
        # any length; 300 samples is only a test length.
        {"n_chans": 10, "n_times": 300, "sfreq": 100.0},
    ),
    (
        "CareSleepNet",
        ["n_chans", "n_outputs", "n_times"],
        # 20 sleep epochs of 300 samples (3 s at 100 Hz) keep the tests small;
        # the reference uses 30 s epochs, n_times=60000. 1 EEG + 2 EOG channels.
        {"n_times": 6000, "sfreq": 100.0},
    ),
    ("SyncNet", ["n_chans", "n_outputs", "n_times"], None),
    ("MSVTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("NICE", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGMiner", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 200.0}),
    ("CTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("MSCFormer", ["n_chans", "n_outputs", "n_times"], None),
    ("TMSANet", ["n_chans", "n_outputs", "n_times"], None),
    ("TCFormer", ["n_chans", "n_outputs", "n_times"], None),
    ("SincShallowNet", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 250.0}),
    ("SCCNet", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 200.0}),
    ("SignalJEPA", ["chs_info"], None),
    ("SignalJEPA_Contextual", ["chs_info", "n_times", "n_outputs"], None),
    ("SignalJEPA_PostLocal", ["n_chans", "n_times", "n_outputs"], None),
    ("SignalJEPA_PreLocal", ["n_chans", "n_times", "n_outputs"], None),
    ("FBCNet", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 200.0}),
    ("FBMSNet", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 200.0}),
    (
        "MetaNeuromotorHand",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {
            "n_chans": 16,
            "n_times": 32000,
            "sfreq": 2000.0,
            "input_window_seconds": 16.0,
            "n_outputs": 100,
        },
    ),
    (
        "EMG2QwertyNet",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {
            "n_chans": 32,
            "n_times": 8000,
            "sfreq": 2000.0,
            "input_window_seconds": 4.0,
            "n_outputs": 99,
        },
    ),
    (
        "VEMG2Pose",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {
            "n_chans": 16,
            "n_times": 11790,
            "sfreq": 2000.0,
            "input_window_seconds": 5.895,
            "n_outputs": 20,
        },
    ),
    (
        "NeuroPose",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {
            "n_chans": 16,
            "n_times": 10000,
            "sfreq": 2000.0,
            "input_window_seconds": 5.0,
            "n_outputs": 20,
        },
    ),
    (
        "SensingDynamics",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {
            "n_chans": 16,
            "n_times": 480,
            "sfreq": 2000.0,
            "input_window_seconds": 480 / 2000.0,
            "n_outputs": 20,
        },
    ),
    (
        "TinyMyo",
        ["n_chans", "n_outputs", "n_times"],
        {
            # The pretraining windows: 1000 samples at 2000 Hz
            "n_chans": 8,
            "n_times": 1000,
            "sfreq": 2000.0,
            "input_window_seconds": 0.5,
            # 2 of the 8 released blocks; 2 heads also split the
            # embed_dim=128 of the embedding test into even widths
            "depth": 2,
            "num_heads": 2,
        },
    ),
    ("FBLightConvNet", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 200.0}),
    ("IFNet", ["n_chans", "n_outputs", "n_times", "sfreq"], {"sfreq": 200.0}),
    ("PBT", ["n_chans", "n_outputs", "n_times"], None),
    ("SSTDPN", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("BrainModule", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("Brant", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    # 3072 samples, the pretrained length: 12 s at 256 Hz or 3 s at 1024 Hz
    ("EpiNT", ["n_chans", "n_outputs", "n_times"], {"n_times": 3072, "sfreq": 256.0}),
    (
        "BENDR",
        ["n_chans", "n_outputs", "n_times"],
        # Callable: BENDR now requires chs_info to match BENDR_CHANNEL_ORDER
        # when supplied; deferred to avoid a circular import (util.py is loaded
        # by base.py before bendr.py).
        lambda: {
            "chs_info": _get_bendr_chs_info(),
            # 1M parameters instead of the released 157M (5.2 GB to script).
            "encoder_h": 64,
            "contextualizer_hidden": 128,
            "transformer_layers": 2,
        },
    ),
    ("LUNA", ["n_chans", "n_times", "n_outputs"], None),
    (
        "MAPA",
        ["chs_info", "n_outputs", "n_times"],
        {
            # 1 s at the paper's 2048 Hz, i.e. 32 frames of the 32 Hz clock.
            "n_times": 2048,
            "sfreq": 2048.0,
            # Clinical sEEG labels, which MAPA reads the array and the contact
            # number off: two arrays of unequal length.
            "chs_info": [
                {"ch_name": name, "kind": "seeg"}
                for name in ("LA1", "LA2", "LA5", "LB3", "LB7")
            ],
        },
    ),
    ("SAMBA", ["chs_info", "n_outputs"], {"n_times": 128, "sfreq": 128.0}),
    ("MEDFormer", ["n_chans", "n_outputs", "n_times"], None),
    ("MEGGPT", ["n_chans", "n_outputs"], None),
    ("MIRepNet", ["n_chans", "n_outputs"], None),
    (
        "TFMTokenizer",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {"sfreq": 200.0, "n_times": 1000},
    ),
    ("STEEGFormer", ["n_chans", "n_outputs", "n_times"], None),
    # 1 s of the 128 Hz low-frequency stream
    (
        "CORTEG",
        ["chs_info", "n_outputs", "n_times"],
        # 2 of 8 blocks; the high-gamma stream merges at the last one
        {"n_times": 128, "sfreq": 128.0, "depth": 2, "n_merge_blocks": 1},
    ),
    (
        "MVPFormer",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {"sfreq": 100.0, "n_times": 2000},
    ),
    (
        "REVE",
        ["n_times", "n_outputs", "n_chans", "chs_info"],
        {
            "sfreq": 200.0,
            "n_chans": 19,
            "n_times": 1_000,
            "depth": 2,  # 2 of 22 blocks
            "chs_info": [{"ch_name": f"E{i + 1}", "kind": "eeg"} for i in range(19)],
        },
    ),
    ("CBraMod", ["n_outputs"], {"n_chans": 3, "n_times": 1000}),
    (
        "CSBrain",
        ["n_outputs"],
        {"n_chans": 3, "n_times": 1000, "head_hidden_dim": 800},
    ),
    (
        "CodeBrain",
        ["n_chans", "n_outputs", "n_times"],
        # 5 patches; test_foundation_models runs the pretraining 19 x 6000 geometry
        {"n_chans": 19, "n_times": 1000},
    ),
    ("DGCNN", ["n_chans", "n_outputs", "n_times", "chs_info"], None),
    ("GCNsNet", ["n_chans", "n_outputs", "n_times"], None),
    ("BrainOmni", ["chs_info", "n_outputs", "n_times", "sfreq"], None),
    # 0.5 s at 250 Hz, the pretraining window
    (
        "BBL",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {"n_times": 125, "sfreq": 250.0},
    ),
    ("BrainTokenizer", ["chs_info", "n_times", "sfreq"], None),
    (
        "MEGXL",
        ["chs_info", "n_outputs", "n_times"],
        # 10 s at the released 50 Hz (42 tokens); 2 of the 8 released blocks.
        {"sfreq": 50.0, "n_times": 500, "num_layers": 2},
    ),
    (
        "DIVER1",
        ["chs_info", "n_outputs", "n_times"],
        {"sfreq": 500.0},
    ),
    ("EEGDINO", ["n_chans", "n_outputs", "n_times"], None),
    (
        "NeuroRVQ",
        ["n_chans", "n_outputs", "n_times", "sfreq"],
        {
            "n_chans": 3,
            "n_times": 600,
            "sfreq": 200.0,
            "chs_info": [
                {"ch_name": name, "kind": "eeg"} for name in ("F3", "F4", "Cz")
            ],
        },
    ),
    (
        "NeuroRVQTokenizer",
        ["n_chans", "n_outputs", "n_times", "sfreq", "chs_info"],
        {
            "n_chans": 3,
            "n_times": 200,
            "sfreq": 200.0,
            "encoder_depth": 2,  # 2 of the 12 released encoder blocks
            "decoder_depth": 2,  # 2 of the 3 released decoder blocks
            "chs_info": [
                {"ch_name": name, "kind": "eeg"} for name in ("F3", "F4", "Cz")
            ],
        },
    ),
    (
        "DANCE",
        ["n_outputs", "n_chans", "n_times", "sfreq", "chs_info"],
        _dance_signal_params,
    ),
    (
        "ZUNA",
        ["chs_info", "n_outputs", "n_times"],
        {
            "n_times": 256,  # 1 s: 8 patches per channel
            "sfreq": 256.0,
            "input_window_seconds": 1.0,
            # 0.4M parameters instead of the released 172M: scripting and
            # serializing the full model took 5.6 GB per test worker.
            "dim": 128,
            "n_layers": 2,
            "n_heads": 2,
        },
    ),
    (
        "SeizureTransformer",
        ["n_chans", "n_outputs", "n_times"],
        {
            "n_chans": 19,
            "n_times": 1024,  # 4 s @ 256 Hz
            "sfreq": 256.0,
            "num_layers": 2,  # 2 of 8 Transformer layers
        },
    ),
    (
        "DeepSOZ",
        ["n_chans", "n_outputs"],
        # 10 one-second tokens of the released model's 19 channels at 256 Hz
        {"n_chans": 19, "n_times": 2560, "sfreq": 256.0},
    ),
    ("Guetschel2026", ["chs_info", "n_outputs", "n_times"], {"sfreq": 200.0}),
    (
        "FNIRSNet",
        ["n_chans", "n_outputs", "n_times"],
        {"n_chans": 4, "n_times": 60, "sfreq": 5.0},
    ),
    (
        "FNIRSCNN1D",
        ["n_chans", "n_outputs"],
        {"n_chans": 4, "n_times": 60, "sfreq": 5.0},
    ),
    (
        "FNIRS4D",
        ["n_chans", "n_outputs", "n_times"],
        lambda: {"chs_info": _fnirs_chs_info(2), "n_times": 60, "sfreq": 5.0},
    ),
    (
        "FNIRST",
        ["n_chans", "n_outputs", "n_times"],
        # 5 pairs: the authors' spatial kernel spans 5 HbO/HbR pairs
        lambda: {"chs_info": _fnirs_chs_info(5), "n_times": 60, "sfreq": 5.0},
    ),
    (
        "FNIRSPreT",
        ["n_chans", "n_outputs", "n_times"],
        # 5 pairs: the authors' spatial kernel spans 5 HbO/HbR pairs
        lambda: {"chs_info": _fnirs_chs_info(5), "n_times": 60, "sfreq": 5.0},
    ),
    (
        "GeoNIRS",
        ["chs_info", "n_outputs", "n_times"],
        lambda: {"chs_info": _fnirs_chs_info(2), "n_times": 150, "sfreq": 5.0},
    ),
    (
        "MtReNIRS",
        ["chs_info", "n_outputs", "n_times"],
        lambda: {"chs_info": _fnirs_chs_info(2), "n_times": 100, "sfreq": 10.0},
    ),
    (
        "JungMAE",
        ["n_chans", "n_outputs", "n_times"],
        lambda: {"chs_info": _fnirs_chs_info(2), "n_times": 150, "sfreq": 5.0},
    ),
]


def _modalities(cell: str) -> set[str]:
    """The modalities of a ``summary.csv`` Modality cell, such as ``"EEG, iEEG"``."""
    return {m.strip() for m in cell.split(",")}


def get_summary_table(dir_name=None):
    import pandas as pd

    # Package data rather than a file path, so a zip archive works too.
    root = files(__package__) if dir_name is None else Path(dir_name)
    with root.joinpath("summary.csv").open("rb") as table:
        return pd.read_csv(table, index_col="Model", skipinitialspace=True)


def _is_real_scalar(val: Any) -> bool:
    return isinstance(val, (int, float, np.number)) and not isinstance(
        val, (bool, np.bool_)
    )


def _extract_xyz_row(loc: Any) -> Optional[np.ndarray]:
    if isinstance(loc, np.ndarray):
        if loc.ndim == 1 and loc.size >= 3 and loc.dtype.kind in "iuf":
            return np.asarray(loc[:3], dtype=np.float32)
        return None
    if isinstance(loc, (list, tuple)) and len(loc) >= 3:
        head = loc[:3]
        if all(_is_real_scalar(v) for v in head):
            return np.asarray(head, dtype=np.float32)
    return None


def extract_channel_locations_from_chs_info(
    chs_info: Optional[Sequence[Dict[str, Any]]],
    num_channels: Optional[int] = None,
    fill_missing: bool = False,
) -> Optional[np.ndarray]:
    """Extract 3D channel locations from MNE-style channel information.

    This function provides a unified approach to extract 3D channel locations
    from MNE channel information. It's compatible with models like SignalJEPA
    and LUNA that need to work with channel spatial information.

    Parameters
    ----------
    chs_info : list of dict or None
        Channel information, typically from ``mne.Info.chs``. Each dict should
        contain a 'loc' key with a 12-element array (MNE format) where indices 0:3
        represent the 3D cartesian coordinates.
    num_channels : int or None
        If specified, only extract the first ``num_channels`` channel locations.
        If None, extract all available channels.
    fill_missing : bool
        If False (default), extraction stops at the first channel whose location
        is missing or malformed, so the result may be shorter than requested,
        and an all-zero montage yields None. If True, each such channel
        contributes a row of ``NaN`` instead, an all-zero montage yields all
        ``NaN``, and the result always has one row per requested channel.

    Returns
    -------
    channel_locations : np.ndarray of shape (n_channels, 3) or None
        Array of 3D channel locations in cartesian coordinates. Returns None if
        no valid locations are found, unless ``fill_missing=True``, in which
        case missing locations are reported as ``NaN`` rows and None is only
        returned when ``chs_info`` is None or no channel was requested.

    Notes
    -----
    - This function handles both 12-element MNE location format (using indices 0:3)
      and 3-element location format (using directly).
    - Invalid or missing locations cause extraction to stop at that point, unless
      ``fill_missing=True``.
    - An all-zero montage is treated as no montage at all, which is also how
      :func:`has_valid_locations` reads it.
    - Returns None if no valid locations can be extracted, except with
      ``fill_missing=True`` as described above.
    - This is a unified utility compatible with models like SignalJEPA and LUNA.

    Examples
    --------
    >>> import mne
    >>> from braindecode.models.util import extract_channel_locations_from_chs_info
    >>> raw = mne.io.read_raw_edf("sample.edf")
    >>> locs = extract_channel_locations_from_chs_info(raw.info['chs'], num_channels=22)
    >>> print(locs.shape)
    (22, 3)
    """
    if chs_info is None:
        return None

    locations = []
    n_to_extract = num_channels if num_channels is not None else len(chs_info)

    for i in range(n_to_extract):
        ch_info = chs_info[i] if i < len(chs_info) else None
        loc = ch_info.get("loc") if isinstance(ch_info, dict) else None
        row = _extract_xyz_row(loc)
        if row is not None:
            coordinates = row
        elif fill_missing:
            coordinates = np.full(3, np.nan, dtype=np.float32)
        else:
            break

        locations.append(coordinates)

    if len(locations) == 0:
        return None

    result = np.stack(locations, axis=0)

    # Check positions are not all zero / degenerate
    if np.allclose(result, 0):
        if not fill_missing:
            return None
        # An all-zero montage carries no position, just like an absent one
        result = np.full_like(result, np.nan)

    return result


def resolve_channel_indices(
    chs_info: Sequence[Dict[str, Any]],
    channel_names: Sequence[str],
    *,
    montage: Optional[str] = None,
    metric: str | Callable = "cosine",
) -> Optional[list[int]]:
    """Resolve input channels into an ordered, case-insensitive vocabulary.

    Exact names take precedence over positions, as in the pretrained channel
    embeddings of SignalJEPA and LaBraM. Optionally, map unknown names to the
    nearest angular standard-montage site. This selects embedding indices; it
    does not interpolate signals or change the supplied channel information.

    Parameters
    ----------
    chs_info : sequence of dict
        MNE channel metadata. Positional fallback requires finite, nonzero
        ``loc[:3]`` in metres in the head frame. Missing ``coord_frame`` means
        head coordinates (for hand-built dictionaries); explicit non-head
        frames are rejected because channel dictionaries lack fiducials needed
        to transform them. Known names do not require valid positions.
    channel_names : sequence of str
        Ordered embedding vocabulary. Multiple inputs may select the same slot.
    montage : str | None
        Standard MNE montage supplying reference sites for unknown names.
        Native montage coordinates are transformed to head coordinates with
        MNE's public fiducial transform. None enables name-only resolution.
    metric : str | callable
        Distance metric accepted by :func:`sklearn.metrics.pairwise_distances`.
        Applied to sphere-centred unit directions, not raw electrode positions.
        The default, "cosine", preserves nearest-angular-site matching. A
        callable takes two direction vectors and returns their distance.
        Used only for positional fallback; exact names always take precedence.

    Returns
    -------
    list of int | None
        One index per input channel, or None if any channel cannot be resolved.
        The caller chooses whether to raise or use a model-specific fallback.

    Notes
    -----
    Angular distances are measured about a least-squares sphere centre fitted
    to the full reference montage, not the selected vocabulary or input subset.
    This removes radius dependence without making the mapping depend on which
    input channels are present. Nearest-site assignment is a heuristic, not an
    exact correspondence to a pretrained sensor layout.
    """
    name_to_index = {name.upper(): i for i, name in enumerate(channel_names)}
    indices = [name_to_index.get(ch.get("ch_name", "").upper()) for ch in chs_info]
    unresolved = [i for i, index in enumerate(indices) if index is None]
    if not unresolved:
        return cast(list[int], indices)
    if montage is None:
        return None

    unknown = [chs_info[i] for i in unresolved]
    if any(
        ch.get("coord_frame", FIFF.FIFFV_COORD_HEAD)
        not in (FIFF.FIFFV_COORD_HEAD, "head")
        for ch in unknown
    ):
        return None
    locations = extract_channel_locations_from_chs_info(unknown)
    if (
        locations is None
        or len(locations) != len(unknown)
        or not np.isfinite(locations).all()
        or np.any(np.linalg.norm(locations, axis=1) == 0)
    ):
        return None

    standard = mne.channels.make_standard_montage(resolve_montage_name(montage))
    standard.apply_trans(
        mne.channels.compute_native_head_t(standard, on_missing="raise")
    )
    positions = {
        name.upper(): xyz for name, xyz in standard.get_positions()["ch_pos"].items()
    }
    site_indices = [
        i for i, name in enumerate(channel_names) if name.upper() in positions
    ]
    if not site_indices:
        return None
    sites = np.array([positions[channel_names[i].upper()] for i in site_indices])
    points = np.array(list(positions.values()))
    design = np.c_[2 * points, np.ones(len(points))]
    solution, *_ = np.linalg.lstsq(design, (points**2).sum(axis=1), rcond=None)
    centre = solution[:3]
    radial = np.concatenate([locations, sites]) - centre
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-7):
        return None
    directions = radial / norms
    from sklearn.metrics import pairwise_distances_argmin

    nearest = pairwise_distances_argmin(
        directions[: len(locations)], directions[len(locations) :], metric=metric
    )
    for i, j in zip(unresolved, nearest):
        indices[i] = site_indices[int(j)]
    return cast(list[int], indices)


def positions_from_chs_info(chs_info):
    """``(n_chans, 2)`` electrode xy normalized to ``[0, 1]`` per axis.

    Takes the raw 3D sensor coordinates ``ch["loc"][:3]``, keeps ``xy`` and
    min-max normalizes each axis to ``[0, 1]`` -- the upstream brainmagick/DANCE
    convention (``dance/example/data.py``). This is **not** MNE's
    ``_find_topomap_coords`` projection (which returns an interpolated scalp grid
    and would change the merger softmax). The ``1e-9`` floor avoids a
    divide-by-zero on a degenerate (constant) axis.

    Parameters
    ----------
    chs_info : list of dict or torch.Tensor
        MNE-style channel info dicts, each with a ``"loc"`` array whose first
        three entries are the head-frame ``x, y, z`` coordinates, or a floating
        coordinate tensor of shape ``(n_chans, 3)`` in consistent units.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        ``(n_chans, 2)`` float positions in ``[0, 1]``. Dictionary inputs retain
        the NumPy float64 result. Tensor inputs preserve device and dtype and
        support autograd and TorchScript. This does not return raw xyz or
        convert metres to millimetres.
    """
    if isinstance(chs_info, torch.Tensor):
        xy = chs_info[:, :2]
        # Preserve the floor and backward intermediates at small half spans.
        if xy.dtype == torch.float16 or xy.dtype == torch.bfloat16:
            xy = xy.float()
        mn, mx = xy.amin(dim=0), xy.amax(dim=0)
        return ((xy - mn) / (mx - mn).clamp_min(1e-9)).to(chs_info.dtype)
    xyz = np.array([ch["loc"][:3] for ch in chs_info], dtype=float)
    xy = xyz[:, :2]
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    return (xy - mn) / np.maximum(mx - mn, 1e-9)


def has_valid_locations(chs_info) -> bool:
    """Whether a montage has finite, not approximately all-zero locations.

    ``chs_info`` accepts MNE channel dictionaries or a coordinate tensor of
    shape ``(n_chans, 3)``. Coordinates retain their input units. The result is
    a montage-wide Python bool, not a per-channel mask; zero rows are allowed
    when another row is nonzero. The all-zero tolerance is ``1e-8``.
    For a TorchScript-compatible per-row check, use :func:`valid_location_mask`.
    """
    if isinstance(chs_info, torch.Tensor):
        return bool(
            chs_info.numel() > 0
            and torch.isfinite(chs_info).all()
            and chs_info.abs().max().item() > 1e-8
        )
    if not isinstance(chs_info, (list, tuple)) or len(chs_info) == 0:
        return False
    rows = []
    for ch in chs_info:
        if not isinstance(ch, dict):
            return False
        row = _extract_xyz_row(ch.get("loc"))
        if row is None:
            return False
        rows.append(row.astype(float))
    xyz = np.stack(rows, axis=0)
    if xyz.size == 0 or not np.isfinite(xyz).all():
        return False
    if np.allclose(xyz, 0.0):
        return False
    return True


def valid_location_mask(locations: torch.Tensor) -> torch.Tensor:
    """Flag finite, exactly nonzero coordinate rows without changing their units.

    Parameters
    ----------
    locations : torch.Tensor
        Coordinates of shape ``(..., 3)``, in any consistent coordinate frame
        and units (e.g. xyz in metres or millimetres, not normalized xy).

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(..., 1)`` on the input device. Unlike
        :func:`has_valid_locations`, this tests each row against exact zero,
        not an approximate montage-wide zero. Supports TorchScript.
    """
    return torch.isfinite(locations).all(dim=-1, keepdim=True) & (locations != 0).any(
        dim=-1, keepdim=True
    )


def chs_info_positions(chs_info, model_name: str) -> Optional[torch.Tensor]:
    """``(n_chans, 3)`` positions of ``chs_info``, or None when it has none.

    A ``chs_info`` with positions for some channels only raises, naming the
    others: a model must neither invent their positions nor ignore the given ones.
    """
    if chs_info is None:
        return None
    locations = torch.as_tensor(
        extract_channel_locations_from_chs_info(chs_info, fill_missing=True),
        dtype=torch.float32,
    )
    placed = valid_location_mask(locations)[:, 0].tolist()
    if not any(placed):
        return None
    if not all(placed):
        names = [ch["ch_name"] for ch, ok in zip(chs_info, placed) if not ok]
        raise ValueError(
            f"{model_name} needs the 3D position of each channel, but chs_info has "
            f"none for {names}; drop these channels, or pass the positions at each "
            "call."
        )
    return locations


def warn_if_sfreq_differs(
    model_name: str, sfreq: Optional[float], pretrained_sfreq: float
) -> None:
    """Warn when ``sfreq`` differs from the rate the released weights expect.

    A randomly initialised model may run at any rate, so this warns instead of
    raising. ``sfreq=None`` (not given) does not warn.
    """
    if sfreq is not None and abs(float(sfreq) - pretrained_sfreq) > 1e-6:
        warnings.warn(
            f"{model_name} was pretrained at {pretrained_sfreq:g} Hz, got "
            f"sfreq={sfreq}. Resample the data to {pretrained_sfreq:g} Hz to use "
            "the pretrained weights.",
            UserWarning,
            stacklevel=3,
        )


def _in_model_resample(
    model_name: str, sfreq: float, model_sfreq: int, n_times: int
) -> tuple[Resample, int]:
    """A :class:`torchaudio.transforms.Resample` from the integer ``sfreq`` to
    ``model_sfreq``, with a warning, and the resampled length of ``n_times``.

    The kernel follows from the two rates, so it stays out of the state dict and
    pretrained weights load into a resampling model unchanged.
    """
    if sfreq != round(sfreq):
        raise ValueError(f"{model_name} resamples integer rates only, got {sfreq}.")
    warnings.warn(
        f"{model_name} was trained at {model_sfreq} Hz; the {sfreq:g} Hz input is "
        "resampled inside the model.",
        UserWarning,
    )
    resample = Resample(int(sfreq), model_sfreq)
    resample.register_buffer("kernel", resample.kernel, persistent=False)
    return resample, math.ceil(n_times * model_sfreq / sfreq)  # torchaudio's length


def channel_types_from_chs_info(
    chs_info: Optional[Sequence[Dict[str, Any]]],
    num_channels: Optional[int] = None,
) -> list[str]:
    """Canonical MNE channel-type name for each channel of ``chs_info``.

    Parameters
    ----------
    chs_info : list of dict or None
        MNE-style channel info dicts, typically from ``mne.Info.chs``. Each
        ``"kind"`` entry may be an MNE FIFF integer code or a plain type string.
    num_channels : int or None
        Number of channels to report. Channels beyond ``chs_info`` yield
        ``"unknown"``, so the returned list always has this length. If None, one
        entry per ``chs_info`` channel is returned (empty when it is None).

    Returns
    -------
    list of str
        Lowercase channel-type names such as ``"eeg"``, ``"seeg"``, ``"ecog"`` or
        ``"dbs"``, and ``"unknown"`` where the kind is missing or is not one of
        the electrode kinds resolved here.

    Notes
    -----
    Only electrode kinds are resolved. An MNE ``"kind"`` code does not identify
    every channel type on its own: code 2 is shared by ``"eeg"`` and ``"csd"``,
    code 1 by ``"grad"`` and ``"mag"``, because MNE disambiguates those with
    ``"unit"`` and ``"coil_type"``. Use :func:`mne.channel_type` with a full
    :class:`mne.Info` when those channel types matter.
    """
    if num_channels is None:
        num_channels = len(chs_info) if chs_info is not None else 0

    types = []
    for i in range(num_channels):
        ch_info = chs_info[i] if chs_info is not None and i < len(chs_info) else None
        kind = ch_info.get("kind") if isinstance(ch_info, dict) else None
        if isinstance(kind, str):
            types.append(kind.lower())
        elif isinstance(kind, int):
            types.append(_KIND_TO_CH_TYPE.get(kind, "unknown"))
        else:
            types.append("unknown")
    return types


def fnirs_chromophores_from_chs_info(chs_info: Sequence[Dict[str, Any]]) -> list[str]:
    """``"hbo"``, ``"hbr"`` or ``"unknown"`` per channel, from MNE's
    ``coil_type`` (``FIFFV_COIL_FNIRS_HBO`` / ``_HBR``), else from the
    ``" hbo"`` / ``" hbr"`` suffix of its name (MNE names them ``"S1_D1 hbo"``)."""
    coils = {FIFF.FIFFV_COIL_FNIRS_HBO: "hbo", FIFF.FIFFV_COIL_FNIRS_HBR: "hbr"}
    out = []
    for ch in chs_info:
        suffix = str(ch.get("ch_name", "")).lower().rsplit(" ", 1)[-1]
        by_name = suffix if suffix in ("hbo", "hbr") else "unknown"
        out.append(coils.get(int(ch.get("coil_type") or 0), by_name))
    return out


def fnirs_channel_order(
    chs_info: Optional[Sequence[Dict[str, Any]]], channel_order: Optional[str] = None
) -> str:
    """HbO/HbR pair layout of an fNIRS montage: ``"stacked"``
    (``[HbO_1..HbO_C, HbR_1..HbR_C]``) or ``"interleaved"`` (MNE's
    ``[HbO_1, HbR_1, ...]``).

    Without ``chs_info`` it returns ``channel_order``, or ``"stacked"``. With
    ``chs_info`` every channel must be HbO or HbR, in equal numbers, and the
    ``i``-th HbO and HbR channels must share their name up to the chromophore
    suffix (``"S1_D1 hbo"`` / ``"S1_D1 hbr"``); the layout is read from it, and
    an explicit ``channel_order`` must agree.

    Raises ``ValueError`` for an unknown ``channel_order``, a ``chs_info``
    that is not paired HbO/HbR, or a ``channel_order`` it contradicts.
    """
    if channel_order not in (None, "stacked", "interleaved"):
        raise ValueError(
            "channel_order must be 'stacked', 'interleaved' or None, "
            f"got {channel_order!r}."
        )
    if not chs_info:
        return channel_order or "stacked"
    chromophores = fnirs_chromophores_from_chs_info(chs_info)
    names = [str(ch.get("ch_name", "")).rsplit(" ", 1)[0] for ch in chs_info]
    n = len(chs_info) // 2
    layouts = {"stacked": (slice(0, n), slice(n, None))}
    layouts["interleaved"] = (slice(0, None, 2), slice(1, None, 2))
    found = [
        order
        for order, (hbo, hbr) in layouts.items()
        if chromophores[hbo] == ["hbo"] * n
        and chromophores[hbr] == ["hbr"] * n
        and names[hbo] == names[hbr]
    ]
    if not found:
        raise ValueError(
            "chs_info must list paired HbO/HbR channels, stacked or interleaved, "
            "with matching names (see mne's reorder_channels); got "
            f"{list(zip(names, chromophores))}."
        )
    if channel_order is not None and channel_order not in found:
        raise ValueError(
            f"channel_order={channel_order!r} contradicts chs_info, which is "
            f"{found[0]}."
        )
    return channel_order or found[0]


def fnirs_split_hbo_hbr(
    x: torch.Tensor, channel_order: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fNIRS channels into their HbO and HbR halves.

    The one HbO/HbR split of the fNIRS models: ``x`` is
    ``(batch, n_chans, ...)`` with an even ``n_chans`` in the ``channel_order``
    given by :func:`fnirs_channel_order`. Returns ``(hbo, hbr)``, each
    ``(batch, n_chans // 2, ...)``, where ``hbo[:, i]`` and ``hbr[:, i]`` are
    the two chromophores of pair ``i``.
    """
    if channel_order == "interleaved":  # [HbO_1, HbR_1, HbO_2, HbR_2, ...]
        return x[:, 0::2], x[:, 1::2]
    if channel_order == "stacked":  # [HbO_1, ..., HbO_C, HbR_1, ..., HbR_C]
        n_pairs = x.shape[1] // 2
        return x[:, :n_pairs], x[:, n_pairs:]
    raise ValueError("channel_order must be 'stacked' or 'interleaved'.")


_SENSOR_CODE = {"eeg": 0, "mag": 1, "grad": 2}
# FIFF coil names by value, so plain-int coil types (serialized chs_info) resolve.
_COIL_NAMES = {
    int(value): name for name, value in FIFF.items() if name.startswith("FIFFV_COIL_")
}


def _coil_name(ch) -> str:
    return _COIL_NAMES.get(int(ch.get("coil_type", 0)), "")


def sensor_type_of(chs_info, index: int) -> str:
    """EEG, MAG or GRAD as released: an MEG coil is MAG iff its name has "MAG".

    So CTF/KIT axial gradiometers are GRAD although MNE types them ``mag``.
    """
    ch = chs_info[index]
    kind = ch.get("kind")
    if kind is not None and not isinstance(kind, str):
        if int(kind) == FIFF.FIFFV_EEG_CH:
            return "eeg"
        if int(kind) == FIFF.FIFFV_MEG_CH and "coil_type" in ch:
            return "mag" if "MAG" in _coil_name(ch) else "grad"
        return mne.channel_type({"chs": chs_info}, index)
    resolved_type = ch.get("ch_type", kind)
    if resolved_type is None:
        resolved_type = mne.channel_type({"chs": chs_info}, index)
    return str(resolved_type).lower()


def geometry_from_chs_info(chs_info):
    """``(pos (n_chans, 6) float32, sensor_type (n_chans,) int64)``, as released.

    Positions are centred and scaled per modality (EEG, MEG); the orientation
    is the coil x-axis for planar gradiometers, the coil normal for other MEG
    coils and zero for EEG.
    """
    types = [sensor_type_of(chs_info, index) for index in range(len(chs_info))]
    xyz = extract_channel_locations_from_chs_info(chs_info)
    if xyz is None or len(xyz) != len(chs_info) or not np.isfinite(xyz).all():
        raise ValueError(
            "chs_info lacks finite sensor positions; call raw.set_montage(...)."
        )
    unsupported = set(types) - set(_SENSOR_CODE)
    if unsupported:
        raise ValueError(
            f"Unsupported channel type(s) {sorted(unsupported)}; pass only EEG/MEG."
        )
    sensor_type = np.array([_SENSOR_CODE[t] for t in types], dtype=np.int64)

    ori = np.zeros((len(chs_info), 3))
    for index in np.flatnonzero(sensor_type > 0):
        planar = "PLANAR" in _coil_name(chs_info[index])
        axis = np.asarray(chs_info[index]["loc"], dtype=np.float64)
        axis = axis[3:6] if sensor_type[index] == 2 and planar else axis[9:12]
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError("chs_info lacks a finite MEG coil orientation.")
        ori[index] = axis

    pos = np.concatenate([xyz, ori], axis=1).astype(np.float32)
    for mask in (sensor_type == 0, sensor_type > 0):
        if mask.any():
            centred = pos[mask, :3] - pos[mask, :3].mean(axis=0, keepdims=True)
            scale = np.sqrt(3 * np.mean(np.sum(centred**2, axis=1)))
            pos[mask, :3] = centred / (scale if scale > 0 else 1.0)
    return pos, sensor_type
