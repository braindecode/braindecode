# Authors: Pierre Guetschel
#          Maciej Sliwowski
#
# License: BSD-3

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import math
import operator
import sys
import warnings
from collections import OrderedDict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from mne.utils import _soft_import
from torch.nn.parameter import UninitializedParameter
from torchinfo import ModelStatistics, summary

from braindecode.functional.patches import pad_or_crop_to_patches, patch_count
from braindecode.models import util as model_util
from braindecode.models.util import (
    _EEG_PARAMS,
    _in_model_resample,
    _is_jsonable,
    _modalities,
    build_model_config,
    resolve_config_type,
    resolve_type_kwargs,
    track_model_init_kwargs,
)
from braindecode.modules.channels import _ELECTRODES, STRATEGIES, ChannelLayer, _resolve
from braindecode.modules.layers import Ensure4d
from braindecode.modules.parametrization import make_parametrizations_scriptable
from braindecode.version import __version__

huggingface_hub = _soft_import(
    "huggingface_hub", "Hugging Face Hub integration", strict=False
)

HAS_HF_HUB = huggingface_hub is not False
log = logging.getLogger(__name__)

# Set while from_pretrained runs: load_state_dict then records on the model
# which parameters came from the checkpoint (``_pretrained_provenance``).
_RECORD_LOADED_KEYS = contextvars.ContextVar("_RECORD_LOADED_KEYS", default=False)

# __init__ args that from_pretrained fills from config.json when not given.
_GEOMETRY_KWARGS = (
    "n_outputs",
    "n_chans",
    "chs_info",
    "n_times",
    "input_window_seconds",
    "sfreq",
)

# Channel strategies without learned or fitted state: adapt_input keeps the
# state_dict keys ("source" only with trainable=False).
_ADAPT_STRATEGIES: tuple[str, ...] = tuple(
    s for s in STRATEGIES if s not in ("wiener", "latent")
)


def _validate_weights_filename(filename):
    """Accept portable relative weight paths, including nested subfolders."""
    if (
        not isinstance(filename, str)
        or not filename
        or "\x00" in filename
        or "\\" in filename
        or PurePosixPath(filename).is_absolute()
        or PureWindowsPath(filename).drive
        or any(part in ("", ".", "..") for part in filename.split("/"))
    ):
        raise ValueError("filename must be a relative weight path without traversal")


def _load_card_data(pretrained_model_name_or_path, **kwargs):
    """Metadata of a checkpoint's model card (``README.md``), or None.

    The card is read only where it is already on disk, with no network request
    of its own: in the checkpoint folder, or in the local Hugging Face cache
    (``cache_dir`` and ``revision`` as given to ``from_pretrained``). For a Hub
    checkpoint, ``from_pretrained`` downloads ``README.md`` with the weights
    (``huggingface_hub.snapshot_download``), unless ``local_files_only`` or
    ``HF_HUB_OFFLINE`` is set. When the card is not there, the result is None and
    the weights load as usual.
    """
    local_dir = Path(pretrained_model_name_or_path)
    if local_dir.is_dir():
        card = local_dir / "README.md"
        if not card.is_file():
            return None
    else:
        cached = huggingface_hub.try_to_load_from_cache(
            str(pretrained_model_name_or_path),
            "README.md",
            cache_dir=kwargs.get("cache_dir"),
            revision=kwargs.get("revision"),
        )
        # A path when cached; None, or a marker for "known absent", otherwise.
        if not isinstance(cached, str):
            return None
        card = Path(cached)
    import yaml

    try:
        return huggingface_hub.ModelCard.load(card, ignore_metadata_errors=True).data
    except yaml.YAMLError:  # an unparsable YAML header must not block the weights
        return None


def _checkpoint_source(pretrained_model_name_or_path, **kwargs):
    """Repository (or folder) and revision a checkpoint was loaded from.

    The commit is read from the local Hub cache, without network access.
    """
    source = str(pretrained_model_name_or_path)
    if Path(source).is_dir():
        return {"repo_id": source, "local": True}
    info = {"repo_id": source, "revision": kwargs.get("revision")}
    for filename in (
        kwargs.get("filename") or "model.safetensors",
        "pytorch_model.bin",
        "config.json",
    ):
        cached = huggingface_hub.try_to_load_from_cache(
            source,
            filename,
            cache_dir=kwargs.get("cache_dir"),
            revision=kwargs.get("revision"),
        )
        if isinstance(cached, str):
            snapshot = Path(cached).parent
            if snapshot.parent.name == "snapshots":
                info["commit"] = snapshot.name
                break
    return info


def _local_weights_path(model_id, filename):
    """Reject local weight paths (including symlinks) outside the model folder."""
    root = Path(model_id).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Weights file must resolve inside the model directory")
    return path


def _channel_dicts(target: list) -> list[dict]:
    """Channel dicts of a montage given as dicts or names."""
    return [
        c if isinstance(c, dict) else {"ch_name": c, "kind": "eeg", "loc": np.zeros(12)}
        for c in target
    ]


@functools.lru_cache(maxsize=None)
def _forward_param_names(cls: type) -> tuple[str, ...]:
    """Ordered input parameter names of ``cls.forward`` (excluding ``self``)."""
    params = list(inspect.signature(cls.forward).parameters.values())
    return tuple(
        p.name
        for p in params[1:]
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    )


_HF_INSTALL_HINT = (
    "requires the `huggingface_hub` package. "
    "Install with: pip install 'braindecode[hub]'"
)


class _BaseHubMixinStub:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ImportError(f"{cls.__name__}.from_pretrained() {_HF_INSTALL_HINT}")

    def push_to_hub(self, *args, **kwargs):
        raise ImportError(f"{type(self).__name__}.push_to_hub() {_HF_INSTALL_HINT}")


# Define base class for hub mixin and docstring metaclass
if TYPE_CHECKING:
    _BaseHubMixin = _BaseHubMixinStub
    _DocstringMetaBase = type
else:
    _BaseHubMixin = (
        huggingface_hub.PyTorchModelHubMixin if HAS_HF_HUB else _BaseHubMixinStub
    )
    _DocstringMetaBase = NumpyDocstringInheritanceInitMeta


def deprecated_args(obj, *old_new_args):
    out_args = []
    for old_name, new_name, old_val, new_val in old_new_args:
        if old_val is None:
            out_args.append(new_val)
        else:
            warnings.warn(
                f"{obj.__class__.__name__}: {old_name!r} is depreciated. Use {new_name!r} instead."
            )
            if new_val is not None:
                raise ValueError(
                    f"{obj.__class__.__name__}: Both {old_name!r} and {new_name!r} were specified."
                )
            out_args.append(old_val)
    return out_args


def deprecated_forward_kwargs(*, removed_in="2.3", old_positional=None, **aliases):
    """Keep the old ``forward`` arguments of braindecode 2.0 working until ``removed_in``.

    Each alias maps an old keyword to its new name, or to ``(new_name, convert)``
    when the value changes meaning too (``convert`` is skipped for ``None``).
    ``old_positional`` names the arguments after ``x`` in their 2.0 order; it is
    used when the argument after ``x`` is a tensor or ``None``, i.e. a 2.0
    positional call. The decorated ``forward`` is returned unchanged, so
    TorchScript, ``torch.export`` and ``torch.compile`` see only the new names:
    :meth:`EEGModuleMixin.__call__` maps the old arguments, with a
    ``FutureWarning``, before the call reaches ``forward``. Calling
    ``model.forward`` directly takes the new names only.
    """

    def mark(forward):
        forward._deprecated_kwargs = (removed_in, aliases, old_positional)
        return forward

    return mark


def deprecated_init_flags(*, removed_in="2.3", identity_head=(), **flags):
    """Keep the constructor feature flags of braindecode 2.0 working until ``removed_in``.

    Each flag maps to its 2.0 output, read from ``model(x, return_features=True)``:
    a key, a tuple or a dict of keys, or a function of that dict (``"logits"``
    falls back to ``model(x)``). Passing a flag warns; when it is true, calling
    the model returns that output, and the last true flag wins. A true flag
    listed in ``identity_head`` also replaces ``final_layer`` by
    ``nn.Identity``, as 2.0 did, so its checkpoints load; :meth:`reset_head`
    puts a head back and clears the flag. Eager only: ``torch.jit.script``
    refuses a model built with a true flag.
    """

    def mark(cls):
        cls._deprecated_init_flags = (removed_in, flags, frozenset(identity_head))
        return cls

    return mark


def _forward_deprecation(cls):
    """The old ``forward`` arguments that ``cls`` still maps, or ``None``.

    Read from the nearest decorated ``forward`` in the MRO, and kept only where
    ``cls``'s own ``forward`` does not take the old name and takes the new one
    (or ``**kwargs``): a subclass whose ``forward`` still says ``input`` keeps it.
    """
    owner = next((k for k in cls.__mro__ if "forward" in vars(k)), None)
    decorated = next(
        (
            vars(k)["forward"]
            for k in cls.__mro__
            if hasattr(vars(k).get("forward"), "_deprecated_kwargs")
        ),
        None,
    )
    if owner is None or decorated is None:
        return None
    removed_in, aliases, old_positional = decorated._deprecated_kwargs
    params = inspect.signature(vars(owner)["forward"]).parameters
    takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
    kept = {
        old: new
        for old, new in aliases.items()
        if old not in params
        and (takes_any or (new[0] if isinstance(new, tuple) else new) in params)
    }
    positional = old_positional if vars(owner)["forward"] is decorated else None
    return (removed_in, kept, positional) if kept or positional else None


def _warn_deprecated_argument(
    owner, old, new, convert=None, *, removed_in="2.3", note="", stacklevel=3
):
    """``FutureWarning`` for an old argument, silent under ``torch.compile``."""
    if torch.compiler.is_compiling():
        return
    value = f"{convert.__name__}({old})" if convert is not None else "..."
    warnings.warn(
        f"Calling {owner} with {old}=... is deprecated and will be removed in "
        f"braindecode {removed_in}; pass {new}={value} instead.{note}",
        FutureWarning,
        stacklevel=stacklevel,
    )


def _map_deprecated_args(name, args, kwargs, removed_in, aliases, old_positional):
    """Map 2.0 positional order and old keywords to the new ones, with a warning."""
    if (
        old_positional
        and len(args) > 1
        and (args[1] is None or torch.is_tensor(args[1]))
    ):
        extra = args[1:]
        if len(extra) > len(old_positional):
            raise TypeError(
                f"{name} takes at most {len(old_positional) + 1} positional "
                f"arguments in the 2.0 order, got {len(args)}."
            )
        for key in old_positional[: len(extra)]:
            if key in kwargs:
                raise TypeError(f"{name} got multiple values for argument {key!r}.")
        kwargs = {**kwargs, **dict(zip(old_positional, extra))}
        args = args[:1]
    renames = [old for old in aliases if old in kwargs]
    if not renames:
        return args, kwargs
    kwargs = dict(kwargs)
    for old in renames:
        new, convert = (
            aliases[old] if isinstance(aliases[old], tuple) else (aliases[old], None)
        )
        if new in kwargs:
            raise TypeError(f"{name} got both {old!r} (deprecated) and {new!r}.")
        value = kwargs.pop(old)
        kwargs[new] = value if convert is None or value is None else convert(value)
        _warn_deprecated_argument(
            name, old, new, convert, removed_in=removed_in, stacklevel=4
        )
    return args, kwargs


class _BraindecodeDocstringMeta(_DocstringMetaBase):
    """Defer ``__init__`` wrapping until after docstring inheritance.

    ``NumpyDocstringInheritanceInitMeta`` uses ``inspect.unwrap()``
    internally, which bypasses ``@wraps`` wrappers.  By wrapping
    ``__init__`` *after* docstring processing, the metaclass sees the
    unwrapped function and correctly inherits ``cls.__doc__``.
    """

    def __init__(cls, class_name, class_bases, class_dict, **kwargs):
        super().__init__(class_name, class_bases, class_dict)
        # Only wrap subclass __init__s, not EEGModuleMixin itself.
        # Wrapping the mixin would cause super().__init__() calls to
        # overwrite _braindecode_init_kwargs captured by the subclass.
        if any(isinstance(b, _BraindecodeDocstringMeta) for b in class_bases):
            track_model_init_kwargs(cls)
        # Once per class: __call__ only reads a plain attribute (torch.compile).
        cls._forward_deprecation = _forward_deprecation(cls)


def _new_head_follows_model(reset_head):
    """Wrap a ``reset_head`` so that the modules it creates take the model's
    floating dtype, device and train/eval mode (a new ``nn.Linear`` is float32
    on the CPU in training mode, whatever the model)."""

    @functools.wraps(reset_head)
    def wrapper(self, *args, **kwargs):
        n_outputs = args[0] if args else kwargs.get("n_outputs")
        # numpy and torch integers count; 0 is left to overrides (Labram, LUNA)
        if not hasattr(n_outputs, "__index__") or operator.index(n_outputs) < 0:
            raise ValueError(
                "n_outputs must be positive (a non-negative integer for heads "
                f"that allow 0); got {n_outputs!r}."
            )
        ref = next((p for p in self.parameters() if p.is_floating_point()), None)
        # Hold the old tensors until the end, so that their ids cannot be
        # reused by the new ones.
        old = {id(t): t for t in [*self.parameters(), *self.buffers()]}
        # Popped here, so an override calling super() sees a model with a head.
        head = self.__dict__.pop("_head_spec", None)
        if head is not None:  # headless build: put back its head, then resize it
            self.final_layer = head[0]
        done = False
        try:
            out = reset_head(self, *args, **kwargs)
            done = True
        finally:
            if head is not None and not done:  # refused: the model stays headless
                self._head_spec, self.final_layer = head, torch.nn.Identity()
        flag = self.__dict__.pop("_legacy_head_flag", None)
        if flag is not None:  # as in 2.0, a new head clears that flag (not a refusal)
            self._update_init_kwargs(**{flag: False})
            self._legacy_flag = self._true_legacy_flag()
        if ref is not None:
            for module in self.modules():
                tensors = [*module.parameters(), *module.buffers()]
                if tensors and not any(id(t) in old for t in tensors):
                    module.to(device=ref.device, dtype=ref.dtype)
                    module.train(self.training)
        provenance = self.__dict__.get("_pretrained_provenance")
        if provenance and provenance.get("loaded") is not None:
            # The new head was not loaded from the checkpoint.
            current = dict(self.named_parameters(remove_duplicate=False))
            provenance["loaded"] = {
                name
                for name in provenance["loaded"]
                if name in current and id(current[name]) in old
            }
        return out

    return wrapper


class EEGModuleMixin(_BaseHubMixin, metaclass=_BraindecodeDocstringMeta):
    """
    Mixin class for all EEG models in braindecode.

    This class integrates with Hugging Face Hub when the ``huggingface_hub`` package
    is installed, enabling models to be pushed to and loaded from the Hub using
    :func:`push_to_hub()` and :func:`from_pretrained()` methods.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model. This is the number of classes
        in the case of classification.
    n_chans : int
        Number of EEG channels.
    chs_info : list of dict
        Information about each individual EEG channel. This should be filled with
        ``info["chs"]``. Refer to :class:`mne.Info` for more details.
    n_times : int
        Number of time samples of the input window.
    input_window_seconds : float
        Length of the input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recordings.
    channel_strategy : str, default="native"
        How any montage reaches the backbone (pretrained models only; see
        :ref:`channel-strategies`). ``"native"`` keeps the model as
        it is. ``"exact"``, ``"zero"``, ``"nearest"``, ``"idw"``, ``"spline"``,
        ``"field"``, ``"source"``, ``"region"``, ``"wiener"`` (call
        ``model.channel_layer.fit`` first) or ``"latent"`` map the montage of ``chs_info`` (or of the
        ``chs_info`` given to ``forward``) onto the backbone's channels with a
        :class:`~braindecode.modules.ChannelLayer`. Saved in the config.
        ``model(x)`` and ``model.forward(x)`` both apply the layer.
    channel_strategy_kwargs : dict or None, default=None
        Options of the strategy (e.g. ``{"reg": 1e-2}`` for ``"spline"``).

    Raises
    ------
    ValueError: If some input signal-related parameters are not specified
                and can not be inferred.

    Notes
    -----
    If some input signal-related parameters are not specified,
    there will be an attempt to infer them from the other parameters.
    """

    #: Hub model card metadata (:class:`huggingface_hub.ModelCardData`) of the
    #: checkpoint, set by :meth:`from_pretrained` from the checkpoint folder,
    #: the local Hub cache, or a download made with the weights (unless offline),
    #: e.g. ``model.card_data.datasets`` lists the data the weights saw.
    card_data = None

    #: Template for model-specific Hub integration notes appended by
    #: ``__init_subclass__``.  ``{name}`` is replaced with the concrete
    #: model class name.
    _HUB_NOTES_TEMPLATE = """
    .. rubric:: Hugging Face Hub integration

    When the optional ``huggingface_hub`` package is installed, all models
    automatically gain the ability to be pushed to and loaded from the
    Hugging Face Hub. Install with::

        pip install braindecode[hub]

    **Pushing a model to the Hub:**

    .. code-block::

        from braindecode.models import {name}

        # Train your model
        model = {name}(n_chans=22, n_outputs=4, n_times=1000)
        # ... training code ...

        # Push to the Hub; ``datasets`` and ``base_model`` go to the card
        model.push_to_hub(
            repo_id="username/my-{name_lower}-model",
            commit_message="Initial model upload",
            model_card_kwargs={{"datasets": ["BNCI2014_001"]}},
        )

    **Loading a model from the Hub:**

    .. code-block::

        from braindecode.models import {name}

        # Load pretrained model
        model = {name}.from_pretrained("username/my-{name_lower}-model")
        # Data the weights were trained on, from the Hub model card
        print(model.card_data.datasets if model.card_data else None)

        # Load with a different number of outputs (head is rebuilt automatically)
        model = {name}.from_pretrained("username/my-{name_lower}-model", n_outputs=4)

    **Extracting features and replacing the head:**

    .. code-block::

        import torch

        x = torch.randn(1, model.n_chans, model.n_times)
        # Extract encoder features and a mean-pooled 2-D embedding
        out = model(x, return_features=True)
        embedding = out["embedding"]  # (batch, dim)
        features = out["features"]

        # Replace the classification head
        model.reset_head(n_outputs=10)

    Note that ``torch.jit.script`` returns features only on models whose
    ``forward`` declares ``return_features``.

    **Saving and restoring full configuration:**

    .. code-block::

        import json

        config = model.get_config()            # all __init__ params
        with open("config.json", "w") as f:
            json.dump(config, f)

        model2 = {name}.from_config(config)    # reconstruct (no weights)

    All model parameters (both EEG-specific and model-specific such as
    dropout rates, activation functions, number of filters) are automatically
    saved to the Hub and restored when loading.

    See :ref:`load-pretrained-models` for a complete tutorial.
    """

    _feature_axis: ClassVar[int] = -1
    _native_features: ClassVar[bool] = False
    #: Head of a model built with ``n_outputs=None``, kept off the module tree
    #: until :meth:`reset_head` attaches it.
    _head_spec: Optional[tuple] = None

    #: Attributes that :func:`torch.jit.script` must not introspect. The
    #: signal-related properties raise :class:`ValueError` when their value was
    #: neither given nor inferable, and ``mapping`` carries a postponed
    #: annotation TorchScript cannot resolve. Either one aborts scripting
    #: before ``forward`` is ever compiled.
    __jit_ignored_attributes__ = [
        *sorted(_EEG_PARAMS),
        "_chs_info",
        "input_shape",
        "mapping",
        "_time_axis",
        "_channel_axis",
        "_feature_axis",
        "_native_features",
        "_head_spec",
        "_time_inputs",
        "_min_n_times",
        "_n_times_step",
        "_n_times_offset",
        "_warned_length_snap",
        "_on_non_divisible_explicit",
        "_snaps_in_prepare_inputs",
        "_has_ensure4d",
        "_materialized_n_chans",
        "_materialized_n_times",
        "_dense_conversion",
        "_input_sfreq",
        "_adapt_config",
    ]
    #: Rich MNE channel dictionaries are intentionally unavailable in scripted
    #: forwards; they remain unchanged on eager models and in saved configs.
    __jit_unused_properties__ = ["chs_info"]

    def __init_subclass__(cls, **kwargs):
        license = kwargs.pop("license", "bsd-3-clause")
        forward = getattr(cls, "forward", None)
        cls._native_features = (
            forward is not None
            and "return_features" in inspect.signature(forward).parameters
        )

        if "reset_head" in cls.__dict__:
            cls.reset_head = _new_head_follows_model(cls.__dict__["reset_head"])

        # TorchScript only honours ``__jit_ignored_attributes__`` for
        # properties defined directly on the concrete class: it collects them
        # with ``vars(type(module))``, which skips inherited ones. Rebinding
        # the very same property objects on each subclass makes them visible
        # there without changing any runtime behaviour.
        for name in cls.__jit_ignored_attributes__:
            if name in cls.__dict__:
                continue
            prop = getattr(cls, name, None)
            if isinstance(prop, property):
                setattr(cls, name, prop)

        # Append model-specific Hub integration notes to the docstring.
        # This runs before the metaclass __init__, so the Hub notes will
        # be included in the docstring that the metaclass processes.
        if cls.__doc__ is not None:
            hub_notes = cls._HUB_NOTES_TEMPLATE.format(
                name=cls.__name__,
                name_lower=cls.__name__.lower(),
            )
            cls.__doc__ = cls.__doc__.rstrip() + "\n" + hub_notes

        if not HAS_HF_HUB:
            super().__init_subclass__(**kwargs)
            return

        base_tags = ["braindecode", cls.__name__]
        user_tags = kwargs.pop("tags", None)
        tags = list(user_tags) if user_tags is not None else []
        for tag in base_tags:
            if tag not in tags:
                tags.append(tag)

        docs_url = kwargs.pop(
            "docs_url",
            f"https://braindecode.org/stable/generated/braindecode.models.{cls.__name__}.html",
        )
        repo_url = kwargs.pop("repo_url", "https://braindecode.org")
        library_name = kwargs.pop("library_name", "braindecode")
        # Register a coder so that type[nn.Module] parameters
        # (e.g. activation=nn.ELU) are serialized as importable
        # strings in config.json and decoded back on load.
        coders = kwargs.pop("coders", None) or {}
        coders.setdefault(
            type,
            (
                lambda t: f"{t.__module__}.{t.__qualname__}",
                resolve_config_type,
            ),
        )

        # TODO: model_card_template can be added in the future for custom model cards
        super().__init_subclass__(
            tags=tags,
            docs_url=docs_url,
            repo_url=repo_url,
            library_name=library_name,
            license=license,
            coders=coders,
            **kwargs,
        )

    def __init__(
        self,
        n_outputs: Optional[int] = None,  # type: ignore[assignment]
        n_chans: Optional[int] = None,  # type: ignore[assignment]
        chs_info=None,  # type: ignore[assignment]
        n_times: Optional[int] = None,  # type: ignore[assignment]
        input_window_seconds: Optional[float] = None,  # type: ignore[assignment]
        sfreq: Optional[float] = None,  # type: ignore[assignment]
        channel_strategy: str = "native",
        channel_strategy_kwargs: Optional[dict] = None,
    ):
        # Deserialize chs_info if it comes as a list of dicts (from Hub)
        if chs_info is not None and isinstance(chs_info, list):
            if len(chs_info) > 0 and isinstance(chs_info[0], dict):
                # Check if it needs deserialization (has 'loc' as list)
                if "loc" in chs_info[0] and isinstance(chs_info[0]["loc"], list):
                    chs_info = self._deserialize_chs_info(chs_info)
                    warnings.warn(
                        "Modifying chs_info argument using the _deserialize_chs_info() method"
                    )

        if (  # under a channel_strategy the input montage changes per call
            channel_strategy == "native"
            and n_chans is not None
            and chs_info is not None
            and len(chs_info) != n_chans
        ):
            raise ValueError(f"{n_chans=} different from {chs_info=} length")
        if (
            n_times is not None
            and input_window_seconds is not None
            and sfreq is not None
            and n_times != round(input_window_seconds * sfreq)
        ):
            raise ValueError(
                f"{n_times=} different from {input_window_seconds=} * {sfreq=}"
            )
        layer = None
        if channel_strategy != "native":  # the backbone is built on the target
            target = self._channel_target or chs_info
            if target is None:
                raise ValueError(
                    f"channel_strategy={channel_strategy!r} needs chs_info."
                )
            kwargs = channel_strategy_kwargs or {}
            layer = ChannelLayer(
                _channel_dicts(target), channel_strategy, chs_info, **kwargs
            )
            chs_info, n_chans = layer.target, None
        elif channel_strategy_kwargs:
            raise ValueError(
                "channel_strategy='native' takes no channel_strategy_kwargs."
            )

        self._input_window_seconds = input_window_seconds  # type: ignore[assignment]
        self._chs_info = chs_info  # type: ignore[assignment]
        self._n_outputs = n_outputs  # type: ignore[assignment]
        # False: built without n_outputs; True once the constructor reads it.
        self._headless_build = False if n_outputs is None else None
        self._n_chans = n_chans  # type: ignore[assignment]
        # TorchScript cannot represent the rich MNE dictionaries in _chs_info.
        # Keep the original eager state above and expose only the derived scalar
        # to scripted signal-property getters.
        self._n_chans_for_jit = (
            len(chs_info) if n_chans is None and chs_info is not None else n_chans
        )
        self._n_times = n_times  # type: ignore[assignment]
        self._sfreq = sfreq  # type: ignore[assignment]

        # Back-fill instance attributes from captured init kwargs (or
        # _hub_mixin_config) for any params the subclass didn't store on self.
        # Skip EEG params (stored as self._*) and descriptors (properties).
        init_config = (
            getattr(self, "_braindecode_init_kwargs", None)
            or getattr(self, "_hub_mixin_config", None)
            or {}
        )
        for key, val in init_config.items():
            if key in _EEG_PARAMS or isinstance(val, (torch.nn.Module, torch.Tensor)):
                continue
            if hasattr(getattr(type(self), key, None), "__get__"):
                continue
            if not hasattr(self, key):
                setattr(self, key, val)

        self._min_n_times: int | None = None
        self._n_times_step: int = 1
        self._n_times_offset: int = 0
        self._warned_length_snap: bool = False
        self._snaps_in_prepare_inputs: bool | str = True
        self._has_ensure4d: bool | None = None
        self._materialized_n_chans: int | None = None
        self._materialized_n_times: int | None = None
        self._on_non_divisible_explicit: bool = "on_non_divisible" in getattr(
            self, "_braindecode_explicit_init_keys", ()
        )
        if not hasattr(self, "on_non_divisible"):
            self.on_non_divisible = "crop"

        super().__init__()
        if layer is not None:
            self.channel_layer = layer
            # Instance attribute: model(x) and model.forward(x) both map x.
            setattr(self, "forward", self._channel_forward)

    def _channel_forward(self, *args, **kwargs):
        """``forward`` with a channel layer or an input resampling: resample the
        time inputs to ``model.sfreq``, map ``x`` (recorded with ``chs_info``,
        default the layer's) onto the backbone's channels, then run the model's
        own ``forward``."""
        if self._input_sfreq is not None:  # check and snap lengths at model.sfreq
            args, kwargs = self._replace_time_inputs(
                args, kwargs, lambda _key, t: self._on_3d(t, self.input_resample)
            )
            args, kwargs = self._prepare_inputs(args, kwargs)
        if getattr(self, "channel_layer", None) is not None:
            args, kwargs = self._map_channels(*args, **kwargs)
        return type(self).forward(self, *args, **kwargs)

    def _map_channels(self, *args, chs_info=None, ch_names=None, **kwargs):
        # pad_channels_collate / set_return_ch_pos: pos and ch_mask describe the
        # input montage of each sample; the backbone gets the full target one.
        ch_mask, pos = kwargs.pop("ch_mask", None), kwargs.pop("pos", None)
        bound = inspect.signature(type(self).forward).bind_partial(
            self, *args, **kwargs
        )
        if bound.arguments.get("ch_mask") is not None:
            raise ValueError(
                "Pass ch_mask by keyword under a channel_strategy: the channel layer "
                "uses it to map each sample, the backbone never sees it."
            )
        if chs_info is None and pos is not None and ch_mask is not None:  # no names
            chs_info = self.channel_layer.montages(pos, ch_mask)
        elif ch_mask is not None and not (chs_info and isinstance(chs_info[0], list)):
            raise ValueError(
                "ch_mask under a channel_strategy needs the montage of each sample: "
                "pos (set_return_ch_pos(True)) or chs_info as a list of montages."
            )
        if chs_info is None and ch_names is not None:  # LaBraM: names of x's channels
            chs_info = [{"ch_name": n} for n in ch_names]

        def map_x(x):
            return self._on_3d(x, lambda t: self.channel_layer(t, chs_info, ch_mask)[0])

        if args:
            args = (map_x(args[0]), *args[1:])
        else:  # x by keyword: the backbone's first input (x, X, eeg...)
            x = _forward_param_names(type(self))[0]
            kwargs[x] = map_x(kwargs[x])
        return args, kwargs

    def _on_3d(self, x: torch.Tensor, fn: Callable) -> torch.Tensor:
        """``fn`` on ``x`` as (batch, channels, time), also for the
        (batch, channels, time, 1) input of an Ensure4d model."""
        if x.ndim == 4 and x.shape[-1] == 1 and self._uses_ensure4d():
            return fn(x[..., 0])[..., None]
        return fn(x)

    def _apply(self, fn, recurse=True):
        # adapt_input keeps its layers out of the children (out of nn.Sequential
        # loops and of the last-child head); they still follow .to() and .half().
        for name in ("channel_layer", "input_resample"):
            if name in self.__dict__:
                self.__dict__[name]._apply(fn, recurse)
        return super()._apply(fn, recurse)

    def _to_model_device(self, module: torch.nn.Module) -> torch.nn.Module:
        """``module`` on the device and dtype of the model's floating tensors."""
        tensors = (*self.parameters(), *self.buffers())
        ref = next((t for t in tensors if t.is_floating_point()), None)
        return module if ref is None else module.to(ref.device, ref.dtype)

    def adapt_input(
        self,
        *,
        chs_info: list[dict] | None = None,
        sfreq: float | None = None,
        channel_strategy: str | None = None,
        channel_strategy_kwargs: dict | None = None,
    ) -> "EEGModuleMixin":
        """Adapt the built model, in place, to another montage or sampling rate.
        The ``state_dict`` keys do not change.

        Parameters
        ----------
        chs_info : list of dict | None
            Montage of the recordings, mapped onto the EEG montage the model was
            built with. With ``chs_info=None`` and a ``channel_strategy``, forward
            maps each sample from the ``pos`` and ``ch_mask`` that
            :func:`~braindecode.datasets.pad_channels_collate` returns.
        sfreq : float | None
            Integer sampling rate of the recordings, resampled inside the model to
            ``model.sfreq``. ``sfreq=model.sfreq`` removes the resampling.
        channel_strategy : str | None
            One of ``_ADAPT_STRATEGIES``. When only ``chs_info`` is given, the
            strategy of an earlier call, else ``"nearest"``.
        channel_strategy_kwargs : dict | None
            Options of ``channel_strategy``.

        Returns
        -------
        EEGModuleMixin
            The model. :meth:`get_config` saves the adaptation, and
            :meth:`from_config` and ``from_pretrained`` apply it again.
            ``model(x)`` and ``model.forward(x)`` apply it; TorchScript does not.
            The added layers are not submodules: ``.modules()`` and ``.apply()``
            do not see them.
        """
        name = type(self).__name__
        config = dict(self._adapt_config or {})
        layer = resample = None  # validate everything, then change the model
        if chs_info is not None and not (
            chs_info and all(isinstance(c, dict) for c in chs_info)
        ):
            raise ValueError(
                f"{name}.adapt_input takes chs_info as a non-empty list of channel "
                f"dicts (raw.info['chs']), got {chs_info!r:.80}."
            )
        if chs_info is not None or channel_strategy is not None:
            if channel_strategy is None:  # keep the stored strategy and its options
                channel_strategy = config.get("channel_strategy", "nearest")
                channel_strategy_kwargs = channel_strategy_kwargs or config.get(
                    "channel_strategy_kwargs"
                )
            strategy = channel_strategy
            kwargs = {  # saved in the config: numpy values as python ones
                k: v.tolist() if isinstance(v, (np.generic, np.ndarray)) else v
                for k, v in (channel_strategy_kwargs or {}).items()
            }
            bad = [k for k, v in kwargs.items() if not _is_jsonable(v)]
            if bad:
                raise ValueError(
                    f"{name}.adapt_input saves channel_strategy_kwargs in the config "
                    f"and needs JSON values; not JSON: {bad}."
                )
            if (strategy in STRATEGIES and strategy not in _ADAPT_STRATEGIES) or (
                strategy == "source" and kwargs.get("trainable")
            ):
                raise ValueError(
                    f"adapt_input takes {_ADAPT_STRATEGIES}: {strategy!r} adds "
                    "learned or fitted state to the state_dict."
                )
            if "channel_layer" in self._modules:  # registered by __init__
                raise ValueError(
                    f"{name} was built with a channel_strategy and maps any montage "
                    "already: pass the montage of x to forward (chs_info=...)."
                )
            target = self._signal_param("chs_info") or self._channel_target
            if target is None:
                raise ValueError(
                    f"{name}.adapt_input needs the montage the model was built "
                    "with (chs_info)."
                )
            target = _channel_dicts(target)
            table = model_util._summary_table
            modality = table.at[name, "Modality"] if name in table.index else "EEG"
            kinds = sorted(set(_resolve(target)[3]) - set(_ELECTRODES))
            if (
                "EEG" not in _modalities(modality)
                or kinds
                or len(self._time_inputs) > 1
            ):
                raise ValueError(
                    f"{name}.adapt_input cannot map channels: the channel layer maps "
                    f"one input of EEG electrodes, and {name} takes {modality} "
                    f"(channel kinds {kinds}, inputs {self._time_inputs})."
                )
            if chs_info is not None and isinstance(chs_info[0].get("loc"), list):
                chs_info = self._deserialize_chs_info(chs_info)  # from a config
            layer = ChannelLayer(target, strategy, chs_info, **kwargs)
            config.update(
                chs_info=chs_info,
                channel_strategy=strategy,
                channel_strategy_kwargs=kwargs or None,
            )
        elif channel_strategy_kwargs:
            raise ValueError("channel_strategy_kwargs needs a channel_strategy.")
        if sfreq is not None:
            model_sfreq = self._signal_param("sfreq")
            if model_sfreq is None or model_sfreq != round(model_sfreq):
                raise ValueError(
                    f"{name}.adapt_input(sfreq=...) needs the integer sampling rate "
                    f"the model was built with, got sfreq={model_sfreq}."
                )
            if not math.isfinite(sfreq) or sfreq <= 0:
                raise ValueError(
                    f"{name}.adapt_input needs a finite sfreq > 0, got {sfreq=}."
                )
            if sfreq != model_sfreq:
                resample = _in_model_resample(
                    name, sfreq, int(model_sfreq), self._signal_param("n_times") or 0
                )[0]
            config["sfreq"] = None if resample is None else float(sfreq)
        # Not children: see _apply.
        for key, module in (("channel_layer", layer), ("input_resample", resample)):
            if module is not None:
                self.__dict__[key] = self._to_model_device(module)
                self.forward = self._channel_forward
        if sfreq is not None:
            if resample is None:
                self.__dict__.pop("input_resample", None)
            self._input_sfreq = config["sfreq"]
        self._adapt_config = {k: v for k, v in config.items() if v is not None} or None
        return self

    @property
    def n_outputs(self) -> int:
        if self._n_outputs is None:
            if not torch.jit.is_scripting():
                if self.__dict__.get("_headless_build") is not None:
                    # Headless build: size the head for one output, then
                    # _drop_headless_head takes it off the model.
                    self._headless_build = True
                    return 1
            raise ValueError("n_outputs not specified.")
        return self._n_outputs

    def _drop_headless_head(self) -> None:
        """Replace the head of a model built with ``n_outputs=None`` by
        ``nn.Identity``, so the model has no unused head parameters."""
        if self.__dict__.pop("_headless_build", None) and isinstance(
            getattr(self, "final_layer", None), torch.nn.Module
        ):
            self._head_spec = (self.final_layer,)
            self.final_layer = torch.nn.Identity()

    @property
    def n_chans(self) -> int:
        if torch.jit.is_scripting():
            if self._n_chans_for_jit is None:
                raise ValueError(
                    "n_chans could not be inferred. Either specify n_chans or chs_info."
                )
            return self._n_chans_for_jit
        if self._n_chans is None and self._chs_info is not None:
            return len(self._chs_info)
        elif self._n_chans is None:
            if not torch.jit.is_scripting():
                materialized = getattr(self, "_materialized_n_chans", None)
                if materialized is not None:
                    return materialized
            raise ValueError(
                "n_chans could not be inferred. Either specify n_chans or chs_info."
            )
        return self._n_chans

    @property
    def chs_info(self) -> list[dict]:
        if self._chs_info is None:
            raise ValueError("chs_info not specified.")
        return self._chs_info

    @property
    def n_times(self) -> int:
        if (
            self._n_times is None
            and self._input_window_seconds is not None
            and self._sfreq is not None
        ):
            return int(round(self._input_window_seconds * self._sfreq))
        elif self._n_times is None:
            if not torch.jit.is_scripting():
                materialized = getattr(self, "_materialized_n_times", None)
                if materialized is not None:
                    return materialized
            raise ValueError(
                "n_times could not be inferred. "
                "Either specify n_times or input_window_seconds and sfreq."
            )
        return self._n_times

    @property
    def input_window_seconds(self) -> float:
        if (
            self._input_window_seconds is None
            and self._n_times is not None
            and self._sfreq is not None
        ):
            return float(self._n_times / self._sfreq)
        elif self._input_window_seconds is None:
            raise ValueError(
                "input_window_seconds could not be inferred. "
                "Either specify input_window_seconds or n_times and sfreq."
            )
        if torch.jit.is_scripting():  # may be an int, e.g. from a Hub config
            return float(self._input_window_seconds)
        return self._input_window_seconds

    @property
    def sfreq(self) -> float:
        if (
            self._sfreq is None
            and self._input_window_seconds is not None
            and self._n_times is not None
        ):
            return float(self._n_times / self._input_window_seconds)
        elif self._sfreq is None:
            raise ValueError(
                "sfreq could not be inferred. "
                "Either specify sfreq or input_window_seconds and n_times."
            )
        if torch.jit.is_scripting():  # may be an int, e.g. from a Hub config
            return float(self._sfreq)
        return self._sfreq

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Input data shape."""
        return (1, self.n_chans, self.n_times)

    def __dir__(self):
        # torch.jit.trace probes every name of dir() with hasattr(), which lets a
        # property's ValueError through: list a signal parameter only when its
        # value is stored (reading a missing one still raises the same error).
        # ponytail: values derived from others (n_chans from chs_info, sfreq from
        # n_times and the window) are left out too.
        stored = {n for n in _EEG_PARAMS if self.__dict__.get(f"_{n}") is not None}
        if {"n_chans", "n_times"} <= stored:  # input_shape reads both
            stored.add("input_shape")
        hidden = {*_EEG_PARAMS, "input_shape"} - stored
        return [n for n in super().__dir__() if n not in hidden]

    #: The true deprecated constructor flag that decides the output (deprecated_init_flags).
    _legacy_flag = None

    def __prepare_scriptable__(self):
        # Called by torch.jit.script before compiling the model.
        if self._adapt_config is not None:
            raise NotImplementedError(
                f"TorchScript does not compile {type(self).__name__}.adapt_input: "
                "use torch.export or torch.jit.trace on the adapted model."
            )
        if self._legacy_flag is not None:
            raise RuntimeError(
                f"{type(self).__name__} was built with a deprecated constructor "
                "feature flag, whose 2.0 output TorchScript does not reproduce; "
                "build it without the flag and call forward(x, return_features=True)."
            )
        return make_parametrizations_scriptable(self)

    def _apply_deprecated_init_flags(self) -> None:
        """Warn for a 2.0 constructor feature flag; remember a true one's output."""
        spec = getattr(type(self), "_deprecated_init_flags", None)
        if spec is None:
            return
        removed_in, flags, identity_head = spec
        for flag in flags:
            if flag in self._braindecode_explicit_init_keys:
                warnings.warn(
                    f"{type(self).__name__}({flag}=...) is deprecated and will be "
                    f"removed in braindecode {removed_in}; call "
                    "model(x, return_features=True) instead.",
                    FutureWarning,
                    stacklevel=3,
                )
            if flag in identity_head and self._braindecode_init_kwargs.get(flag):
                if self._head_spec is None:  # 2.0 built no head: like n_outputs=None
                    self._head_spec = (self.final_layer,)
                    self.final_layer = torch.nn.Identity()
                self._legacy_head_flag = flag
        self._legacy_flag = self._true_legacy_flag()

    def _true_legacy_flag(self):
        """The last true deprecated flag, which decides the output (as in 2.0)."""
        flags = type(self)._deprecated_init_flags[1]
        kwargs = self._braindecode_init_kwargs
        return next((f for f in reversed(list(flags)) if kwargs.get(f)), None)

    def _legacy_call(self, args, kwargs):
        """The 2.0 output of the true constructor feature flag."""
        output = type(self)._deprecated_init_flags[1][self._legacy_flag]
        kwargs = {k: v for k, v in kwargs.items() if k != "return_features"}
        out = self(*args, return_features=True, **kwargs)
        if callable(output):
            return output(out)
        if isinstance(output, str):
            return out[output]
        if isinstance(output, tuple):
            return tuple(out[k] for k in output)
        return {k: out.get(v) for k, v in output.items()}

    def _signal_param(self, name: str) -> Any | None:
        """Return signal parameter ``name`` or ``None`` if unset and not inferable."""
        if name == "n_outputs":
            return getattr(self, "_n_outputs", None)
        if name == "chs_info":
            return getattr(self, "_chs_info", None)
        if name == "n_chans":
            n_chans = getattr(self, "_n_chans", None)
            if n_chans is not None:
                return n_chans
            chs_info = getattr(self, "_chs_info", None)
            if chs_info is not None:
                return len(chs_info)
            return getattr(self, "_materialized_n_chans", None)
        if name == "n_times":
            n_times = getattr(self, "_n_times", None)
            if n_times is not None:
                return n_times
            win = getattr(self, "_input_window_seconds", None)
            sfreq = getattr(self, "_sfreq", None)
            if win is not None and sfreq is not None:
                return int(round(win * sfreq))
            return getattr(self, "_materialized_n_times", None)
        if name == "input_window_seconds":
            win = getattr(self, "_input_window_seconds", None)
            if win is not None:
                return win
            n_times = getattr(self, "_n_times", None)
            sfreq = getattr(self, "_sfreq", None)
            if n_times is not None and sfreq is not None:
                return float(n_times / sfreq)
            return None
        if name == "sfreq":
            sfreq = getattr(self, "_sfreq", None)
            if sfreq is not None:
                return sfreq
            win = getattr(self, "_input_window_seconds", None)
            n_times = getattr(self, "_n_times", None)
            if win is not None and n_times is not None:
                return float(n_times / win)
            return None
        raise KeyError(f"Unknown signal parameter {name!r}.")

    def _call_arg(self, args: tuple, kwargs: dict, name: str, default=None) -> Any:
        """Value of the ``forward`` argument ``name`` in this call (a keyword the
        signature lacks, such as ``return_features`` that ``__call__`` serves,
        is read from ``kwargs``)."""
        names = _forward_param_names(type(self))
        i = names.index(name) if name in names else len(args)
        return args[i] if len(args) > i else kwargs.get(name, default)

    def _first_input(self, args: tuple, kwargs: dict) -> torch.Tensor | None:
        """The primary signal tensor of this call, passed by position or keyword."""
        names = _forward_param_names(type(self))
        first = args[0] if args else (kwargs.get(names[0]) if names else None)
        return first if isinstance(first, torch.Tensor) else None

    def _accepts_any_n_chans(self, args: tuple, kwargs: dict) -> bool:
        """Whether this call accepts an input channel count other than ``n_chans``."""
        return getattr(self, "channel_layer", None) is not None

    def _accepts_any_n_times(self, args: tuple, kwargs: dict) -> bool:
        """Whether this call accepts a window length other than ``n_times``."""
        return False

    @staticmethod
    def _snap_to_length(x: torch.Tensor, target: int) -> torch.Tensor:
        return F.pad(x[..., :target], (0, max(0, target - x.shape[-1])))

    def _grid(self) -> tuple[int, int]:
        """First patch length and step of the length grid."""
        step = getattr(self, "_n_times_step", 1)
        offset = getattr(self, "_n_times_offset", 0)
        return (offset, step) if offset > 0 else (step, 0)

    def _snap_grid_target(self, got_times: int) -> int:
        patch, step = self._grid()
        on_non_divisible = getattr(self, "on_non_divisible", "crop")
        n_patches = patch_count(got_times, patch, on_non_divisible, step=step)
        return patch + (n_patches - 1) * (step or patch)

    def _snap_grid_tensor(self, name: str, x: torch.Tensor) -> torch.Tensor:
        patch, step = self._grid()
        on_non_divisible = getattr(self, "on_non_divisible", "crop")
        return pad_or_crop_to_patches(x, patch, on_non_divisible, step=step)

    def _replace_time_inputs(
        self, args: tuple, kwargs: dict, fn: Callable[[str, torch.Tensor], torch.Tensor]
    ) -> tuple[tuple, dict]:
        names = _forward_param_names(type(self))
        # One time input: the first forward argument, whatever its name (X, eeg...).
        time_inputs = self._time_inputs if len(self._time_inputs) > 1 else names[:1]

        def snap(name, value):
            is_time = name in time_inputs and isinstance(value, torch.Tensor)
            return fn(name, value) if is_time else value

        args = tuple(snap(n, a) for n, a in zip(names, args)) + args[len(names) :]
        return args, {k: snap(k, v) for k, v in kwargs.items()}

    def _uses_ensure4d(self) -> bool:
        cached = getattr(self, "_has_ensure4d", None)
        if cached is None:
            cached = any(isinstance(m, Ensure4d) for m in self.modules())
            self._has_ensure4d = cached
        return cached

    def _input_chans_and_times(self, x: torch.Tensor) -> tuple[int, int | None, bool]:
        if x.ndim < 3:
            return x.shape[1], None, False
        if x.ndim == 4 and x.shape[3] == 1 and self._uses_ensure4d():
            return x.shape[1], x.shape[2], False
        return x.shape[1], x.shape[-1], True

    def _check_input_shape(
        self, x: torch.Tensor, args: tuple, kwargs: dict
    ) -> str | None:
        """Validate ``x`` against declared channel/time contracts.

        Returns ``"fixed_snap"`` or ``"grid_snap"`` when trailing samples
        should be snapped in :meth:`_prepare_inputs`, or ``None`` when no
        snapping is needed.
        """
        time_axis = getattr(self, "_time_axis", None)
        channel_axis = getattr(self, "_channel_axis", None)
        if (time_axis is None and channel_axis is None) or x.ndim < 2:
            return None
        name = type(self).__name__
        got_chans, got_times, can_snap_last_dim = self._input_chans_and_times(x)

        if channel_axis == "fixed" and not self._accepts_any_n_chans(args, kwargs):
            n_chans = self._signal_param("n_chans")
            if n_chans is not None and got_chans != n_chans:
                raise ValueError(
                    f"{name} was built for n_chans={n_chans} and got {got_chans} "
                    f"channels (input shape {tuple(x.shape)}). Rebuild it with "
                    f"n_chans={got_chans}, or map the montage with "
                    "model.adapt_input(chs_info=...)."
                )

        if got_times is None:
            return None

        min_n_times = getattr(self, "_min_n_times", None)
        if min_n_times is not None and got_times < min_n_times:
            raise ValueError(
                f"{name} needs at least {min_n_times} time samples and got "
                f"{got_times} (input shape {tuple(x.shape)})."
            )

        on_non_divisible = getattr(self, "on_non_divisible", "crop")
        step = getattr(self, "_n_times_step", 1)
        offset = getattr(self, "_n_times_offset", 0)

        if time_axis == "fixed" and not self._accepts_any_n_times(args, kwargs):
            n_times = self._signal_param("n_times")
            if n_times is not None and got_times != n_times:
                diff = abs(got_times - n_times)
                if (
                    on_non_divisible != "error"
                    and diff <= max(1, n_times // 100)
                    and diff <= max(1, step)
                    and can_snap_last_dim
                ):
                    return "fixed_snap"
                resampled = (
                    f" after resampling from {self._input_sfreq:g} Hz"
                    if self._input_sfreq is not None
                    else ""
                )
                raise ValueError(
                    f"{name} was built for n_times={n_times} and got {got_times} "
                    f"samples{resampled} (input shape {tuple(x.shape)}). Rebuild it "
                    f"with n_times={got_times}, cut the signal into windows of "
                    f"{n_times} samples, or call model.adapt_input(sfreq=...)."
                )
            return None

        if step > 1 and can_snap_last_dim:
            rem = (got_times - offset) % step if got_times >= offset else -1
            if rem != 0:
                if on_non_divisible == "error":
                    patch = offset if offset > 0 else step
                    raise ValueError(
                        f"Time dimension ({got_times}) is not divisible by "
                        f"patch_size ({patch})"
                        + ("." if step == patch else f" with patch step {step}.")
                    )
                return "grid_snap"
        return None

    def _prepare_inputs(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Validate input axes and apply configured length snapping before forward."""
        x = self._first_input(args, kwargs)
        if x is None:
            return args, kwargs
        action = self._check_input_shape(x, args, kwargs)
        if action is None:
            return args, kwargs
        name = type(self).__name__
        got_times = x.shape[-1]
        explicit_policy = getattr(self, "_on_non_divisible_explicit", False)
        if action == "fixed_snap":
            n_times = self._signal_param("n_times")
            assert n_times is not None
            args, kwargs = self._replace_time_inputs(
                args, kwargs, lambda _name, t: self._snap_to_length(t, n_times)
            )
            if not explicit_policy and not getattr(self, "_warned_length_snap", False):
                self._warned_length_snap = True
                warnings.warn(
                    f"{name} snapped input length {got_times} to n_times={n_times}.",
                    UserWarning,
                    stacklevel=3,
                )
            return args, kwargs

        nominal = self._signal_param("n_times")
        on_non_divisible = getattr(self, "on_non_divisible", "crop")
        snapped_len = self._snap_grid_target(got_times)
        snap_mode = getattr(self, "_snaps_in_prepare_inputs", True)
        snap_here = snap_mode is True or (
            snap_mode == "pad_only" and on_non_divisible == "pad"
        )
        snapped_anywhere = snap_here or snap_mode is False
        if snap_here:
            args, kwargs = self._replace_time_inputs(
                args, kwargs, self._snap_grid_tensor
            )
        if (
            snapped_anywhere
            and not explicit_policy
            and got_times != nominal
            and snapped_len != got_times
            and not getattr(self, "_warned_length_snap", False)
        ):
            self._warned_length_snap = True
            warnings.warn(
                f"{name} snapped input length {got_times} to {snapped_len} "
                f"(step={getattr(self, '_n_times_step', 1)}, "
                f"on_non_divisible={on_non_divisible!r}).",
                UserWarning,
                stacklevel=3,
            )
        return args, kwargs

    def _head_linear(self) -> torch.nn.Module | None:
        """Return the last linear or convolutional layer inside ``final_layer``."""
        final_layer = getattr(self, "final_layer", None)
        if not isinstance(final_layer, torch.nn.Module):
            return None
        last: torch.nn.Module | None = None
        for module in final_layer.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                last = module
        return last

    def _pool_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Mean-pool ``features`` to a 2-D ``(batch, dim)`` embedding."""
        if features.ndim == 2:
            return features
        axis = self._feature_axis % features.ndim
        if axis != features.ndim - 1:
            features = features.movedim(axis, -1)
        return features.mean(dim=tuple(range(1, features.ndim - 1)))

    def _record_materialized_axes(
        self,
        args: tuple,
        kwargs: dict,
        had_lazy: bool,
        need_chans: bool,
        need_times: bool,
    ) -> None:
        if had_lazy and not any(
            isinstance(p, UninitializedParameter) for p in self.parameters()
        ):
            x = self._first_input(args, kwargs)
            if x is not None and x.ndim >= 3:
                n_chans, n_times, _ = self._input_chans_and_times(x)
                if need_chans and not self._accepts_any_n_chans(args, kwargs):
                    self._materialized_n_chans = n_chans
                if need_times and not self._accepts_any_n_times(args, kwargs):
                    if self._input_sfreq is not None:  # the length at model.sfreq
                        n_times = -(
                            -n_times
                            * int(self._signal_param("sfreq"))
                            // int(self._input_sfreq)
                        )
                    # the length the weights saw: on the model's length grid
                    self._materialized_n_times = self._snap_grid_target(n_times)

    def __call__(self, *args, **kwargs):
        spec = type(self)._forward_deprecation
        if spec is not None:
            args, kwargs = _map_deprecated_args(
                type(self).__name__, args, kwargs, *spec
            )
        # 2.0: an explicit return_features=False or None kept the flag's output.
        if self._legacy_flag is not None and not kwargs.get("return_features"):
            return self._legacy_call(args, kwargs)
        if self._head_spec is not None and not self._call_arg(
            args, kwargs, "return_features", False
        ):
            raise ValueError(
                f"{type(self).__name__} was built with n_outputs=None (headless); "
                "call model(x, return_features=True) or attach a head with "
                "model.reset_head(n_outputs)."
            )
        if self._input_sfreq is None:  # else forward checks after resampling
            args, kwargs = self._prepare_inputs(args, kwargs)
        need_chans = (
            getattr(self, "_channel_axis", None) == "fixed"
            and self._signal_param("n_chans") is None
        )
        need_times = (
            getattr(self, "_time_axis", None) == "fixed"
            and self._signal_param("n_times") is None
        )
        had_lazy = (need_chans or need_times) and any(
            isinstance(p, UninitializedParameter) for p in self.parameters()
        )
        if not self._native_features and "return_features" in kwargs:
            kwargs = dict(kwargs)
            want_features = bool(kwargs.pop("return_features"))
            if want_features:
                name = type(self).__name__
                captured: list[Any] = []

                def _capture(
                    _module: torch.nn.Module, hook_args: tuple[Any, ...]
                ) -> None:
                    captured.append(hook_args[0] if hook_args else None)

                handle = self.final_layer.register_forward_pre_hook(_capture)
                try:
                    super().__call__(*args, **kwargs)
                finally:
                    handle.remove()
                self._record_materialized_axes(
                    args, kwargs, had_lazy, need_chans, need_times
                )
                if len(captured) != 1:
                    raise ValueError(
                        f"{name}: final_layer ran {len(captured)} times in one "
                        "forward; return_features needs a native implementation "
                        "for this model."
                    )
                feat = captured[0]
                x = self._first_input(args, kwargs)
                if (
                    not isinstance(feat, torch.Tensor)
                    or feat.ndim < 2
                    or (x is not None and feat.shape[0] != x.shape[0])
                ):
                    raise ValueError(
                        f"{name}: the head does not read one batched tensor; "
                        "return_features needs a native implementation for this "
                        "model."
                    )
                return {
                    "features": feat,
                    "embedding": self._pool_embedding(feat),
                    "cls_token": None,
                }

        out = super().__call__(*args, **kwargs)
        self._record_materialized_axes(args, kwargs, had_lazy, need_chans, need_times)
        if self._native_features and isinstance(out, dict) and "features" in out:
            out.setdefault("cls_token", None)
            if out.get("embedding") is None and isinstance(
                out["features"], torch.Tensor
            ):
                out["embedding"] = self._pool_embedding(out["features"])
        return out

    def get_output_shape(self) -> tuple[int, ...]:
        """Returns shape of neural network output for batch size equal 1.

        Returns
        -------
        output_shape : tuple[int, ...]
            shape of the network output for `batch_size==1` (1, ...)
        """
        # Run in eval mode, so that the probe input does not update BatchNorm
        # running stats or draw dropout masks; restore each module's own mode
        # afterwards (some may have been put in eval mode on purpose).
        modes = {module: module.training for module in self.modules()}
        self.eval()
        with torch.inference_mode():
            try:
                ref = next(self.parameters(), torch.empty(0))
                probe = torch.zeros(
                    self.input_shape, dtype=ref.dtype, device=ref.device
                )
                (probe,), _ = self._prepare_inputs((probe,), {})
                return tuple(
                    type(self)
                    .forward(  # the backbone, on its own channels
                        self,
                        probe,
                    )
                    .shape
                )
            finally:
                for module, training in modes.items():
                    module.training = training

    def get_config(self) -> dict:
        """Return a JSON-serializable dict of all ``__init__`` parameters.

        The returned dictionary can be saved to a JSON file and later
        used with :meth:`from_config` to reconstruct the model (without
        weights).  It is also used internally by :meth:`push_to_hub` to
        persist the full model configuration.

        Returns
        -------
        dict
            All ``__init__`` parameters, JSON-serializable.
            ``type[nn.Module]`` parameters (e.g. ``activation``) are
            encoded as importable dotted-path strings.

        Examples
        --------
        >>> import json
        >>> from braindecode.models import EEGNet
        >>> model = EEGNet(n_chans=22, n_times=1000, n_outputs=4, F1=16)
        >>> config = model.get_config()
        >>> config["F1"]
        16
        >>> # Save to disk
        >>> with open("config.json", "w") as f:
        ...     json.dump(config, f)

        .. versionadded:: 1.4
        """
        return build_model_config(self)

    @classmethod
    def from_config(cls, config: dict) -> "EEGModuleMixin":
        """Create a model instance from a configuration dict.

        This is the inverse of :meth:`get_config`.  Weights are **not**
        loaded -- use :meth:`from_pretrained` for that.

        Parameters
        ----------
        config : dict
            Configuration dict as returned by :meth:`get_config`.

        Returns
        -------
        EEGModuleMixin
            A new model instance.

        Examples
        --------
        >>> import json
        >>> from braindecode.models import EEGNet
        >>> model = EEGNet(n_chans=22, n_times=1000, n_outputs=4, F1=16)
        >>> config = model.get_config()
        >>> # Reconstruct (without weights)
        >>> model2 = EEGNet.from_config(config)
        >>> model2.F1
        16
        >>> # Or from a JSON file
        >>> with open("config.json") as f:
        ...     config = json.load(f)
        >>> model3 = EEGNet.from_config(config)

        .. versionadded:: 1.4
        """
        config = dict(config)  # shallow copy
        config.pop("braindecode_version", None)
        adaptation = config.pop("input_adaptation", None)
        cls._fill_legacy_defaults(config)
        resolve_type_kwargs(cls, config)
        model = cls(**config)
        return model.adapt_input(**adaptation) if adaptation else model

    @classmethod
    def _fill_legacy_defaults(cls, kwargs: dict) -> None:
        """Rebuild configs saved by earlier versions as they were.

        Models that used to crop a time axis not divisible by their patch size
        set ``_legacy_on_non_divisible = "crop"``. A saved config or checkpoint
        without the ``on_non_divisible`` key then loads with ``"crop"``, so its
        head and position tables keep their shapes; a new model defaults to
        ``"pad"``. A config saved while a model that reads its input natively
        still took ``channel_strategy`` (2.1 development builds) loses its
        ``"native"`` strategy keys. Configs of released weights also lose the
        constructor feature flags (``return_feats`` and the like) that are false,
        the default, so that loading them does not warn, and those removed since.
        """
        params = inspect.signature(cls.__init__).parameters
        takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for flag in (
            "return_feats",
            "return_feature",
            "return_encoder_output",
            "return_features",
        ):
            if not kwargs.get(flag) or (not takes_any and flag not in params):
                kwargs.pop(flag, None)
        if not takes_any and "channel_strategy" not in params:
            if kwargs.get("channel_strategy", "native") == "native" and not kwargs.get(
                "channel_strategy_kwargs"
            ):
                kwargs.pop("channel_strategy", None)
                kwargs.pop("channel_strategy_kwargs", None)
        legacy = getattr(cls, "_legacy_on_non_divisible", None)
        if legacy is None or "on_non_divisible" in kwargs:
            return
        # A subclass with its own signature may not take the argument.
        if takes_any or "on_non_divisible" in params:
            kwargs["on_non_divisible"] = legacy

    def _set_n_outputs(self, n_outputs: int) -> None:
        """Record a new ``n_outputs`` on the model and its saved configs.

        Helper for :meth:`reset_head` implementations: validates the value,
        then keeps the ``n_outputs`` property, the braindecode init kwargs
        and the Hugging Face hub config in sync, so a re-serialized model
        reports the head it actually has.

        Parameters
        ----------
        n_outputs : int
            New number of outputs. Must be positive.

        .. versionadded:: 1.8
        """
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        self._n_outputs = n_outputs
        self._update_init_kwargs(n_outputs=n_outputs)

    def _update_init_kwargs(self, **init_kwargs) -> None:
        """Record constructor arguments that change after construction.

        Helper for :meth:`reset_head` implementations that also change a
        constructor argument, for example turning a feature extractor into a
        classifier. Keeps the braindecode init kwargs and the Hugging Face hub
        config in sync, so :meth:`get_config` and ``save_pretrained`` rebuild
        the architecture the model actually has.

        Parameters
        ----------
        **init_kwargs
            Constructor argument names mapped to their new values.
        """
        for config in (
            getattr(self, "_braindecode_init_kwargs", None),
            getattr(self, "_hub_mixin_config", None),
        ):
            if config is None:
                continue
            for name, value in init_kwargs.items():
                if name in config:
                    config[name] = value

    def reset_parameters(self) -> None:
        """Draw new initial weights, the way the constructor does.

        Every submodule with a ``reset_parameters`` method (PyTorch layers,
        and blocks with their own initialization) is reset, children before
        parents, then the model applies its own initialization
        (:meth:`_init_parameters`). Seed with :func:`torch.manual_seed` for a
        reproducible draw.
        """

        def reset(module):
            if module is self:
                return
            # PyTorch layers use reset_parameters; nn.MultiheadAttention and
            # nn.Transformer use _reset_parameters.
            for name in ("reset_parameters", "_reset_parameters"):
                if callable(getattr(module, name, None)):
                    getattr(module, name)()
                    break

        self.apply(reset)
        self._init_parameters()
        provenance = self.__dict__.get("_pretrained_provenance")
        if provenance is not None:
            # Conservatively invalidate attribution after model-wide initialization.
            provenance["loaded"] = set()
            provenance.pop("mixed", None)
            if "supplementary" in provenance:
                provenance["supplementary"]["loaded"] = set()

    def _init_parameters(self) -> None:
        """Model-specific initialization, run by the constructor after the
        layers are built and by :meth:`reset_parameters`. The default keeps the
        layers' PyTorch initialization."""

    def _can_reset_head(self) -> bool:
        """Whether :meth:`reset_head` can attach a new head to this model."""
        return (
            type(self).reset_head is not EEGModuleMixin.reset_head
            or self._head_swap_refusal() is None
        )

    def _head_swap_refusal(self) -> str | None:
        """Why the default :meth:`reset_head` cannot swap the last layer of the
        head for a new one, or ``None`` when it can."""
        head = self._head_spec[0] if self._head_spec is not None else self.final_layer
        layers = list(head) if isinstance(head, torch.nn.Sequential) else [head]
        # The last layer with weights; parameter-free ones (squeeze, softmax) follow.
        last = next((m for m in reversed(layers) if list(m.parameters())), None)
        if not isinstance(last, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            return "final_layer does not end with a Linear or Conv layer"
        if torch.nn.utils.parametrize.is_parametrized(last):
            return "the weight of its last layer is constrained"
        sized = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
        if sum(isinstance(m, sized) for m in head.modules()) > 1:
            return "final_layer holds more than one Linear or Conv layer"
        return None

    @_new_head_follows_model
    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new number of outputs.

        This is called automatically by :meth:`from_pretrained` when the
        user passes an ``n_outputs`` that differs from the saved config, and
        attaches a head to a model built with ``n_outputs=None``. The default
        replaces the last linear or convolutional layer of ``final_layer`` by
        a new one with ``n_outputs`` outputs, and raises
        :class:`NotImplementedError` when that layer is constrained, when
        ``final_layer`` holds more than one such layer, or when it does not end
        with one. Override in subclasses that need
        a model-specific head structure. Implementations keep changed
        constructor arguments in sync with ``self._update_init_kwargs``, so
        that a saved model can be loaded back. Implementations requiring
        positive outputs can also use ``self._set_n_outputs`` to validate and
        record the new value.

        Parameters
        ----------
        n_outputs : int
            New number of output classes.

        Examples
        --------
        >>> from braindecode.models import BENDR
        >>> model = BENDR(n_chans=22, n_times=1000, n_outputs=4)
        >>> model.reset_head(10)
        >>> model.n_outputs
        10

        .. versionadded:: 1.4
        """
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        reason = self._head_swap_refusal()
        if reason is not None:
            raise NotImplementedError(
                f"{type(self).__name__} needs its own reset_head: {reason}."
            )
        old = self._head_linear()
        bias = old.bias is not None
        if isinstance(old, torch.nn.LazyLinear):  # not sized yet (GCNsNet)
            new = torch.nn.LazyLinear(n_outputs, bias=bias)
        elif isinstance(old, torch.nn.Linear):
            new = torch.nn.Linear(old.in_features, n_outputs, bias=bias)
        else:
            conv = (
                torch.nn.Conv1d if isinstance(old, torch.nn.Conv1d) else torch.nn.Conv2d
            )
            new = conv(
                old.in_channels,
                n_outputs,
                old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                bias=bias,
            )
        name = next(n for n, m in self.final_layer.named_modules() if m is old)
        if name:
            parent, _, child = name.rpartition(".")
            setattr(self.final_layer.get_submodule(parent), child, new)
        else:
            self.final_layer = new
        self._set_n_outputs(n_outputs)

    mapping: Optional[Dict[str, str]] = None
    #: Channels the backbone consumes under a ``channel_strategy`` (names or
    #: ``chs_info``); ``None``: the model's own ``chs_info``.
    _channel_target: Optional[list] = None
    _time_axis: str | None = None
    _channel_axis: str | None = None
    _time_inputs: ClassVar[tuple[str, ...]] = ("x",)
    #: Whether :meth:`to_dense_prediction_model` can convert this model.
    _dense_conversion: ClassVar[bool] = False
    # Set by adapt_input; class defaults also serve unpickled v2.0.0 models.
    _input_sfreq: Optional[float] = None
    _adapt_config: Optional[dict] = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        own = super().state_dict()
        new_state_dict = OrderedDict(state_dict)
        # An old name never replaces a current one: strict loads report it.
        for old, current in (self.mapping or {}).items():
            if old in new_state_dict and current not in new_state_dict:
                new_state_dict[current] = new_state_dict.pop(old)
        # Keep the module versions saved in the checkpoint (BatchNorm's, for one).
        if hasattr(state_dict, "_metadata"):
            new_state_dict._metadata = state_dict._metadata
        # A backbone checkpoint has no weights for a trainable channel layer.
        # Load hooks may expand tensors with locally initialized values. Only
        # attribute whole tensors whose checkpoint shape already matches.
        from_checkpoint = {
            k
            for k, v in new_state_dict.items()
            if k in own
            and (
                not isinstance(v, torch.Tensor)
                or torch.nn.parameter.is_lazy(own[k])
                or v.shape == own[k].shape
            )
        }
        mixed = (set(new_state_dict) & set(own)) - from_checkpoint
        for k, v in own.items():
            if k.startswith("channel_layer.") and k not in new_state_dict:
                new_state_dict[k] = v
        if not _RECORD_LOADED_KEYS.get():
            # PyTorch may copy matching tensors before a strict load raises.
            self.__dict__.pop("_pretrained_provenance", None)
        result = super().load_state_dict(new_state_dict, *args, **kwargs)
        if _RECORD_LOADED_KEYS.get():
            provenance = self.__dict__.setdefault("_pretrained_provenance", {})
            loaded = provenance.get("loaded") or set()
            provenance["loaded"] = (loaded - mixed) | (
                from_checkpoint - set(result.unexpected_keys) - set(result.missing_keys)
            )
            provenance["mixed"] = (
                provenance.get("mixed", set()) | mixed
            ) - from_checkpoint
        return result

    def to_dense_prediction_model(self, axis: tuple[int, ...] | int = (2, 3)) -> None:
        """
        Transform a sequential model with strides to a model that outputs.

        dense predictions by removing the strides and instead inserting dilations.
        Modifies model in-place.

        Parameters
        ----------
        axis : int or (int,int)
            Axis to transform (in terms of intermediate output axes)
            can either be 2, 3, or (2,3).

        Notes
        -----
        Does not yet work correctly for average pooling.
        Prior to version 0.1.7, there had been a bug that could move strides
        backwards one layer.

        Raises
        ------
        ValueError
            If the model does not support dense conversion.
        """
        if not self._dense_conversion:
            reason = (
                "it already gives dense predictions"
                if self._time_axis == "dense"
                else "it pools, tokenizes or fixes its window"
            )
            raise ValueError(
                f"{type(self).__name__} cannot give dense predictions: {reason}"
            )
        if not hasattr(axis, "__iter__"):
            axis = (axis,)
        assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"  # type: ignore[union-attr]
        axis = np.array(axis) - 2
        stride_so_far = np.array([1, 1])
        for module in self.modules():  # type: ignore
            if hasattr(module, "dilation"):
                assert module.dilation == 1 or (module.dilation == (1, 1)), (
                    "Dilation should equal 1 before conversion, maybe the model is "
                    "already converted?"
                )
                new_dilation = [1, 1]
                for ax in axis:  # type: ignore[union-attr]
                    new_dilation[ax] = int(stride_so_far[ax])
                module.dilation = tuple(new_dilation)
            if hasattr(module, "stride"):
                if not hasattr(module.stride, "__len__"):
                    module.stride = (module.stride, module.stride)
                stride_so_far *= np.array(module.stride)
                new_stride = list(module.stride)
                for ax in axis:  # type: ignore[union-attr]
                    new_stride[ax] = 1
                module.stride = tuple(new_stride)

    def get_torchinfo_statistics(
        self,
        col_names: Optional[Iterable[str]] = (
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
        ),
        row_settings: Optional[Iterable[str]] = ("var_names", "depth"),
    ) -> ModelStatistics:
        """Generate table describing the model using torchinfo.summary.

        Parameters
        ----------
        col_names : tuple, optional
            Specify which columns to show in the output, see torchinfo for details, by default
            ("input_size", "output_size", "num_params", "kernel_size")
        row_settings : tuple, optional
             Specify which features to show in a row, see torchinfo for details, by default
             ("var_names", "depth")

        Returns
        -------
        torchinfo.ModelStatistics
            ModelStatistics generated by torchinfo.summary.
        """
        n_chans = self._signal_param("n_chans") or 1
        n_times = self._signal_param("n_times") or 200

        statistics = summary(
            self,
            input_size=(1, n_chans, n_times),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )
        # torchinfo 1.8 also stores the summary layers in a module-level dict,
        # keyed by class name, whatever ``cache_forward_pass`` is: it kept the
        # last printed instance of each class alive. braindecode never reads it.
        cache = getattr(sys.modules[summary.__module__], "_cached_forward_pass", {})
        cache.pop(type(self).__name__, None)
        return statistics

    def _pretrained_summary(self):
        """Display data shared by the text and notebook representations."""
        from torch.nn.parameter import UninitializedParameter

        provenance = self.__dict__.get("_pretrained_provenance", {})
        loaded = provenance.get("loaded", set())
        # Tied parameters may be saved under only one of their names.
        loaded_ids = {
            id(p)
            for n, p in self.named_parameters(remove_duplicate=False)
            if n in loaded
        }
        unbuilt = [
            n
            for n, p in self.named_parameters()
            if isinstance(p, UninitializedParameter)
        ]
        rows, seen = [], set()
        for name, module in self.named_modules():
            params = [
                (n, p)
                for n, p in module.named_parameters(recurse=False)
                if id(p) not in seen and not isinstance(p, UninitializedParameter)
            ]
            seen.update(id(p) for _, p in params)
            if params:
                total = sum(p.numel() for _, p in params)
                count = sum(p.numel() for _, p in params if id(p) in loaded_ids)
                rows.append(
                    (name or type(self).__name__, type(module).__name__, total, count)
                )
        total = sum(row[2] for row in rows)
        count = sum(row[3] for row in rows)
        datasets = getattr(self.card_data, "datasets", None) or []
        if isinstance(datasets, str):
            datasets = [datasets]
        geometry = {}
        for attr in ("n_chans", "n_times", "sfreq", "n_outputs"):
            val = self._signal_param(attr)
            if val is not None:
                geometry[attr] = val
        if hasattr(self, "channel_layer"):
            channels = self.channel_layer.chs_info
            geometry["n_chans"] = (
                len(channels) if channels is not None else "set at forward"
            )
        return dict(
            source=provenance,
            datasets=datasets,
            rows=rows,
            unbuilt=unbuilt,
            geometry=geometry,
            total=total,
            loaded=count,
            percent=100.0 if count == total else min(99.9, 100 * count / max(total, 1)),
            new=[name for name, _, n, k in rows if k < n],
        )

    def __str__(self) -> str:
        table = str(self.get_torchinfo_statistics())
        if not self.__dict__.get("_pretrained_provenance"):
            return table
        data = self._pretrained_summary()  # torchinfo may initialise a lazy head
        source = data["source"]
        repo = source["repo_id"]
        revision = source.get("commit") or source.get("revision")
        if revision and not source.get("local"):
            repo += f" @ {revision[:7]}"
        supplement = source.get("supplementary")
        if supplement:
            repo += f"; supplementary: {supplement['repo_id']}"
            revision = supplement.get("commit") or supplement.get("revision")
            if revision and not supplement.get("local"):
                repo += f" @ {revision[:7]}"
        label = "new or adapted" if source.get("mixed") else "new"
        new = f"; {label}: " + ", ".join(data["new"]) if data["new"] else ""
        scope = (
            "parameters in fully loaded tensors; adapted tensors excluded"
            if source.get("mixed")
            else "parameters"
        )
        if data["unbuilt"]:
            new += "; unbuilt: " + ", ".join(data["unbuilt"])
        return (
            f"{table}\nPretrained weights: {repo}\n"
            f"Trained on (model card): {', '.join(data['datasets']) or 'not listed'}\n"
            f"Loaded from the listed checkpoint(s): {data['loaded']:,} of {data['total']:,} "
            f"{scope} ({data['percent']:.1f}% of materialised parameters){new}\n"
            "Cite Braindecode (Zenodo): https://doi.org/10.5281/zenodo.8214376"
        )

    def _repr_html_(self) -> str:
        """Notebook model card, rendered from an autoescaped Jinja template (as in MNE)."""
        from jinja2 import Environment, PackageLoader

        template = Environment(
            loader=PackageLoader("braindecode.models", "."),
            autoescape=True,
        ).get_template("model.html.jinja")
        return template.render(
            model=self,
            name=type(self).__name__,
            data=self._pretrained_summary(),
            architecture=torch.nn.Module.__repr__(self),
        )

    @staticmethod
    def _serialize_chs_info(chs_info):
        """Serialize MNE channel info (``info["chs"]``) to JSON-compatible dicts."""
        if chs_info is None:
            return None
        _INT_FIELDS = ("kind", "coil_type", "unit", "coord_frame")
        _FLOAT_FIELDS = ("cal", "range")
        serialized = []
        for ch in chs_info:
            ch_dict = {"ch_name": ch.get("ch_name", "")}
            if ch.get("ch_type") is not None:
                ch_dict["ch_type"] = ch["ch_type"]
            for key in _INT_FIELDS:
                val = ch.get(key)
                if val is not None:
                    ch_dict[key] = val if isinstance(val, str) else int(val)
            for key in _FLOAT_FIELDS:
                val = ch.get(key)
                if val is not None:
                    ch_dict[key] = float(val)
            if "loc" in ch and ch["loc"] is not None:
                ch_dict["loc"] = (
                    ch["loc"].tolist()
                    if hasattr(ch["loc"], "tolist")
                    else list(ch["loc"])
                )
            serialized.append(ch_dict)
        return serialized

    @staticmethod
    def _deserialize_chs_info(chs_info_dict):
        """Deserialize JSON channel dicts back to MNE-compatible format."""
        if chs_info_dict is None:
            return None
        deserialized = []
        for ch_dict in chs_info_dict:
            ch = ch_dict.copy()
            if "loc" in ch and ch["loc"] is not None:
                ch["loc"] = np.array(ch["loc"])
            deserialized.append(ch)
        return deserialized

    def _save_pretrained(self, save_directory):
        """
        Save model configuration and weights to the Hub.

        This method is called by PyTorchModelHubMixin.push_to_hub() to save
        model-specific configuration alongside the model weights.

        Parameters
        ----------
        save_directory : str or Path
            Directory where the configuration should be saved.
        """
        if not HAS_HF_HUB:
            return

        save_directory = Path(save_directory)

        config = build_model_config(self)
        config["braindecode_version"] = __version__

        # Save to config.json
        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save model weights with standard Hub filename
        weights_path = save_directory / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)

        # Also save in safetensors format using parent's implementation
        try:
            from safetensors.torch import _find_shared_tensors, _is_complete
        except ImportError as e:
            (save_directory / "model.safetensors").unlink(missing_ok=True)
            warnings.warn(
                f"Could not save model in safetensors format: {e}. "
                "Model weights saved in pytorch_model.bin instead.",
                stacklevel=2,
            )
            return

        state_dict = self.state_dict()
        incomplete = next(
            (
                shared
                for shared in _find_shared_tensors(state_dict)
                if not any(_is_complete(state_dict[name]) for name in shared)
            ),
            None,
        )
        if incomplete is not None:
            # Fall back to pytorch_model.bin, and drop the model.safetensors of a
            # previous save, which would be loaded in its place.
            (save_directory / "model.safetensors").unlink(missing_ok=True)
            warnings.warn(
                "Could not save model in safetensors format: "
                "found shared tensors with no complete storage amongst "
                f"{sorted(incomplete)}. "
                "Model weights saved in pytorch_model.bin instead.",
                stacklevel=2,
            )
            return

        super()._save_pretrained(save_directory)

    if HAS_HF_HUB:

        @classmethod
        def _decode_arg(cls, expected_type, value):
            # Hub eagerly decodes plain/optional type fields before overrides.
            # Defer only our default decoder to the effective-kwargs resolver;
            # explicitly registered application coders retain Hub's behavior.
            if (
                expected_type in (type, Optional[type])
                and next(
                    decoder
                    for coder_type, (_, decoder) in cls._hub_mixin_coders.items()
                    if issubclass(type, coder_type)
                )
                is resolve_config_type
            ):
                return value
            return super()._decode_arg(expected_type, value)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            if kwargs.get("filename") is not None:
                _validate_weights_filename(kwargs["filename"])
            # The Hub mixin fills only the __init__ args whose keys are absent,
            # so drop explicit Nones (e.g. forwarded by a wrapper) to let the
            # saved values fill them. Then pin the derived geometry arg so it
            # cannot clash with the caller's one (e.g. caller chs_info vs saved
            # n_chans) in __init__'s checks.
            for key in _GEOMETRY_KWARGS:
                if key in kwargs and kwargs[key] is None:
                    del kwargs[key]
            if "chs_info" in kwargs:
                kwargs.setdefault("n_chans", len(kwargs["chs_info"]))
            elif "n_chans" in kwargs:
                kwargs.setdefault("chs_info", None)
            if "input_window_seconds" in kwargs:
                # Keep a new duration from clashing with the saved n_times (or the
                # saved sfreq, when the caller gives n_times); _from_pretrained
                # completes the window from what the save holds.
                kwargs.setdefault("sfreq" if "n_times" in kwargs else "n_times", None)
            elif "n_times" in kwargs or "sfreq" in kwargs:
                kwargs.setdefault("input_window_seconds", None)
            token = _RECORD_LOADED_KEYS.set(True)
            try:
                model = super().from_pretrained(*args, **kwargs)
            finally:
                _RECORD_LOADED_KEYS.reset(token)
            if isinstance(model, torch.nn.Module):
                provenance = model.__dict__.setdefault("_pretrained_provenance", {})
                provenance.update(_checkpoint_source(*args, **kwargs))
                if not model.__dict__.get("_missing_encoder_keys"):
                    cls._warn_if_unloaded(model, provenance)
            card_data = _load_card_data(*args, **kwargs)
            if card_data is not None:
                model.card_data = card_data
                if card_data.datasets:
                    log.info(
                        "%s weights were trained on (model card): %s",
                        cls.__name__,
                        card_data.datasets,
                    )
            return model

        @classmethod
        def _warn_if_unloaded(cls, model, provenance):
            params = list(model.named_parameters(remove_duplicate=False))
            loaded = provenance.get("loaded") or set()
            # Tied parameters may be saved under only one of their names.
            loaded_ids = {id(p) for n, p in params if n in loaded}
            # A new head is initialised here: only the rest must load.
            body = [n for n, _ in params if not n.startswith("final_layer.")]
            unloaded = [
                n
                for n, p in params
                if id(p) not in loaded_ids and not n.startswith("final_layer.")
            ]
            # Most of the body unloaded: the checkpoint is in another format.
            # A few fresh layers (Labram's fc_norm, CORTEG's channel_adapter)
            # are by design and stay silent.
            if 2 * len(unloaded) > len(body):
                more = ", ..." if len(unloaded) > 10 else ""
                warnings.warn(
                    f"{len(unloaded)} of {len(params)} parameters of "
                    f"{cls.__name__} were not loaded from "
                    f"{provenance['repo_id']} (the checkpoint keys do not "
                    f"match the model): {', '.join(unloaded[:10])}{more}",
                    UserWarning,
                    stacklevel=2,
                )

        @classmethod
        def _load_supplementary_safetensors(
            cls,
            model,
            source,
            keys,
            *,
            revision=None,
            cache_dir=None,
            local_files_only=False,
            token=None,
        ):
            from safetensors.torch import load_file

            source = str(source)
            if Path(source).is_dir():
                path = _local_weights_path(source, "model.safetensors")
            else:
                path = huggingface_hub.hf_hub_download(
                    source,
                    "model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    token=token,
                )
            state_dict = load_file(path)
            to_load = {k: state_dict[k] for k in keys if k in state_dict}
            rec_token = _RECORD_LOADED_KEYS.set(True)
            try:
                model.load_state_dict(to_load, strict=False)
            finally:
                _RECORD_LOADED_KEYS.reset(rec_token)
            provenance = model.__dict__.setdefault("_pretrained_provenance", {})
            supplementary = {
                **_checkpoint_source(source, revision=revision, cache_dir=cache_dir),
                "loaded": set(to_load),
            }
            snapshot = Path(path).parent
            if not supplementary.get("local") and snapshot.parent.name == "snapshots":
                supplementary["commit"] = snapshot.name
            provenance["supplementary"] = supplementary
            cls._warn_if_unloaded(model, provenance)
            return model

        def generate_model_card(self, *args, **kwargs):
            # ``model_card_kwargs`` only fill the card's Jinja template; also
            # write these two to its YAML header so the Hub indexes them.
            card = super().generate_model_card(*args, **kwargs)
            for key in ("datasets", "base_model"):
                if key in kwargs:
                    setattr(card.data, key, kwargs[key])
            return card

        @classmethod
        def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: Optional[str],
            cache_dir: Optional[Union[str, Path]],
            force_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",
            strict: bool = False,
            config: Optional[dict] = None,
            **model_kwargs,
        ):
            if (
                config is not None
                and "config" in cls._hub_mixin_init_parameters
                and "config" not in model_kwargs
            ):
                model_kwargs["config"] = config
            model_kwargs.pop("braindecode_version", None)
            adaptation = model_kwargs.pop("input_adaptation", None)
            filename = model_kwargs.pop("filename", None)
            if filename is not None:
                _validate_weights_filename(filename)
            # The Hub mixin passes only the keys of config.json (plus the
            # caller's), so a missing key means a pre-on_non_divisible save.
            cls._fill_legacy_defaults(model_kwargs)
            resolve_type_kwargs(cls, model_kwargs)

            # The Hub mixin read config.json (local or Hub) and passed it as
            # ``config``: it gives n_outputs (to detect a head swap) and the
            # saved time geometry.
            is_local = Path(model_id).is_dir()
            saved_config = config or {}
            adaptation = adaptation or saved_config.get("input_adaptation")

            # A save spells its window with any two of these three quantities.
            # When only one is known, complete the pair from the save, preferring
            # its sample count.
            time_keys = ("n_times", "sfreq", "input_window_seconds")
            if sum(model_kwargs.get(k) is not None for k in time_keys) == 1:
                samples, rate, duration = (saved_config.get(k) for k in time_keys)
                if duration:
                    if samples is None and rate is not None:
                        samples = round(duration * rate)
                    elif rate is None and samples is not None:
                        rate = samples / duration
                for key, value in zip(time_keys, (samples, rate, duration)):
                    if model_kwargs.get(key) is None and value is not None:
                        model_kwargs[key] = value
                        break

            saved_n_outputs = saved_config.get("n_outputs")
            requested_n_outputs = model_kwargs.get("n_outputs")

            # If the user requests different n_outputs, load with the
            # saved value first (so weights match), then swap the head.
            if (
                saved_n_outputs is not None
                and requested_n_outputs is not None
                and requested_n_outputs != saved_n_outputs
            ):
                model_kwargs["n_outputs"] = saved_n_outputs

            # The parent loads only model.safetensors from a local directory; use
            # the pytorch_model.bin that save_pretrained writes when there is none.
            if filename is None and is_local:
                preferred = _local_weights_path(model_id, "model.safetensors")
                if (
                    not preferred.exists()
                    and (Path(model_id) / "pytorch_model.bin").is_file()
                ):
                    filename = "pytorch_model.bin"

            # Bring the model card next to the weights. A repo without README.md
            # matches no file: nothing is downloaded and nothing is raised. The card
            # is metadata only, so a failed download (network down, rate limit)
            # warns and the weights load anyway; a missing repo or revision is
            # raised again by the weight download right below.
            if (
                not is_local
                and not local_files_only
                and not huggingface_hub.constants.HF_HUB_OFFLINE
            ):
                try:
                    huggingface_hub.snapshot_download(
                        model_id,
                        allow_patterns=["README.md"],
                        revision=revision,
                        cache_dir=cache_dir,
                        token=token,
                        force_download=force_download,
                    )
                except OSError as exc:
                    warnings.warn(
                        f"Could not download the model card of {model_id} "
                        f"({exc}); loading the weights without it, so "
                        "model.card_data stays None.",
                        UserWarning,
                        stacklevel=2,
                    )

            if filename is None:
                model = super()._from_pretrained(  # type: ignore
                    model_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    map_location=map_location,
                    strict=strict,
                    **model_kwargs,
                )
            else:
                # Hub <1.0 forwards these download options to _from_pretrained.
                # They are not constructor kwargs; only pass them to versions
                # of hf_hub_download that supplied them in the first place.
                download_kwargs = {
                    key: model_kwargs.pop(key)
                    for key in ("proxies", "resume_download")
                    if key in model_kwargs
                }
                if is_local:
                    model_file = str(_local_weights_path(model_id, filename))
                else:
                    model_file = huggingface_hub.hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        token=token,
                        local_files_only=local_files_only,
                        **download_kwargs,
                    )
                # save_pretrained falls back to a pickle when safetensors fails.
                is_pickle = is_local and filename == "pytorch_model.bin"
                load = cls._load_as_pickle if is_pickle else cls._load_as_safetensor
                model = load(cls(**model_kwargs), model_file, map_location, strict)

            if (
                saved_n_outputs is not None
                and requested_n_outputs is not None
                and requested_n_outputs != saved_n_outputs
            ):
                try:
                    model.reset_head(requested_n_outputs)
                except NotImplementedError:
                    raise ValueError(
                        f"{type(model).__name__} does not support changing "
                        f"n_outputs after loading. Saved model has "
                        f"n_outputs={saved_n_outputs}, but "
                        f"n_outputs={requested_n_outputs} was requested."
                    ) from None

            return model.adapt_input(**adaptation) if adaptation else model
