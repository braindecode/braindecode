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
import os
import re
import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from mne.utils import _soft_import
from torchaudio.transforms import Resample
from torchinfo import ModelStatistics, summary

from braindecode.functional.patches import snap_time_axis
from braindecode.models.util import (
    _EEG_PARAMS,
    _in_model_resample,
    build_model_config,
    resolve_config_type,
    resolve_type_kwargs,
    track_model_init_kwargs,
)
from braindecode.modules.channels import ChannelLayer
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

# Stateless channel strategies allowed in adapt_input() (never add state_dict keys).
_ADAPT_STRATEGIES: tuple[str, ...] = (
    "exact",
    "zero",
    "nearest",
    "idw",
    "spline",
    "field",
    "source",
    "region",
)


def _resample_out_length(length: int, orig_freq: int, new_freq: int) -> int:
    """Output sample count produced by ``torchaudio.transforms.Resample``."""
    g = math.gcd(orig_freq, new_freq)
    return math.ceil((new_freq // g) * length / (orig_freq // g))


def _apply_resample(
    resample: Resample, x: torch.Tensor, time_dim: int = -1
) -> torch.Tensor:
    """Apply ``resample`` along ``time_dim`` (torchaudio reshapes with ``view``)."""
    moved = x if time_dim == -1 else x.movedim(time_dim, -1)
    out = resample(moved.contiguous())
    return out if time_dim == -1 else out.movedim(-1, time_dim)


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


_NAME_MAX = os.pathconf("/", "PC_NAME_MAX") if hasattr(os, "pathconf") else 255
_PATH_MAX = os.pathconf("/", "PC_PATH_MAX") if hasattr(os, "pathconf") else 4096


def _is_local_dir(path_or_id: Union[str, os.PathLike]) -> bool:
    """True when ``path_or_id`` names an existing directory without ENAMETOOLONG."""
    raw = os.fspath(path_or_id)
    if not raw or "\x00" in raw or len(os.fsencode(raw)) >= _PATH_MAX:
        return False
    p = Path(raw)
    if any(len(os.fsencode(part)) > _NAME_MAX for part in p.parts):
        return False
    return p.is_dir()


_REPO_ID_REGEX = re.compile(
    r"(\b[\w\-.]{1,96}\b/)?\b[\w\-.]{1,96}\b",
    re.ASCII,
)


def _is_valid_repo_id(repo_id: str) -> bool:
    """Validate Hub repository ID structure matching ``huggingface_hub`` rules."""
    return (
        isinstance(repo_id, str)
        and repo_id.count("/") <= 1
        and _REPO_ID_REGEX.fullmatch(repo_id) is not None
        and "--" not in repo_id
        and ".." not in repo_id
        and not repo_id.endswith(".git")
    )


def _resolve_hub_or_local_file(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    filename: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    token: Union[str, bool, None] = None,
    local_files_only: bool = False,
) -> Optional[Union[str, Path]]:
    """Locate ``filename`` in a local directory, Hub cache, or remote Hub repo."""
    if _is_local_dir(pretrained_model_name_or_path):
        candidate = Path(pretrained_model_name_or_path) / filename
        return candidate if candidate.is_file() else None
    repo_id = os.fspath(pretrained_model_name_or_path)
    if not HAS_HF_HUB or not _is_valid_repo_id(repo_id):
        return None
    if not force_download:
        cached = huggingface_hub.try_to_load_from_cache(
            repo_id,
            filename,
            cache_dir=cache_dir,
            revision=revision,
        )
        if isinstance(cached, str):
            return cached
    offline = local_files_only or huggingface_hub.is_offline_mode()
    if offline:
        return None
    if not huggingface_hub.file_exists(
        repo_id,
        filename,
        revision=revision,
        token=token,
    ):
        return None
    return huggingface_hub.hf_hub_download(
        repo_id,
        filename,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
    )


def _load_card_data(pretrained_model_name_or_path, **kwargs):
    """Metadata of a checkpoint's model card (``README.md``), or None.

    Reads from a local checkpoint directory, the local Hub cache when
    ``local_files_only=True`` or ``HF_HUB_OFFLINE=1`` (or when already cached
    with ``force_download=False``), or the Hub when online. A missing card
    never blocks loading weights; malformed card YAML raises directly.
    """
    if not HAS_HF_HUB:
        return None
    card = _resolve_hub_or_local_file(
        pretrained_model_name_or_path,
        "README.md",
        revision=kwargs.get("revision"),
        cache_dir=kwargs.get("cache_dir"),
        force_download=bool(kwargs.get("force_download", False)),
        token=kwargs.get("token"),
        local_files_only=bool(kwargs.get("local_files_only", False)),
    )
    if card is None:
        return None
    return huggingface_hub.ModelCard.load(card, ignore_metadata_errors=True).data


def _checkpoint_source(pretrained_model_name_or_path, **kwargs):
    """Repository (or folder) and revision a checkpoint was loaded from.

    The commit is read from the local Hub cache, without network access.
    """
    source = os.fspath(pretrained_model_name_or_path)
    if _is_local_dir(source):
        return {"repo_id": source, "local": True}
    info = {"repo_id": source, "revision": kwargs.get("revision")}
    if not HAS_HF_HUB or not _is_valid_repo_id(source):
        return info
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


@functools.lru_cache(maxsize=None)
def _forward_param_names(forward_fn) -> tuple[str, ...]:
    """Positional/keyword parameter names of ``forward`` after ``self``."""
    params = list(inspect.signature(forward_fn).parameters.values())
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


def _new_head_follows_model(reset_head):
    """Wrap a ``reset_head`` so that the modules it creates take the model's
    floating dtype, device and train/eval mode (a new ``nn.Linear`` is float32
    on the CPU in training mode, whatever the model)."""

    @functools.wraps(reset_head)
    def wrapper(self, *args, **kwargs):
        ref = next((p for p in self.parameters() if p.is_floating_point()), None)
        # Hold the old tensors until the end, so that their ids cannot be
        # reused by the new ones.
        old = {id(t): t for t in [*self.parameters(), *self.buffers()]}
        out = reset_head(self, *args, **kwargs)
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
    #: checkpoint, set by :meth:`from_pretrained` when the card can be read,
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
        # Extract encoder features (consistent dict across all models)
        out = model(x, return_features=True)
        features = out["features"]

        # Replace the classification head
        model.reset_head(n_outputs=10)

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

    #: Attributes that :func:`torch.jit.script` must not introspect. The
    #: signal-related properties raise :class:`ValueError` when their value was
    #: neither given nor inferable, and ``mapping`` carries a postponed
    #: annotation TorchScript cannot resolve. Either one aborts scripting
    #: before ``forward`` is ever compiled.
    __jit_ignored_attributes__ = [
        *sorted(_EEG_PARAMS),
        "_adapt_channel_layer",
        "_adapt_channel_strategy",
        "_adapt_channel_strategy_kwargs",
        "_adapt_chs_info",
        "_adapted_n_times",
        "_channel_axis",
        "_chs_info",
        "_ctor_chs_info",
        "_input_sfreq",
        "_min_n_times",
        "_time_axis",
        "_time_input_steps",
        "_time_inputs",
        "input_shape",
        "mapping",
    ]
    #: Rich MNE channel dictionaries are intentionally unavailable in scripted
    #: forwards; they remain unchanged on eager models and in saved configs.
    __jit_unused_properties__ = ["chs_info"]

    def __init_subclass__(cls, **kwargs):
        license = kwargs.pop("license", "bsd-3-clause")

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
            target = [
                c
                if isinstance(c, dict)
                else {"ch_name": c, "kind": "eeg", "loc": np.zeros(12)}
                for c in target
            ]
            kwargs = channel_strategy_kwargs or {}
            layer = ChannelLayer(target, channel_strategy, chs_info, **kwargs)
            chs_info, n_chans = layer.target, None
        elif channel_strategy_kwargs:
            raise ValueError(
                "channel_strategy='native' takes no channel_strategy_kwargs."
            )

        self._input_window_seconds = input_window_seconds  # type: ignore[assignment]
        self._chs_info = chs_info  # type: ignore[assignment]
        self._n_outputs = n_outputs  # type: ignore[assignment]
        self._n_chans = n_chans  # type: ignore[assignment]
        # TorchScript cannot represent the rich MNE dictionaries in _chs_info.
        # Keep the original eager state above and expose only the derived scalar
        # to scripted signal-property getters.
        self._n_chans_for_jit = (
            len(chs_info) if n_chans is None and chs_info is not None else n_chans
        )
        self._n_times = n_times  # type: ignore[assignment]
        self._sfreq = sfreq  # type: ignore[assignment]

        # Back-fill instance attributes from _hub_mixin_config for any
        # params the subclass didn't store on self.  Skip EEG params
        # (stored as self._*) and descriptors (properties).
        for key, val in getattr(self, "_hub_mixin_config", {}).items():
            if key in _EEG_PARAMS:
                continue
            if hasattr(getattr(type(self), key, None), "__get__"):
                continue
            if key == "on_non_divisible" or not hasattr(self, key):
                setattr(self, key, val)

        self._min_n_times = None
        self._n_times_step = 1
        self._n_times_offset = 0
        self._time_input_steps = {}
        self._defer_step_snap = False
        self._warned_length_snap = False
        self._input_sfreq = None
        self._adapted_n_times = None
        self._adapt_channel_layer = False
        self._adapt_chs_info = None
        self._adapt_channel_strategy = None
        self._adapt_channel_strategy_kwargs = None
        self._ctor_chs_info = layer.chs_info if layer is not None else None
        if "on_non_divisible" not in self.__dict__ or self.on_non_divisible is None:
            self.on_non_divisible = "crop"
        if not hasattr(self, "_explicit_on_non_divisible"):
            self._explicit_on_non_divisible = False

        super().__init__()
        if layer is not None:
            self.channel_layer = layer
            # Instance attribute: model(x) and model.forward(x) both map x.
            setattr(self, "forward", self._channel_forward)

    @torch.jit.unused
    def _forward_backbone(self, *args, **kwargs):
        """Run the backbone forward pass, skipping adaptation submodules."""
        if type(self).forward is torch.nn.Sequential.forward:
            x = args[0] if args else next(iter(kwargs.values()))
            for name, module in self._modules.items():
                if module is not None and name not in (
                    "channel_layer",
                    "input_resample",
                ):
                    x = module(x)
            return x
        return type(self).forward(self, *args, **kwargs)

    @torch.jit.unused
    def _apply_channel_layer(
        self,
        args: tuple,
        kwargs: dict,
        *,
        chs_info=None,
        ch_names=None,
    ) -> tuple[tuple, dict]:
        """Map channel-bearing inputs through ``self.channel_layer``."""
        kwargs = dict(kwargs)
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

        def _map_signal(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 4 and t.shape[3] == 1:
                return self.channel_layer(t.squeeze(3), chs_info, ch_mask)[0].unsqueeze(
                    3
                )
            if (
                t.ndim == 4
                and t.shape[1] == 1
                and type(self).__name__ == "SleepStagerChambon2018"
            ):
                return self.channel_layer(t.squeeze(1), chs_info, ch_mask)[0].unsqueeze(
                    1
                )
            return self.channel_layer(t, chs_info, ch_mask)[0]

        if args:
            args = (_map_signal(args[0]), *args[1:])
        else:  # x by keyword: the backbone's first input (x, X, eeg...)
            x = _forward_param_names(type(self).forward)[0]
            kwargs[x] = _map_signal(kwargs[x])

        # Secondary time inputs (CORTEG's x_hi) carry the same channels as x.
        for idx, inp_name in enumerate(self._time_inputs[1:], start=1):
            t = self._get_time_input(idx, inp_name, args, kwargs)
            if t is not None:
                args, kwargs = self._set_time_input(
                    idx, inp_name, _map_signal(t), args, kwargs
                )
        return args, kwargs

    def _channel_forward(self, *args, chs_info=None, ch_names=None, **kwargs):
        """``forward`` under a channel strategy: map ``x`` (recorded with
        ``chs_info``, default the model's) onto the target, then run the backbone."""
        args, kwargs = self._apply_channel_layer(
            args, kwargs, chs_info=chs_info, ch_names=ch_names
        )
        return self._forward_backbone(*args, **kwargs)

    @torch.jit.unused
    def _check_montage_adaptation(self, chs_info: Optional[list[dict]]) -> None:
        """Hook for model-specific montage adaptation refusals."""
        return None

    @torch.jit.unused
    def _validate_adapted_n_times(self, n_times: int) -> None:
        """Hook for model-specific checks when adapting fixed-time heads."""
        return None

    @torch.jit.unused
    def _pool_adapted_time(
        self, x: torch.Tensor, target: int, dim: int = -1
    ) -> torch.Tensor:
        """Pool time/token axis ``dim`` to ``target`` only when ``adapt_input(n_times=...)`` is active."""
        if self._adapted_n_times is None or x.shape[dim] == target:
            return x
        dim = dim % x.ndim
        moved = x.movedim(dim, -1)
        shape = moved.shape[:-1]
        pooled = F.adaptive_avg_pool1d(moved.reshape(-1, 1, moved.shape[-1]), target)
        return pooled.reshape(*shape, target).movedim(-1, dim)

    @torch.jit.unused
    def _reference_device_dtype(self) -> Optional[tuple[torch.device, torch.dtype]]:
        """Device and floating dtype of the backbone (before adaptation submodules)."""
        skip_ids = set()
        for attr in ("channel_layer", "input_resample"):
            sub = self._modules.get(attr)
            if sub is not None:
                for t in (*sub.parameters(), *sub.buffers()):
                    skip_ids.add(id(t))
        for p in self.parameters():
            if id(p) not in skip_ids and p.is_floating_point():
                return p.device, p.dtype
        for b in self.buffers():
            if id(b) not in skip_ids and b.is_floating_point():
                return b.device, b.dtype
        return None

    @torch.jit.unused
    def adapt_input(
        self,
        *,
        chs_info: Optional[list[dict]] = None,
        sfreq: Optional[float] = None,
        n_times: Optional[int] = None,
        channel_strategy: Optional[str] = None,
        channel_strategy_kwargs: Optional[dict] = None,
    ) -> "EEGModuleMixin":
        """Adapt a built model in place to a new montage, sampling rate, or window length.

        All adaptations keep the backbone weights and ``state_dict`` keys intact
        and are persisted in :meth:`get_config` and Hub saves under
        ``"input_adaptation"``.

        Parameters
        ----------
        chs_info : list of dict or None, default=None
            Channel metadata of the incoming recordings. Mapped onto the
            montage the model was built with using ``channel_strategy``
            (default ``"nearest"`` when ``chs_info`` is given).
        sfreq : float or None, default=None
            Sampling frequency of the incoming recordings in Hz. Inputs are
            resampled inside forward when this differs from ``self.sfreq``.
            Passing ``sfreq == self.sfreq`` clears any active resampler.
        n_times : int or None, default=None
            Expected window length in samples for incoming recordings. Free and
            dense models accept this directly; supported fixed-time models
            adaptively pool before their classification head while remaining
            bit-identical at their build ``n_times``.
        channel_strategy : str or None, default=None
            Stateless channel mapping strategy from :data:`_ADAPT_STRATEGIES`.
            When set with ``chs_info=None``, per-sample ``(pos, ch_mask)``
            batches (from ``pad_channels_collate``) are mapped dynamically.
        channel_strategy_kwargs : dict or None, default=None
            Extra keyword arguments passed to
            :class:`~braindecode.modules.ChannelLayer`.

        Returns
        -------
        self : EEGModuleMixin
            The model instance, modified in place.
        """
        name = type(self).__name__
        if (
            channel_strategy_kwargs is not None
            and chs_info is None
            and channel_strategy is None
        ):
            raise ValueError(
                "channel_strategy_kwargs requires chs_info or channel_strategy."
            )

        ref = self._reference_device_dtype()

        # --- Montage validation and ChannelLayer build ---
        adapt_montage = chs_info is not None or channel_strategy is not None
        new_layer = None
        resolved_strategy = None
        resolved_kw: Optional[dict] = None
        parsed_chs: Optional[list[dict]] = None
        if adapt_montage:
            if not self._supports_montage_adaptation:
                raise ValueError(
                    f"{name} reads non-EEG channels, not an EEG montage: "
                    "montage adaptation is unsupported."
                )
            strategy = "nearest" if channel_strategy is None else channel_strategy
            kw = dict(channel_strategy_kwargs or {})
            if strategy not in _ADAPT_STRATEGIES or (
                strategy == "source" and bool(kw.get("trainable", False))
            ):
                raise ValueError(
                    f"adapt_input takes {_ADAPT_STRATEGIES}: {strategy!r} "
                    f"adds learned or fitted state to the state_dict."
                )
            if strategy == "source":
                kw["trainable"] = False
            if (
                chs_info is not None
                and isinstance(chs_info, list)
                and len(chs_info) > 0
                and isinstance(chs_info[0], dict)
                and "loc" in chs_info[0]
                and isinstance(chs_info[0]["loc"], list)
            ):
                parsed_chs = self._deserialize_chs_info(chs_info)
            else:
                parsed_chs = list(chs_info) if chs_info is not None else None
            self._check_montage_adaptation(parsed_chs)
            if self._chs_info is None:
                raise ValueError(
                    f"{name}.adapt_input(...) needs the montage the model was "
                    f"built with; rebuild it with chs_info=..."
                )
            new_layer = ChannelLayer(self._chs_info, strategy, parsed_chs, **kw)
            if parsed_chs is None and np.isnan(new_layer._tpos).any():
                raise ValueError(
                    f"{name}.adapt_input(channel_strategy={strategy!r}) needs valid "
                    "3D channel locations ('loc' or standard 10-05 names) on the "
                    "model montage to map per-sample batches dynamically."
                )
            if ref is not None:
                new_layer = new_layer.to(device=ref[0], dtype=ref[1])
            resolved_strategy = strategy
            resolved_kw = (
                dict(channel_strategy_kwargs) if channel_strategy_kwargs else None
            )

        # --- Sampling rate validation and Resample build ---
        adapt_sfreq = sfreq is not None
        new_resample = None
        new_input_sfreq: Optional[float] = None
        if adapt_sfreq:
            if (
                isinstance(sfreq, bool)
                or not isinstance(sfreq, (int, float, np.integer, np.floating))
                or not math.isfinite(float(sfreq))
                or float(sfreq) <= 0
            ):
                raise ValueError(
                    f"{name}.adapt_input sfreq must be positive, got {sfreq!r}."
                )
            model_sfreq = self._signal_param("sfreq")
            if model_sfreq is None:
                raise ValueError(
                    f"{name}.adapt_input(sfreq=...) requires the model's sfreq "
                    "to be specified at construction."
                )
            if float(model_sfreq) != round(float(model_sfreq)):
                raise ValueError(
                    f"{name} has non-integer model sfreq={model_sfreq}; "
                    "in-model resampling requires integer rates."
                )
            sfreq_val = float(sfreq)
            if sfreq_val != float(model_sfreq):
                new_resample, _ = _in_model_resample(
                    name,
                    sfreq_val,
                    int(round(float(model_sfreq))),
                    self._signal_param("n_times") or 0,
                )
                if ref is not None:
                    new_resample = new_resample.to(device=ref[0], dtype=ref[1])
                new_input_sfreq = sfreq_val

        # --- Window length validation ---
        adapt_times = n_times is not None
        new_adapted_n_times: Optional[int] = None
        if adapt_times:
            if (
                isinstance(n_times, bool)
                or not isinstance(n_times, (int, np.integer))
                or int(n_times) <= 0
            ):
                raise ValueError(
                    f"{name}.adapt_input n_times must be a positive integer, got {n_times!r}."
                )
            target_n_times = int(n_times)
            time_axis = self._effective_time_axis()
            if time_axis in {"free", "dense"}:
                if self._min_n_times is not None and target_n_times < self._min_n_times:
                    raise ValueError(
                        f"{name} needs at least {self._min_n_times} time samples; "
                        f"got n_times={target_n_times}."
                    )
                new_adapted_n_times = target_n_times
            elif time_axis == "fixed":
                if not self._supports_n_times_adaptation:
                    param = self._fixed_time_param or "internal layers"
                    raise ValueError(
                        f"{name} fixes {param} to n_times={self._signal_param('n_times')}; "
                        "adapt_input(n_times=...) cannot resize fixed inner layers."
                    )
                if self._signal_param("n_times") is None:
                    raise ValueError(
                        f"{name}.adapt_input(n_times=...) needs the build n_times to be known."
                    )
                if self._min_n_times is not None and target_n_times < self._min_n_times:
                    raise ValueError(
                        f"{name} needs at least {self._min_n_times} time samples; "
                        f"got n_times={target_n_times}."
                    )
                self._validate_adapted_n_times(target_n_times)
                new_adapted_n_times = target_n_times
            else:
                new_adapted_n_times = target_n_times

        # --- Apply validated changes in-place (clean replacement) ---
        if adapt_montage:
            assert new_layer is not None
            self.channel_layer = new_layer
            self._adapt_channel_layer = True
            self._adapt_chs_info = parsed_chs
            self._adapt_channel_strategy = resolved_strategy
            self._adapt_channel_strategy_kwargs = resolved_kw

        if adapt_sfreq:
            if new_resample is None:
                if hasattr(self, "input_resample"):
                    del self.input_resample
                self._input_sfreq = None
            else:
                self.input_resample = new_resample
                self._input_sfreq = new_input_sfreq

        if adapt_times:
            self._adapted_n_times = new_adapted_n_times

        if (
            hasattr(self, "input_resample")
            or self._adapt_channel_layer
            or (self._adapted_n_times is not None and self._supports_n_times_adaptation)
        ):
            setattr(self, "forward", self._adapted_forward)
        elif hasattr(self, "channel_layer"):
            setattr(self, "forward", self._channel_forward)
        else:
            self.__dict__.pop("forward", None)
        return self

    def _adapted_forward(self, *args, **kwargs):
        """``forward`` after :meth:`adapt_input`: resample, map montage, run backbone."""
        if hasattr(self, "input_resample"):
            kwargs = dict(kwargs)
            time_axis = self._effective_time_axis()
            first = self._first_input(args, kwargs)
            bypass_time = first is not None and self._bypass_time_check(
                first, args, kwargs
            )
            for idx, inp_name in enumerate(self._time_inputs):
                if inp_name == "x_hi":
                    continue
                t = self._get_time_input(idx, inp_name, args, kwargs)
                if t is None or t.ndim < 2:
                    continue
                time_dim = 2 if (idx == 0 and t.ndim == 4 and t.shape[3] == 1) else -1
                t = _apply_resample(self.input_resample, t, time_dim=time_dim)
                target = self._check_length_rule(
                    t.shape[time_dim],
                    idx=idx,
                    inp_name=inp_name,
                    shape=tuple(t.shape),
                    time_axis=time_axis,
                    bypass_time=bypass_time,
                    already_resampled=True,
                )
                if target is not None and t.shape[time_dim] != target:
                    t = (
                        self._snap_tensor_time(t, target)
                        if idx == 0
                        else snap_time_axis(t, target)
                    )
                args, kwargs = self._set_time_input(idx, inp_name, t, args, kwargs)

        if hasattr(self, "channel_layer"):
            kwargs = dict(kwargs)
            chs_info = kwargs.pop("chs_info", None)
            ch_names = kwargs.pop("ch_names", None)
            args, kwargs = self._apply_channel_layer(
                args, kwargs, chs_info=chs_info, ch_names=ch_names
            )
        return self._forward_backbone(*args, **kwargs)

    @torch.jit.unused
    def _adaptation_config(self) -> Optional[dict[str, Any]]:
        """Serialized ``input_adaptation`` payload, or ``None`` when unadapted."""
        if (
            not self._adapt_channel_layer
            and self._input_sfreq is None
            and self._adapted_n_times is None
        ):
            return None
        return {
            "chs_info": (
                self._serialize_chs_info(self._adapt_chs_info)
                if self._adapt_chs_info is not None
                else None
            ),
            "sfreq": self._input_sfreq,
            "n_times": self._adapted_n_times,
            "channel_strategy": self._adapt_channel_strategy,
            "channel_strategy_kwargs": deepcopy(self._adapt_channel_strategy_kwargs),
        }

    @torch.jit.unused
    def _apply_input_adaptation_config(self, spec: dict[str, Any]) -> None:
        """Re-apply a saved ``input_adaptation`` config dict on this model."""
        kwargs: dict[str, Any] = {
            k: spec[k]
            for k in (
                "sfreq",
                "n_times",
                "channel_strategy",
                "channel_strategy_kwargs",
            )
            if spec.get(k) is not None
        }
        if spec.get("chs_info") is not None:
            kwargs["chs_info"] = self._deserialize_chs_info(spec["chs_info"])
        if kwargs:
            self.adapt_input(**kwargs)

    @property
    def n_outputs(self) -> int:
        if self._n_outputs is None:
            raise ValueError("n_outputs not specified.")
        return self._n_outputs

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

    def __prepare_scriptable__(self):
        # Called by torch.jit.script before compiling the model.
        return make_parametrizations_scriptable(self)

    @torch.jit.unused
    def _signal_param(self, name: str) -> Any:
        """Return the value of a signal property, or ``None`` if unspecified."""
        if name == "n_outputs":
            return self._n_outputs
        if name == "chs_info":
            return self._chs_info
        if name == "n_chans":
            if self._n_chans is not None:
                return self._n_chans
            if self._chs_info is not None:
                return len(self._chs_info)
            return None
        if name == "n_times":
            if self._n_times is not None:
                return self._n_times
            if self._input_window_seconds is not None and self._sfreq is not None:
                return int(round(self._input_window_seconds * self._sfreq))
            return None
        if name == "input_window_seconds":
            if self._input_window_seconds is not None:
                return self._input_window_seconds
            if self._n_times is not None and self._sfreq is not None:
                return float(self._n_times / self._sfreq)
            return None
        if name == "sfreq":
            if self._sfreq is not None:
                return self._sfreq
            if self._input_window_seconds is not None and self._n_times is not None:
                return float(self._n_times / self._input_window_seconds)
            return None
        raise ValueError(f"Unknown signal parameter: {name!r}.")

    @torch.jit.unused
    def _first_input(self, args: tuple, kwargs: dict) -> Optional[torch.Tensor]:
        """Return the primary input tensor from ``args`` or ``kwargs``."""
        if args:
            first = args[0]
            return first if torch.is_tensor(first) else None
        names = _forward_param_names(type(self).forward)
        if names and names[0] in kwargs:
            first = kwargs[names[0]]
            return first if torch.is_tensor(first) else None
        return None

    @torch.jit.unused
    def _call_arg(
        self, param_name: str, args: tuple, kwargs: dict, default: Any = None
    ) -> Any:
        """Read ``param_name`` from ``kwargs`` or its positional slot in ``args``."""
        if param_name in kwargs:
            return kwargs[param_name]
        names = _forward_param_names(type(self).forward)
        if param_name in names:
            idx = names.index(param_name)
            if idx < len(args):
                return args[idx]
        return default

    @torch.jit.unused
    def _bypass_channel_check(self, x: torch.Tensor, args: tuple, kwargs: dict) -> bool:
        """Whether per-call spatial metadata or encoder-only mode frees channels."""
        return False

    @torch.jit.unused
    def _bypass_time_check(self, x: torch.Tensor, args: tuple, kwargs: dict) -> bool:
        """Whether per-call flags or pooling mode frees the time axis."""
        return False

    @torch.jit.unused
    def _effective_channel_axis(self) -> Optional[str]:
        return self._channel_axis

    @torch.jit.unused
    def _effective_time_axis(self) -> Optional[str]:
        return self._time_axis

    @torch.jit.unused
    def _input_n_chans(self, x: torch.Tensor) -> int:
        if (
            x.ndim == 4
            and x.shape[1] == 1
            and type(self).__name__ == "SleepStagerChambon2018"
        ):
            return x.shape[2]
        return x.shape[1]

    @torch.jit.unused
    def _input_n_times(self, x: torch.Tensor) -> int:
        if x.ndim == 4 and x.shape[3] == 1:
            return x.shape[2]
        return x.shape[-1]

    @torch.jit.unused
    def _get_time_input(
        self, idx: int, inp_name: str, args: tuple, kwargs: dict
    ) -> Optional[torch.Tensor]:
        if idx == 0:
            return self._first_input(args, kwargs)
        val = self._call_arg(inp_name, args, kwargs)
        return val if torch.is_tensor(val) else None

    @torch.jit.unused
    def _set_time_input(
        self,
        idx: int,
        inp_name: str,
        value: torch.Tensor,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        names = _forward_param_names(type(self).forward)
        if idx == 0:
            if args:
                return (value, *args[1:]), kwargs
            if names and names[0] in kwargs:
                kwargs = dict(kwargs)
                kwargs[names[0]] = value
                return args, kwargs
        if inp_name in kwargs:
            kwargs = dict(kwargs)
            kwargs[inp_name] = value
            return args, kwargs
        if inp_name in names:
            pos = names.index(inp_name)
            if pos < len(args):
                args_list = list(args)
                args_list[pos] = value
                return tuple(args_list), kwargs
        return args, kwargs

    @torch.jit.unused
    def _snap_tensor_time(self, t: torch.Tensor, target: int) -> torch.Tensor:
        time_dim = 2 if (t.ndim == 4 and t.shape[3] == 1) else -1
        return snap_time_axis(t, target, time_dim=time_dim)

    @torch.jit.unused
    def _post_resample_length(self, t_len: int) -> int:
        if self._input_sfreq is None:
            return t_len
        return _resample_out_length(
            t_len,
            int(round(self._input_sfreq)),
            int(round(float(self.sfreq))),
        )

    @torch.jit.unused
    def _check_length_rule(
        self,
        t_len: int,
        *,
        idx: int,
        inp_name: str,
        shape: tuple[int, ...],
        time_axis: Optional[str],
        bypass_time: bool,
        already_resampled: bool = False,
    ) -> Optional[int]:
        """Validate ``t_len`` (after resampling if active) and return target snap length."""
        name = type(self).__name__
        resampling = self._input_sfreq is not None and inp_name != "x_hi"
        eff_len = (
            t_len
            if (already_resampled or not resampling)
            else self._post_resample_length(t_len)
        )
        suffix = (
            f" after resampling from {self._input_sfreq:g} Hz" if resampling else ""
        )
        if time_axis == "fixed" and not bypass_time:
            adapted = self._adapted_n_times is not None
            expected = (
                self._adapted_n_times if adapted else self._signal_param("n_times")
            )
            if expected is not None and len(shape) >= 2:
                if eff_len == expected:
                    return None
                step = (self._time_input_steps or {}).get(inp_name, self._n_times_step)
                cap = min(max(1, expected // 100), max(1, step))
                if self.on_non_divisible != "error" and abs(eff_len - expected) <= cap:
                    return expected
                label = "" if idx == 0 else f" ({inp_name})"
                verb = "adapted" if adapted else "built"
                raise ValueError(
                    f"{name} was {verb} for n_times={expected} and got "
                    f"{eff_len} samples{suffix}{label} (input shape {shape}). "
                    f"Rebuild it with n_times={eff_len}, cut the signal "
                    f"into windows of {expected} samples, or call "
                    f"model.adapt_input(sfreq=...) / "
                    f"model.adapt_input(n_times=...)."
                )
        elif time_axis in {"free", "dense"} and (
            self._n_times_step > 1 or self._time_input_steps
        ):
            step = (self._time_input_steps or {}).get(inp_name, self._n_times_step)
            if step <= 1 or len(shape) < 2:
                return None
            if eff_len < 1:
                min_req = 1 if self.on_non_divisible == "pad" else step
                raise ValueError(
                    f"{name} needs at least {min_req} time samples for "
                    f"{inp_name} (step {step}) and got {eff_len}{suffix} "
                    f"(input shape {shape})."
                )
            rem = (eff_len - self._n_times_offset) % step
            if rem == 0:
                return None
            if self.on_non_divisible == "error":
                raise ValueError(
                    f"Time dimension ({eff_len}) of {name} is not divisible by "
                    f"step {step}{suffix} (input shape {shape})."
                )
            target = (
                eff_len + (step - rem)
                if self.on_non_divisible == "pad"
                else eff_len - rem
            )
            if not self._defer_step_snap:
                if target < 1:
                    raise ValueError(
                        f"{name} needs at least {step} time samples for "
                        f"{inp_name} (step {step}) and got {eff_len}{suffix} "
                        f"(input shape {shape})."
                    )
                return target
        return None

    @torch.jit.unused
    def _check_input_shape(self, x: torch.Tensor, args: tuple, kwargs: dict) -> None:
        """Validate channel count and minimum window length before forward."""
        name = type(self).__name__
        if (
            self._effective_channel_axis() == "fixed"
            and not hasattr(self, "channel_layer")
            and x.ndim >= 2
            and not self._bypass_channel_check(x, args, kwargs)
        ):
            n_chans = self._signal_param("n_chans")
            if n_chans is not None:
                got_chans = self._input_n_chans(x)
                if got_chans != n_chans:
                    raise ValueError(
                        f"{name} was built for n_chans={n_chans} and got {got_chans} "
                        f"channels (input shape {tuple(x.shape)}). Rebuild it with "
                        f"n_chans={got_chans}, or map the montage with "
                        f"model.adapt_input(chs_info=...)."
                    )
        if self._min_n_times is not None and (
            x.ndim == 3
            or (
                x.ndim == 4
                and (
                    x.shape[3] == 1
                    or name in ("USleep", "SleepStagerChambon2018")
                    or (
                        name == "MAPA"
                        and getattr(self, "normalization", None) == "session"
                    )
                )
            )
        ):
            got_times = self._post_resample_length(self._input_n_times(x))
            suffix = (
                f" after resampling from {self._input_sfreq:g} Hz"
                if self._input_sfreq is not None
                else ""
            )
            if got_times < self._min_n_times:
                raise ValueError(
                    f"{name} needs at least {self._min_n_times} time samples and got "
                    f"{got_times}{suffix} (input shape {tuple(x.shape)})."
                )

    @torch.jit.unused
    def _prepare_inputs(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Validate input axes and snap window lengths within policy bounds."""
        x = self._first_input(args, kwargs)
        if x is None:
            return args, kwargs
        self._check_input_shape(x, args, kwargs)
        name = type(self).__name__
        bypass_time = self._bypass_time_check(x, args, kwargs)
        time_axis = self._effective_time_axis()

        if time_axis == "fixed" and not bypass_time and x.ndim >= 2:
            snaps: list[tuple[int, str, torch.Tensor, int, int]] = []
            for idx, inp_name in enumerate(self._time_inputs):
                t = self._get_time_input(idx, inp_name, args, kwargs)
                if t is None or t.ndim < 2:
                    continue
                t_len = self._input_n_times(t) if idx == 0 else t.shape[-1]
                target = self._check_length_rule(
                    t_len,
                    idx=idx,
                    inp_name=inp_name,
                    shape=tuple(t.shape),
                    time_axis=time_axis,
                    bypass_time=bypass_time,
                )
                if target is not None:
                    eff_len = (
                        t_len
                        if (self._input_sfreq is None or inp_name == "x_hi")
                        else self._post_resample_length(t_len)
                    )
                    snaps.append((idx, inp_name, t, eff_len, target))
            if snaps:
                if self._input_sfreq is None:
                    for idx, inp_name, t, _, target in snaps:
                        snapped = (
                            self._snap_tensor_time(t, target)
                            if idx == 0
                            else snap_time_axis(t, target)
                        )
                        args, kwargs = self._set_time_input(
                            idx, inp_name, snapped, args, kwargs
                        )
                if not self._warned_length_snap and not self._explicit_on_non_divisible:
                    self._warned_length_snap = True
                    first_len, first_target = snaps[0][3], snaps[0][4]
                    warnings.warn(
                        f"{name} snapped input length from {first_len} to "
                        f"{first_target} samples. Pass on_non_divisible='error' "
                        f"to reject mismatched window lengths.",
                        UserWarning,
                        stacklevel=3,
                    )
        elif time_axis in {"free", "dense"} and (
            self._n_times_step > 1 or self._time_input_steps
        ):
            build_n_times = self._signal_param("n_times")
            for idx, inp_name in enumerate(self._time_inputs):
                t = self._get_time_input(idx, inp_name, args, kwargs)
                if t is None or t.ndim < 2:
                    continue
                t_len = self._input_n_times(t) if idx == 0 else t.shape[-1]
                target = self._check_length_rule(
                    t_len,
                    idx=idx,
                    inp_name=inp_name,
                    shape=tuple(t.shape),
                    time_axis=time_axis,
                    bypass_time=bypass_time,
                )
                if target is None:
                    continue
                resampling = self._input_sfreq is not None and inp_name != "x_hi"
                eff_len = self._post_resample_length(t_len) if resampling else t_len
                if not resampling:
                    snapped = (
                        self._snap_tensor_time(t, target)
                        if idx == 0
                        else snap_time_axis(t, target)
                    )
                    args, kwargs = self._set_time_input(
                        idx, inp_name, snapped, args, kwargs
                    )
                if (
                    not self._warned_length_snap
                    and not self._explicit_on_non_divisible
                    and (idx > 0 or eff_len != build_n_times)
                ):
                    self._warned_length_snap = True
                    step = (self._time_input_steps or {}).get(
                        inp_name, self._n_times_step
                    )
                    warnings.warn(
                        f"{name} snapped input length from {eff_len} to {target} "
                        f"samples (step {step}, "
                        f"on_non_divisible={self.on_non_divisible!r}).",
                        UserWarning,
                        stacklevel=3,
                    )
        return args, kwargs

    def __call__(self, *args, **kwargs):
        args, kwargs = self._prepare_inputs(args, kwargs)
        x = self._first_input(args, kwargs)
        need_lazy_geometry = (
            x is not None
            and x.ndim >= 3
            and (
                (
                    self._effective_channel_axis() == "fixed"
                    and self._signal_param("n_chans") is None
                    and not hasattr(self, "channel_layer")
                )
                or (
                    self._time_axis == "fixed" and self._signal_param("n_times") is None
                )
            )
        )
        had_lazy = need_lazy_geometry and any(
            isinstance(p, torch.nn.UninitializedParameter) for p in self.parameters()
        )
        out = super().__call__(*args, **kwargs)
        if had_lazy and not any(
            isinstance(p, torch.nn.UninitializedParameter) for p in self.parameters()
        ):
            assert x is not None
            if (
                self._effective_channel_axis() == "fixed"
                and self._signal_param("n_chans") is None
                and not hasattr(self, "channel_layer")
            ):
                self._n_chans = self._input_n_chans(x)
                self._n_chans_for_jit = self._n_chans
            if self._time_axis == "fixed" and self._signal_param("n_times") is None:
                raw_times = self._input_n_times(x)
                self._n_times = (
                    self._post_resample_length(raw_times)
                    if self._input_sfreq is not None
                    else raw_times
                )
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
                probe = torch.zeros(
                    self.input_shape,
                    dtype=next(self.parameters()).dtype,  # type: ignore
                    device=next(self.parameters()).device,  # type: ignore
                )
                self._check_input_shape(probe, (probe,), {})
                return tuple(self._forward_backbone(probe).shape)
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
        input_adaptation = config.pop("input_adaptation", None)
        cls._fill_legacy_defaults(config)
        resolve_type_kwargs(cls, config)
        model = cls(**config)
        if input_adaptation is not None:
            model._apply_input_adaptation_config(input_adaptation)
        return model

    @classmethod
    def _fill_legacy_defaults(cls, kwargs: dict) -> None:
        """Rebuild configs saved by earlier versions as they were.

        Models that used to crop a time axis not divisible by their patch size
        set ``_legacy_on_non_divisible = "crop"``. A saved config or checkpoint
        without the ``on_non_divisible`` key then loads with ``"crop"``, so its
        head and position tables keep their shapes; a new model defaults to
        ``"pad"``. A config saved while a model that reads its input natively
        still took ``channel_strategy`` (2.1 development builds) loses its
        ``"native"`` strategy keys.
        """
        params = inspect.signature(cls.__init__).parameters
        takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
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

    def reset_head(self, n_outputs):
        """Replace the classification head for a new number of outputs.

        This is called automatically by :meth:`from_pretrained` when the
        user passes an ``n_outputs`` that differs from the saved config.
        Override in subclasses that need a model-specific head structure.
        Implementations keep changed constructor arguments in sync with
        ``self._update_init_kwargs``, so that a saved model can be loaded
        back. Implementations requiring positive outputs can also use
        ``self._set_n_outputs`` to validate and record the new value.

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
        raise NotImplementedError(
            f"{type(self).__name__} does not implement reset_head(). "
            "Override this method to support changing n_outputs after "
            "loading pretrained weights."
        )

    mapping: Optional[Dict[str, str]] = None
    #: Channels the backbone consumes under a ``channel_strategy`` (names or
    #: ``chs_info``); ``None``: the model's own ``chs_info``.
    _channel_target: Optional[list] = None
    _time_axis: ClassVar[Optional[str]] = None
    _channel_axis: ClassVar[Optional[str]] = None
    _time_inputs: ClassVar[tuple[str, ...]] = ("x",)
    _min_n_times: Optional[int] = None
    _n_times_step = 1
    _n_times_offset = 0
    _time_input_steps = None
    _defer_step_snap = False
    _warned_length_snap = False
    on_non_divisible = "crop"
    _explicit_on_non_divisible = False
    _supports_montage_adaptation: ClassVar[bool] = True
    _supports_n_times_adaptation: ClassVar[bool] = False
    _fixed_time_param: ClassVar[Optional[str]] = None
    _input_sfreq: Optional[float] = None
    _adapted_n_times: Optional[int] = None
    _adapt_channel_layer: bool = False
    _adapt_chs_info: Optional[list[dict]] = None
    _adapt_channel_strategy: Optional[str] = None
    _adapt_channel_strategy_kwargs: Optional[dict] = None
    _ctor_chs_info: Optional[list[dict]] = None

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
        """
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
        if self._min_n_times is not None:
            time_idx = (
                0 if type(self).__name__ in ("Deep4Net", "ShallowFBCSPNet") else -1
            )
            need = 1
            for m in reversed(
                [mod for mod in self.modules() if hasattr(mod, "kernel_size")]
            ):
                k = (
                    m.kernel_size[time_idx]
                    if hasattr(m.kernel_size, "__len__")
                    else m.kernel_size
                )
                d_attr = getattr(m, "dilation", 1)
                d = d_attr[time_idx] if hasattr(d_attr, "__len__") else d_attr
                s_attr = getattr(m, "stride", 1)
                s = s_attr[time_idx] if hasattr(s_attr, "__len__") else s_attr
                p_attr = getattr(m, "padding", 0)
                if p_attr == "same":
                    need = max(1, s * (need - 1) + 1)
                else:
                    p = p_attr[time_idx] if hasattr(p_attr, "__len__") else p_attr
                    need = max(1, s * (need - 1) + (k - 1) * d + 1 - 2 * p)
            self._min_n_times = need

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
        known_chans = self._signal_param("n_chans")
        n_chans = 1 if known_chans is None else known_chans
        known_times = self._signal_param("n_times")
        n_times = 200 if known_times is None else known_times

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

    @staticmethod
    def _can_view_flat(tensor: torch.Tensor) -> bool:
        """Whether ``tensor.view(-1)`` succeeds (matches safetensors ``_end_ptr``)."""
        dims = [(s, st) for s, st in zip(tensor.shape, tensor.stride()) if s != 1]
        if len(dims) <= 1:
            return True
        return all(
            dims[i][1] == dims[i + 1][0] * dims[i + 1][1] for i in range(len(dims) - 1)
        )

    @classmethod
    def _safetensors_ineligible_reason(
        cls, state_dict: dict[str, torch.Tensor]
    ) -> Optional[str]:
        """Return a reason if ``safetensors`` cannot save ``state_dict``, else None."""
        from torch.nn.parameter import UninitializedParameter

        groups: dict[tuple[torch.device, int, int], list[torch.Tensor]] = {}
        for tensor in state_dict.values():
            if isinstance(tensor, UninitializedParameter):
                return "model contains uninitialized parameters"
            if tensor.is_meta:
                continue
            storage = tensor.untyped_storage()
            storage_ptr = storage.data_ptr()
            storage_nbytes = storage.nbytes()
            if storage_ptr == 0 or storage_nbytes == 0:
                continue
            groups.setdefault((tensor.device, storage_ptr, storage_nbytes), []).append(
                tensor
            )

        for (_, storage_ptr, storage_nbytes), tensors in groups.items():
            if len(tensors) < 2:
                clusters: list[list[tuple[int, int, torch.Tensor]]] = [
                    [(0, 0, tensors[0])]
                ]
            else:
                areas: list[tuple[int, int, torch.Tensor]] = []
                for tensor in tensors:
                    if tensor.nelement() > 0 and not cls._can_view_flat(tensor):
                        return (
                            "shared non-contiguous tensor cannot be flattened by "
                            "safetensors"
                        )
                    start = tensor.data_ptr()
                    if tensor.nelement() > 0:
                        span = sum(
                            (size - 1) * stride
                            for size, stride in zip(tensor.shape, tensor.stride())
                        )
                        stop = start + (span + 1) * tensor.element_size()
                    else:
                        stop = start
                    areas.append((start, stop, tensor))
                areas.sort(key=lambda item: (item[0], item[1]))
                clusters = [[areas[0]]]
                last_stop = areas[0][1]
                for item in areas[1:]:
                    if item[0] >= last_stop:
                        clusters.append([item])
                    else:
                        clusters[-1].append(item)
                    # Mirror safetensors.torch._filter_shared_not_shared's stop update.
                    last_stop = item[1]
            for cluster in clusters:
                has_complete = any(
                    t.data_ptr() == storage_ptr
                    and t.nelement() * t.element_size() == storage_nbytes
                    for _, _, t in cluster
                )
                if not has_complete:
                    return (
                        "some tensors share storage without a complete tensor "
                        "covering the underlying storage"
                    )
        return None

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
            import safetensors.torch  # noqa: F401
        except ImportError as e:
            # Fall back to pytorch_model.bin, and drop the model.safetensors of a
            # previous save, which would be loaded in its place.
            (save_directory / "model.safetensors").unlink(missing_ok=True)
            warnings.warn(
                f"Could not save model in safetensors format: {e}. "
                "Model weights saved in pytorch_model.bin instead.",
                stacklevel=2,
            )
            return

        reason = self._safetensors_ineligible_reason(self.state_dict())
        if reason is not None:
            (save_directory / "model.safetensors").unlink(missing_ok=True)
            warnings.warn(
                f"Could not save model in safetensors format: {reason}. "
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
            **model_kwargs,
        ):
            model_kwargs.pop("braindecode_version", None)
            input_adaptation = model_kwargs.pop("input_adaptation", None)
            filename = model_kwargs.pop("filename", None)
            if filename is not None:
                _validate_weights_filename(filename)
            # The Hub mixin passes only the keys of config.json (plus the
            # caller's), so a missing key means a pre-on_non_divisible save.
            cls._fill_legacy_defaults(model_kwargs)
            resolve_type_kwargs(cls, model_kwargs)

            # config.json gives n_outputs (to detect a head swap) and the saved
            # time geometry, for local directories and Hub repo IDs alike.
            is_local = _is_local_dir(model_id)
            saved_config = {}
            config_file = _resolve_hub_or_local_file(
                model_id,
                "config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
            if config_file is not None:
                with open(config_file, "r") as f:
                    saved_config = json.load(f)
            if input_adaptation is None:
                input_adaptation = saved_config.get("input_adaptation")

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

            if input_adaptation is not None:
                model._apply_input_adaptation_config(input_adaptation)

            return model
