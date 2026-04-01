# Authors: Pierre Guetschel
#          Maciej Sliwowski
#
# License: BSD-3

from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Type, Union

import numpy as np
import torch
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from mne.utils import _soft_import
from torchinfo import ModelStatistics, summary

from braindecode.models.util import (
    _EEG_PARAMS,
    _IMPORT_ADAPTER,
    build_model_config,
    resolve_type_kwargs,
    track_model_init_kwargs,
)
from braindecode.version import __version__

huggingface_hub = _soft_import(
    "huggingface_hub", "Hugging Face Hub integration", strict=False
)

HAS_HF_HUB = huggingface_hub is not False


class _BaseHubMixin:
    pass


# Define base class for hub mixin
if HAS_HF_HUB:
    _BaseHubMixin: Type = huggingface_hub.PyTorchModelHubMixin  # type: ignore


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


class EEGModuleMixin(_BaseHubMixin, metaclass=NumpyDocstringInheritanceInitMeta):
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

    Raises
    ------
    ValueError: If some input signal-related parameters are not specified
                and can not be inferred.

    Notes
    -----
    If some input signal-related parameters are not specified,
    there will be an attempt to infer them from the other parameters.
    """

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

        # Push to the Hub
        model.push_to_hub(
            repo_id="username/my-{name_lower}-model",
            commit_message="Initial model upload",
        )

    **Loading a model from the Hub:**

    .. code-block::

        from braindecode.models import {name}

        # Load pretrained model
        model = {name}.from_pretrained("username/my-{name_lower}-model")

    The integration automatically handles EEG-specific parameters (n_chans,
    n_times, sfreq, chs_info, etc.) by saving them in a config file alongside
    the model weights. This ensures that loaded models are correctly configured
    for their original data specifications.

    All model parameters (both EEG-specific and model-specific such as
    dropout rates, activation functions, number of filters) are automatically
    saved to the Hub and restored when loading.
    """

    def __init_subclass__(cls, **kwargs):
        # Append model-specific Hub integration notes to the docstring.
        # This runs after the metaclass, so we concatenate rather than
        # override any existing Notes section in the subclass.
        if cls.__doc__ is not None:
            hub_notes = cls._HUB_NOTES_TEMPLATE.format(
                name=cls.__name__,
                name_lower=cls.__name__.lower(),
            )
            cls.__doc__ = cls.__doc__.rstrip() + "\n" + hub_notes

        if not HAS_HF_HUB:
            super().__init_subclass__(**kwargs)
            track_model_init_kwargs(cls)
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
        license = kwargs.pop("license", "bsd-3-clause")

        # Register a coder so that type[nn.Module] parameters
        # (e.g. activation=nn.ELU) are serialized as importable
        # strings in config.json and decoded back on load.
        coders = kwargs.pop("coders", None) or {}
        coders.setdefault(
            type,
            (
                lambda t: f"{t.__module__}.{t.__qualname__}",
                lambda data: _IMPORT_ADAPTER.validate_python(data),
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
        track_model_init_kwargs(cls)

    def __init__(
        self,
        n_outputs: Optional[int] = None,  # type: ignore[assignment]
        n_chans: Optional[int] = None,  # type: ignore[assignment]
        chs_info=None,  # type: ignore[assignment]
        n_times: Optional[int] = None,  # type: ignore[assignment]
        input_window_seconds: Optional[float] = None,  # type: ignore[assignment]
        sfreq: Optional[float] = None,  # type: ignore[assignment]
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

        if n_chans is not None and chs_info is not None and len(chs_info) != n_chans:
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

        self._input_window_seconds = input_window_seconds  # type: ignore[assignment]
        self._chs_info = chs_info  # type: ignore[assignment]
        self._n_outputs = n_outputs  # type: ignore[assignment]
        self._n_chans = n_chans  # type: ignore[assignment]
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
            if not hasattr(self, key):
                setattr(self, key, val)

        super().__init__()

    @property
    def n_outputs(self) -> int:
        if self._n_outputs is None:
            raise ValueError("n_outputs not specified.")
        return self._n_outputs

    @property
    def n_chans(self) -> int:
        if self._n_chans is None and self._chs_info is not None:
            return len(self._chs_info)
        elif self._n_chans is None:
            raise ValueError(
                "n_chans could not be inferred. Either specify n_chans or chs_info."
            )
        return self._n_chans

    @property
    def chs_info(self) -> list[str]:
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
            return round(self._input_window_seconds * self._sfreq)
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
        return self._sfreq

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Input data shape."""
        return (1, self.n_chans, self.n_times)

    def get_output_shape(self) -> tuple[int, ...]:
        """Returns shape of neural network output for batch size equal 1.

        Returns
        -------
        output_shape : tuple[int, ...]
            shape of the network output for `batch_size==1` (1, ...)
        """
        with torch.inference_mode():
            try:
                return tuple(
                    self.forward(  # type: ignore
                        torch.zeros(
                            self.input_shape,
                            dtype=next(self.parameters()).dtype,  # type: ignore
                            device=next(self.parameters()).device,  # type: ignore
                        )
                    ).shape
                )
            except RuntimeError as exc:
                if str(exc).endswith(
                    (
                        "Output size is too small",
                        "Kernel size can't be greater than actual input size",
                    )
                ):
                    msg = (
                        "During model prediction RuntimeError was thrown showing that at some "
                        f"layer `{str(exc).split('.')[-1]}` (see above in the stacktrace). This "
                        "could be caused by providing too small `n_times`/`input_window_seconds`. "
                        "Model may require longer chunks of signal in the input than "
                        f"{self.input_shape}."
                    )
                    raise ValueError(msg) from exc
                raise exc

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
        resolve_type_kwargs(cls, config)
        return cls(**config)

    mapping: Optional[Dict[str, str]] = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        mapping = self.mapping if self.mapping else {}
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in mapping:
                new_state_dict[mapping[k]] = v
            else:
                new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, *args, **kwargs)

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
        try:
            n_chans = self.n_chans
        except ValueError:
            n_chans = 1
        try:
            n_times = self.n_times
        except ValueError:
            n_times = 200

        return summary(
            self,
            input_size=(1, n_chans, n_times),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )

    def __str__(self) -> str:
        return str(self.get_torchinfo_statistics())

    @staticmethod
    def _serialize_chs_info(chs_info):
        """Serialize MNE channel info (``info["chs"]``) to JSON-compatible dicts."""
        if chs_info is None:
            return None
        _INT_FIELDS = ("kind", "coil_type", "unit")
        _FLOAT_FIELDS = ("cal", "range")
        serialized = []
        for ch in chs_info:
            ch_dict = {"ch_name": ch.get("ch_name", "")}
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
            super()._save_pretrained(save_directory)
        except (ImportError, RuntimeError) as e:
            # Fallback to pytorch_model.bin if safetensors saving fails
            warnings.warn(
                f"Could not save model in safetensors format: {e}. "
                "Model weights saved in pytorch_model.bin instead.",
                stacklevel=2,
            )

    if HAS_HF_HUB:

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
            resolve_type_kwargs(cls, model_kwargs)
            return super()._from_pretrained(  # type: ignore
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
