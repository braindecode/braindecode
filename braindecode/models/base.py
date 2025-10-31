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

    .. rubric:: Hugging Face Hub integration

    When the optional ``huggingface_hub`` package is installed, all models
    automatically gain the ability to be pushed to and loaded from the
    Hugging Face Hub. Install with::

        pip install braindecode[hug]

    **Pushing a model to the Hub:**

    .. code-block:: python

        from braindecode.models import EEGNetv4

        # Train your model
        model = EEGNetv4(n_chans=22, n_outputs=4, n_times=1000)
        # ... training code ...

        # Push to the Hub
        model.push_to_hub(
            repo_id="username/my-eegnet-model", commit_message="Initial model upload"
        )

    **Loading a model from the Hub:**

    .. code-block:: python

        from braindecode.models import EEGNetv4

        # Load pretrained model
        model = EEGNetv4.from_pretrained("username/my-eegnet-model")

    The integration automatically handles EEG-specific parameters (n_chans,
    n_times, sfreq, chs_info, etc.) by saving them in a config file alongside
    the model weights. This ensures that loaded models are correctly configured
    for their original data specifications.

    .. important::
        Currently, only EEG-specific parameters (n_outputs, n_chans, n_times,
        input_window_seconds, sfreq, chs_info) are saved to the Hub. Model-specific
        parameters (e.g., dropout rates, activation functions, number of filters)
        are not preserved and will use their default values when loading from the Hub.

        To use non-default model parameters, specify them explicitly when calling
        :func:`from_pretrained()`::

            model = EEGNet.from_pretrained("user/model", dropout=0.3, activation='relu')

        Full parameter serialization will be addressed in a future update.
    """

    def __init_subclass__(cls, **kwargs):
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
        license = kwargs.pop("license", "bsd-3-clause")
        # TODO: model_card_template can be added in the future for custom model cards
        super().__init_subclass__(
            tags=tags,
            docs_url=docs_url,
            repo_url=repo_url,
            library_name=library_name,
            license=license,
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
            and n_times != int(input_window_seconds * sfreq)
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
            return int(self._input_window_seconds * self._sfreq)
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
        output_shape: tuple[int, ...]
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
        Transform a sequential model with strides to a model that outputs
        dense predictions by removing the strides and instead inserting dilations.
        Modifies model in-place.

        Parameters
        ----------
        axis: int or (int,int)
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
        return summary(
            self,
            input_size=(1, self.n_chans, self.n_times),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )

    def __str__(self) -> str:
        return str(self.get_torchinfo_statistics())

    @staticmethod
    def _serialize_chs_info(chs_info):
        """
        Serialize MNE channel info to JSON-compatible format.

        Parameters
        ----------
        chs_info : list of dict or None
            Channel information from MNE Info object.

        Returns
        -------
        list of dict or None
            Serialized channel information that can be saved to JSON.
        """
        if chs_info is None:
            return None

        serialized = []
        for ch in chs_info:
            # Extract serializable fields from MNE channel info
            ch_dict = {
                "ch_name": ch.get("ch_name", ""),
            }

            # Handle kind field - can be either string or integer
            kind_val = ch.get("kind")
            if kind_val is not None:
                ch_dict["kind"] = (
                    kind_val if isinstance(kind_val, str) else int(kind_val)
                )

            # Add numeric fields with safe conversion
            coil_type = ch.get("coil_type")
            if coil_type is not None:
                ch_dict["coil_type"] = int(coil_type)

            unit = ch.get("unit")
            if unit is not None:
                ch_dict["unit"] = int(unit)

            cal = ch.get("cal")
            if cal is not None:
                ch_dict["cal"] = float(cal)

            range_val = ch.get("range")
            if range_val is not None:
                ch_dict["range"] = float(range_val)

            # Serialize location array if present
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
        """
        Deserialize channel info from JSON-compatible format to MNE-like structure.

        Parameters
        ----------
        chs_info_dict : list of dict or None
            Serialized channel information.

        Returns
        -------
        list of dict or None
            Deserialized channel information compatible with MNE.
        """
        if chs_info_dict is None:
            return None

        deserialized = []
        for ch_dict in chs_info_dict:
            ch = ch_dict.copy()
            # Convert location back to numpy array if present
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

        # Collect EEG-specific configuration
        config = {
            "n_outputs": self._n_outputs,
            "n_chans": self._n_chans,
            "n_times": self._n_times,
            "input_window_seconds": self._input_window_seconds,
            "sfreq": self._sfreq,
            "chs_info": self._serialize_chs_info(self._chs_info),
            "braindecode_version": __version__,
        }

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
