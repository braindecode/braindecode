from collections.abc import Callable
from inspect import signature
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import numpy as np
from mne.utils import _soft_import
from typing_extensions import TypedDict

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import SigArgName, models_dict, models_mandatory_parameters

pydantic = _soft_import(name="pydantic", purpose="model configuration", strict=False)

try:
    from numpydantic import NDArray, Shape
except ImportError:
    # we can't use soft import for numpydantic because numpydantic does not define its version in __init__
    NDArray = Any  # type: ignore
    Shape = Any  # type: ignore


class ChsInfoType(TypedDict, total=False, closed=True):  # type: ignore[call-arg]
    cal: float
    ch_name: str
    coil_type: int
    coord_frame: int
    kind: str
    loc: NDArray[Shape["12"], np.float64]  # type: ignore[misc]
    logno: int
    range: float
    scanno: int
    unit: int
    unit_mul: int


def _replace_type_hints(type_hint: Any) -> Any:
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if origin is type or origin is Callable or type_hint is Callable:
        return pydantic.ImportString
    if origin is None:
        return type_hint
    replaced_args = tuple(_replace_type_hints(arg) for arg in args)
    if origin is UnionType:
        origin = Union
    return origin[replaced_args]


SIGNAL_ARGS_TYPES = {
    "n_chans": int,
    "n_times": int,
    "sfreq": float,
    "input_window_seconds": float,
    "n_outputs": int,
    "chs_info": list[ChsInfoType],
}


class BaseBraindecodeModelConfig(pydantic.BaseModel):  # type: ignore
    def create_instance(self) -> EEGModuleMixin:
        model_cls = models_dict[self.model_name_]
        kwargs = self.model_dump(mode="python", exclude={"model_name_"})
        if kwargs.get("n_chans") is not None and kwargs.get("chs_info") is not None:
            kwargs.pop("n_chans")
        if (
            kwargs.get("n_times") is not None
            and kwargs.get("input_window_seconds") is not None
            and kwargs.get("sfreq") is not None
        ):
            kwargs.pop("n_times")
        return model_cls(**kwargs)


def make_model_config(
    model_class: type[EEGModuleMixin],
    required: list[SigArgName],
) -> type[BaseBraindecodeModelConfig]:
    """Create a pydantic model config for a given model class.

    Parameters
    ----------
    model_class : type[EEGModuleMixin]
        The model class for which to create the config.
    required : list of SigArgName
        The required signal arguments for the model.

    Returns
    -------
    type
        A pydantic BaseModel subclass representing the model config.
    """
    if not pydantic:
        raise ImportError(
            "pydantic is required to use make_model_config. "
            "Please install braindecode[typing]."
        )

    # ironically, we need to ignore the type here to have the soft dependency.

    @pydantic.model_validator(mode="before")
    def validate_signal_params(cls, data: Any):
        n_outputs = data.get("n_outputs")
        n_chans = data.get("n_chans")
        chs_info = data.get("chs_info")
        n_times = data.get("n_times")
        input_window_seconds = data.get("input_window_seconds")
        sfreq = data.get("sfreq")

        # Check that required parameters are provided or can be inferred
        if "n_outputs" in required and n_outputs is None:
            raise ValueError("n_outputs is a required parameter but was not provided.")
        if "n_chans" in required and n_chans is None and chs_info is None:
            raise ValueError(
                "n_chans is required and could not be inferred. Either specify n_chans or chs_info."
            )
        if "chs_info" in required and chs_info is None:
            raise ValueError("chs_info is a required parameter but was not provided.")
        if "n_times" in required and (
            n_times is None and (sfreq is None or input_window_seconds is None)
        ):
            raise ValueError(
                "n_times is required and could not be inferred."
                "Either specify n_times or input_window_seconds and sfreq."
            )
        if "sfreq" in required and (
            sfreq is None and (n_times is None or input_window_seconds is None)
        ):
            raise ValueError(
                "sfreq is required and could not be inferred."
                "Either specify sfreq or input_window_seconds and n_times."
            )
        if "input_window_seconds" in required and (
            input_window_seconds is None and (n_times is None or sfreq is None)
        ):
            raise ValueError(
                "input_window_seconds is required and could not be inferred."
                "Either specify input_window_seconds or n_times and sfreq."
            )

        # Infer missing parameters if possible, and check consistency
        if chs_info is not None:
            if n_chans is None:
                data["n_chans"] = len(chs_info)
            elif n_chans != len(chs_info):
                raise ValueError(
                    f"Provided {n_chans=} does not match length of chs_info: {len(chs_info)}."
                )
        if (
            n_times is not None
            and sfreq is not None
            and input_window_seconds is not None
        ):
            if n_times != int(input_window_seconds * sfreq):
                raise ValueError(
                    f"Provided {n_times=} does not match {input_window_seconds=} * {sfreq=}."
                )
        elif n_times is None and sfreq is not None and input_window_seconds is not None:
            data["n_times"] = int(input_window_seconds * sfreq)
        elif sfreq is None and n_times is not None and input_window_seconds is not None:
            data["sfreq"] = n_times / input_window_seconds
        elif input_window_seconds is None and n_times is not None and sfreq is not None:
            data["input_window_seconds"] = n_times / sfreq
        return data

    signature_params = signature(model_class.__init__, eval_str=True).parameters
    has_args = any(p.kind == p.VAR_POSITIONAL for p in signature_params.values())
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature_params.values())
    if has_args:
        raise ValueError("Model __init__ methods cannot have *args")

    extra = "allow" if has_kwargs else "forbid"
    fields = {}
    for name, p in signature_params.items():
        if name == "self" or p.kind == p.VAR_KEYWORD:
            continue

        annot = p.annotation
        if annot is p.empty:
            annot = Any
        # case with type[nn.Module] or callable
        else:
            annot = _replace_type_hints(annot)
        # Most models did not specify types for signal args, so we add them here
        if name in SIGNAL_ARGS_TYPES:
            annot = SIGNAL_ARGS_TYPES[name] | None

        fields[name] = (annot, p.default) if p.default is not p.empty else annot

    name = model_class.__name__
    model_config = pydantic.create_model(
        f"{name}Config",
        model_name_=(Literal[name], name),
        __config__=pydantic.ConfigDict(
            arbitrary_types_allowed=True, extra=extra, validate_default=True
        ),
        __doc__=f"Pydantic config of model {model_class.__name__}\n\n{model_class.__doc__}",
        __base__=BaseBraindecodeModelConfig,
        __module__="braindecode.models.config",
        __validators__={"validate_signal_params": validate_signal_params},
        **fields,
    )
    return model_config


# Automatically generate and add classes to the global namespace
# and define __all__ based on generated classes
__all__ = ["make_model_config"]

if not pydantic:
    pass
else:
    models_configs: list[type[BaseBraindecodeModelConfig]] = []
    for model_name, req, _ in models_mandatory_parameters:
        model_cls = models_dict[model_name]
        model_cfg = make_model_config(model_cls, req)
        globals()[model_cfg.__name__] = model_cfg
        __all__.append(model_cfg.__name__)
        models_configs.append(model_cfg)

    BraindecodeModelConfig = Annotated[  # type: ignore
        Union[tuple(models_configs)],
        pydantic.Field(
            discriminator="model_name_", description="Braindecode model configuration"
        ),
    ]

# # Example usage:
#
# class DummyConfigWithModel(pydantic.BaseModel):
#     model: BraindecodeModelConfig
#
# DummyConfigWithModel.model_validate({'model': dict(model_name_='EEGNet', n_chans=16, n_outputs=1, n_times=200)})
