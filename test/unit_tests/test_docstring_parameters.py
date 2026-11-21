"""Documented parameters of augmentations and datasets must be real arguments."""

import inspect

import pytest

import braindecode.augmentation as augmentation
import braindecode.datasets as datasets

from .models.test_models import _documented_parameters


def _public_classes(module):
    return sorted(
        (
            cls
            for name in module.__all__
            if inspect.isclass(cls := getattr(module, name))
            and cls.__module__.startswith("braindecode")
        ),
        key=lambda cls: cls.__qualname__,
    )


@pytest.mark.parametrize(
    "cls",
    _public_classes(augmentation) + _public_classes(datasets),
    ids=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
)
def test_documented_parameters_exist_in_signature(cls):
    parameters = inspect.signature(cls.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        pytest.skip(f"{cls.__name__} forwards **kwargs, any name is accepted")

    unknown = sorted(set(_documented_parameters(cls)) - set(parameters))

    assert not unknown, (
        f"{cls.__name__} documents parameters its constructor does not accept: "
        f"{unknown}"
    )
