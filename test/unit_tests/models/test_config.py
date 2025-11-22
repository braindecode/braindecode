import pytest
from mne.utils import _soft_import

pydantic = _soft_import(name="pydantic", purpose="model config testing", strict=False)

try:
    # NumPyDantic does not work with soft imports because the version is not within the __init__ of the package.
    import numpydantic  # noqa: F401
    numpydantic_imported = True
except ImportError:
    numpydantic_imported = False

if pydantic is False or numpydantic_imported is False:
    pytest.skip("pydantic or numpydantic not installed, skipping", allow_module_level=True)

import numpy as np

from braindecode.models.config import make_model_config
from braindecode.models.util import (
    _get_possible_signal_params,
    _get_signal_params,
    models_dict,
    models_mandatory_parameters,
)


@pytest.mark.parametrize(
    "model_name, required, signal_params", models_mandatory_parameters
)
def test_make_model_config_instantiation(model_name, required, signal_params):
    """Test the make_model_config function for instance creation."""
    model_class = models_dict[model_name]
    ModelConfig = make_model_config(model_class, required)

    sp = _get_signal_params(signal_params)
    model_kwargs_list = _get_possible_signal_params(sp, required)
    for model_kwargs in model_kwargs_list:
        cfg = ModelConfig(**model_kwargs)
        model = cfg.create_instance()
        assert isinstance(model, model_class)


@pytest.mark.parametrize(
    "model_name, required, signal_params", models_mandatory_parameters
)
def test_make_model_config_json_serialization(model_name, required, signal_params):
    """Test the make_model_config function for serialization.

    If this test fails, this is probably due to a bad typing of the corresponding model's arguments.
    """
    model_class = models_dict[model_name]
    ModelConfig = make_model_config(model_class, required)

    sp = _get_signal_params(signal_params)
    model_kwargs_list = _get_possible_signal_params(sp, required)
    for model_kwargs in model_kwargs_list:
        cfg = ModelConfig(**model_kwargs)
        serialized = cfg.model_dump(mode="json")
        cfg_from_serialized = ModelConfig.model_validate(serialized)
        np.testing.assert_equal(
            cfg_from_serialized.model_dump(mode="python"), cfg.model_dump(mode="python")
        )
