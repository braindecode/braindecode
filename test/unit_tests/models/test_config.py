import pytest
from mne.utils import _soft_import

pydantic = _soft_import("pydantic", strict=False, purpose="model config testing")
if pydantic is None:
    pytest.skip("pydantic not installed, skipping", allow_module_level=True)

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
def test_make_model_config(model_name, required, signal_params):
    """Test the make_model_config function."""
    model_class = models_dict[model_name]
    ModelConfig = make_model_config(model_class, required)

    sp = _get_signal_params(signal_params)
    model_kwargs_list = _get_possible_signal_params(sp, required)
    for model_kwargs in model_kwargs_list:
        cfg = ModelConfig(**model_kwargs)
        model = cfg.create_instance()
        assert isinstance(model, model_class)
