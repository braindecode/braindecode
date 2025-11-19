import pytest
from mne.utils import _soft_import

pydantic = _soft_import("pydantic", strict=False, purpose="model config testing")
if pydantic is None:
    pytest.skip("pydantic not installed, skipping", allow_module_level=True)

from braindecode.models.config import make_model_config
from braindecode.models.util import _get_signal_params as get_sp
from braindecode.models.util import models_dict, models_mandatory_parameters


@pytest.mark.parametrize(
    "model_name, required, signal_params", models_mandatory_parameters
)
def test_make_model_config(model_name, required, signal_params):
    """Test the make_model_config function."""
    model_class = models_dict[model_name]
    ModelConfig = make_model_config(model_class, required)

    sp = get_sp(signal_params, required)
    cfg = ModelConfig(**sp)
    model = cfg.create_instance()
    assert isinstance(model, model_class)
