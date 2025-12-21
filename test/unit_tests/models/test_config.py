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


@pytest.mark.parametrize(
    "n_times, input_window_seconds, sfreq",
    [
        (1001, 4.004, 250.0),  # Issue example: 4.004 * 250.0 = 1001.0
        (751, 3.004, 250.0),   # 3.004 * 250.0 = 751.0
        (501, 2.004, 250.0),   # 2.004 * 250.0 = 501.0
        (101, 0.404, 250.0),   # 0.404 * 250.0 = 101.0
    ],
)
def test_fractional_input_window_seconds_config(n_times, input_window_seconds, sfreq):
    """Test that config accepts fractional input_window_seconds when consistent.

    This test validates the fix for the bug where int() truncation rejected
    valid configurations in the pydantic validator.
    """
    from braindecode.models import EEGNetv4

    ModelConfig = make_model_config(EEGNetv4, ["n_chans", "n_outputs", "n_times"])

    # Should not raise ValueError
    cfg = ModelConfig(
        n_chans=22,
        n_outputs=4,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    assert cfg.n_times == n_times
    assert cfg.input_window_seconds == input_window_seconds
    assert cfg.sfreq == sfreq


@pytest.mark.parametrize(
    "n_times, input_window_seconds, sfreq, expected_n_times",
    [
        (None, 4.004, 250.0, 1001),  # Infer n_times with rounding
        (None, 3.004, 250.0, 751),   # Infer n_times with rounding
        (None, 2.004, 250.0, 501),   # Infer n_times with rounding
        (None, 0.404, 250.0, 101),   # Infer n_times with rounding
    ],
)
def test_fractional_input_window_seconds_inference_config(
    n_times, input_window_seconds, sfreq, expected_n_times
):
    """Test that config correctly infers n_times with fractional input_window_seconds.

    This test validates that the pydantic validator uses round() instead of int()
    when inferring n_times.
    """
    from braindecode.models import EEGNetv4

    ModelConfig = make_model_config(EEGNetv4, ["n_chans", "n_outputs", "n_times"])

    cfg = ModelConfig(
        n_chans=22,
        n_outputs=4,
        n_times=n_times,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    )
    assert cfg.n_times == expected_n_times


@pytest.mark.parametrize(
    "model_name, required, signal_params", models_mandatory_parameters
)
def test_make_model_config_json_serialization_with_serialize_as_any(
    model_name, required, signal_params
):
    """Test serialization with serialize_as_any=True for the activation field.
    
    This test validates that configs can be serialized with serialize_as_any=True,
    which is required by frameworks like Exca. This validates the fix for the
    Pydantic v2 serialization failure bug with type fields (like activation).
    
    Note: This test only validates models without chs_info, as chs_info contains
    numpy arrays which have a separate serialization issue beyond the scope of
    the activation field fix.
    
    See: https://github.com/braindecode/braindecode/issues/XXX
    """
    # Skip tests with chs_info as numpy arrays have separate serialization issues
    sp = _get_signal_params(signal_params)
    if sp.get('chs_info') is not None:
        pytest.skip("Skipping test with chs_info due to numpy array serialization issues")
    
    model_class = models_dict[model_name]
    ModelConfig = make_model_config(model_class, required)

    model_kwargs_list = _get_possible_signal_params(sp, required)
    for model_kwargs in model_kwargs_list:
        cfg = ModelConfig(**model_kwargs)
        
        # This should not raise PydanticSerializationError for type fields
        serialized = cfg.model_dump(mode="json", serialize_as_any=True)
        
        # Verify we can reconstruct from the serialized data
        cfg_from_serialized = ModelConfig.model_validate(serialized)
        
        # Verify the reconstructed config has the same values
        np.testing.assert_equal(
            cfg_from_serialized.model_dump(mode="python"), cfg.model_dump(mode="python")
        )


def test_activation_field_serialization_with_serialize_as_any():
    """Test that activation fields serialize correctly with serialize_as_any=True.
    
    This is a focused test for the main bug fix: activation fields (type objects)
    should serialize to strings even when serialize_as_any=True is used.
    This was the core issue preventing Exca integration.
    """
    from braindecode.models import EEGNet
    import torch.nn as nn
    
    ModelConfig = make_model_config(EEGNet, ["n_chans", "n_outputs", "n_times"])
    
    # Test 1: Create config with type object (default)
    cfg = ModelConfig(n_chans=22, n_outputs=4, n_times=1000)
    
    # Test 2: Verify serialization with serialize_as_any=True works
    serialized = cfg.model_dump(mode="json", serialize_as_any=True)
    assert 'activation' in serialized
    assert serialized['activation'] == 'torch.nn.modules.activation.ELU'
    assert isinstance(serialized['activation'], str)
    
    # Test 3: Verify reconstruction from serialized data
    cfg_reconstructed = ModelConfig.model_validate(serialized)
    assert cfg_reconstructed.activation == cfg.activation
    
    # Test 4: Verify create_instance works with string representation
    model = cfg.create_instance()
    assert isinstance(model.activation, type)
    assert model.activation == nn.ELU
    
    # Test 5: Test with explicit string input
    cfg2 = ModelConfig(
        n_chans=22,
        n_outputs=4,
        n_times=1000,
        activation='torch.nn.modules.activation.ReLU'
    )
    assert cfg2.activation == 'torch.nn.modules.activation.ReLU'
    serialized2 = cfg2.model_dump(mode="json", serialize_as_any=True)
    assert serialized2['activation'] == 'torch.nn.modules.activation.ReLU'
