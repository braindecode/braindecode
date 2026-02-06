# Authors: Kuntal Kokate
#          Bruno Aristimunha
#
# License: BSD-3

"""
Tests for Hugging Face Hub integration with braindecode models.

These tests verify that models can be pushed to and loaded from the
Hugging Face Hub when the optional huggingface_hub dependency is installed.
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from braindecode.models import EEGNet
from braindecode.models.base import HAS_HF_HUB, EEGModuleMixin

# importing some fixtures/utilities to help with testing
from braindecode.models.util import (
    models_dict,
    models_mandatory_parameters,
    non_classification_models,
)

from .test_integration import get_sp

# Skip all tests in this file if huggingface_hub is not installed
pytestmark = pytest.mark.skipif(
    not HAS_HF_HUB,
    reason="huggingface_hub not installed. Install with: pip install braindecode[hug]"
)

@pytest.fixture(scope="module", params=models_mandatory_parameters, ids=lambda p: p[0])
def sample_model(request):
    """Instantiated model."""
    name, req, sig_params = request.param
    sp = get_sp(sig_params, req)
    model = models_dict[name](**sp)

    model.eval()
    return model, name, sp


@pytest.fixture
def sample_chs_info():
    """Create sample channel information."""
    return [
        {
            'ch_name': f'EEG {i:03d}',
            'coil_type': 0,
            'kind': 2,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': np.random.randn(12),
        }
        for i in range(22)
    ]


def test_models_work_without_hf_hub():
    """Ensure soft dependencies do not break model usage."""
    model = EEGNet(n_chans=22, n_outputs=4, n_times=1000)
    assert model is not None
    assert isinstance(model, EEGModuleMixin)


def test_has_hf_hub_flag_enabled():
    """Verify the HAS_HF_HUB flag matches the skip condition."""
    assert HAS_HF_HUB is True


def test_serialize_none_returns_none():
    result = EEGModuleMixin._serialize_chs_info(None)
    assert result is None


def test_deserialize_none_returns_none():
    result = EEGModuleMixin._deserialize_chs_info(None)
    assert result is None


def test_serialize_chs_info(sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)

    assert serialized is not None
    assert len(serialized) == len(sample_chs_info)
    assert isinstance(serialized, list)

    first_ch = serialized[0]
    assert 'ch_name' in first_ch
    assert 'coil_type' in first_ch
    assert 'kind' in first_ch
    assert 'unit' in first_ch
    assert 'cal' in first_ch
    assert 'range' in first_ch
    assert 'loc' in first_ch
    assert isinstance(first_ch['loc'], list)


def test_deserialize_chs_info(sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)
    deserialized = EEGModuleMixin._deserialize_chs_info(serialized)

    assert deserialized is not None
    assert len(deserialized) == len(sample_chs_info)
    assert isinstance(deserialized[0]['loc'], np.ndarray)


def test_roundtrip_serialization(sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)
    deserialized = EEGModuleMixin._deserialize_chs_info(serialized)

    for orig, deser in zip(sample_chs_info, deserialized):
        assert orig['ch_name'] == deser['ch_name']
        assert orig['coil_type'] == deser['coil_type']
        assert orig['kind'] == deser['kind']
        assert orig['unit'] == deser['unit']
        assert np.allclose(orig['loc'], deser['loc'])


def test_json_serialization(tmp_path, sample_chs_info):
    serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)

    temp_path = tmp_path / 'chs_info.json'
    with open(temp_path, 'w') as tmp_file:
        json.dump({'chs_info': serialized}, tmp_file)

    with open(temp_path, 'r') as file:
        loaded = json.load(file)

    assert 'chs_info' in loaded
    assert len(loaded['chs_info']) == len(sample_chs_info)


def test_save_pretrained_creates_config(tmp_path, sample_model):
    sample_model, name, _ = sample_model
    # TODO: fix for AttnSleep/DeepSleepNet/DeepSleepNet/SincShallowNet
    if name in non_classification_models + ["AttnSleep","DeepSleepNet","SincShallowNet"]:
        pytest.skip(f"Skipping config test for non-classification model: {name}")

    if any(isinstance(p, nn.UninitializedParameter) for p in sample_model.parameters()):
        pytest.skip(f"Skipping config test for model with uninitialized parameters: {name}")

    sample_model._save_pretrained(tmp_path)

    config_path = tmp_path / f'config.json'
    assert config_path.exists()

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Use getattr with exception handling to safely get model properties
    # Some models may raise ValueError when properties are not properly initialized
    def safe_get_property(obj, prop_name):
        try:
            return getattr(obj, prop_name)
        except (ValueError, AttributeError):
            return None

    n_times = safe_get_property(sample_model, 'n_times')
    n_chans = safe_get_property(sample_model, 'n_chans')
    n_outputs = safe_get_property(sample_model, 'n_outputs')

    if n_chans is not None and name != "SignalJEPA_Contextual":
        if name == "Labram":
            assert config['n_chans'] in (n_chans, None)
        else:
            assert config['n_chans'] == n_chans
    if n_times is not None:
        assert config['n_times'] == n_times
    if n_outputs is not None:
        if name == "Labram":
            assert config['n_outputs'] in (n_outputs, None)
        else:
            assert config['n_outputs'] == n_outputs




def test_config_contains_all_parameters(tmp_path, sample_model):
    sample_model, _, _ = sample_model

    if any(isinstance(p, nn.UninitializedParameter) for p in sample_model.parameters()):
        pytest.skip("Skipping config parameter test for model with uninitialized parameters.")

    sample_model._save_pretrained(tmp_path)

    config_path = tmp_path / 'config.json'
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    expected_keys = {'n_outputs', 'n_chans', 'n_times', 'input_window_seconds', 'sfreq', 'chs_info', 'braindecode_version'}
    assert expected_keys == config.keys()


def test_local_push_and_pull_roundtrip(tmp_path, sample_model):
    """Roundtrip through local Hub save/load mimics push/pull."""
    model, name, sp = sample_model
    # TODO: fix for AttnSleep-DeepSleepNet/SincShallowNet
    if name in non_classification_models+["AttnSleep","DeepSleepNet","SincShallowNet"]:
        pytest.skip(f"Skipping Hugging Face Hub test for non-classification model: {name}")
    assert hasattr(model, 'from_pretrained')
    assert callable(getattr(model, 'from_pretrained'))
    model.eval()

    if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
        pytest.skip(f"Skipping Hugging Face Hub test for model with uninitialized parameters: {name}")

    repo_dir = tmp_path / f'hf_local_repo_{name}'
    repo_dir.mkdir()
    model._save_pretrained(repo_dir)

    # Load the model from the saved config using the class method
    model_class = models_dict[name]
    restored = model_class.from_pretrained(repo_dir)
    restored.eval()

    n_times = sp.get('n_times', 1000)
    n_chans = sp.get('n_chans')
    if n_chans is None and sp.get('chs_info') is not None:
        n_chans = len(sp['chs_info'])
    if n_chans is None:
        n_chans = 22
    # TODO: small adjust necessary for SignalJEPA_Contextual
    if name == "SignalJEPA_Contextual":
        n_chans = 3
    torch.manual_seed(42)

    sample_input = torch.randn(2, n_chans, n_times)

    out_original = model(sample_input)

    out_restored = restored(sample_input)
    torch.testing.assert_close(out_restored, out_original)


def test_serialize_chs_info_with_string_kind():
    """Test serialization when kind field is a string."""
    chs_info = [
        {
            'ch_name': 'EEG 001',
            'kind': 'eeg',  # String instead of int
            'coil_type': 0,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': np.array([0.0] * 12),
        }
    ]
    serialized = EEGModuleMixin._serialize_chs_info(chs_info)
    assert serialized[0]['kind'] == 'eeg'


def test_serialize_chs_info_without_optional_fields():
    """Test serialization when optional fields are missing."""
    chs_info = [
        {
            'ch_name': 'EEG 001',
            # Missing coil_type, unit, cal, range, loc, kind
        }
    ]
    serialized = EEGModuleMixin._serialize_chs_info(chs_info)
    assert serialized[0]['ch_name'] == 'EEG 001'
    # Optional fields should not be in the serialized output
    assert 'coil_type' not in serialized[0] or serialized[0].get('coil_type') is None


def test_init_with_serialized_chs_info():
    """Test model initialization with serialized channel info from Hub."""
    serialized_chs_info = [
        {
            'ch_name': 'EEG 001',
            'kind': 2,
            'coil_type': 0,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # List, not ndarray
        }
    ]
    # Model should deserialize this automatically in __init__
    model = EEGNet(n_chans=1, n_outputs=2, n_times=100, chs_info=serialized_chs_info)
    # Check that chs_info was deserialized (loc should be ndarray now)
    assert isinstance(model.chs_info[0]['loc'], np.ndarray)


def test_init_with_already_deserialized_chs_info():
    """Test model initialization with already-deserialized channel info."""
    deserialized_chs_info = [
        {
            'ch_name': 'EEG 001',
            'kind': 2,
            'coil_type': 0,
            'unit': 107,
            'cal': 1.0,
            'range': 1.0,
            'loc': np.array([0.0] * 12),  # Already ndarray
        }
    ]
    # Model should handle this without error
    model = EEGNet(n_chans=1, n_outputs=2, n_times=100, chs_info=deserialized_chs_info)
    assert isinstance(model.chs_info[0]['loc'], np.ndarray)


def test_save_pretrained_without_hf_hub():
    """Test that _save_pretrained returns early when HF Hub is not available."""
    from braindecode.models import base
    original_has_hf_hub = base.HAS_HF_HUB
    try:
        # Temporarily set HAS_HF_HUB to False
        base.HAS_HF_HUB = False
        model = EEGNet(n_chans=22, n_outputs=4, n_times=1000)
        # Should return early without error
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model._save_pretrained(tmpdir)
    finally:
        base.HAS_HF_HUB = original_has_hf_hub


def test_init_subclass_without_hf_hub():
    """Test that __init_subclass__ works when HF Hub is not available."""
    from braindecode.models import base
    original_has_hf_hub = base.HAS_HF_HUB
    try:
        base.HAS_HF_HUB = False
        # Creating a subclass should work without HF Hub
        class TestModel(base.EEGModuleMixin):
            pass
        # Should complete without error
        assert TestModel is not None
    finally:
        base.HAS_HF_HUB = original_has_hf_hub
