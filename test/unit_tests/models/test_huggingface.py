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
    sample_model, _, _ = sample_model

    sample_model._save_pretrained(tmp_path)

    config_path = tmp_path / 'config.json'
    assert config_path.exists()

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    assert config['n_outputs'] == sample_model.n_outputs
    assert config['n_times'] == sample_model.n_times




def test_config_contains_all_parameters(tmp_path, sample_model):
    sample_model, _, _ = sample_model
    sample_model._save_pretrained(tmp_path)

    config_path = tmp_path / 'config.json'
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    expected_keys = {'n_outputs', 'n_chans', 'n_times', 'input_window_seconds', 'sfreq', 'chs_info', 'braindecode_version'}
    assert expected_keys == config.keys()


def test_local_push_and_pull_roundtrip(tmp_path, sample_model):
    """Roundtrip through local Hub save/load mimics push/pull."""
    model, name, sp = sample_model
    # TODO: fix for AttnSleep/DeepSleepNet/DeepSleepNet/SincShallowNet
    if name in non_classification_models+["AttnSleep","DeepSleepNet","SincShallowNet"]:
        pytest.skip(f"Skipping Hugging Face Hub test for non-classification model: {name}")
    assert hasattr(model, 'from_pretrained')
    assert callable(getattr(model, 'from_pretrained'))
    model.eval()

    repo_dir = tmp_path / f'hf_local_repo_{name}'
    repo_dir.mkdir()
    model._save_pretrained(repo_dir)

    # Load the model from the saved config using the class method
    model_class = models_dict[name]
    restored = model_class.from_pretrained(repo_dir)
    restored.eval()
    n_times = sp.get('n_times', 1000)
    n_chans = sp.get('n_chans', 22)  # Default to 22 if not specified

    torch.manual_seed(42)
    sample_input = torch.randn(2, n_chans, n_times)

    out_original = model(sample_input)

    out_restored = restored(sample_input)
    torch.testing.assert_close(out_restored, out_original)
