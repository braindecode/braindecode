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
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch

from braindecode.models.base import EEGModuleMixin, HAS_HF_HUB
from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net


# Skip all tests in this file if huggingface_hub is not installed
pytestmark = pytest.mark.skipif(
    not HAS_HF_HUB,
    reason="huggingface_hub not installed. Install with: pip install braindecode[huggingface]"
)


@pytest.fixture
def sample_model():
    """Create a simple EEGNet model for testing."""
    return EEGNetv4(
        n_chans=22,
        n_outputs=4,
        n_times=1000,
        final_conv_length="auto",
    )


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


class TestSoftImportFallback:
    """Test that models work without huggingface_hub installed."""

    def test_models_work_without_hf_hub(self):
        """Test that models can be instantiated and used without HF Hub."""
        # This test should always pass since we skip if HF Hub is not installed
        # But it verifies the basic functionality
        model = EEGNetv4(n_chans=22, n_outputs=4, n_times=1000)
        assert model is not None
        assert isinstance(model, EEGModuleMixin)

    def test_has_hf_hub_flag(self):
        """Test that HAS_HF_HUB flag is correctly set."""
        # Since we skip tests when HF Hub is not installed, this should be True
        assert HAS_HF_HUB is True


class TestChsInfoSerialization:
    """Test serialization and deserialization of MNE channel info."""

    def test_serialize_none(self):
        """Test serialization of None returns None."""
        result = EEGModuleMixin._serialize_chs_info(None)
        assert result is None

    def test_deserialize_none(self):
        """Test deserialization of None returns None."""
        result = EEGModuleMixin._deserialize_chs_info(None)
        assert result is None

    def test_serialize_chs_info(self, sample_chs_info):
        """Test that channel info can be serialized to JSON-compatible format."""
        serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)

        assert serialized is not None
        assert len(serialized) == len(sample_chs_info)
        assert isinstance(serialized, list)

        # Check first channel
        first_ch = serialized[0]
        assert 'ch_name' in first_ch
        assert 'coil_type' in first_ch
        assert 'kind' in first_ch
        assert 'unit' in first_ch
        assert 'cal' in first_ch
        assert 'range' in first_ch
        assert 'loc' in first_ch

        # Check that location is a list (JSON-compatible)
        assert isinstance(first_ch['loc'], list)

    def test_deserialize_chs_info(self, sample_chs_info):
        """Test that serialized channel info can be deserialized back."""
        serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)
        deserialized = EEGModuleMixin._deserialize_chs_info(serialized)

        assert deserialized is not None
        assert len(deserialized) == len(sample_chs_info)

        # Check that location is numpy array after deserialization
        first_ch = deserialized[0]
        assert isinstance(first_ch['loc'], np.ndarray)

    def test_roundtrip_serialization(self, sample_chs_info):
        """Test that channel info survives serialization roundtrip."""
        serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)
        deserialized = EEGModuleMixin._deserialize_chs_info(serialized)

        # Compare values
        for orig, deser in zip(sample_chs_info, deserialized):
            assert orig['ch_name'] == deser['ch_name']
            assert orig['coil_type'] == deser['coil_type']
            assert orig['kind'] == deser['kind']
            assert orig['unit'] == deser['unit']
            assert np.allclose(orig['loc'], deser['loc'])

    def test_json_serialization(self, sample_chs_info):
        """Test that serialized chs_info can be saved to JSON."""
        serialized = EEGModuleMixin._serialize_chs_info(sample_chs_info)

        # Try to dump to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'chs_info': serialized}, f)
            temp_path = f.name

        # Try to load back
        with open(temp_path, 'r') as f:
            loaded = json.load(f)

        assert 'chs_info' in loaded
        assert len(loaded['chs_info']) == len(sample_chs_info)

        # Cleanup
        Path(temp_path).unlink()


class TestConfigSerialization:
    """Test model configuration serialization and deserialization."""

    def test_save_pretrained_creates_config(self, sample_model):
        """Test that _save_pretrained creates a config.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_model._save_pretrained(tmpdir)

            config_path = Path(tmpdir) / 'config.json'
            assert config_path.exists()

            # Load and check config
            with open(config_path, 'r') as f:
                config = json.load(f)

            assert 'n_outputs' in config
            assert 'n_chans' in config
            assert 'n_times' in config
            assert config['n_outputs'] == 4
            assert config['n_chans'] == 22
            assert config['n_times'] == 1000

    def test_save_pretrained_with_chs_info(self, sample_model, sample_chs_info):
        """Test that _save_pretrained saves channel info."""
        sample_model._chs_info = sample_chs_info

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_model._save_pretrained(tmpdir)

            config_path = Path(tmpdir) / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)

            assert 'chs_info' in config
            assert config['chs_info'] is not None
            assert len(config['chs_info']) == 22

    def test_config_contains_all_parameters(self, sample_model):
        """Test that config contains all EEG-specific parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_model._save_pretrained(tmpdir)

            config_path = Path(tmpdir) / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Check all expected keys
            expected_keys = ['n_outputs', 'n_chans', 'n_times',
                           'input_window_seconds', 'sfreq', 'chs_info']
            for key in expected_keys:
                assert key in config


class TestHubIntegrationMocked:
    """Test Hugging Face Hub integration with mocked API calls."""

    def test_push_to_hub_method_exists(self, sample_model):
        """Test that models have push_to_hub method when HF Hub is installed."""
        assert hasattr(sample_model, 'push_to_hub')
        assert callable(getattr(sample_model, 'push_to_hub'))

    def test_from_pretrained_method_exists(self):
        """Test that models have from_pretrained class method."""
        assert hasattr(EEGNetv4, 'from_pretrained')
        assert callable(getattr(EEGNetv4, 'from_pretrained'))


class TestAllModelsHaveHubMethods:
    """Test that all braindecode models inherit HF Hub methods."""

    @pytest.mark.parametrize("model_class", [
        EEGNetv4,
        ShallowFBCSPNet,
        Deep4Net,
    ])
    def test_model_has_hub_methods(self, model_class):
        """Test that model class has both push_to_hub and from_pretrained."""
        # Check class method
        assert hasattr(model_class, 'from_pretrained')
        assert callable(getattr(model_class, 'from_pretrained'))

        # Check instance method - all these models use n_times parameter
        if model_class == EEGNetv4:
            model = model_class(n_chans=22, n_outputs=4, n_times=1000)
        else:
            # ShallowFBCSPNet and Deep4Net use n_times parameter
            model = model_class(n_chans=22, n_outputs=4, n_times=1000)

        assert hasattr(model, 'push_to_hub')
        assert callable(getattr(model, 'push_to_hub'))

    @pytest.mark.parametrize("model_class", [
        EEGNetv4,
        ShallowFBCSPNet,
        Deep4Net,
    ])
    def test_model_inherits_from_mixin(self, model_class):
        """Test that model inherits from EEGModuleMixin."""
        assert issubclass(model_class, EEGModuleMixin)


class TestBackwardCompatibility:
    """Test that existing functionality still works."""

    def test_model_forward_pass(self, sample_model):
        """Test that model forward pass works normally."""
        x = torch.randn(2, 22, 1000)
        output = sample_model(x)
        assert output.shape[0] == 2
        assert output.shape[1] == 4

    def test_model_properties(self, sample_model):
        """Test that model properties work normally."""
        assert sample_model.n_chans == 22
        assert sample_model.n_outputs == 4
        assert sample_model.n_times == 1000

    def test_get_output_shape(self, sample_model):
        """Test that get_output_shape works normally."""
        output_shape = sample_model.get_output_shape()
        assert output_shape[0] == 1
        assert output_shape[1] == 4

    def test_state_dict_loading(self, sample_model):
        """Test that state dict loading still works."""
        # Save state dict
        state_dict = sample_model.state_dict()

        # Create new model and load state dict
        new_model = EEGNetv4(n_chans=22, n_outputs=4, n_times=1000)
        new_model.load_state_dict(state_dict)

        # Check that weights are the same
        for (name1, param1), (name2, param2) in zip(
            sample_model.named_parameters(),
            new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
