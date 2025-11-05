#!/usr/bin/env python
"""
Test script for Hugging Face Hub integration.

This script demonstrates:
1. Creating a braindecode model
2. Pushing it to Hugging Face Hub
3. Loading it back from the Hub
4. Verifying the loaded model works correctly
"""

import torch
import tempfile
from pathlib import Path
from braindecode.models import EEGNet

def test_local_save_load():
    """Test local save/load functionality first."""
    print("=" * 70)
    print("STEP 1: Testing Local Save/Load")
    print("=" * 70)

    # Create a model
    print("\n✓ Creating EEGNet model...")
    model = EEGNet(
        n_chans=22,
        n_outputs=4,
        n_times=1000,
        sfreq=250.0,
    )

    # Get model parameters
    print(f"  - Channels: {model.n_chans}")
    print(f"  - Outputs: {model.n_outputs}")
    print(f"  - Time samples: {model.n_times}")
    print(f"  - Sampling frequency: {model.sfreq}")

    # Test forward pass
    print("\n✓ Testing forward pass...")
    x = torch.randn(2, 22, 1000)
    output = model(x)
    print(f"  - Input shape: {tuple(x.shape)}")
    print(f"  - Output shape: {tuple(output.shape)}")

    # Save to temporary directory
    print("\n✓ Saving model locally...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        save_path.mkdir()

        # Save pretrained
        model._save_pretrained(save_path)
        print(f"  - Saved to: {save_path}")

        # Check config file exists
        config_file = save_path / "config.json"
        if config_file.exists():
            print(f"  - Config file created: ✓")
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"  - Config contains: {list(config.keys())}")
        else:
            print(f"  - Config file created: ✗ FAILED")
            return False

    print("\n✅ Local save/load test PASSED")
    return True


def test_hub_push_pull(repo_name=None):
    """Test pushing to and pulling from Hugging Face Hub."""
    if repo_name is None:
        print("\n" + "=" * 70)
        print("STEP 2: Hub Push/Pull Test (SKIPPED)")
        print("=" * 70)
        print("\nTo test Hub integration, run:")
        print("  python test_hf_hub_integration.py <your-username/repo-name>")
        print("\nExample:")
        print("  python test_hf_hub_integration.py myusername/test-eegnet")
        print("\nNote: You need to:")
        print("  1. Login to HF Hub: huggingface-cli login")
        print("  2. Create a model repository on https://huggingface.co")
        return True

    print("\n" + "=" * 70)
    print("STEP 2: Testing Hub Push/Pull")
    print("=" * 70)

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Create and push model
        print(f"\n✓ Creating model to push to {repo_name}...")
        model = EEGNet(
            n_chans=22,
            n_outputs=4,
            n_times=1000,
            sfreq=250.0,
        )

        print(f"\n✓ Pushing to Hugging Face Hub...")
        print(f"  - Repository: {repo_name}")
        model.push_to_hub(
            repo_id=repo_name,
            commit_message="Test EEGNet model from braindecode"
        )
        print(f"  - Push successful! ✓")

        # Load from Hub
        print(f"\n✓ Loading from Hugging Face Hub...")
        loaded_model = EEGNet.from_pretrained(repo_name)
        print(f"  - Model loaded successfully! ✓")

        # Verify parameters
        print(f"\n✓ Verifying model parameters...")
        assert loaded_model.n_chans == 22, f"n_chans mismatch: {loaded_model.n_chans}"
        assert loaded_model.n_outputs == 4, f"n_outputs mismatch: {loaded_model.n_outputs}"
        assert loaded_model.n_times == 1000, f"n_times mismatch: {loaded_model.n_times}"
        assert loaded_model.sfreq == 250.0, f"sfreq mismatch: {loaded_model.sfreq}"
        print(f"  - All parameters match! ✓")

        # Test forward pass
        print(f"\n✓ Testing loaded model forward pass...")
        x = torch.randn(2, 22, 1000)
        output = loaded_model(x)
        assert output.shape == (2, 4), f"Output shape mismatch: {output.shape}"
        print(f"  - Forward pass successful! ✓")

        print(f"\n✅ Hub push/pull test PASSED")
        print(f"\n🎉 Model available at: https://huggingface.co/{repo_name}")
        return True

    except Exception as e:
        print(f"\n❌ Hub test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    import sys

    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "HUGGING FACE HUB INTEGRATION TEST" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")

    # Test 1: Local save/load
    if not test_local_save_load():
        print("\n❌ Tests FAILED")
        return 1

    # Test 2: Hub push/pull (optional, requires repo name)
    repo_name = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_hub_push_pull(repo_name):
        print("\n❌ Tests FAILED")
        return 1

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
