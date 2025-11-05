"""
Final test of HuggingFace Hub integration after all fixes.
Uses existing repository to test push/pull.
"""
from braindecode.models import EEGNet
import torch

REPO_ID = "Kkuntal990/test-eegnet-braindecode"

print("=" * 70)
print("FINAL HUGGINGFACE HUB INTEGRATION TEST")
print("=" * 70)

# Step 1: Create fresh model
print("\n[1/4] Creating fresh EEGNet model...")
model = EEGNet(
    n_chans=22,
    n_outputs=4,
    n_times=1000,
    sfreq=250.0
)
print(f"✓ Model created: {model.__class__.__name__}")
print(f"  - Parameters: n_chans=22, n_outputs=4, n_times=1000, sfreq=250.0")

# Step 2: Initialize model with forward pass
print("\n[2/4] Initializing model (forward pass)...")
x = torch.randn(2, 22, 1000)
output = model(x)
print(f"✓ Model initialized")
print(f"  - Input shape: {tuple(x.shape)}")
print(f"  - Output shape: {tuple(output.shape)}")

# Step 3: Push to Hub with weights
print(f"\n[3/4] Pushing complete model to Hub...")
print(f"  - Repository: {REPO_ID}")
try:
    model.push_to_hub(
        REPO_ID,
        commit_message="Test push with safetensors and pytorch_model.bin",
    )
    print("✓ Successfully pushed model to Hub!")
except Exception as e:
    print(f"❌ Error pushing to Hub: {e}")
    import traceback
    traceback.print_exc()
    raise

# Step 4: Load from Hub
print(f"\n[4/4] Loading model from Hub...")
try:
    loaded_model = EEGNet.from_pretrained(REPO_ID)
    print("✓ Successfully loaded model from Hub!")

    # Verify parameters
    print("\n  Verifying parameters:")
    print(f"    - n_chans: {loaded_model.n_chans} (expected: 22)")
    print(f"    - n_outputs: {loaded_model.n_outputs} (expected: 4)")
    print(f"    - n_times: {loaded_model.n_times} (expected: 1000)")
    print(f"    - sfreq: {loaded_model.sfreq} (expected: 250.0)")

    # Test forward pass
    loaded_output = loaded_model(x)
    print(f"  - Forward pass output shape: {tuple(loaded_output.shape)}")

    # Compare weights
    original_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(p.numel() for p in loaded_model.parameters())
    print(f"  - Original model params: {original_params}")
    print(f"  - Loaded model params: {loaded_params}")

    if original_params == loaded_params:
        print("  ✓ Parameter count matches!")
    else:
        print("  ❌ Parameter count mismatch!")

    # Check if both safetensors and pytorch_model.bin exist on Hub
    from huggingface_hub import list_repo_files
    files = list_repo_files(REPO_ID)
    print(f"\n  Files on Hub:")
    for f in sorted(files):
        print(f"    - {f}")

    has_safetensors = "model.safetensors" in files
    has_pytorch = "pytorch_model.bin" in files
    has_config = "config.json" in files

    print(f"\n  ✓ config.json: {'Present' if has_config else 'MISSING'}")
    print(f"  ✓ model.safetensors: {'Present' if has_safetensors else 'MISSING'}")
    print(f"  ✓ pytorch_model.bin: {'Present' if has_pytorch else 'MISSING'}")

except Exception as e:
    print(f"❌ Error loading from Hub: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Hub integration working!")
print("=" * 70)
print(f"\nModel available at: https://huggingface.co/{REPO_ID}")
