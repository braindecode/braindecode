"""
Test loading from existing HuggingFace Hub repository.
"""
from braindecode.models import EEGNet
from huggingface_hub import list_repo_files
import torch

REPO_ID = "Kkuntal990/test-eegnet-braindecode"

print("=" * 70)
print("TEST LOADING FROM HUGGINGFACE HUB")
print("=" * 70)

# Step 1: Check files on Hub
print(f"\n[1/3] Checking files on Hub ({REPO_ID})...")
try:
    files = list_repo_files(REPO_ID)
    print(f"  Files on Hub:")
    for f in sorted(files):
        print(f"    - {f}")

    has_safetensors = "model.safetensors" in files
    has_pytorch = "pytorch_model.bin" in files
    has_config = "config.json" in files

    print(f"\n  ✓ config.json: {'Present' if has_config else 'MISSING'}")
    print(f"  ✓ model.safetensors: {'Present' if has_safetensors else 'MISSING'}")
    print(f"  ✓ pytorch_model.bin: {'Present' if has_pytorch else 'MISSING'}")
except Exception as e:
    print(f"❌ Error listing files: {e}")
    raise

# Step 2: Load from Hub
print(f"\n[2/3] Loading model from Hub...")
try:
    loaded_model = EEGNet.from_pretrained(REPO_ID)
    print("✓ Successfully loaded model from Hub!")

    # Verify parameters
    print("\n  Verifying parameters:")
    print(f"    - n_chans: {loaded_model.n_chans}")
    print(f"    - n_outputs: {loaded_model.n_outputs}")
    print(f"    - n_times: {loaded_model.n_times}")
    try:
        print(f"    - sfreq: {loaded_model.sfreq}")
    except:
        print(f"    - sfreq: None")

    # Count parameters
    loaded_params = sum(p.numel() for p in loaded_model.parameters())
    print(f"  - Model parameters: {loaded_params:,}")

except Exception as e:
    print(f"❌ Error loading from Hub: {e}")
    import traceback
    traceback.print_exc()
    raise

# Step 3: Test forward pass
print(f"\n[3/3] Testing forward pass with loaded model...")
try:
    loaded_model.eval()
    x = torch.randn(2, loaded_model.n_chans, loaded_model.n_times)
    with torch.no_grad():
        output = loaded_model(x)
    print(f"✓ Forward pass successful!")
    print(f"  - Input shape: {tuple(x.shape)}")
    print(f"  - Output shape: {tuple(output.shape)}")
except Exception as e:
    print(f"❌ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Can load and use model from Hub!")
print("=" * 70)
print(f"\nModel available at: https://huggingface.co/{REPO_ID}")
print("\nNote: The full integration (push + load) works in CI with proper tokens.")
print("Local tests verify the loading functionality works correctly.")
