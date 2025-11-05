"""
Script to fix Hub upload by deleting old repo and pushing fresh model with weights.
"""
from braindecode.models import EEGNet
from huggingface_hub import delete_repo
import torch

REPO_ID = "Kkuntal990/test-eegnet-braindecode"

print("=" * 70)
print("FIXING HUGGING FACE HUB UPLOAD")
print("=" * 70)

# Step 1: Delete the old repo
print("\n[1/4] Deleting old repository...")
try:
    delete_repo(REPO_ID, repo_type="model")
    print(f"✓ Deleted repository: {REPO_ID}")
except Exception as e:
    print(f"Note: {e}")
    if "404" in str(e):
        print("✓ Repository doesn't exist or already deleted")

# Step 2: Create fresh model
print("\n[2/4] Creating fresh EEGNet model...")
model = EEGNet(
    n_chans=22,
    n_outputs=4,
    n_times=1000,
    sfreq=250.0
)
print(f"✓ Model created: {model.__class__.__name__}")
print(f"  - Parameters: n_chans=22, n_outputs=4, n_times=1000, sfreq=250.0")

# Step 3: Initialize model with forward pass
print("\n[3/4] Initializing model (forward pass)...")
x = torch.randn(2, 22, 1000)
output = model(x)
print(f"✓ Model initialized")
print(f"  - Input shape: {tuple(x.shape)}")
print(f"  - Output shape: {tuple(output.shape)}")

# Step 4: Push to Hub with weights
print(f"\n[4/4] Pushing complete model to Hub...")
print(f"  - Repository: {REPO_ID}")
try:
    model.push_to_hub(
        REPO_ID,
        commit_message="Complete model with weights and config",
    )
    print("✓ Successfully pushed model to Hub!")
    print("\n" + "=" * 70)
    print("SUCCESS! Model uploaded with weights.")
    print("=" * 70)
    print(f"\nYou can now run: python test_hf_hub_integration.py {REPO_ID}")
except Exception as e:
    print(f"❌ Error pushing to Hub: {e}")
    raise
