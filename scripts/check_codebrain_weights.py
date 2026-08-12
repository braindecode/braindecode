"""Verify that upstream CodeBrain weights load into the braindecode model.

Downloads CodeBrain.pth from https://huggingface.co/YjMajy/CodeBrain
(a DataParallel-saved checkpoint) and loads it via CodeBrain.load_state_dict,
which handles all key remapping automatically.

The upstream checkpoint contains only backbone weights (no classification head),
so final_layer.* keys are expected to be missing.
"""

import torch
from huggingface_hub import hf_hub_download

from braindecode.models.codebrain import CodeBrain

# ---------- 1. Download upstream weights ----------
print("Downloading CodeBrain.pth from HuggingFace...")
path = hf_hub_download(repo_id="YjMajy/CodeBrain", filename="CodeBrain.pth")
# Checkpoint was saved from DataParallel (keys have "module." prefix)
upstream = torch.load(path, map_location="cpu", weights_only=True)

# ---------- 2. Instantiate braindecode model with matching config ----------
# Upstream pretrained config: 19 channels, 6000 time points, 8 res layers
# n_outputs is arbitrary — the upstream checkpoint has no classification head
model = CodeBrain(n_chans=19, n_outputs=2, n_times=6000)

# ---------- 3. Load with automatic key remapping ----------
missing, unexpected = model.load_state_dict(upstream, strict=False)

print(f"\nMissing keys:    {len(missing)}")
for k in sorted(missing):
    print(f"  {k}")
print(f"\nUnexpected keys: {len(unexpected)}")
for k in sorted(unexpected):
    print(f"  {k}")

# ---------- 4. Validate ----------
# Only classification head keys should be missing (randomly initialized for fine-tuning)
expected_missing = {k for k in missing if k.startswith("final_layer.")}
unexpected_missing = sorted(set(missing) - expected_missing)

if unexpected_missing:
    print(f"\nERROR: {len(unexpected_missing)} unexpected missing keys:")
    for k in unexpected_missing:
        print(f"  {k}")
    raise RuntimeError("Weight loading failed — unexpected missing keys")

if unexpected:
    print(f"\nERROR: {len(unexpected)} unexpected keys in checkpoint")
    raise RuntimeError("Weight loading failed — unexpected keys")

print(f"\nAll backbone weights loaded successfully!")
print(f"({len(expected_missing)} classification head keys randomly initialized)")

# ---------- 5. Smoke test forward pass ----------
x = torch.randn(1, 19, 6000)
model.eval()
with torch.no_grad():
    out = model(x)
print(f"Forward pass OK: output shape = {out.shape}")
