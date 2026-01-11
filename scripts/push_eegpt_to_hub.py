
import sys
from pathlib import Path

import torch

# Add project root to sys.path to allow imports from root and braindecode package
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from braindecode.models import EEGPT
from simplify_channel_selection import get_eegpt_chs_info


def push_eegpt():
    print("Preparing to push EEGPT model to Hugging Face Hub...")

    # 1. Get channel info with better location data
    chs_info = get_eegpt_chs_info()
    n_chans = len(chs_info)
    print(f"Generated channel info for {n_chans} channels.")

    passed_channels = {ch['ch_name'] for ch in chs_info}
    print(f"Channels: {passed_channels}")

    # 2. Instantiate the model
    # Note: We must ensure n_chans matches the pretrained weights.
    # The EEGPT code expects 62 channels by default (len(CHANNEL_DICT)).
    # We use n_times=1000 and sfreq=250 as reasonable foundation model defaults.
    model = EEGPT(
        n_chans=n_chans,
        n_outputs=1,
        n_times=1000,
        chs_info=chs_info,
        sfreq=250,
        return_encoder_output=True
    )

    # 3. Load pretrained weights if available
    ckpt_path = root_dir / "EEGPT" / "checkpoint" / "eegpt_mcae_58chs_4s_large4E.ckpt"

    if ckpt_path.exists():
        print(f"Loading weights from {ckpt_path}")
        # Use simple torch load with weights_only=False (trusted source)
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Handle Lightning checkpoint format
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Clean up keys if necessary (e.g. remove prefixes)
        # But try standard load first.
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("Weights loaded.")
            print(f"Missing keys: {len(missing)}")
            if len(missing) > 0:
                print(f"Missing keys list: {missing}")
            print(f"Unexpected keys: {len(unexpected)}")

            # Check for critical keys
            critical_keys = ["target_encoder.chan_embed.weight"]
            for k in critical_keys:
                if k in missing:
                    print(f"CRITICAL WARNING: {k} is MISSING! Channel embeddings will be random!")

            # If large mismatch, might indicate wrong config or model version
            if len(missing) > 20:
                print("WARNING: Large number of missing keys. Review model configuration.")
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            return
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}. Model will be initialized with random weights!")
        print("Aborting push to prevent overwriting pretrained model with random weights.")
        return

    # 4. Push to Hub
    repo_id = "braindecode/eegpt-pretrained"
    print(f"Pushing to Hub repo: {repo_id}")

    try:
        model.push_to_hub(
            repo_id=repo_id,
            commit_message="Update channel configuration with standard 1005 locations"
        )
        print("Successfully pushed model to Hugging Face Hub.")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")
        print("Ensure you are logged in using `huggingface-cli login` or have HF_TOKEN env var set.")

if __name__ == "__main__":
    push_eegpt()
