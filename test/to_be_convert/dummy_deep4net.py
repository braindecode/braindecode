
import torch
from braindecode.models import Deep4Net
batch_size = 64

parameters_list = [
    {"n_times": 256, "n_chans": 8, "sfreq": 256.0, "n_outputs": 2},
    {"n_times": 204, "n_chans": 8, "sfreq": 256.0, "n_outputs": 2},
    {"n_times": 125, "n_chans": 32, "sfreq": 500.0, "n_outputs": 2},
    {"n_times": 204, "n_chans": 16, "sfreq": 256.0, "n_outputs": 2},
    {"n_times": 128, "n_chans": 16, "sfreq": 128.0, "n_outputs": 2},
    {"n_times": 384, "n_chans": 14, "sfreq": 128.0, "n_outputs": 5},
    {"n_times": 153, "n_chans": 8, "sfreq": 512.0, "n_outputs": 2},
]
for i, params in enumerate(parameters_list):
    try:
        model = Deep4Net(
            n_times=params["n_times"],
            n_chans=params["n_chans"],
            sfreq=params["sfreq"],
            n_outputs=params["n_outputs"],
        )
    except Exception as e:
        print(f"Failed to create model for n_times={params['n_times']}: {e}")
        continue
    
    # Create a dummy input tensor with the shape (batch_size, n_chans, n_times)
    dummy_input = torch.randn(batch_size, params["n_chans"], params["n_times"])
    
    # Forward pass
    output = model(dummy_input)
    print(f"Model created successfully for n_times={params['n_times']}")

