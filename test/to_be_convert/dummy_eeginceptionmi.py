
import torch
from braindecode.models import EEGInceptionMI
batch_size = 64

# NOT okay Parameters
parameters_list = [{"n_times": 2000, "n_chans": 63, "sfreq": 500.0, "n_outputs": 4},{"n_times": 125, "n_chans": 32, "sfreq": 500.0, "n_outputs": 2},{"n_times": 128, "n_chans": 16, "sfreq": 128.0, "n_outputs": 2},{"n_times": 1000, "n_chans": 61, "sfreq": 500.0, "n_outputs": 4},{"n_times": 3500, "n_chans": 128, "sfreq": 500.0, "n_outputs": 2},{"n_times": 1024, "n_chans": 30, "sfreq": 1024.0, "n_outputs": 2},{"n_times": 384, "n_chans": 14, "sfreq": 128.0, "n_outputs": 5},{"n_times": 2048, "n_chans": 32, "sfreq": 2048.0, "n_outputs": 2},{"n_times": 2000, "n_chans": 29, "sfreq": 500.0, "n_outputs": 2},{"n_times": 800, "n_chans": 60, "sfreq": 200.0, "n_outputs": 7},{"n_times": 614, "n_chans": 64, "sfreq": 2048.0, "n_outputs": 2},{"n_times": 2000, "n_chans": 30, "sfreq": 200.0, "n_outputs": 2},{"n_times": 2000, "n_chans": 128, "sfreq": 500.0, "n_outputs": 4},{"n_times": 899, "n_chans": 31, "sfreq": 1000.0, "n_outputs": 2},{"n_times": 4000, "n_chans": 62, "sfreq": 1000.0, "n_outputs": 4},{"n_times": 4000, "n_chans": 62, "sfreq": 1000.0, "n_outputs": 2},{"n_times": 1000, "n_chans": 62, "sfreq": 1000.0, "n_outputs": 2},{"n_times": 480, "n_chans": 64, "sfreq": 160.0, "n_outputs": 5},{"n_times": 500, "n_chans": 8, "sfreq": 500.0, "n_outputs": 2},{"n_times": 1200, "n_chans": 31, "sfreq": 1000.0, "n_outputs": 2}]
for i, params in enumerate(parameters_list):
    model = EEGInceptionMI(
        n_times=params["n_times"],
        n_chans=params["n_chans"],
        sfreq=params["sfreq"],
        n_outputs=params["n_outputs"],
    )
    # compile the model because it too slowly
    model = torch.compile(model, mode="max-autotune")
    # Create a dummy input tensor with the shape (batch_size, n_chans, n_times)
    dummy_input = torch.randn(batch_size, params["n_chans"], params["n_times"])
    
    # Forward pass
    output = model(dummy_input)
    print(f"Model created successfully for n_times={params['n_times']}")

