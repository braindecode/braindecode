import torch
from braindecode.models import BIOT
batch_size = 64
# Okay
parameters_list = [{"n_times": 204, "n_chans": 8, "sfreq": 256.0, "n_outputs": 2},{"n_times": 125, "n_chans": 32, "sfreq": 500.0, "n_outputs": 2},{"n_times": 204, "n_chans": 16, "sfreq": 256.0, "n_outputs": 2},{"n_times": 614, "n_chans": 64, "sfreq": 2048.0, "n_outputs": 2},{"n_times": 899, "n_chans": 31, "sfreq": 1000.0, "n_outputs": 2},{"n_times": 153, "n_chans": 8, "sfreq": 512.0, "n_outputs": 2}]
for params in parameters_list:
    model = BIOT(
        n_times=params["n_times"],
        n_chans=params["n_chans"],
        sfreq=params["sfreq"],
        n_outputs=params["n_outputs"],
    )
    # Create a dummy input tensor with the shape (batch_size, n_chans, n_times)
    dummy_input = torch.randn(batch_size, params["n_chans"], params["n_times"])
    # Forward pass
    output = model(dummy_input)
    print(model)
