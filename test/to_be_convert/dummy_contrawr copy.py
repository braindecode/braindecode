import torch
from braindecode.models import ContraWR
batch_size = 64
# Okay
parameters_list = [{"n_times": 1536, "n_chans": 16, "sfreq": 512.0, "n_outputs": 3},{"n_times": 512, "n_chans": 16, "sfreq": 512.0, "n_outputs": 2},{"n_times": 125, "n_chans": 32, "sfreq": 500.0, "n_outputs": 2},{"n_times": 2560, "n_chans": 32, "sfreq": 512.0, "n_outputs": 2},{"n_times": 2560, "n_chans": 13, "sfreq": 512.0, "n_outputs": 2},{"n_times": 512, "n_chans": 32, "sfreq": 512.0, "n_outputs": 2},{"n_times": 2560, "n_chans": 15, "sfreq": 512.0, "n_outputs": 2},{"n_times": 1024, "n_chans": 30, "sfreq": 1024.0, "n_outputs": 2},{"n_times": 5120, "n_chans": 16, "sfreq": 512.0, "n_outputs": 2},{"n_times": 1536, "n_chans": 64, "sfreq": 512.0, "n_outputs": 2},{"n_times": 2048, "n_chans": 32, "sfreq": 2048.0, "n_outputs": 2},{"n_times": 614, "n_chans": 64, "sfreq": 2048.0, "n_outputs": 2},{"n_times": 1536, "n_chans": 61, "sfreq": 512.0, "n_outputs": 7},{"n_times": 899, "n_chans": 31, "sfreq": 1000.0, "n_outputs": 2},{"n_times": 4000, "n_chans": 62, "sfreq": 1000.0, "n_outputs": 4},{"n_times": 4000, "n_chans": 62, "sfreq": 1000.0, "n_outputs": 2},{"n_times": 153, "n_chans": 8, "sfreq": 512.0, "n_outputs": 2},{"n_times": 1000, "n_chans": 62, "sfreq": 1000.0, "n_outputs": 2},{"n_times": 1200, "n_chans": 31, "sfreq": 1000.0, "n_outputs": 2}]
for params in parameters_list:
    model = ContraWR(
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
