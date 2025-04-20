# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np

from braindecode.models.util import (
    models_dict,
    models_mandatory_parameters,
    non_classification_models,
)

rng = np.random.default_rng(12)

chs_info = [
    {
        "ch_name": f"C{i}",
        "kind": "eeg",
        "loc": rng.random(12),
    }
    for i in range(1, 4)
]

default_signal_params = dict(
    n_times=1000,
    sfreq=250,
    n_outputs=2,
    chs_info=chs_info,
)

for model in model_list:
    # Create a random input tensorp
    input_tensor = torch.randn(1, model.n_chans, model.n_times)

    # Compile the model
    compiled_model = torch.compile(model)

    # Perform a forward pass with the compiled model
    output = compiled_model(input_tensor)

    # Check if the output shape is correct
    assert output.shape == (1, model.n_outputs)
