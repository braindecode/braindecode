from copy import deepcopy
from typing import Any, Dict, List, Optional
from warnings import simplefilter

import numpy as np
import torch

from braindecode.models import *
from braindecode.models.modules import Expression
from braindecode.models.util import (
    models_dict,
    models_mandatory_parameters,
    non_classification_models,
)

simplefilter(action='ignore')

rng = np.random.default_rng(12)
# Generating the channel info
chs_inf_img = [
    {
        "ch_name": f"C{i}",
        "kind": "eeg",
        "loc": rng.random(12),
    }
    for i in range(1, 5)
]
chs_inf_sleep = [
    {
        "ch_name": f"C{i}",
        "kind": "eeg",
        "loc": rng.random(12),
    }
    for i in range(1, 2)
]
chs_inf_sleep_2 = [
    {
        "ch_name": f"C{i}",
        "kind": "eeg",
        "loc": rng.random(12),
    }
    for i in range(1, 5)
]
# Generating the signal parameters
params_img = dict(
    n_times=1000,
    sfreq=250,
    n_outputs=2,
    n_chans=len(chs_inf_img),
    input_window_seconds=4,
    chs_info=chs_inf_img,
)
#chs_info=chs_info,
params_sleep = dict(
    n_times=30*100,
    sfreq=100,
    n_outputs=4,
    n_chans=len(chs_inf_sleep),
    input_window_seconds=30,
    chs_info=chs_inf_sleep,
)

params_sleep_2 = dict(
    n_times=30*100,
    sfreq=100,
    n_outputs=4,
    n_chans=len(chs_inf_sleep_2),
    input_window_seconds=30,
    chs_info=chs_inf_sleep_2,
)



#model_instances = build_model_list()

for model in models_dict.keys():

    model_name = model
    if "SignalJEPA" in model_name:
        continue 

    #print(model_name, end=" " )
    try:
        if "Sleep" in model_name:
            if "SleepStagerBlanco2020" != model_name:
                default_signal_params = deepcopy(params_sleep)
            else:
                default_signal_params = deepcopy(params_sleep_2)
        else:
            default_signal_params = deepcopy(params_img)


        model_class = models_dict[model]
        model = model_class(**default_signal_params)
        
        x = torch.randn(2, model.n_chans, model.n_times)

        out = model(x)
        # if "Sleep" in model_name:
        model_script = torch.jit.script(model)
        #print(model)
        print(f"Successfully scripted {model.__class__.__name__}")
        print(" " * 200 + "\n\n", end="\r")
    except Exception as ex:

        #variable_name = model
        print(f"Failed scripted {model_name}:", ex)
        continue


