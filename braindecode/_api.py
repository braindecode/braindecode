
from typing import Optional

from .models.shallow_fbcsp import ShallowFBCSPNet
from .pretrained import WeightsEnum

MODELS = {
    'shallowfbcspnet': ShallowFBCSPNet
}

from warnings import warn

# enter moabb str name for dataset, and itn for subject id

def get_model(name: str, weights: Optional[WeightsEnum] = None, **config):  # Need to verify that the correct weights have been passed

    name = name.lower()
    model = MODELS[name]

    if isinstance(weights, WeightsEnum):
        init_params = weights.fetch_init_params()

        # check for args conflict
        for k in config.keys():
            if k in init_params:
                if config[k] != init_params[k]:
                    warn(f'You are overiding the non-default pretrained parameter {k}={init_params[k]}'
                         f' with {config[k]}. If this parameter is necessary for correct model'
                         f'initalisation, initialisation will fail.')

        config = {**init_params, **config}

        # initialise
        model = model(**config)

        state_dict = weights.fetch_state_dict()
        model.load_state_dict(state_dict)

    elif weights is None:
        model = model(**config)
    else:
        raise ValueError

    return model


