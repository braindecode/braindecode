from warnings import warn

from .models.shallow_fbcsp import ShallowFBCSPNet, ShallowFBCSPNetWeights

MODELS_AND_WEIGHTS = {
    'shallowfbcspnet': {
        'model': ShallowFBCSPNet,
        'weights': ShallowFBCSPNetWeights
    }
    # Other models go here
}


def get_model(name: str, dataset_name: str = None, subject_id: int = None, **init_params):

    name = name.lower()
    model = MODELS_AND_WEIGHTS[name]['model']

    if dataset_name is None and subject_id is None:
        return model(**init_params)

    weights_enum = MODELS_AND_WEIGHTS[name]['weights']

    dset_sub = f'{dataset_name}_{subject_id}'
    if dset_sub not in weights_enum:
        raise ValueError(f'Pretrained weights for {dataset_name} & {subject_id} not available!')

    weights = weights_enum[f'{dataset_name}_{subject_id}']
    pretrained_init_params = weights.fetch_init_params()
    init_params = _check_params(pretrained_init_params, init_params)

    # initialise
    model = model(**init_params)
    state_dict = weights.fetch_state_dict()
    model.load_state_dict(state_dict)

    return model


def _check_params(pretrained_init: dict, passed_init: dict) -> dict:
    """

    Parameters
    ----------
    pretrained_init: dict, parameters used to initalise the model for pretraining,
     from Weights.fetch_init_params()
    passed_init: dict, parameters used has passed to initialise the model.
     They will overwrite parameters in pretrained_init

    Returns
    -------
    init_params: dict, parameters to initialise model
    """
    for k in passed_init.keys():
        if k in pretrained_init and passed_init[k] != pretrained_init[k]:
            warn(f'You are overiding the non-default pretrained parameter {k}={pretrained_init[k]}'
                 f' with {passed_init[k]}. If this parameter is necessary for correct model '
                 f'initalisation, initialisation will fail.')
    return {**pretrained_init, **passed_init}


