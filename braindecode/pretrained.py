from _warnings import warn

from .models.util import models_dict, weights_dict


def initialize_model(
    name: str, weights_id: None, **init_params
):
    """
    Initialize and return a model specified by the `name` parameter. If `weights_id` is provided,
    pretrained weights will be downloaded and used for initialization; otherwise, random
    initialization will be performed.

    TODO: add weight_id naming guide once we've decided on a setup

    Specific initialization parameters can be passed via `**init_params`. When using pretrained
    weights, any parameters provided via `init_params` will override the associated parameters used
    during pretraining. Whether this is desired depends on the specific model, dataset, and use case.

    Parameters
    ----------
    name : str
        Model name.
    weights_id : str or None, optional
        Subject identifier for downloading pretrained weights. Default is None.
    init_params : kwargs, optional
        Additional parameters to pass to the model for initialization.

    Returns
    -------
    model
        The initialized model.
    """

    model = models_dict[name]
    weights = weights_dict[name][weights_id]

    pretrained_init_params = weights.fetch_init_params()
    init_params = _check_params(pretrained_init_params, init_params)

    model = model(**init_params)
    state_dict = weights.fetch_state_dict()
    model.load_state_dict(state_dict)

    return model


def _check_params(pretrained_init: dict, passed_init: dict) -> dict:
    """

    Parameters
    ----------
    pretrained_init: dict
        parameters used to initalise the model for pretraining,
        from Weights.fetch_init_params()
    passed_init: dict
        parameters used has passed to initialise the model.
        They will overwrite parameters in pretrained_init

    Returns
    -------
    init_params: dict
        parameters to initialise model
    """
    for k in passed_init.keys():
        if k in pretrained_init and passed_init[k] != pretrained_init[k]:
            warn(
                f"You are overiding the non-default pretrained parameter {k}={pretrained_init[k]}"
                f" with {passed_init[k]}. If this parameter is necessary for correct model "
                f"initalisation, initialisation will fail."
            )
    return {**pretrained_init, **passed_init}
