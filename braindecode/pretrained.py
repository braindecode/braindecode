from _warnings import warn

from .models.util import models_dict, weights_dict


def initialize_model(
    name: str, dataset_name: str = None, subject_id: int = None, **init_params
):
    """
    Initialize and return a model specified by the `name` parameter. If `dataset_name` and
    `subject_id` are provided, pretrained weights associated with those parameters will be downloaded
    and used for initialization; otherwise, random initialization will be performed.

    Specific initialization parameters can be passed via `**init_params`. When using pretrained
    weights, any parameters provided via `init_params` will override the associated parameters used
    during pretraining. Whether this is desired depends on the specific model, dataset, and use case.

    Parameters
    ----------
    name : str
        Model name.
    dataset_name : str or None, optional
        Dataset name (corresponding to MOABB) for downloading pretrained weights. Default is None.
    subject_id : int or None, optional
        Subject identifier for downloading pretrained weights. Default is None.
    init_params : kwargs, optional
        Additional parameters to pass to the model for initialization.

    Returns
    -------
    model
        The initialized model.
    """

    model = models_dict[name]

    if dataset_name is None and subject_id is None:
        return model(**init_params)
    elif dataset_name is None or subject_id is None:
        raise ValueError(
            "If using pretrained weights need to specify both"
            " dataset name and subject id"
        )

    weights_enum = weights_dict[name]

    try:
        weights = weights_enum[f"{dataset_name}_{subject_id}"]
    except KeyError as e:
        raise KeyError(
            f"Pretrained weights for {dataset_name} & {subject_id} not available!"
        )

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