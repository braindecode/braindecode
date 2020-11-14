import numpy as np


def mixup_beta(datum, params):

    alpha = params["alpha"]
    if "epoch_counter" not in params.keys():
        params["acc_in_epoch"] = 0
    if alpha > 0:
        if params["beta_per_sample"]:
            lam = np.random.beta(alpha, alpha)
        else:
            if not (params["acc_in_epoch"] %
                    datum.required_variables["true_len_of_ds"]):
                params["beta"] = np.random.beta(alpha, alpha)
                lam = params["beta"]
            else:
                lam = params["beta"]
            params["acc_in_epoch"] += 1
    else:
        lam = 1
    other_data_idx = np.random.randint(len(datum.ds))
    other_data = datum.ds[other_data_idx]
    datum.X = lam * datum.X + (1 - lam) * other_data[0]
    for label in datum.y.keys():
        datum.y[label] *= lam
    if other_data[1] in datum.y.keys():
        datum.y[other_data[1]] += (1 - lam)
    else:
        datum.y[other_data[1]] = (1 - lam)
    return datum


MIXUP_BETA_REQUIRED_VARIABLES = {"true_len_of_ds": lambda dataset, transf_list:
                                 len(dataset) * len(transf_list)}
