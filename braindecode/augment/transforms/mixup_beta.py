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
    datum.y = list(datum.y)
    for label_idx in range(len(datum.y)):
        datum.y[label_idx] *= lam
        if datum.list_of_labels[other_data[1]] == label_idx:
            datum.y[other_data[1]] += (1 - lam)
    datum.y = tuple(datum.y)
    return datum


MIXUP_BETA_REQUIRED_VARIABLES = {"true_len_of_ds": lambda dataset, transf_list:
                                 len(dataset) * len(transf_list)}
