import json

from huggingface_hub import hf_hub_download

# from braindecode.classifier import EEGClassifier

# like to be able to load independent of skorch
# toggle skorch?
# mimic torchvision style

# def fetch_pretrained_model(model, clf_args=None, dataset_name='BNCI2014001', subject_id=3, ):
#     repo_id = "dcwil/test-repo"  # swap for braindecode repo
#
#     model_pkl = hf_hub_download(repo_id=repo_id, filename=f"{dataset_name}/{subject_id}/test_model.pkl")
#     optimizer_pkl = hf_hub_download(repo_id=repo_id, filename=f"{dataset_name}/{subject_id}/test_optimizer.pkl")
#     history_json = hf_hub_download(repo_id=repo_id, filename=f"{dataset_name}/{subject_id}/test_history.json")
#
#     if clf_args is None:
#         clf_args = dict()
#
#     clf = EEGClassifier(model, **clf_args)
#     clf.initialize()
#     clf.load_params(f_params=model_pkl, f_optimizer=optimizer_pkl, f_history=history_json)
#     return clf
#
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

import torch


fetch_from_hf_hub = partial(hf_hub_download, 'dcwil/test-repo')

@dataclass
class Weights:
    path: str


class WeightsEnum(Enum):

    def fetch_state_dict(self):
        p = fetch_from_hf_hub(filename=str(Path(self.path).joinpath('state_dict.pkl')))
        return torch.load(p, map_location=torch.device('cpu'))

    def fetch_init_params(self):
        p = fetch_from_hf_hub(filename=str(Path(self.path).joinpath('init_params.json')))
        with open(p, 'r') as h:
            init_params = json.load(h)
        return init_params



    @property
    def path(self):
        return self.value.path
