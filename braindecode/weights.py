import json
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

fetch_from_hf_hub = partial(
    hf_hub_download, "dcwil/test-repo"
)  # Swap for braindecode repo


@dataclass
class Weights:
    path: str


class WeightsEnum(Enum):
    # TODO: add torch.device to uploaded file, so map location can be automatically set

    def fetch_state_dict(self):
        p = fetch_from_hf_hub(filename=str(Path(self.path).joinpath("state_dict.pkl")))
        return torch.load(p, map_location=torch.device("cpu"))

    def fetch_init_params(self):
        p = fetch_from_hf_hub(
            filename=str(Path(self.path).joinpath("init_params.json"))
        )
        with open(p, "r") as h:
            init_params = json.load(h)
        return init_params

    @property
    def path(self):
        return self.value.path