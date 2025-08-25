from typing import Sequence

import torch
from torch.utils.data import Dataset

from braindecode.datasets.base import BaseConcatDataset
from braindecode.samplers.experimental import DatasetIndex


class SamplerWindowedConcatDataset(Dataset):
    """
    Tiny dataset that uses the sampler's mapping to slice Raw on-the-fly.
    Returns (X, p_factor, crop_inds, infos) just like your original wrapper.
    """

    def __init__(
        self, concat_ds: BaseConcatDataset, index_mapping: Sequence[DatasetIndex]
    ):
        self.concat_ds = concat_ds
        self.mapping = list(index_mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, virtual_idx: int):
        m = self.mapping[virtual_idx]
        ds = self.concat_ds.datasets[m.ds_idx]
        raw = ds.raw

        # Slice Raw into (n_channels, n_times) for the requested window
        # mne.io.Raw.get_data uses 'start' inclusive and 'stop' exclusive
        X = raw._slice(start=m.start, stop=m.stop)  # np.ndarray (C, T)

        # Label
        p_factor = float(ds.description["p_factor"])

        # Extra infos
        desc = ds.description
        infos = {
            "subject": desc["subject"],
            "sex": desc.get("sex", desc.get("gender", "")),
            "age": float(desc["age"]),
            "task": desc["task"],
            "session": desc.get("session", "") or "",
            "run": desc.get("run", "") or "",
        }

        # Braindecode-style crop indices (absolute sample positions)
        crop_inds = (m.i_window_in_trial, m.start, m.stop)

        # Return tensors for PyTorch (DataLoader will stack)
        return (
            torch.from_numpy(X),
            torch.tensor(p_factor, dtype=torch.float32),
            crop_inds,
            infos,
        )
