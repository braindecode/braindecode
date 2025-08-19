from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Sampler

from braindecode.datasets.base import BaseConcatDataset


@dataclass(frozen=True)
class DatasetIndex:
    ds_idx: int  # which sub-dataset inside the BaseConcatDataset
    start: int  # absolute sample start in the Raw
    stop: int  # absolute sample stop (exclusive)
    i_window_in_trial: int  # window index within that recording (for traceability)


class FixedWindowSampler(Sampler[int]):
    """
    Enumerates fixed-length windows over each Raw in a BaseConcatDataset,
    matching Braindecode's create_fixed_length_windows semantics for
    window_size/stride and drop_last behavior.

    Optional 'jitter' applies a single left-offset per recording in [0, stride),
    akin to a global temporal shift (disabled by default to exactly match
    Braindecode's deterministic placement).
    """

    def __init__(
        self,
        concat_ds: BaseConcatDataset,
        *,
        window_size_samples: int,
        window_stride_samples: int,
        drop_last_window: bool = True,
        shuffle: bool = True,
        jitter: bool = False,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.concat_ds = concat_ds
        self.window_size = int(window_size_samples)
        self.stride = int(window_stride_samples)
        self.drop_last = drop_last_window
        self.shuffle = shuffle
        self.jitter = jitter
        self.seed = 0 if seed is None else int(seed)
        self.mapping: List[DatasetIndex] = []
        self._build_mapping()

        if generator is None:
            self.generator = torch.Generator().manual_seed(self.seed)
        elif isinstance(generator, torch.Generator):
            self.generator = generator

    def _build_mapping(self):
        self.mapping.clear()
        g = self.generator
        for ds_idx, ds in enumerate(self.concat_ds.datasets):
            n_times = int(ds.raw.n_times)
            if n_times < self.window_size:
                continue

            # No jitter (exact Braindecode behavior) unless requested.
            left_offset = 0
            if self.jitter:
                # uniform integer in [0, stride)
                left_offset = int(
                    torch.randint(
                        low=0, high=max(1, self.stride), size=(1,), generator=g
                    ).item()
                )

            last_start = n_times - self.window_size
            if last_start < 0:
                continue

            starts = list(range(left_offset, last_start + 1, self.stride))
            if not self.drop_last and (len(starts) == 0 or starts[-1] != last_start):
                starts.append(last_start)

            for i_win, s in enumerate(starts):
                self.mapping.append(
                    DatasetIndex(
                        ds_idx=ds_idx,
                        start=s,
                        stop=s + self.window_size,
                        i_window_in_trial=i_win,
                    )
                )

    def __len__(self) -> int:
        return len(self.mapping)

    def set_epoch(self, epoch: int):
        # Optional: call this from your training loop to get different shuffles per epoch
        self._epoch = int(epoch)

    def __iter__(self):
        # Yield indices into 'WindowedConcatDataset' (0..len(mapping)-1)
        if not self.shuffle:
            yield from range(len(self.mapping))
            return

        # Epoch-aware shuffling for reproducibility
        epoch = getattr(self, "_epoch", 0)
        self.generator.manual_seed(self.seed + epoch)
        perm = torch.randperm(len(self.mapping), generator=self.generator).tolist()
        for idx in perm:
            yield idx
