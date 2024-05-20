# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Alexandre Gramfort
#          Pierre Guetschel
#
# License: BSD-3

import mne

from braindecode.datasets import BaseConcatDataset, BaseDataset
from braindecode.datasets.moabb import fetch_data_with_moabb
from braindecode.preprocessing.windowers import create_windows_from_events

bnci_kwargs = {
    "n_sessions": 2,
    "n_runs": 3,
    "n_subjects": 9,
    "paradigm": "imagery",
    "duration": 3869,
    "sfreq": 250,
    "event_list": ("left", "right"),
    "channels": ("C5", "C3", "C1"),
}


def concat_ds_targets():
    raws, description = fetch_data_with_moabb(
        dataset_name="FakeDataset", subject_ids=1,
        dataset_kwargs=bnci_kwargs
    )

    events, _ = mne.events_from_annotations(raws[0])
    targets = events[:, -1] - 1
    ds = [BaseDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)
    return concat_ds, targets


def concat_windows_dataset(concat_ds_targets):
    concat_ds, targets = concat_ds_targets
    windows_ds = create_windows_from_events(
        concat_ds=concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=750,
        window_stride_samples=100,
        drop_last_window=False,
    )

    return windows_ds
