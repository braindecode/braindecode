"""Dataset objects for some public datasets."""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import warnings
from typing import Any

import mne
import pandas as pd

from braindecode.util import _update_moabb_docstring

from .base import BaseConcatDataset, BaseDataset


def _find_dataset_in_moabb(dataset_name, dataset_kwargs=None):
    # soft dependency on moabb
    from moabb.datasets.utils import dataset_list

    for dataset in dataset_list:
        if dataset_name == dataset.__name__:
            # return an instance of the found dataset class
            if dataset_kwargs is None:
                return dataset()
            else:
                return dataset(**dataset_kwargs)
    raise ValueError(f"{dataset_name} not found in moabb datasets")


def _fetch_and_unpack_moabb_data(dataset, subject_ids=None, dataset_load_kwargs=None):
    if dataset_load_kwargs is None:
        data = dataset.get_data(subject_ids)
    else:
        data = dataset.get_data(subjects=subject_ids, **dataset_load_kwargs)

    raws, subject_ids, session_ids, run_ids = [], [], [], []
    for subj_id, subj_data in data.items():
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                annots = _annotations_from_moabb_stim_channel(raw, dataset)
                raw.set_annotations(annots)
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    description = pd.DataFrame(
        {"subject": subject_ids, "session": session_ids, "run": run_ids}
    )
    return raws, description


def _annotations_from_moabb_stim_channel(raw, dataset):
    # find events from the stim channel
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    if len(stim_channels) > 0:
        # returns an empty array if none found
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        event_id = dataset.event_id
    else:
        events, event_id = mne.events_from_annotations(raw, verbose=False)

    # get annotations from events
    event_desc = {k: v for v, k in event_id.items()}
    annots = mne.annotations_from_events(events, raw.info["sfreq"], event_desc)

    # set trial on and offset given by moabb
    onset, offset = dataset.interval
    annots.onset += onset
    annots.duration += offset - onset
    return annots


def fetch_data_with_moabb(
    dataset_name: str,
    subject_ids: list[int] | int | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
    dataset_load_kwargs: dict[str, Any] | None = None,
) -> tuple[list[mne.io.Raw], pd.DataFrame]:
    # ToDo: update path to where moabb downloads / looks for the data
    """Fetch data using moabb.

    Parameters
    ----------
    dataset_name: str | moabb.datasets.base.BaseDataset
        the name of a dataset included in moabb
    subject_ids: list(int) | int
        (list of) int of subject(s) to be fetched
    dataset_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset when instantiating it.
    data_load_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset's load_data method.
        Allows using the moabb cache_config=None and
        process_pipeline=None.

    Returns
    -------
    raws: mne.Raw
    info: pandas.DataFrame
    """
    if isinstance(dataset_name, str):
        dataset = _find_dataset_in_moabb(dataset_name, dataset_kwargs)
    else:
        from moabb.datasets.base import BaseDataset

        if isinstance(dataset_name, BaseDataset):
            dataset = dataset_name

    subject_id = [subject_ids] if isinstance(subject_ids, int) else subject_ids
    return _fetch_and_unpack_moabb_data(
        dataset, subject_id, dataset_load_kwargs=dataset_load_kwargs
    )


class MOABBDataset(BaseConcatDataset):
    """A class for moabb datasets.

    Parameters
    ----------
    dataset_name: str
        name of dataset included in moabb to be fetched
    subject_ids: list(int) | int | None
        (list of) int of subject(s) to be fetched. If None, data of all
        subjects is fetched.
    dataset_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset when instantiating it.
    dataset_load_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset's load_data method.
        Allows using the moabb cache_config=None and
        process_pipeline=None.
    """

    def __init__(
        self,
        dataset_name: str,
        subject_ids: list[int] | int | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        dataset_load_kwargs: dict[str, Any] | None = None,
    ):
        # soft dependency on moabb
        from moabb import __version__ as moabb_version  # type: ignore

        if moabb_version == "1.0.0":
            warnings.warn(
                "moabb version 1.0.0 generates incorrect annotations. "
                "Please update to another version, version 0.5 or 1.1.0 "
            )

        raws, description = fetch_data_with_moabb(
            dataset_name,
            subject_ids,
            dataset_kwargs,
            dataset_load_kwargs=dataset_load_kwargs,
        )
        all_base_ds = [
            BaseDataset(raw, row) for raw, (_, row) in zip(raws, description.iterrows())
        ]
        super().__init__(all_base_ds)


class BNCI2014001(MOABBDataset):
    doc = """See moabb.datasets.bnci.BNCI2014001

    Parameters
    ----------
    subject_ids: list(int) | int | None
        (list of) int of subject(s) to be fetched. If None, data of all
        subjects is fetched.
    """
    try:
        from moabb.datasets import BNCI2014001

        __doc__ = _update_moabb_docstring(BNCI2014001, doc)
    except ModuleNotFoundError:
        pass  # keep moabb soft dependency, otherwise crash on loading of datasets.__init__.py

    def __init__(self, subject_ids):
        super().__init__("BNCI2014001", subject_ids=subject_ids)


class HGD(MOABBDataset):
    doc = """See moabb.datasets.schirrmeister2017.Schirrmeister2017

    Parameters
    ----------
    subject_ids: list(int) | int | None
        (list of) int of subject(s) to be fetched. If None, data of all
        subjects is fetched.
    """
    try:
        from moabb.datasets import Schirrmeister2017

        __doc__ = _update_moabb_docstring(Schirrmeister2017, doc)
    except ModuleNotFoundError:
        pass  # keep moabb soft dependency, otherwise crash on loading of datasets.__init__.py

    def __init__(self, subject_ids):
        super().__init__("Schirrmeister2017", subject_ids=subject_ids)
