"""SeizeIT2: wearable behind-the-ear EEG for seizure detection.

OpenNeuro ds005873, released under CC0 in the same BIDS layout and SzCORE
events format as :class:`braindecode.datasets.CHBMIT` and
:class:`braindecode.datasets.SIENA`.
"""

# Authors: Bhagubai, Miguel (Data collector)
#          Aayushya Patel
#
# License: BSD (3-clause)
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import pooch
from mne.datasets.utils import _get_path

from braindecode.datasets.utils import _ZenodoBIDSDataset

SEIZEIT2_URL = "https://s3.amazonaws.com/openneuro.org/"
SEIZEIT2_ACCESSION = "ds005873"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _list_files(prefix, delimiter=None):
    """``(key, md5 or None)`` of every OpenNeuro file whose key starts with
    ``prefix``. ``delimiter="/"`` lists one folder level only."""
    params = {"list-type": 2, "prefix": prefix}
    if delimiter is not None:
        params["delimiter"] = delimiter
    while True:
        with urlopen(f"{SEIZEIT2_URL}?{urlencode(params)}", timeout=60) as page:
            listing = ET.parse(page).getroot()
        for item in listing.iterfind("s3:Contents", _S3_NS):
            etag = item.findtext("s3:ETag", namespaces=_S3_NS).strip('"')
            # The ETag of a multipart upload is not the MD5 of the file.
            md5 = None if "-" in etag else etag
            yield item.findtext("s3:Key", namespaces=_S3_NS), md5
        token = listing.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        if token is None:
            return
        params["continuation-token"] = token


def _fetch_seizeit2(subjects=None, path=None):
    """Download the top-level files and the EEG runs of ``subjects`` (all
    subjects if None) and return the local BIDS root. Files already on disk
    with the listed MD5 are not downloaded again."""
    root = os.path.join(
        _get_path(path, "MNE_DATASETS_SEIZEIT2_PATH", "SeizeIT2"), SEIZEIT2_ACCESSION
    )

    def fetch(files):
        for key, md5 in files:
            local = os.path.join(root, key.removeprefix(f"{SEIZEIT2_ACCESSION}/"))
            pooch.retrieve(
                SEIZEIT2_URL + key,
                known_hash=None if md5 is None else f"md5:{md5}",
                fname=os.path.basename(local),
                path=os.path.dirname(local),
            )

    fetch(_list_files(f"{SEIZEIT2_ACCESSION}/", delimiter="/"))
    if subjects is None:
        participants = pd.read_csv(os.path.join(root, "participants.tsv"), sep="\t")
        subjects = participants["participant_id"].str.removeprefix("sub-")
    for subject in [subjects] if isinstance(subjects, str) else subjects:
        # The ECG, EMG and movement recordings take 60% of the dataset: skip them.
        fetch(
            (key, md5)
            for key, md5 in _list_files(f"{SEIZEIT2_ACCESSION}/sub-{subject}/")
            if "/eeg/" in key
        )
    return root


class SeizeIT2(_ZenodoBIDSDataset):
    """The SeizeIT2 wearable EEG dataset.

    SeizeIT2 :footcite:`Bhagubai2025` is a seizure monitoring dataset of 125
    children and adults with focal epilepsy, recorded in five European epilepsy
    monitoring units with the Sensor Dot, a wearable device with electrodes
    behind the ears. It holds about 11,640 hours of wearable data and 886
    annotated focal seizures.

    **Description of the contents of the dataset:**

    The recording of each subject is split into runs, at 256 Hz. Every EEG run
    has two channels out of ``BTEleft SD`` and ``BTEright SD`` (each behind
    one ear) and ``CROSStop SD`` (across the head). The dataset also has ECG,
    EMG and movement recordings, which this class neither downloads nor reads.

    Events are in the SzCORE format :footcite:`Dan2025`, as for
    :class:`CHBMIT` and :class:`SIENA`: each annotation is named after the
    ``eventType`` column, ``"sz..."`` for a seizure (with its ILAE type, e.g.
    ``"sz_foc_ia"``) and ``"impd"`` for an impedance check.

    The dataset is released under CC0 on OpenNeuro, accession ds005873
    (doi:10.18112/openneuro.ds005873). If ``root`` is None, only the EEG runs
    of the requested ``subjects`` are downloaded, into the folder set by the
    ``MNE_DATASETS_SEIZEIT2_PATH`` MNE config key (``~/mne_data`` by default).
    The EEG of all subjects is about 48 GB, the whole dataset 126 GB.

    .. versionadded:: 2.1

    Parameters
    ----------
    root : pathlib.Path | str | None
        The root of the BIDS path. If None, download the data of ``subjects``
        from OpenNeuro.
    subjects : str | array-like of str | None
        The subject ID, e.g. ``"001"``. Corresponds to "sub". If None, all
        subjects.
    **kwargs
        Other parameters of :class:`braindecode.datasets.BIDSDataset`, e.g.
        ``runs``, ``preload`` or ``n_jobs``. ``datatypes`` defaults to
        ``"eeg"``.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, root=None, subjects=None, **kwargs):
        if root is None:
            root = _fetch_seizeit2(subjects)
        kwargs.setdefault("datatypes", "eeg")
        super().__init__(root, subjects=subjects, **kwargs)
