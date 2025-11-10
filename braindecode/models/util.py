# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

import braindecode.models as models

models_dict = {}

# For the models inside the init model, go through all the models
# check those have the EEGMixin class inherited. If they are, add them to the
# list.


def _init_models_dict():
    for m in inspect.getmembers(models, inspect.isclass):
        if (
            issubclass(m[1], models.base.EEGModuleMixin)
            and m[1] != models.base.EEGModuleMixin
        ):
            if m[1].__name__ == "EEGNetv4":
                continue
            models_dict[m[0]] = m[1]


################################################################
# Test cases for models
#
# This list should be updated whenever a new model is added to
# braindecode (otherwise `test_completeness__models_test_cases`
# will fail).
# Each element in the list should be a tuple with structure
# (model_class, required_params, signal_params), such that:
#
# model_name: str
#   The name of the class of the model to be tested.
# required_params: list[str]
#   The signal-related parameters that are needed to initialize
#   the model.
# signal_params: dict | None
#   The characteristics of the signal that should be passed to
#   the model tested in case the default_signal_params are not
#   compatible with this model.
#   The keys of this dictionary can only be among those of
#   default_signal_params.
################################################################
models_mandatory_parameters = [
    ("ATCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("BDTCN", ["n_chans", "n_outputs"], None),
    ("Deep4Net", ["n_chans", "n_outputs", "n_times"], None),
    ("DeepSleepNet", ["n_outputs"], None),
    ("EEGConformer", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGInceptionERP", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGInceptionMI", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGITNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ShallowFBCSPNet", ["n_chans", "n_outputs", "n_times"], None),
    (
        "SleepStagerBlanco2020",
        ["n_chans", "n_outputs", "n_times"],
        dict(n_chans=4),  # n_chans dividable by n_groups=2
    ),
    ("SleepStagerChambon2018", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    (
        "AttnSleep",
        ["n_outputs", "n_times", "sfreq"],
        dict(sfreq=100.0, n_times=3000, chs_info=[dict(ch_name="C1", kind="eeg")]),
    ),  # 1 channel
    ("TIDNet", ["n_chans", "n_outputs", "n_times"], None),
    ("USleep", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=128.0)),
    ("BIOT", ["n_chans", "n_outputs", "sfreq", "n_times"], None),
    ("AttentionBaseNet", ["n_chans", "n_outputs", "n_times"], None),
    ("Labram", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGSimpleConv", ["n_chans", "n_outputs", "sfreq"], None),
    ("SPARCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ContraWR", ["n_chans", "n_outputs", "sfreq", "n_times"], dict(sfreq=200.0)),
    ("EEGNeX", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGSym", ["chs_info", "n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("TSception", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200.0)),
    ("EEGTCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SyncNet", ["n_chans", "n_outputs", "n_times"], None),
    ("MSVTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGMiner", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200.0)),
    ("CTNet", ["n_chans", "n_outputs", "n_times"], None),
    ("SincShallowNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=250.0)),
    ("SCCNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200.0)),
    ("SignalJEPA", ["chs_info"], None),
    ("SignalJEPA_Contextual", ["chs_info", "n_times", "n_outputs"], None),
    ("SignalJEPA_PostLocal", ["n_chans", "n_times", "n_outputs"], None),
    ("SignalJEPA_PreLocal", ["n_chans", "n_times", "n_outputs"], None),
    ("FBCNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200.0)),
    ("FBMSNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200.0)),
    ("FBLightConvNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200.0)),
    ("IFNet", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=200.0)),
    ("PBT", ["n_chans", "n_outputs", "n_times"], None),
    ("SSTDPN", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("BENDR", ["n_chans", "n_outputs", "n_times"], None),
    ("LUNA", ["n_chans", "n_times", "n_outputs"], None),
    ("MEDFormer", ["n_chans", "n_outputs", "n_times"], None),
]

################################################################
# List of models that are not meant for classification
#
# Their output shape may difer from the expected output shape
# for classification models.
################################################################
non_classification_models = [
    "SignalJEPA",
]


################################################################
def get_summary_table(dir_name=None):
    if dir_name is None:
        dir_path = Path(__file__).parent
    else:
        dir_path = Path(dir_name) if not isinstance(dir_name, Path) else dir_name

    path = dir_path / "summary.csv"

    df = pd.read_csv(
        path,
        header=0,
        index_col="Model",
        skipinitialspace=True,
    )
    return df


def extract_channel_locations_from_chs_info(
    chs_info: Optional[Sequence[Dict[str, Any]]],
    num_channels: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Extract 3D channel locations from MNE-style channel information.

    This function provides a unified approach to extract 3D channel locations
    from MNE channel information. It's compatible with models like SignalJEPA
    and LUNA that need to work with channel spatial information.

    Parameters
    ----------
    chs_info : list of dict or None
        Channel information, typically from ``mne.Info.chs``. Each dict should
        contain a 'loc' key with a 12-element array (MNE format) where indices 3:6
        represent the 3D cartesian coordinates.
    num_channels : int or None
        If specified, only extract the first ``num_channels`` channel locations.
        If None, extract all available channels.

    Returns
    -------
    channel_locations : np.ndarray of shape (n_channels, 3) or None
        Array of 3D channel locations in cartesian coordinates. Returns None if
        no valid locations are found.

    Notes
    -----
    - This function handles both 12-element MNE location format (using indices 3:6)
      and 3-element location format (using directly).
    - Invalid or missing locations cause extraction to stop at that point.
    - Returns None if no valid locations can be extracted.
    - This is a unified utility compatible with models like SignalJEPA and LUNA.

    Examples
    --------
    >>> import mne
    >>> from braindecode.models.util import extract_channel_locations_from_chs_info
    >>> raw = mne.io.read_raw_edf("sample.edf")
    >>> locs = extract_channel_locations_from_chs_info(raw.info['chs'], num_channels=22)
    >>> print(locs.shape)
    (22, 3)
    """
    if chs_info is None:
        return None

    locations = []
    n_to_extract = num_channels if num_channels is not None else len(chs_info)

    for i, ch_info in enumerate(chs_info[:n_to_extract]):
        if not isinstance(ch_info, dict):
            break

        loc = ch_info.get("loc")
        if loc is None:
            break

        try:
            loc_array = np.asarray(loc, dtype=np.float32)

            # MNE format: 12-element array with coordinates at indices 3:6
            if loc_array.ndim == 1 and loc_array.size >= 6:
                if loc_array.size == 12:
                    # Standard MNE format
                    coordinates = loc_array[3:6]
                else:
                    # Assume first 3 elements are coordinates
                    coordinates = loc_array[:3]
            else:
                break

            locations.append(coordinates)
        except (ValueError, TypeError):
            break

    if len(locations) == 0:
        return None

    return np.stack(locations, axis=0)


_summary_table = get_summary_table()
