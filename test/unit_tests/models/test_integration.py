# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#          Alexandre Gramfort
#          Pierre Guetschel
#
# License: BSD-3

from copy import deepcopy

import mne
import torch
import pytest

from skorch.dataset import ValidSplit

from braindecode.models.util import models_dict
from braindecode import EEGClassifier
from braindecode.datasets import BaseDataset, BaseConcatDataset
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
# Generating the channel info
chs_info = [dict(ch_name=f"C{i}", kind="eeg") for i in range(1, 4)]
# Generating the signal parameters
default_signal_params = dict(
    n_times=1000,
    sfreq=250,
    n_outputs=2,
    chs_info=chs_info,
)


@pytest.fixture(scope="module")
def concat_ds_targets():
    raws, description = fetch_data_with_moabb(
        dataset_name="FakeDataset", subject_ids=1, dataset_kwargs=bnci_kwargs
    )

    events, _ = mne.events_from_annotations(raws[0])
    targets = events[:, -1] - 1
    ds = [BaseDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)
    return concat_ds, targets


@pytest.fixture(scope="module")
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


@pytest.mark.parametrize("model_name", models_dict.keys())
def test_model_list(model_name, concat_windows_dataset):

    model_class = models_dict[model_name]

    LEARNING_RATE = 0.0625 * 0.01
    BATCH_SIZE = 2
    EPOCH = 1
    seed = 2409
    valid_split = 0.2

    clf = EEGClassifier(
        module=model_class,
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=EPOCH,
        classes=[0, 1],
        train_split=ValidSplit(valid_split, random_state=seed),
        verbose=0,
    )

    clf.fit(X=concat_windows_dataset)


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
models_test_cases = [
    ("ATCNet", ["n_chans", "n_outputs", "n_times"], None),
    ("Deep4Net", ["n_chans", "n_outputs", "n_times"], None),
    ("DeepSleepNet", ["n_outputs"], None),
    ("EEGConformer", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGInception", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGInceptionERP", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGInceptionMI", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    ("EEGITNet", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNetv1", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGNetv4", ["n_chans", "n_outputs", "n_times"], None),
    ("EEGResNet", ["n_chans", "n_outputs", "n_times"], None),
    ("HybridNet", ["n_chans", "n_outputs", "n_times"], None),
    ("ShallowFBCSPNet", ["n_chans", "n_outputs", "n_times"], None),
    (
        "SleepStagerBlanco2020",
        ["n_chans", "n_outputs", "n_times"],
        # n_chans dividable by n_groups=2:
        dict(chs_info=[dict(ch_name=f"C{i}", kind="eeg") for i in range(1, 5)]),
    ),
    ("SleepStagerChambon2018", ["n_chans", "n_outputs", "n_times", "sfreq"], None),
    (
        "SleepStagerEldele2021",
        ["n_outputs", "n_times", "sfreq"],
        dict(sfreq=100, n_times=3000, chs_info=[dict(ch_name="C1}", kind="eeg")]),
    ),  # 1 channel
    ("TCN", ["n_chans", "n_outputs"], None),
    ("TIDNet", ["n_chans", "n_outputs", "n_times"], None),
    ("USleep", ["n_chans", "n_outputs", "n_times", "sfreq"], dict(sfreq=128)),
]


def test_completeness__models_test_cases():
    models_tested = set(x[0] for x in models_test_cases)
    all_models = set(models_dict.keys())
    assert (
        all_models == models_tested
    ), f"Models missing from models_test_cases: {all_models - models_tested}"


@pytest.mark.parametrize(
    "model_name, required_params, signal_params", models_test_cases
)
def test_model_integration(model_name, required_params, signal_params):
    """
    Verifies that all models can be initialized with all their parameters at
    default, except eventually the signal-related parameters.
    """
    # Verify that the parameters are correct:
    model_class = models_dict[model_name]
    assert isinstance(required_params, list)
    assert set(required_params) <= {
        "n_times",
        "sfreq",
        "chs_info",
        "n_outputs",
        "input_window_seconds",
        "n_chans",
    }
    assert signal_params is None or isinstance(signal_params, dict)
    if signal_params is not None:
        assert set(signal_params.keys()) <= set(default_signal_params.keys())

    sp = deepcopy(default_signal_params)
    if signal_params is not None:
        sp.update(signal_params)
    sp["n_chans"] = len(sp["chs_info"])
    sp["input_window_seconds"] = sp["n_times"] / sp["sfreq"]

    # create input data
    batch_size = 5
    X = torch.randn(
        batch_size,
        len(sp["chs_info"]),
        sp["n_times"],
    )

    # List possible model kwargs:
    output_kwargs = []
    output_kwargs.append(dict(n_outputs=sp["n_outputs"]))
    if "n_outputs" not in required_params:
        output_kwargs.append(dict(n_outputs=None))
    channel_kwargs = []
    channel_kwargs.append(dict(chs_info=sp["chs_info"], n_chans=None))
    if "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=sp["n_chans"], chs_info=None))
    if "n_chans" not in required_params and "chs_info" not in required_params:
        channel_kwargs.append(dict(n_chans=None, chs_info=None))
    time_kwargs = []
    time_kwargs.append(
        dict(n_times=sp["n_times"], sfreq=sp["sfreq"], input_window_seconds=None)
    )
    time_kwargs.append(
        dict(
            n_times=None,
            sfreq=sp["sfreq"],
            input_window_seconds=sp["input_window_seconds"],
        )
    )
    time_kwargs.append(
        dict(
            n_times=sp["n_times"],
            sfreq=None,
            input_window_seconds=sp["input_window_seconds"],
        )
    )
    if "n_times" not in required_params and "sfreq" not in required_params:
        time_kwargs.append(
            dict(
                n_times=None,
                sfreq=None,
                input_window_seconds=sp["input_window_seconds"],
            )
        )
    if (
        "n_times" not in required_params
        and "input_window_seconds" not in required_params
    ):
        time_kwargs.append(
            dict(n_times=None, sfreq=sp["sfreq"], input_window_seconds=None)
        )
    if "sfreq" not in required_params and "input_window_seconds" not in required_params:
        time_kwargs.append(
            dict(n_times=sp["n_times"], sfreq=None, input_window_seconds=None)
        )
    if (
        "n_times" not in required_params
        and "sfreq" not in required_params
        and "input_window_seconds" not in required_params
    ):
        time_kwargs.append(dict(n_times=None, sfreq=None, input_window_seconds=None))
    model_kwargs_list = [
        dict(**o, **c, **t)
        for o in output_kwargs
        for c in channel_kwargs
        for t in time_kwargs
    ]

    for model_kwargs in model_kwargs_list:
        # test initialisation:
        model = model_class(**model_kwargs)
        # test forward pass:
        out = model(X)
        # test output shape
        assert out.shape[:2] == (batch_size, sp["n_outputs"])
        # We add a "[:2]" because some models return a 3D tensor.
