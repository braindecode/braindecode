# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import logging
from typing import Any, Literal

import mne
import numpy as np
from skorch.helper import SliceDataset
from skorch.utils import is_dataset

from braindecode.datasets.base import BaseConcatDataset, WindowsDataset
from braindecode.models.util import SigArgName

log = logging.getLogger(__name__)


def ms_to_samples(ms, fs):
    """
    Compute milliseconds to number of samples.

    Parameters
    ----------
    ms: number
        Milliseconds
    fs: number
        Sampling rate

    Returns
    -------
    n_samples: int
        Number of samples

    """
    return ms * fs / 1000.0


def samples_to_ms(n_samples, fs):
    """
    Compute milliseconds to number of samples.

    Parameters
    ----------
    n_samples: number
        Number of samples
    fs: number
        Sampling rate

    Returns
    -------
    milliseconds: int
    """
    return n_samples * 1000.0 / fs


def _get_n_outputs(y, classes, mode):
    if mode == "classification":
        classes_y = np.unique(y)
        if classes is not None:
            assert set(classes_y) <= set(classes)
        else:
            classes = classes_y
        return len(classes)
    elif mode == "regression":
        if y is None:
            return None
        if y.ndim == 1:
            return 1
        else:
            return y.shape[-1]
    else:
        raise ValueError(f"Unknown mode {mode}")


def infer_signal_properties(
    X,
    y=None,
    mode: Literal["classification", "regression"] = "classification",
    classes: list | None = None,
) -> dict[SigArgName, Any]:
    """Infers signal properties from the data.

    The extracted signal properties are:

    + n_chans: number of channels
    + n_times: number of time points
    + n_outputs: number of outputs
    + chs_info: channel information
    + sfreq: sampling frequency

    The returned dictionary can serve as kwargs for model initialization.

    Depending on the type of input passed, not all properties can be inferred.

    Parameters
    ----------
    X: array-like or mne.BaseEpochs or Dataset
        Input data
    y: array-like or None
        Targets
    mode: "classification" or "regression"
        Mode of the task
    classes: list or None
        List of classes for classification

    Returns
    -------
    signal_kwargs: dict
        Dictionary with signal-properties. Can serve as kwargs for model
        initialization.
    """
    signal_kwargs: dict[SigArgName, Any] = {}
    # Using shape to work both with torch.tensor and numpy.array:
    if (
        isinstance(X, mne.BaseEpochs)
        or (hasattr(X, "shape") and len(X.shape) >= 2)
        or isinstance(X, SliceDataset)
    ):
        if y is None:
            raise ValueError("y must be specified if X is array-like.")
        signal_kwargs["n_outputs"] = _get_n_outputs(y, classes, mode)
        if isinstance(X, mne.BaseEpochs):
            log.info("Using mne.Epochs to find signal-related parameters.")
            signal_kwargs["n_times"] = len(X.times)
            signal_kwargs["sfreq"] = X.info["sfreq"]
            signal_kwargs["chs_info"] = X.info["chs"]
        elif isinstance(X, SliceDataset):
            log.info("Using SliceDataset to find signal-related parameters.")
            Xshape = X[0].shape
            signal_kwargs["n_times"] = Xshape[-1]
            signal_kwargs["n_chans"] = Xshape[-2]
        else:
            log.info("Using array-like to find signal-related parameters.")
            signal_kwargs["n_times"] = X.shape[-1]
            signal_kwargs["n_chans"] = X.shape[-2]
    elif is_dataset(X):
        log.info(f"Using Dataset {X!r} to find signal-related parameters.")
        X0 = X[0][0]
        Xshape = X0.shape
        signal_kwargs["n_times"] = Xshape[-1]
        signal_kwargs["n_chans"] = Xshape[-2]
        if isinstance(X, BaseConcatDataset) and all(
            ds.targets_from == "metadata" for ds in X.datasets
        ):
            y_target = X.get_metadata().target
            signal_kwargs["n_outputs"] = _get_n_outputs(y_target, classes, mode)
        elif isinstance(X, WindowsDataset) and X.targets_from == "metadata":
            y_target = X.windows.metadata.target
            signal_kwargs["n_outputs"] = _get_n_outputs(y_target, classes, mode)
    else:
        log.warning(
            f"Can only infer signal shape of array-like and Datasets, got {type(X)!r}."
        )
    return signal_kwargs
