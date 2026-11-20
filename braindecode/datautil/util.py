# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import logging
from typing import Any, Literal

import mne
import numpy as np
from skorch.helper import SliceDataset
from skorch.utils import is_dataset

from braindecode.datasets.base import (
    BaseConcatDataset,
    EEGWindowsDataset,
    WindowsDataset,
)
from braindecode.models.util import SigArgName

log = logging.getLogger(__name__)


def ms_to_samples(ms, fs):
    """
    Compute milliseconds to number of samples.

    Parameters
    ----------
    ms : number
        Milliseconds
    fs : number
        Sampling rate

    Returns
    -------
    n_samples : int
        Number of samples
    """
    return ms * fs / 1000.0


def samples_to_ms(n_samples, fs):
    """
    Compute milliseconds to number of samples.

    Parameters
    ----------
    n_samples : number
        Number of samples
    fs : number
        Sampling rate

    Returns
    -------
    milliseconds : int
    """
    return n_samples * 1000.0 / fs


def _get_n_outputs(y, classes, mode, stored=None):
    """Number of outputs for the targets ``y`` the model trains on.

    Classification: 1-D labels count the distinct labels (or ``classes``), 2-D
    targets ``(n, k)`` their width ``k``, whatever the values. ``classes`` names
    the labels as ``stored``, before any ``target_transform`` (``y`` by default).
    """
    if mode == "classification":
        y = np.asarray(y)
        labels = y if stored is None else np.asarray(stored)
        if classes is not None and labels.ndim == 1:
            missing = sorted(set(np.unique(labels).tolist()) - set(classes))
            if missing:
                raise ValueError(
                    f"Labels {missing} are not in classes {list(classes)}. classes "
                    "names the labels as stored, before any target_transform."
                )
        if y.ndim > 1:  # (n, k, ...): class probabilities, one-hot or multi-label
            if classes is not None and len(classes) != y.shape[1]:
                raise ValueError(
                    f"The number of classes ({len(classes)}) must match the "
                    f"target width ({y.shape[1]})."
                )
            return y.shape[1]
        return len(np.unique(y)) if classes is None else len(classes)
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
    n_outputs: int | None = None,
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
    X : array-like or mne.BaseEpochs or Dataset
        Input data
    y : array-like or None
        Targets of array-like ``X``, ignored for a dataset, which holds its own
        (after its ``target_transform``). In classification, 1-D labels give one
        output per distinct label, and 2-D targets ``(n, k)`` (class probabilities,
        one-hot, multi-label) their width ``k``. Dense class indices
        ``(n, n_times)`` are 2-D too: pass ``n_outputs``.
    mode : "classification" or "regression"
        Mode of the task
    classes : list or None
        List of classes for classification, as stored, before any
        ``target_transform``. ``n_outputs`` is its length.
    n_outputs : int or None
        Number of outputs, if known: it is returned as is and the targets are
        not read.

    Returns
    -------
    signal_kwargs : dict
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
        signal_kwargs["n_outputs"] = (
            n_outputs if n_outputs is not None else _get_n_outputs(y, classes, mode)
        )
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
        log.info("Using %s to find signal-related parameters.", type(X).__name__)
        X0 = X[0][0]
        Xshape = X0.shape
        signal_kwargs["n_times"] = Xshape[-1]
        signal_kwargs["n_chans"] = Xshape[-2]
        datasets = X.datasets if isinstance(X, BaseConcatDataset) else [X]
        if n_outputs is not None:
            signal_kwargs["n_outputs"] = n_outputs
        elif all(
            isinstance(ds, (EEGWindowsDataset, WindowsDataset))
            and ds.targets_from == "metadata"
            for ds in datasets
        ):
            # One label per window, as stored (both window backends, lazy metadata
            # included) and as the model receives it, after any target_transform.
            stored = [ds.y[i] for ds in datasets for i in range(len(ds))]
            targets = stored
            if isinstance(X, BaseConcatDataset) and X.target_transform is not None:
                targets = [X.target_transform(t) for t in stored]
            signal_kwargs["n_outputs"] = _get_n_outputs(
                np.asarray(targets), classes, mode, stored=np.asarray(stored)
            )
    else:
        log.warning(
            f"Can only infer signal shape of array-like and Datasets, got {type(X)!r}."
        )
    return signal_kwargs
