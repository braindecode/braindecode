""".. _temporal-generalization:

Temporal generalization with Braindecode
========================================

In this tutorial, we will show you how to use the Braindecode library to decode
EEG data over time. The problem of decoding EEG data over time is formulated as fitting 
a multivariate predictive model on each time point of the signal and then evaluating the performance
of the model at the same time point in new epoched data. Specifically, given a pair of features $X$ 
and targets $y$, where $X$ has more than 2 dimensions, we want to fit a model $f$ to $X$ and $y$ and
evaluate the performance of $f$ on a new pair of features $X'$ and targets $y'$. Typically, $X$ is 
in the shape of $n_{epochs} \times n_{channels} \times n_{time}$ and $y$ is in the shape of 
$n_{epochs} \times n_{classes}$. This tutorial is based on the MNE tutorial:
https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#temporal-decoding
For more information on the problem of temporal generalization, visit MNE [1]_.

"""

# Authors: Matthew Chen <matt.chen42601@gmail.com>
# License: BSD (3-clause)

###################################################################################################
# Loading and preprocessing the data
# ----------------------------------
## Loading the dataset
# We will use subject 1 from the EEGBCI dataset on MNE, which is also available from PhysioNet [2]_. We will specifically decode 
# EEG data related to imaging opening and closing of both firsts or both feet (associated with runs 6, 10, and 14).
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import pick_types
from mne.channels import make_standard_montage

fnames = eegbci.load_data(subjects=[1], runs=(6, 10, 14))
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in fnames])

## Preprocessing the dataset

eegbci.standardize(raw)
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.annotations.rename(dict(T1="hands", T2="feet")) # rename descriptions to be easier to interpret
raw.set_eeg_reference(projection=True)
raw.rename_channels(lambda x: x.strip(".")) # remove periods from channel names

# Original data was sampled at 160 Hz, but low pass filtered to 80 Hz
# We specifically bandpass filter to 4-38 Hz to remove high frequency noise
raw.load_data().filter(4.0, 38.0, fir_design="firwin", skip_by_annotation="edge")
tmin, tmax = -0.5, 2 # nmber of seconds before/after event to extract epochs from
event_ids = dict(hands=2, feet=3) # map event IDs to tasks

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

epochs = mne.Epochs(
    raw,
    event_id=["hands", "feet"],
    tmin=tmin,
    tmax=tmax,
    picks=picks,
    baseline=None,
    preload=True,
)
del raw

X = epochs.get_data(copy=False)
y = epochs.events[:, -1] - 2
y_encod = LabelEncoder().fit_transform(y)
print("X shape: ", X.shape, "y_encod shape: ", y_encod.shape)
"""

###################################################################################################
# References
# ----------
#
# .. [1] Jean-Rémi King, Laura Gwilliams, Chris Holdgraf, Jona Sassenhagen,
#        Alexandre Barachant, Denis Engemann, Eric Larson, and Alexandre Gramfort.
#        "Encoding and decoding neuronal dynamics: methodological
#        framework to uncover the algorithms of cognition." hal-01848442, 2018.
#        URL: https://hal.archives-ouvertes.fr/hal-01848442.
# .. [2] Ary L. Goldberger, Luis A. N. Amaral, Leon Glass, Jeffrey M. Hausdorff,
#        Plamen Ch. Ivanov, Roger G. Mark, Joseph E. Mietus, George B. Moody, 
#        Chung-Kang Peng, and H. Eugene Stanley. PhysioBank, PhysioToolkit,
#        and PhysioNet: Components of a new research resource for complex
#        physiologic signals. Circulation, 2000.
#        URL: https://www.ahajournals.org/doi/10.1161/01.CIR.101.23.e215.
#
