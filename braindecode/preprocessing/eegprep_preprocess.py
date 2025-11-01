"""Preprocessors using the EEGPrep package."""

# Authors: Christian Kothe <christian.kothe@intheon.io>
#
# License: BSD-3

import logging
from abc import abstractmethod
from typing import Any, Sequence

import mne
import numpy as np
from mne.io import BaseRaw

from .preprocess import Preprocessor
from .util import mne_load_metadata, mne_store_metadata

log = logging.getLogger(__name__)

__all__ = [
    "EEGPrep",
    "RemoveDCOffset",
    "Resampling",
    "RemoveFlatChannels",
    "RemoveDrifts",
    "RemoveBadChannels",
    "RemoveBadChannelsNoLocs",
    "RemoveBursts",
    "RemoveBadWindows",
    "ReinterpolateRemovedChannels",
    "RemoveCommonAverageReference",
]

try:
    import eegprep
except ImportError:
    eegprep = None


class EEGPrepBasePreprocessor(Preprocessor):
    """Abstract base class for EEGPrep preprocessors, implementing shared functionality.

    Parameters
    ----------
    can_change_duration : bool | str
        Whether the preprocessor can change the duration of the data during processing;
        can also be the name of some sub-operation that does so for display in a more
        actionable error message.
    record_orig_chanlocs : bool
        Whether to record the EEG channel locations before processing
        in the MNE Raw structure for later use. This will not override any already
        present channel location information, so this can safely be used multiple times
        to record whatever were the original channel locations.
    force_dtype : np.dtype | str | None
        Optionally for the in/out data to be converted to this dtype before and after
        processing. Can help ensure consistent dtypes across different preprocessors.

    """

    # payload key under which we store our original channel locations
    _chanlocs_key = "original_chanlocs"

    def __init__(
        self,
        *,
        can_change_duration: str | bool = False,
        record_orig_chanlocs: bool = False,
        force_dtype: np.dtype | str | None = None,
    ):
        super().__init__(
            fn=self._apply_op,
            apply_on_array=False,
        )
        if can_change_duration is True:
            can_change_duration = self.__class__.__name__
        self.can_change_duration = can_change_duration
        self.record_orig_chanlocs = record_orig_chanlocs
        self.force_dtype = np.dtype(force_dtype) if force_dtype is not None else None

    @abstractmethod
    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure. Overridden by subclass."""
        ...

    def _apply_op(self, raw: BaseRaw) -> None:
        """Internal method that does the actual work; this is called by Preprocessor.apply()."""
        # handle error if eegprep is not available
        if eegprep is None:
            raise RuntimeError(
                "The eegprep package is required to use the EEGPrep preprocessor.\n"
                "  Please install braindecode with the [eegprep] extra added as in\n"
                "  'pip install braindecode[eegprep]' to use this functionality,\n"
                "  or run 'pip install eegprep[eeglabio]' directly."
            )

        opname = self.__class__.__name__
        if isinstance(raw, mne.BaseEpochs):
            raise ValueError(
                f"{opname} is meant to be used on Raw (continuous) data, "
                f"not Epochs. Use before epoching."
            )

        # preserve the data description for restoration later
        description = raw.info["description"]

        # split off non-EEG channels since apply_eeg() expects only EEG channels
        chn_order = raw.ch_names.copy()
        eeg_idx = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_idx) < len(chn_order):
            eeg = raw.copy().pick(eeg_idx)
            non_eeg = raw.drop_channels(eeg.ch_names)
        else:
            eeg = raw
            non_eeg = None

        eeg = eegprep.mne2eeg(eeg)

        # back up channel locations for potential later use
        orig_chanlocs = [cl.copy() for cl in eeg["chanlocs"]]

        # ensure all events in EEG structure have a 'duration' field; this is
        # necessary for some of the EEGPrep operations to succeed
        if not all("duration" in ev for ev in eeg["event"]):
            for ev in eeg["event"]:
                if "duration" not in ev:
                    ev["duration"] = 1

        if self.force_dtype is not None:
            eeg["data"] = eeg["data"].astype(self.force_dtype)

        # actual operation happens here
        eeg = self.apply_eeg(eeg, raw)

        if self.force_dtype is not None:
            eeg["data"] = eeg["data"].astype(self.force_dtype)

        # rename EEGLAB-type boundary events to a form that's recognized by MNE so they
        # (or intersecting epochs) are ignored during potential downstream epoching
        # done by braindecode pipelines
        for ev in eeg["event"]:
            if ev["type"] == "boundary":
                ev["type"] = "BAD boundary"

        # set some minimal chaninfo fields, which are otherwise not necessarily
        # guaranteed to be there
        if not isinstance(eeg["chaninfo"], dict) or not eeg["chaninfo"]:
            eeg["chaninfo"] = {"nosedir": "+X"}

        # actual conversion
        proc = eegprep.eeg2mne(eeg)

        # restore the description field, since it may not survive the roundtrip
        # conversion (and we don't have control over all the code involved there)
        proc.info["description"] = description

        if self.record_orig_chanlocs:
            # stash original channel locations if not already present
            mne_store_metadata(
                raw=proc,
                payload=orig_chanlocs,
                key=self._chanlocs_key,
                no_overwrite=True,
            )

        if non_eeg is not None:
            # try to add back any non-EEG channels that were originally present

            if offending_op_name := self.can_change_duration:
                # while this may work due to luck on one or another session, in
                # general we will not try to add them back in in this case; in future
                # we may try to work out exactly what sample mask was removed from the
                # EEG channels and apply the same to non-EEG channels via MNE
                affected_chans = ", ".join(non_eeg.ch_names)
                if offending_op_name == opname:
                    detail = f"the {opname} Preprocessor"
                else:
                    detail = f"{offending_op_name} in the {opname} Preprocessor"
                log.error(
                    f"Could not add back non-EEG channels ({affected_chans}) after"
                    f" {opname} processing; these will be omitted from the processed"
                    f" data. If you want to retain these channels, you will have to"
                    f" disable {detail}; you may perform that step using other"
                    f" methods before and after {opname}, respectively."
                )
            else:
                # re-insert non-EEG channels, and restore original channel order
                proc.add_channels([non_eeg], force_update_info=True)
                if proc.ch_names != chn_order:
                    proc.reorder_channels(chn_order)

        # write result back into raw, discard proc (_apply_op() is in-place)
        if not proc.preload:
            proc.load_data()
        raw.__dict__ = proc.__dict__

    @classmethod
    def _get_orig_chanlocs(cls, raw: BaseRaw) -> list[dict[str, Any]] | None:
        """Retrieve original channel locations stashed in the given MNE Raw
        structure, if any."""
        return mne_load_metadata(raw, key=cls._chanlocs_key)


class EEGPrep(EEGPrepBasePreprocessor):
    """Preprocessor for an MNE Raw object that applies the EEGPrep pipeline.
    This is based on [Mullen2015]_.

    .. figure:: https://cdn.ncbi.nlm.nih.gov/pmc/blobs/a79a/4710679/675fc2dee929/nihms733482f9.jpg
       :align: center
       :alt: Before/after comparison of EEGPrep processing on EEG data.

    This pipeline involves the stages:

    - DC offset subtraction (:class:`RemoveDCOffset`)
    - Optional resampling (:class:`Resampling`)
    - Flatline channel detection and removal (:class:`RemoveFlatChannels`)
    - High-pass filtering (:class:`RemoveDrifts`)
    - Bad channel detection and removal using correlation and HF noise criteria
      (:class:`RemoveBadChannels` with fallback to :class:`RemoveBadChannelsNoLocs`)
    - Burst artifact removal using ASR (Artifact Subspace Reconstruction)
      (:class:`RemoveBursts`)
    - Detection and removal of residual bad time windows (:class:`RemoveBadWindows`)
    - Optional reinterpolation of removed channels
      (:class:`ReinterpolateRemovedChannels`)
    - Optional common average referencing (:class:`RemoveCommonAverageReference`)

    These steps are also individually available as separate preprocessors in this module
    if you want to apply only a subset of them or customize some beyond the parameters
    available here. Note that it is important to apply them in the order given above;
    other orderings may lead to suboptimal results.

    Typically no signal processing (except potentially resampling or removal of unused
    channels or time windows) should be done before this pipeline. It is recommended to
    follow this with at least a low-pass filter to remove high-frequency artifacts
    (e.g., 40-45 Hz transition band).

    The main processing parameters can each be set to None to skip the respective
    stage (or False for boolean switches). Note this pipeline will only affect the
    EEG channels in your data, and will leave other channels unaffected. It is
    recommended to remove these channels yourself beforehand if you don't want them
    included in your downstream analysis.

    .. Note::
        This implementation of the pipeline is best used in the context of
        cross-session prediction; when using this with a within-session split, there is
        a risk of data leakage since the artifact removal will be calibrated on statistics
        of the entire session (and thus test sets). In practice the effect may be minor,
        unless your downstream analysis is strongly driven by artifacts (e.g., if you
        are trying to decode eye movements or muscle activity), but paper reviewers may
        not be convinced by that.


    Parameters
    ----------
    resample_to : float | None = None
        Optionally resample to this sampling rate (in Hz) before processing.
        Good choices are 200, 250, 256 Hz (consider keeping it a power of two
        if it was originally), but one may go as low as 100-128 Hz if memory, compute,
        or model complexity limitations demand it.
    flatline_maxdur : float | None
        Remove channels that are flat for longer than this duration (in seconds).
        This stage is almost never triggered in practice but can help with the
        occasional strange EEG configuration.
    highpass_frequencies : tuple[float, float] | None
        Tuple of lower and upper bound of the *transition band* for high-pass filtering
        before processing. This means that full suppression will be reached at the
        lower bound, and the upper bound is where the passband begins.
    bad_channel_corr_threshold : float | None
        Threshold for correlation-based bad channel detection. A good default range
        is 0.75-0.8. Becomes quite aggressive at and beyond 0.8; also, consider using
        lower values (eg 0.7-0.75) for <32ch EEG and higher (0.8-0.85) for >128ch.
    burst_removal_cutoff : float | None
        Amplitude threshold for burst artifact removal using ASR
        (Artifact Subspace Reconstruction). This parameter tends to have a large effect
        on the performance of downstream ML. 10-15 is a good range for ML pipelines
        (lower is more aggressive); for neuroscience analysis, more conservative values
        like 20-30 may be better. The unit is z-scores relative to a Gaussian component
        of background EEG, but real EEG can be super-Gaussian, thus the large values.
    bad_window_max_bad_channels : float | None
        Threshold for rejection of bad time windows based on fraction of simultaneously
        noisy channels. Lower is more aggressive. Typical values are 0.15 (quite
        aggressive) to 0.3 (quite lax).
    bad_channel_reinterpolate : bool
        Whether to reinterpolate bad channels that were detected and removed. Usually
        required when doing cross-session analysis (to have a consistent channel set).
    common_avg_ref : bool
        Whether to apply a common average reference after processing. Recommended
        when doing cross-study analysis to have a consistent referencing scheme.
    bad_channel_hf_threshold : float
        Threshold for high-frequency (>=45 Hz) noise-based bad channel detection,
        in z-scores. Lower is more aggressive. Default is 4.0. This is rarely tuned,
        but data with unusual higher-frequency activity could benefit from exploration
        in the 3.5-5.0 range.
    bad_channel_max_broken_time : float
        Max fraction of session length during which a channel may be bad before
        it is removed. Default is 0.4 (40%), max is 0.5 (breakdown point of stats).
        Pretty much never tuned.
    bad_window_tolerances : tuple[float, float]
        (min, max) z-score tolerance for identifying bad time window/channel pairs.
        This typically does not need to be changed (instead one may change the max
        bad channels that cross this threshold), but different implementations
        use different values here. The max value is the main parameter, where
        EEGLAB/EEGPrep uses 7 while the original pipeline [1] used 5.5, and NeuroPype
        uses 6. Lower values are more aggressive. The min value is only triggered if the
        EEG data has signal dropouts (very low amplitude, e.g. due to something becoming
        unplugged) which is rare; some choices are (-inf, EEGPrep; -3.5, BCILAB;
        -4 NeuroPype).
    refdata_max_bad_channels : float | None
        Same function as bad_window_max_bad_channels, but used only to determine
        calibration data for burst removal. Usually more aggressive than the former
        (0.05-0.1) to get clean calibration data. This can be set to None to skip this
        and force all data to be used for calibration.
    refdata_max_tolerances : tuple[float, float]
        Same as bad_window_tolerances, but used only to determine calibration data for
        burst removal. Almost never touched, and defaults to a fairly aggressive
        (-inf, 5.5) to get clean calibration data.
    num_samples : int
        Number of channel subsets to draw for the RANSAC reconstruction during bad
        channel identification. Higher can be more robust but slower to calibrate.
        Default is 50.
    subset_size : float
        Size of channel subsets for RANSAC, as fraction (0-1) or count. Default 0.25.
        For higher-density EEG (e.g., 64-128ch), one can achieve somewhat better
        robustness to clusters of bad channels by setting this to 0.15 and increasing
        num_samples to 200.
    bad_channel_nolocs_threshold : float
        A fallback correlation threshold for bad-channel removal that is applied when
        no channel location information is available. The value here typically needs to
        be fairly low, e.g., 0.45-0.5 (lower is more aggressive). Ideally you have
        channel locations so that this fallback is not needed.
    bad_channel_nolocs_exclude_frac : float
        A fraction of most correlated channels to exclude in the case where no channel
        location information is available. Used to reject pairs of shorted or otherwise
        highly correlated sets of bad channels.
    max_mem_mb : int
        Max memory that ASR can use, in MB. Larger values can reduce overhead during
        processing, but usually 64MB is sufficient.

    References
    ----------
    .. [Mullen2015] Mullen, T.R., Kothe, C.A., Chi, Y.M., Ojeda, A., Kerth, T.,
       Makeig, S., Jung, T.P. and Cauwenberghs, G., 2015. Real-time neuroimaging and
       cognitive monitoring using wearable dry EEG. IEEE Transactions on Biomedical
       Engineering, 62(11), pp.2553-2567.

    """

    def __init__(
        self,
        *,
        # (main processing parameters)
        resample_to: float | None = None,
        flatline_maxdur: float | None = 5.0,
        highpass_frequencies: tuple[float, float] | None = (0.25, 0.75),
        bad_channel_corr_threshold: float | None = 0.8,
        burst_removal_cutoff: float | None = 10.0,
        bad_window_max_bad_channels: float | None = 0.25,
        bad_channel_reinterpolate: bool = True,
        common_avg_ref: bool = True,
        # additional tuning parameters
        bad_channel_max_broken_time: float = 0.4,
        bad_channel_hf_threshold: float | None = 4.0,
        bad_window_tolerances: tuple[float, float] | None = (-np.inf, 7),
        refdata_max_bad_channels: float | None = 0.075,
        refdata_max_tolerances: tuple[float, float] | None = (-np.inf, 5.5),
        num_samples: int = 50,
        subset_size: float = 0.25,
        bad_channel_nolocs_threshold: float = 0.45,
        bad_channel_nolocs_exclude_frac: float = 0.1,
        max_mem_mb: int = 64,
    ):
        can_change_duration = " and ".join(
            opname
            for opname in (
                "resample" if resample_to else "",
                "bad time window removal" if bad_window_max_bad_channels else "",
            )
            if opname
        )
        super().__init__(
            can_change_duration=can_change_duration or False,
        )
        self.resample_to = resample_to
        self.reinterpolate = bad_channel_reinterpolate
        self.common_avg_ref = common_avg_ref
        self.burst_removal_cutoff = burst_removal_cutoff
        self.bad_window_max_bad_channels = bad_window_max_bad_channels

        if bad_channel_corr_threshold is None:
            line_noise_crit = None
        else:
            line_noise_crit = bad_channel_hf_threshold
        self.clean_artifacts_params = dict(
            ChannelCriterion=bad_channel_corr_threshold,
            LineNoiseCriterion=line_noise_crit,
            BurstCriterion=burst_removal_cutoff,
            WindowCriterion=bad_window_max_bad_channels,
            Highpass=highpass_frequencies,
            ChannelCriterionMaxBadTime=bad_channel_max_broken_time,
            BurstCriterionRefMaxBadChns=refdata_max_bad_channels,
            BurstCriterionRefTolerances=refdata_max_tolerances,
            WindowCriterionTolerances=bad_window_tolerances,
            FlatlineCriterion=flatline_maxdur,
            NumSamples=num_samples,
            SubsetSize=subset_size,
            NoLocsChannelCriterion=bad_channel_nolocs_threshold,
            NoLocsChannelCriterionExcluded=bad_channel_nolocs_exclude_frac,
            MaxMem=max_mem_mb,
            # For reference, the function additionally accepts these (legacy etc.)
            # arguments, which we're not exposing here (current defaults as below):
            # BurstRejection='off',
            # Distance='euclidian',
            # Channels=None,
            # Channels_ignore=None,
            # availableRAM_GB=None,
        )

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        # remove per-channel DC offset (can be huge)
        eeg["data"] -= np.median(eeg["data"], axis=1, keepdims=True)

        # optional resampling
        if (srate := self.resample_to) is not None:
            eeg = eegprep.resample(eeg, srate)

        # do a check if the data has a supported sampling rate
        if self.burst_removal_cutoff is not None:
            supported_rates = (100, 128, 200, 250, 256, 300, 500, 512)
            cur_srate = int(eegprep.utils.round_mat(eeg["srate"]))
            if cur_srate not in supported_rates:
                # note: technically the method will run if you disable this error,
                #   but you're likely getting (potentially quite) suboptimal results
                raise NotImplementedError(
                    f"The dataset has an uncommon sampling rate of {cur_srate} Hz,"
                    f" which is not supported by the EEGPrep Preprocessor"
                    f" implementation. Please enable resampling to"
                    f" resample the data to one of the supported rates"
                    f" ({', '.join(str(r) for r in supported_rates)})."
                )

        # preserve input channel locations for reinterpolation later
        orig_chanlocs = [channel_loc.copy() for channel_loc in eeg["chanlocs"]]

        # artifact removal stage
        eeg, *_ = eegprep.clean_artifacts(eeg, **self.clean_artifacts_params)

        if self.force_dtype != np.float64:
            # cast to float32 for equivalence with multi-stage EEGPrep pipeline
            eeg["data"] = eeg["data"].astype(np.float32)

        # optionally reinterpolate dropped channels
        if self.reinterpolate and (len(orig_chanlocs) > len(eeg["chanlocs"])):
            eeg = eegprep.eeg_interp(eeg, orig_chanlocs)

        # optionally apply common average reference
        if self.common_avg_ref:
            eeg = eegprep.reref(eeg, [])

        return eeg


class RemoveFlatChannels(EEGPrepBasePreprocessor):
    """Removes EEG channels that flat-line for extended periods of time.
    Follows [Mullen2015]_.

    This is an automated artifact rejection function which ensures that
    the data contains no flat-lined channels. This is very rarely the case, but the
    preprocessor exists since the presence of such channels may throw off downstream
    preproc steps.

    This step is best placed very early in a preprocessing pipeline, before any
    filtering (since filter pre/post ringing can mask flatlines).

    A channel :math:`c` is flagged as flat if there exists a time interval
    :math:`[t_1, t_2]` where:

    .. math::

        |X_{c,t+1} - X_{c,t}| < \\varepsilon_{\\text{jitter}} \\quad \\forall t \\in [t_1, t_2]

        \\text{and} \\quad t_2 - t_1 > T_{\\text{max}}

    where :math:`\\varepsilon_{\\text{jitter}} = \\text{max_allowed_jitter} \\times \\varepsilon`
    (with :math:`\\varepsilon` being machine epsilon for float64), and
    :math:`T_{\\text{max}} = \\text{max_flatline_duration} \\times f_s` (with :math:`f_s`
    being the sampling rate).

    Parameters
    ----------
    max_flatline_duration : float
        Maximum tolerated flatline duration. In seconds. If a channel has a longer
        flatline than this, it will be considered abnormal. Defaults to 5.0.
    max_allowed_jitter : float
        Maximum tolerated jitter during flatlines. As a multiple of epsilon for the
        64-bit float data type (np.finfo(np.float64).eps). Defaults to 20.

    References
    ----------
    .. [Mullen2015] Mullen, T.R., Kothe, C.A., Chi, Y.M., Ojeda, A., Kerth, T.,
       Makeig, S., Jung, T.P. and Cauwenberghs, G., 2015. Real-time neuroimaging and
       cognitive monitoring using wearable dry EEG. IEEE Transactions on Biomedical
       Engineering, 62(11), pp.2553-2567.

    """

    def __init__(
        self,
        *,
        max_flatline_duration: float = 5.0,
        max_allowed_jitter: float = 20.0,
    ):
        super().__init__(record_orig_chanlocs=True)
        self.max_flatline_duration = max_flatline_duration
        self.max_allowed_jitter = max_allowed_jitter

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        eeg = eegprep.clean_flatlines(
            eeg,
            max_flatline_duration=self.max_flatline_duration,
            max_allowed_jitter=self.max_allowed_jitter,
        )

        return eeg


class RemoveDCOffset(EEGPrepBasePreprocessor):
    """Remove the DC offset from the EEG data by subtracting the per-channel median.

    This preprocessor mainly exists because some EEG data (depending on the electrical
    characteristics of the hardware) can have such a large DC offset that highpass
    filters do not necessarily fully remove it, unless some care is taken with filter
    settings (noted in EEGLAB documentation [Delorme2004]_).

    The operation performed is:

    .. math::

        X'_{c,t} = X_{c,t} - \\text{median}_t(X_{c,t})

    where :math:`c` indexes the channel and :math:`t` indexes time.

    References
    ----------
    .. [Delorme2004] Delorme, A. and Makeig, S., 2004. EEGLAB: an open source toolbox
       for analysis of single-trial EEG dynamics including independent component
       analysis. Journal of Neuroscience Methods, 134(1), pp.9-21.

    """

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        # note this might as well be implemented directly on the MNE data structure,
        # but was kept this way since we have the EEGPrep machinery here already.
        eeg["data"] -= np.median(eeg["data"], axis=1, keepdims=True)
        return eeg


class RemoveDrifts(EEGPrepBasePreprocessor):
    """Remove drifts from the EEG data using a forward-backward high-pass filter.
    See [Oppenheim1999]_.

    .. figure:: ../../docs/_static/preprocess/highpass.png
       :align: center
       :alt: Magnitude response for this filter with default settings.

    Note that MNE has its own suite of filters for this that offers more choices; use
    this filter if you are specifically interested in matching the EEGLAB and EEGPrep
    behavior, for example if you're building an EEGPrep-like pipeline from individual
    steps, e.g., to customize parts that are not exposed by the top-level EEGPrep
    preprocessor.

    .. Note::
       If your method involves causal analysis, either with applications to real-time
       single-trial brain-computer interfacing or for example involving autoregressive
       modeling or other causal measures, consider using a strictly causal highpass
       filter instead.

    Parameters
    ----------
    transition : Sequence[float]
        The transition band in Hz, i.e. lower and upper edge of the transition as in
        (lo, hi). Defaults to (0.25, 0.75). Choosing this can be tricky when your data
        contains long-duration event-related potentials that your method exploits, in
        which case you may need to carefully lower this somewhat to avoid attenuating
        them.
    attenuation : float
        The stop-band attenuation, in dB. Defaults to 80.0.
    method : str
        The method to use for filtering ('fft' or 'fir'). Defaults to 'fft' (uses more
        memory but is much faster than 'fir').

    References
    ----------
    .. [Oppenheim1999] Oppenheim, A.V., 1999. Discrete-time signal processing.
       Pearson Education India.

    """

    def __init__(
        self,
        transition: Sequence[float] = (0.25, 0.75),
        *,
        attenuation: float = 80.0,
        method: str = "fft",
    ):
        super().__init__()
        self.transition = transition
        self.attenuation = attenuation
        self.method = method

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        eeg = eegprep.clean_drifts(
            eeg,
            transition=self.transition,
            attenuation=self.attenuation,
            method=self.method,
        )

        return eeg


class Resampling(EEGPrepBasePreprocessor):
    """Resample the data to a specified rate (EEGPrep version).
    Based on [Proakis2007]_ and included for equivalence with EEGPrep.

    .. figure:: ../../docs/_static/preprocess/downsample.png
       :align: center
       :alt: Example of resampling a time series.

    MNE has its resampling routine (use as `Preprocessor("resample", sfreq=rate)`)
    but this will not necessarily match EEGPrep's behavior exactly. Typical
    differences include edge padding, the exact design rule for the filter kernel
    and its window function, and handling of resampling ratios with large rational
    factors.

    It's not necessarily clear which of the two implementations is "better" (likely
    both are fine for typical EEG applications). Use this one if you try to match
    EEGPrep and EEGLAB behavior specifically, for example when you migrate from a
    simple pipeline that uses the high-level EEGPrep preprocessor to a more
    custom pipeline built from individual steps and want to ensure identical
    results (up to float precision issues).

    Resampling can be placed quite early in a preprocessing pipeline to cut down on
    compute time and memory usage of downstram steps, e.g., before filtering, but
    note the sampling rate interacts with e.g. temporal convolution kernel sizes;
    when reproducing literature, ideally you first resample to the same rate as
    used there.

    .. Note::
        There can be a small timing accuracy penalty when resampling on continuous data
        (before epoching) when doing event-locked analysis, since epoch windows will be
        snapped to the nearest sample. However, this jitter is typically fairly minor
        relative to timing variability in the brain responses themselves, so will often
        not be a problem in practice.

    Parameters
    ----------
    sfreq : float | None
        The desired sampling rate in Hz. Skipped if set to None.


    References
    ----------
    .. [Proakis2007] Proakis, J.G., 2007. Digital signal processing: principles,
       algorithms, and applications, 4/E. Pearson Education India.

    """

    def __init__(
        self,
        sfreq: float | None,
    ):
        super().__init__(can_change_duration=True)
        self.sfreq = sfreq

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        if self.sfreq is not None:
            eeg = eegprep.resample(eeg, self.sfreq)

        return eeg


class RemoveBadChannels(EEGPrepBasePreprocessor):
    """Removes EEG channels with problematic data; variant that uses channel locations.
    Implemented as in [Kothe2013]_.

    .. figure:: https://www.mdpi.com/sensors/sensors-22-07314/article_deploy/html/images/sensors-22-07314-g003.png
       :align: center
       :alt: Conceptual image of bad-channel removal.

    This is an automated artifact rejection function which ensures that the data
    contains no channels that record only noise for extended periods of time. This uses
    a hybrid criterion involving correlation and high-frequency noise thresholds:

    a) if a channel has lower correlation to its robust estimate (based on other
       channels) than a given threshold for a minimum period of time (or percentage of
       the recording), it will be removed.
    b) if a channel has more (high-frequency) noise relative relative to the (robust)
       population of other channels than a given threshold (in standard deviations),
       it will be removed.

    This method requires channels to have an associated location; when a location
    is not known or could not be inferred (e.g., from channel labels if using a standard
    montage such as the 10-20 system), use the :class:`RemoveBadChannelsNoLocs`
    preprocessor instead.

    Preconditions:

    - One of :class:`RemoveDrifts` or :class:`braindecode.preprocessing.Filter` (
      configured as a highpass filter) must have been applied beforehand.
    - 3D channel locations must be available in the data (can be automatic with some
      file types, but may require some MNE operations with others).
    - Consider applying :class:`RemoveDCOffset` beforehand as a general precaution.

    Parameters
    ----------
    corr_threshold : float
        Correlation threshold. If a channel over a short time window is correlated at
        less than this value to its robust estimate (based on other channels), it is
        considered abnormal during that time. A good default range is 0.75-0.8 and the
        default is 0.8. Becomes quite aggressive at and beyond 0.8; also, consider
        using lower values (eg 0.7-0.75) for <32ch EEG and higher (0.8-0.85) for >128ch.
        This is the main tunable parameter of the method.
    noise_threshold : float
        Threshold for high-frequency (>=45 Hz) noise-based bad channel detection,
        in robust z-scores (i.e., st. devs.). Lower is more aggressive. Default is 4.0.
        This is rarely tuned, but data with unusual higher-frequency activity could
        benefit from exploration in the 3.5-5.0 range.
    window_len : float
        Length of the time windows (in seconds) for which correlation statistics
        are computed; ideally short enough to reasonably capture periods of global
        artifacts or intermittent sensor dropouts, but not shorter (for statistical
        reasons). Default is 5.0 sec.
    subset_size : float
        Size of random channel subsets to compute robust reconstructions. This can be
        given as a fraction (0-1) of the total number of channels, or as an absolute
        number. Multiple (pseudo-)random subsets are sampled in a RANSAC-like process
        to obtain a robust reference estimate for each channel. Default is 0.25 (25% of
        channels). For higher-density EEG (e.g., 64-128ch) with potential clusters
        of bad channels, one can achieve somewhat better robustness by setting this
        to 0.15 and increasing num_samples to 200.
    num_samples : int
        Number of samples generated for the robust channel reconstruction. This is the
        number of samples to generate in a RANSAC-like process. The larger
        this value, the more robust but also slower the initial identification of
        bad channels will be. Default is 50.
    max_broken_time : float
        Maximum time (either in seconds or as fraction of the recording) during which
        a channel is allowed to have artifacts. If a channel exceeds this, it will be
        removed. Not usually tuned. Default is 0.4 (40%), max is 0.5 (breakdown point
        of stats). Pretty much never tuned.

    References
    ----------
    .. [Kothe2013] Kothe, C.A. and Makeig, S., 2013. BCILAB: a platform for
       brain–computer interface development. Journal of Neural Engineering, 10(5),
       p.056014.

    """

    def __init__(
        self,
        *,
        corr_threshold: float = 0.8,
        noise_threshold: float = 4.0,
        window_len: float = 5,
        max_broken_time: float = 0.4,
        subset_size: float = 0.25,
        num_samples: int = 50,
    ):
        super().__init__(record_orig_chanlocs=True)
        self.corr_threshold = corr_threshold
        self.noise_threshold = noise_threshold
        self.window_len = window_len
        self.max_broken_time = max_broken_time
        self.num_samples = num_samples
        self.subset_size = subset_size

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        eeg = eegprep.clean_channels(
            eeg,
            corr_threshold=self.corr_threshold,
            noise_threshold=self.noise_threshold,
            window_len=self.window_len,
            max_broken_time=self.max_broken_time,
            num_samples=self.num_samples,
            subset_size=self.subset_size,
        )

        return eeg


class RemoveBadChannelsNoLocs(EEGPrepBasePreprocessor):
    """Remove EEG channels with problematic data; variant that does not use channel
    locations. Implemented as in [Kothe2013]_.

    .. figure:: https://www.mdpi.com/sensors/sensors-22-07314/article_deploy/html/images/sensors-22-07314-g003.png
       :align: center
       :alt: Conceptual image of bad-channel removal.

    This is an automated artifact rejection function which ensures that the data
    contains no channels that record only noise for extended periods of time.
    The criterion is based on correlation: if a channel is decorrelated from all others
    (pairwise correlation < a given threshold), excluding a given fraction of most
    correlated channels, and if this holds on for a sufficiently long fraction of the
    data set, then the channel is removed.

    This method does not require or take into account channel locations; if you do have
    locations, you may get better results with the RemoveBadChannels preprocessor
    instead.

    Preconditions:

    - One of :class:`RemoveDrifts` or :class:`braindecode.preprocessing.Filter` (
      configured as a highpass filter) must have been applied beforehand.
    - Consider applying :class:`RemoveDCOffset` beforehand as a general precaution.

    Parameters
    ----------
    min_corr : float
        Minimum correlation between a channel and any other channel (in a short
        period of time) below which the channel is considered abnormal for that time
        period. Reasonable range: 0.4 (very lax) to 0.6 (quite aggressive).
        Default is 0.45.
    ignored_quantile : float
        Fraction of channels that need to have at least the given min_corr value w.r.t.
        the channel under consideration. This allows to deal with channels or small
        groups of channels that measure the same noise source. Reasonable
        range: 0.05 (rather lax) to 0.2 (tolerates many disconnected/shorted channels).
    window_len : float
        Length of the windows (in seconds) over which correlation stats are computed.
        Reasonable values are 1.0 sec (more noisy estimates) to 5.0 sec (more reliable,
        but can miss brief artifacts). Default is 2.0 sec.
    max_broken_time : float
        Maximum time (either in seconds or as fraction of the recording) during which
        a channel is allowed to have artifacts. If a channel exceeds this, it will be
        removed. Not usually tuned. Default is 0.4 (40%), max is 0.5 (breakdown point
        of stats). Pretty much never tuned.
    linenoise_aware : bool
        Whether the operation should be performed in a line-noise
        aware manner. If enabled, the correlation measure will not be affected
        by the presence or absence of line noise (using a temporary notch filter).

    References
    ----------
    .. [Kothe2013] Kothe, C.A. and Makeig, S., 2013. BCILAB: a platform for
       brain–computer interface development. Journal of Neural Engineering, 10(5),
       p.056014.

    """

    def __init__(
        self,
        *,
        min_corr: float = 0.45,
        ignored_quantile: float = 0.1,
        window_len: float = 2.0,
        max_broken_time: float = 0.4,
        linenoise_aware: bool = True,
    ):
        super().__init__(record_orig_chanlocs=True)
        self.min_corr = min_corr
        self.ignored_quantile = ignored_quantile
        self.window_len = window_len
        self.max_broken_time = max_broken_time
        self.linenoise_aware = linenoise_aware

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        eeg, _ = eegprep.clean_channels_nolocs(
            eeg,
            min_corr=self.min_corr,
            ignored_quantile=self.ignored_quantile,
            window_len=self.window_len,
            max_broken_time=self.max_broken_time,
            linenoise_aware=self.linenoise_aware,
        )

        return eeg


class RemoveBursts(EEGPrepBasePreprocessor):
    """Run the Artifact Subspace Reconstruction (ASR) method on EEG data to
    remove burst-type artifacts. Follows [Mullen2015]_.

    .. figure:: https://cdn.ncbi.nlm.nih.gov/pmc/blobs/a79a/4710679/675fc2dee929/nihms733482f9.jpg
       :align: center
       :alt: Before/after comparison of ASR applied to EEG data.

    This is an automated artifact rejection function that ensures that the data
    contains no events that have abnormally strong power; the subspaces on which
    those events occur are reconstructed (interpolated) based on the rest of the
    EEG signal during these time periods.

    Preconditions:

    - One of :class:`RemoveDrifts` or :class:`braindecode.preprocessing.Filter` (
      configured as a highpass filter) must have been applied beforehand.
    - Must have removed flat-line channels beforehand with :class:`RemoveFlatChannels`.
    - If you are removing bad channels (:class:`RemoveBadChannels` or
      :class:`RemoveBadChannelsNoLocs`), use those before this step.
    - Consider applying :class:`RemoveDCOffset` beforehand as a general best practice.
    - If you are re-referencing to common average (:class:`RemoveCommonAverageReference`),
      this should normally *NOT* be done before this step, but after it.

    Parameters
    ----------
    cutoff : float
        Threshold for artifact rejection. Data portions whose variance is larger than
        this threshold relative to the calibration data are considered artifactual
        and removed. There is a fair amount of literature on what constitutes a good
        value. 7.5 is very aggressive, 10-15 is a good range for ML pipelines, 20-30
        is more forgiving and is more common in neuroscience applications. The unit is
        z-scores relative to a Gaussian component of background EEG, but since EEG
        phenomena of interest can stand out from the Gaussian background, typical
        thresholds are considerably larger than for a purely Gaussian distribution.
        Default is 10.0.
    window_len : float | None
        Length of the statistics window in seconds. Should not be much longer
        than artifact timescale. The number of samples in the window should
        be >= 1.5x channels. Default: max(0.5, 1.5 * nbchan / srate).
    step_size : int | None
        Step size for processing in samples. The reconstruction matrix is updated every
        this many samples. If None, defaults to window_len / 2 samples.
    max_dims : float
        Maximum dimensionality/fraction of dimensions to reconstruct. Default: 0.66.
        This can be understood to be the number of simultaneous artifact components that
        may be removed; normally needs no tuning, but on very low-channel data (e.g.,
        4ch) one may exploring small integers between 1 and #channels-1.
    ref_maxbadchannels : float | None
        Parameter that controls automatic calibration data selection. This represents
        the max fraction (0-1) of bad channels tolerated in a window for it to be used
        as calibration data. Lower is more aggressive (e.g., 0.05). Default: 0.075.
        The parameter has the same meaning as the max_bad_channels parameter in the
        RemoveBadWindows preprocessor, but note that this stage is used here as a
        subroutine to identify calibration data only. The overall method will always
        output a data matrix of the same shape as the input data. If set to None,
        all data is used for calibration.
    ref_tolerances : tuple[float, float]
        Power tolerances (lower, upper) in SDs from robust EEG power for a channel to
        be considered 'bad' during calibration data selection. This parameter goes hand
        in hand with ref_maxbadchannels. Default: (-inf, 5.5).
    ref_wndlen : float
        Window length in seconds for calibration data selection granularity.
        Default: 1.0.
    maxmem : int
        Maximum memory (in MB) to use during processing. Larger values can reduce
        overhead during processing, but usually 64MB is sufficient.

    References
    ----------
    .. [Mullen2015] Mullen, T.R., Kothe, C.A., Chi, Y.M., Ojeda, A., Kerth, T.,
       Makeig, S., Jung, T.P. and Cauwenberghs, G., 2015. Real-time neuroimaging and
       cognitive monitoring using wearable dry EEG. IEEE Transactions on Biomedical
       Engineering, 62(11), pp.2553-2567.

    """

    def __init__(
        self,
        *,
        cutoff: float = 10.0,
        window_len: float | None = None,
        step_size: int | None = None,
        max_dims: float = 0.66,
        ref_maxbadchannels: float | None = 0.075,
        ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
        ref_wndlen: float = 1.0,
        maxmem: int = 64,
    ):
        super().__init__(can_change_duration=True)
        self.cutoff = cutoff
        self.window_len = window_len
        self.step_size = step_size
        self.max_dims = max_dims
        self.ref_maxbadchannels = ref_maxbadchannels
        self.ref_tolerances = ref_tolerances
        self.ref_wndlen = ref_wndlen
        self.maxmem = maxmem

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        eeg = eegprep.clean_asr(
            eeg,
            cutoff=self.cutoff,
            window_len=self.window_len,
            step_size=self.step_size,
            max_dims=self.max_dims,
            ref_maxbadchannels=self.ref_maxbadchannels,
            ref_tolerances=self.ref_tolerances,
            ref_wndlen=self.ref_wndlen,
            maxmem=self.maxmem,
        )

        return eeg


class RemoveBadWindows(EEGPrepBasePreprocessor):
    """Remove periods with abnormally high-power content from continuous data.
     Implemented as in [Kothe2013]_.

    .. figure:: https://www.jove.com/files/ftp_upload/65829/65829fig13.jpg
       :align: center
       :alt: Before/after comparison of bad-window removal.

    This function cuts segments from the data which contain high-power (or low-power)
    artifacts. Specifically, only time windows are retained which have less than a
    certain fraction of *bad* channels, where a channel is bad in a window if its RMS
    power is above or below some z-score threshold relative to a robust estimate
    of clean-EEG power in that channel.

    .. Note::
      When your method is meant to produce predictions for all time points
      in your continuous data (or all epochs of interest), you may not want to use this
      preprocessor, and enabling it may give you rosy performance estimates that do not
      reflect how your method works when used on gap-free data. It can nevertheless be
      useful to apply this to training data only in such cases, however, to get an
      artifact-unencumbered model.

    Preconditions:

    - One of :class:`RemoveDrifts` or :class:`braindecode.preprocessing.Filter` (
      configured as a highpass filter) must have been applied beforehand.

    Parameters
    ----------
    max_bad_channels : int | float
        Threshold for rejection of bad time windows based on fraction of simultaneously
        noisy channels. This is the main tuning parameter; lower is more aggressive.
        Typical values are 0.15 (quite aggressive) to 0.3 (quite lax). Can also be
        specified as an absolute number of channels. Default is 0.25 (25% of channels).
    zthresholds : tuple(float, float)
        (min, max) z-score tolerance for identifying bad time window/channel pairs.
        This typically does not need to be changed (instead one may change the max
        bad channels that cross this threshold), but different implementations
        use different values here. The max value is the main parameter, where
        EEGLAB/EEGPrep uses 7 while the original pipeline [1] used 5.5, and NeuroPype
        uses 6. Lower values are more aggressive. The min value is only triggered if the
        EEG data has signal dropouts (very low amplitude, e.g. due to something becoming
        unplugged) which is rare; some choices are (-inf, EEGPrep; -3.5, BCILAB;
        -4, NeuroPype).
    window_len : float
        The window length that is used to check the data for artifact content, in
        seconds. This is ideally as long as the expected time scale of the artifacts,
        but short enough for there to be enough windows to compute statistics over.
        Default is 1.0 sec, but this may be lowered to 0.5 sec to catch very brief
        artifacts.
    window_overlap : float
        Fractional overlap between consecutive windows (0-1). Higher overlap
        finds more artefacts but is slower. Default is 0.66 (about 2/3 overlap).
    max_dropout_fraction : float
        Maximum fraction of windows that may have arbitrarily low amplitude
        (e.g. sensor unplugged). Default is 0.1.
    min_clean_fraction : float
        Minimum fraction of windows expected to be clean (essentially
        uncontaminated EEG). Default is 0.25.
    truncate_quant : tuple(float, float)
        Quantile range of the truncated Gaussian to fit (default (0.022,0.6)).
    step_sizes : tuple(float, float)
        Grid-search step sizes in quantiles for lower/upper edge. Default is (0.01,0.01)
    shape_range : sequence(float)
        Range for the beta shape parameter in the generalised Gaussian used
        for distribution fitting. Default is np.arange(1.7, 3.6, 0.15).

    References
    ----------
    .. [Kothe2013] Kothe, C.A. and Makeig, S., 2013. BCILAB: a platform for
       brain–computer interface development. Journal of Neural Engineering, 10(5),
       p.056014.

    """

    def __init__(
        self,
        *,
        max_bad_channels: int | float = 0.25,
        zthresholds: tuple[float, float] = (-np.inf, 7),
        window_len: float = 1.0,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        truncate_quant: tuple[float, float] = (0.022, 0.6),
        step_sizes: tuple[float, float] = (0.01, 0.01),
        shape_range: np.ndarray | Sequence[float] = np.arange(1.7, 3.6, 0.15),
    ):
        super().__init__(can_change_duration=True)
        self.max_bad_channels = max_bad_channels
        self.zthresholds = zthresholds
        self.window_len = window_len
        self.window_overlap = window_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.truncate_quant = truncate_quant
        self.step_sizes = step_sizes
        self.shape_range = shape_range

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        eeg, _ = eegprep.clean_windows(
            eeg,
            max_bad_channels=self.max_bad_channels,
            zthresholds=self.zthresholds,
            window_len=self.window_len,
            window_overlap=self.window_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            truncate_quant=self.truncate_quant,
            step_sizes=self.step_sizes,
            shape_range=self.shape_range,
        )

        return eeg


class ReinterpolateRemovedChannels(EEGPrepBasePreprocessor):
    """Reinterpolate previously removed EEG channels to restore original channel set.

    .. figure:: ../../docs/_static/preprocess/sph_spline_interp.png
       :align: center
       :alt: Spherical spline interpolation example.

    This reinterpolates EEG channels that were previously dropped via one of the EEGPrep
    channel removal operations and restores the original order of EEG channels. This
    is typically necessary when you are using automatic channel removal but you need
    a consistent channel set across multiple recordings/sessions. Uses spherical-spline
    interpolation (based on [Perrin1989]_).

    The typical place to perform this is after all other EEGPrep-related artifact
    removal steps, except re-referencing. If no channel locations were recorded,
    this preprocessor has no effect.

    Preconditions:

    - Must have 3D channel locations.
    - This filter will only have an effect if one or more of the preceding steps
      recorded original channel locations (e.g., :class:`RemoveBadChannels`,
      :class:`RemoveBadChannelsNoLocs`, or :class:`RemoveFlatChannels`).
    - If you are re-referencing to common average (:class:`RemoveCommonAverageReference`),
      this should normally *NOT* be done before this step, but after it (otherwise
      your reference will depend on which channels were removed).

    References
    ----------
    .. [Perrin1989] Perrin, F., Pernier, J., Bertrand, O. and Echallier, J.F., 1989.
       Spherical splines for scalp potential and current density mapping.
       Electroencephalography and Clinical Neurophysiology, 72(2), pp.184-187.


    """

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        orig_chanlocs = self._get_orig_chanlocs(raw)
        if orig_chanlocs is None:
            log.info(
                "ReinterpolateRemovedChannels: No original channel locations were "
                "recorded by a preceding step; skipping reinterpolation."
            )
        elif len(orig_chanlocs) > len(eeg["chanlocs"]):
            eeg = eegprep.eeg_interp(eeg, orig_chanlocs)

        return eeg


class RemoveCommonAverageReference(EEGPrepBasePreprocessor):
    """Subtracts the common average reference from the EEG data (EEGPrep version).
    This is useful for having a consistent referencing scheme across recordings
    (cf. [Offner1950]_).

    Generally, common average re-referencing is `data -= mean(data, axis=0)`, but
    both EEGLAB/eegprep and to a greater extent MNE have additional bookkeeping around
    re-referencing, in the latter case due to its focus on source localization. This
    will have little effect on most machine-learning use cases; nevertheless, this
    operation is included here to allow users to mirror the behavior of the end-to-end
    EEGPrep pipeline by means of individual operations (for example when migrating
    from one to the other form) without introducing perhaps unexpected side effects
    on the MNE data structure.

    The operation performed is:

    .. math::

        X'_{c,t} = X_{c,t} - \\frac{1}{C}\\sum_{c=1}^{C} X_{c,t}

    where :math:`C` is the number of channels, :math:`c` indexes the channel, and
    :math:`t` indexes time.

    References
    ----------
    .. [Offner1950] Offner, F. F. (1950). The EEG as potential mapping: the value of the
       average monopolar reference. Electroencephalography and Clinical Neurophysiology,
       2(2), 213-214.

    """

    def apply_eeg(self, eeg: dict[str, Any], raw: BaseRaw) -> dict[str, Any]:
        """Apply the preprocessor to an EEGLAB EEG structure."""
        return eegprep.reref(eeg, [])
