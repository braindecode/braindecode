"""Preprocessor using the EEGPrep package."""

# Authors: Christian Kothe <christian.kothe@intheon.io>
#
# License: BSD-3

import logging
import contextlib
from typing import Any

import numpy as np
from mne.io import BaseRaw

from braindecode.preprocessing.preprocess import Preprocessor

log = logging.getLogger(__name__)


class EEGPrep(Preprocessor):
    """Preprocessor for an MNE Raw object that applies the EEGPrep pipeline. This
    pipeline involves the stages:
     * Optional resampling
     * Flatline channel detection and removal
     * High-pass filtering
     * Bad channel detection and removal using correlation and HF noise criteria
     * Burst artifact removal using ASR (Artifact Subspace Reconstruction)
     * Residual bad time window detection and removal
     * Optional reinterpolation of removed channels
     * Optional common average referencing

    It is recommended to follow this with at least a low-pass filter to remove
    high-frequency artifacts (e.g., 40-45 Hz transition band).

    The main processing parameters can each be set to None to skip the respective
    stage (or False for boolean switches). Note this pipeline will only affect the
    EEG channels in your data, and will leave other channels unaffected. It is
    recommended to remove these channels yourself beforehand if you don't want them
    included in your downstream analysis.

    Note that this implementation of the pipeline is best used in the context of
    cross-session prediction; when using this with a within-session split, there is
    a risk of data leakage since the artifact removal will be calibrated on statistics
    of the entire session (and thus test sets). In practice the effect may be minor,
    unless your downstream analysis is in strongly driven by artifacts (e.g., if you
    are trying to decode eye movements or muscle activity), but reviewers may not buy
    that.

    Parameters
    ----------
    (main processing parameters, corresponding to processing stages)
    resample_to: float | None = None
        Optionally resample to this sampling rate (in Hz) before processing.
        Good choices are 200, 250, 256 Hz (consider keeping it a power of two
        if it was originally), but one may go as low as 100-128 Hz if memory, compute,
        or model complexity limitations demand it.
    flatline_maxdur: float | None
        Remove channels that are flat for longer than this duration (in seconds).
        This stage is almost never triggered in practice but can help with the
        occasional strange EEG configuration.
    highpass_frequencies: tuple[float, float] | None
        Tuple of lower and upper bound of the *transition band* for high-pass filtering
        before processing. This means that full suppression will be reached at the
        lower bound, and the upper bound is where the passband begins.
    bad_channel_corr_threshold: float | None
        Threshold for correlation-based bad channel detection. A good default range
        is 0.75-0.8. Becomes quite aggressive at and beyond 0.8; also, consider using
        lower values (eg 0.7-0.75) for <32ch EEG and higher (0.8-0.85) for >128ch.
    burst_removal_cutoff: float | None
        Amplitude threshold for burst artifact removal using ASR
        (Artifact Subspace Reconstruction). This parameter tends to have a large effect
        on the performance of downstream ML. 10-15 is a good range for ML pipelines
        (lower is more aggressive); for neuroscience analysis, more conservative values
        like 20-30 may be better. The unit is z-scores relative to a Gaussian component
        of background EEG, but real EEG can be super-Gaussian, thus the large values.
    bad_window_max_bad_channels: float | None
        Threshold for rejection of bad time windows based on fraction of simultaneously
        noisy channels. Lower is more aggressive. Typical values are 0.15 (quite
        aggressive) to 0.3 (quite lax).
    bad_channel_reinterpolate: bool
        Whether to reinterpolate bad channels that were detected and removed. Usually
        required when doing cross-session analysis (to have a consistent channel set).
    common_avg_ref: bool
        Whether to apply a common average reference after processing. Recommended
        when doing cross-study analysis to have a consistent referencing scheme.

    (additional tuning parameters)
    bad_channel_hf_threshold: float
        Threshold for high-frequency (>=45 Hz) noise-based bad channel detection,
        in z-scores. Lower is more aggressive. Default is 4.0. This is rarely tuned,
        but data with unusual higher-frequency activity could benefit from exploration
        in the 3.5-5.0 range.
    bad_channel_max_broken_time: float
        Max fraction of session length during which a channel may be bad before
        it is removed. Default is 0.4 (40%), max is 0.5 (breakdown point of stats).
        Pretty much never tuned.
    bad_window_tolerances: tuple[float, float]
        (min, max) z-score tolerance for identifying bad time window/channel pairs.
        This typically does not need to be changed (instead one may change the max
        bad channels that cross this threshold), but different implementations
        use different values here. The max value is the main parameter, where
        EEGLAB/EEGPrep uses 7 while the original pipeline [1] used 5.5, and NeuroPype
        uses 6. Lower values are more aggressive. The min value is only triggered if the
        EEG data has signal dropouts (very low amplitude, e.g. due to something becoming
        unplugged) which is rare; some choices are (-inf, EEGPrep; -3.5, BCILAB;
        -4 NeuroPype).
    refdata_max_bad_channels: float | None
        Same function as bad_window_max_bad_channels, but used only to determine
        calibration data for burst removal. Usually more aggressive than the former
        (0.05-0.1) to get clean calibration data. This can be set to None to skip this
        and force all data to be used for calibration.
    refdata_max_tolerances: tuple[float, float]
        Same as bad_window_tolerances, but used only to determine calibration data for
        burst removal. Almost never touched, and defaults to a fairly aggressive
        (-inf, 5.5) to get clean calibration data.
    num_samples: int
        Number of channel subsets to draw for the RANSAC reconstruction during bad
        channel identification. Higher can be more robust but slower to calibrate.
        Default is 50.
    subset_size: float
        Size of channel subsets for RANSAC, as fraction (0-1) or count. Default 0.25.
        For higher-density EEG (e.g., 64-128ch), one can achieve somewhat better
        robustness to clusters of bad channels by setting this to 0.15 and increasing
        num_samples to 200.
    bad_channel_nolocs_threshold: float
        A fallback correlation threshold for bad-channel removal that is applied when
        no channel location information is available. The value here typically needs to
        be fairly low, e.g., 0.45-0.5 (lower is more aggressive). Ideally you have
       channel locations so that this fallback is not needed.
    bad_channel_nolocs_exclude_frac: float = 0.1,
        A fraction of most correlated channels to exclude in the case where no channel
        location information is available. Used to reject pairs of shorted or otherwise
        highly correlated sets of bad channels.
    max_mem_mb: int
        Max memory that ASR can use, in MB. Larger values can reduce overhead during
        processing, but usually 64MB is sufficient.

    References
    ----------
    [1] Mullen, T. R., Kothe, C. A., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S., ...
       & Cauwenberghs, G. (2015). Real-time neuroimaging and cognitive monitoring using
       wearable dry EEG. IEEE Transactions on Biomedical Engineering, 62(11), 2553-2567.

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
        super().__init__(
            fn=self._apply_eegprep,
            apply_on_array=False,
        )
        self.resample_to = resample_to
        self.reinterpolate = bad_channel_reinterpolate
        self.common_avg_ref = common_avg_ref
        self.burst_removal_cutoff = burst_removal_cutoff
        self.bad_window_max_bad_channels = bad_window_max_bad_channels

        def off_if_none(x, *, switch: Any = -1):
            """Helper to convert None to 'off'; can also use a different variable to switch."""
            return 'off' if x is None or switch is None else x

        self.clean_artifacts_params = dict(
            ChannelCriterion=off_if_none(bad_channel_corr_threshold),
            LineNoiseCriterion=off_if_none(bad_channel_hf_threshold, switch=bad_channel_corr_threshold),
            BurstCriterion=off_if_none(burst_removal_cutoff),
            WindowCriterion=off_if_none(bad_window_max_bad_channels),
            Highpass=off_if_none(highpass_frequencies),

            ChannelCriterionMaxBadTime=bad_channel_max_broken_time,
            BurstCriterionRefMaxBadChns=off_if_none(refdata_max_bad_channels),
            BurstCriterionRefTolerances=refdata_max_tolerances,
            # BurstRejection='off', # we don't use that one; left at default
            WindowCriterionTolerances=bad_window_tolerances,
            FlatlineCriterion=off_if_none(flatline_maxdur),
            NumSamples=num_samples,
            SubsetSize=subset_size,
            NoLocsChannelCriterion=bad_channel_nolocs_threshold,
            NoLocsChannelCriterionExcluded=bad_channel_nolocs_exclude_frac,
            MaxMem=max_mem_mb,
            # Distance='euclidian',
            # Channels=None,
            # Channels_ignore=None,
            # availableRAM_GB=None,
            # (no other args)
        )

    def _apply_eegprep(self, raw: BaseRaw) -> BaseRaw:
        """Internal method that does the actual work; this is called by Preprocessor.apply()."""
        if 'Epochs' in raw.__class__.__name__:
            raise ValueError("EEGPrep can only be applied to Raw (continuous) data, "
                             "not Epochs. Use before epoching.")
        try:
            import eegprep
        except ImportError as e:
            raise ImportError("The eegprep package is required to use the "
                              "EEGPrep preprocessor.\n"
                              "Please install it via 'pip install eegprep eeglabio', "
                              "or (if you use uv) 'uv pip install eegprep eeglabio'.") from e

        # split off non-EEG channels
        orig_chs = raw.ch_names.copy()
        EEG = raw.copy().pick_types(eeg=True, exclude=[])
        if len(EEG.ch_names) < len(raw.ch_names):
            nonEEG = raw.drop_channels(EEG.ch_names)
        else:
            nonEEG = None

        # --- convert to EEGLAB structure and prepare for processing ---
        EEG = eegprep.mne2eeg(EEG)

        # make sure all events have a 'duration' field (mne2eeg doesn't enforce it)
        if not all('duration' in ev for ev in EEG['event']):
            for ev in EEG['event']:
                if 'duration' not in ev:
                    ev['duration'] = 1

        try:
            # noinspection PyUnresolvedReferences
            from threadpoolctl import threadpool_limits
            # this is done because some processing steps might cause churn otherwise
            thread_limit_ctx = threadpool_limits(limits=4, user_api='blas')
        except ImportError:
            # should not happen since eegprep implicitly pulls it in
            log.warning("threadpoolctl not installed, using default thread limits.")
            thread_limit_ctx = contextlib.nullcontext()

        with thread_limit_ctx:
            # remove per-channel DC offset (can be huge)
            EEG['data'] -= np.median(EEG['data'], axis=1, keepdims=True)

            # optional resampling
            if (srate := self.resample_to) is not None:
                EEG = eegprep.resample(EEG, srate)

            # do a check if the data has a supported sampling rate
            if self.burst_removal_cutoff is not None:
                from eegprep.utils import round_mat
                supported_rates = (100, 128, 200, 250, 256, 300, 500, 512)
                if (sr_current := int(round_mat(EEG['srate']))) not in supported_rates:
                    # note: technically the method will run if you disable this error,
                    #   but you're likely getting (potentially quite) suboptimal results
                    raise NotImplementedError(
                        f"The dataset has an uncommon sampling rate of {sr_current} Hz,"
                        f" which is not supported by the EEGPrep Preprocessor"
                        f" implementation. Please enable resampling to"
                        f" resample the data to one of the supported rates"
                        f" ({', '.join(str(r) for r in supported_rates)}).")

            orig_chanlocs = EEG['chanlocs']

            # artifact removal stage
            EEG, *_ = eegprep.clean_artifacts(EEG, **self.clean_artifacts_params)

            # optionally reinterpolate dropped channels
            if self.reinterpolate and (len(orig_chanlocs) > len(EEG['chanlocs'])):
                EEG = eegprep.eeg_interp(EEG, list(orig_chanlocs))

            # optionally apply common average reference
            if self.common_avg_ref:
                EEG = eegprep.reref(EEG, [])

        # rename boundary events so they are ignored during downstream MNE epoching
        for ev in EEG['event']:
            if ev['type'] == 'boundary':
                ev['type'] = 'BAD boundary'

        # convert back to MNE Raw
        proc = eegprep.eeg2mne(EEG)

        # try to add back any non-EEG channels that were originally present
        if nonEEG is not None:
            if self.resample_to is not None or self.bad_window_max_bad_channels is not None:
                # while this may work by accident on one or another session, in
                # general we will not try to add them back in in this case
                affected_chans = ', '.join(nonEEG.ch_names)
                log.error(
                    f"Could not add back non-EEG channels ({affected_chans}) after"
                    f" EEGPrep processing; these will be omitted from the processed"
                    f" data. If you want to retain these channels, you will have to"
                    f" disable resampling and bad time window removal in the"
                    f" EEGPrep Preprocessor; you may perform these steps using other"
                    f" methods before and after EEGPrep, respectively.")
            else:
                proc.add_channels([nonEEG], force_update_info=True)
                if proc.ch_names != orig_chs:
                    proc.reorder_channels(orig_chs)

        return proc
