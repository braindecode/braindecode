import logging

import numpy as np

from braindecode.datautil.iterators import get_balanced_batches
from braindecode.util import wrap_reshape_apply_fn, corr

log = logging.getLogger(__name__)


def phase_perturbation(amps, phases, rng=None):
    """Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Shifts spectral phases randomly U(-pi,pi) for input and frequencies, but same for all channels

    Parameters
    ----------
    amps : numpy array
        Spectral amplitude (not used)
    phases : numpy array
        Spectral phases
    rng : object
        Random Generator

    Returns
    -------
    amps : numpy array
        Input amps (not modified)
    phases_pert : numpy array
        Shifted phases
    pert_vals : numpy array
        Absolute phase shifts
    """
    if rng is None:
        rng = np.random.RandomState()
    noise_shape = list(phases.shape)
    noise_shape[1] = 1  # Do not sample noise for channels individually

    # Sample phase perturbation noise
    phase_noise = rng.uniform(-np.pi, np.pi, noise_shape).astype(np.float32)
    phase_noise_rep = phase_noise.repeat(phases.shape[1], axis=1)
    # Apply noise to inputs
    phases_pert = phases + phase_noise_rep
    phases_pert[phases_pert < -np.pi] += 2 * np.pi
    phases_pert[phases_pert > np.pi] -= 2 * np.pi

    pert_vals = np.abs(phase_noise)
    return amps, phases_pert, pert_vals


def amp_perturbation_additive(amps, phases, rng=None):
    """Takes amplitudes and phases of BxCxF with B input, C channels, F frequencies
    Adds additive noise N(0,0.02) to amplitudes

    Parameters
    ----------
    amps : numpy array
        Spectral amplitude
    phases : numpy array
        Spectral phases (not used)
    rng : object
        Random Seed

    Returns
    -------
    amps_pert : numpy array
        Scaled amplitudes
    phases_pert : numpy array
        Input phases (not modified)
    pert_vals : numpy array
        Amplitude noise
    """
    if rng is None:
        rng = np.random.RandomState()
    amp_noise = rng.normal(0, 1, amps.shape).astype(np.float32)
    amps_pert = amps + amp_noise
    amps_pert[amps_pert < 0] = 0
    amp_noise = amps_pert - amps
    return amps_pert, phases, amp_noise


def amp_perturbation_multiplicative(amps, phases, rng=None):
    """Takes amplitude and phases of BxCxF with B input, C channels, F frequencies
    Adds multiplicative noise N(1,0.02) to amplitudes

    Parameters
    ----------
    amps : numpy array
        Spectral amplitude
    phases : numpy array
        Spectral phases (not used)
    rng : object
        Random Seed

    Returns
    -------
    amps_pert : numpy array
        Scaled amplitudes
    phases_pert : numpy array
        Input phases (not modified)
    pert_vals : numpy array
        Amplitude scaling factor
    """
    if rng is None:
        rng = np.random.RandomState()
    amp_noise = rng.normal(1, 0.02, amps.shape).astype(np.float32)
    amps_pert = amps * amp_noise
    amps_pert[amps_pert < 0] = 0
    return amps_pert, phases, amp_noise


def correlate_feature_maps(x, y):
    """Takes two activation matrices of the form Bx[F]xT where B is batch size, F number of filters (optional) and T time points
    Returns correlations of the corresponding activations over T

    Parameters
    ----------
    x : numpy array
        Activations Bx[F]xT
    y : numpy array
        Activations Bx[F]xT

    Returns
    correlations : numpy array
        Correlations of `x` and `y` Bx[F]
    """
    shape_x = x.shape
    shape_y = y.shape
    assert np.array_equal(shape_x, shape_y)
    assert len(shape_x) < 4
    x = x.reshape((-1, shape_x[-1]))
    y = y.reshape((-1, shape_y[-1]))

    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)

    tmp_corr = x * y
    corr_ = tmp_corr.sum(axis=1)
    # corr_ = np.zeros((x.shape[0]))
    # for i in range(x.shape[0]):
    #    # Correlation of standardized variables
    #    corr_[i] = np.correlate((x[i]-x[i].mean())/x[i].std(),(y[i]-y[i].mean())/y[i].std())

    correlations = corr_.reshape(*shape_x[:-1])
    return correlations


def mean_diff_feature_maps(x, y):
    """Takes two activation matrices of the form BxFxT where B is batch size, F number of filters and T time points
    Returns mean difference between feature map activations

    Parameters
    ----------
    x : numpy array
        Activations Bx[F]xT
    y : numpy array
        Activations Bx[F]xT

    Returns
    mean_diff : numpy array
        Mean difference between `x` and `y` Bx[F]
    """
    mean_diff = np.mean(x - y, axis=2)
    return mean_diff


def spectral_perturbation_correlation(
    pert_fn,
    diff_fn,
    pred_fn,
    n_layers,
    inputs,
    n_iterations,
    batch_size=30,
    seed=((2017, 7, 10)),
):
    """Calculates perturbation correlations for layers in network by perturbing either amplitudes or phases

    Parameters
    ----------
    pert_fn : function
        Function that perturbs spectral phase and amplitudes of inputs
    diff_fn : function
        Function that calculates difference between original and perturbed activations
    pred_fn : function
        Function that returns a list of activations.
        Each entry in the list corresponds to the output of 1 layer in a network
    n_layers : int
        Number of layers pred_fn returns activations for.
    inputs : numpy array
        Original inputs that are used for perturbation [B,X,T,1]
        Phase perturbations are sampled for each input individually, but applied to all X of that input
    n_iterations : int
        Number of iterations of correlation computation. The higher the better
    batch_size : int
        Number of inputs that are used for one forward pass. (Concatenated for all inputs)

    Returns
    -------
    pert_corrs : numpy array
        List of length n_layers containing average perturbation correlations over iterations
        L  x  CxFrxFi (Channels,Frequencies,Filters)
    """
    rng = np.random.RandomState(seed)

    # Get batch indeces
    batch_inds = get_balanced_batches(
        n_trials=len(inputs), rng=rng, shuffle=False, batch_size=batch_size
    )
    # Calculate layer activations and reshape
    log.info("Compute original predictions...")
    orig_preds = [pred_fn(inputs[inds]) for inds in batch_inds]
    use_shape = []
    for l in range(n_layers):
        tmp = list(orig_preds[0][l].shape)
        tmp.extend([1] * (4 - len(tmp)))
        tmp[0] = len(inputs)
        use_shape.append(tmp)
    orig_preds_layers = [
        np.concatenate([orig_preds[o][l] for o in range(len(orig_preds))]).reshape(
            use_shape[l]
        )
        for l in range(n_layers)
    ]

    # Compute FFT of inputs
    fft_input = np.fft.rfft(inputs, n=inputs.shape[2], axis=2)
    amps = np.abs(fft_input)
    phases = np.angle(fft_input)

    pert_corrs = [0] * n_layers
    for i in range(n_iterations):
        log.info("Iteration {:d}...".format(i))
        log.info("Sample perturbation...")
        amps_pert, phases_pert, pert_vals = pert_fn(amps, phases, rng=rng)

        # Compute perturbed inputs
        log.info("Compute perturbed complex inputs...")
        fft_pert = amps_pert * np.exp(1j * phases_pert)
        log.info("Compute perturbed real inputs...")
        inputs_pert = np.fft.irfft(fft_pert, n=inputs.shape[2], axis=2).astype(
            np.float32
        )

        # Calculate layer activations for perturbed inputs
        log.info("Compute new predictions...")
        new_preds = [pred_fn(inputs_pert[inds]) for inds in batch_inds]
        new_preds_layers = [
            np.concatenate([new_preds[o][l] for o in range(len(new_preds))]).reshape(
                use_shape[l]
            )
            for l in range(n_layers)
        ]

        for l in range(n_layers):
            log.info("Layer {:d}...".format(l))
            # Calculate difference of original and perturbed feature map activations
            log.info("Compute activation difference...")
            preds_diff = diff_fn(
                new_preds_layers[l][:, :, :, 0], orig_preds_layers[l][:, :, :, 0]
            )

            # Calculate feature map differences with perturbations
            log.info("Compute correlation...")
            pert_corrs_tmp = wrap_reshape_apply_fn(
                corr, pert_vals[:, :, :, 0], preds_diff, axis_a=(0,), axis_b=(0)
            )
            pert_corrs[l] += pert_corrs_tmp

    pert_corrs = [
        pert_corrs[l] / n_iterations for l in range(n_layers)
    ]  # mean over iterations
    return pert_corrs


def compute_amplitude_prediction_correlations(
    pred_fn,
    examples,
    n_iterations,
    perturb_fn=amp_perturbation_additive,
    batch_size=30,
    seed=((2017, 7, 10)),
):
    """
    Perturb input amplitudes and compute correlation between amplitude
    perturbations and prediction changes when pushing perturbed input through
    the prediction function.

    For more details, see [EEGDeepLearning]_.

    Parameters
    ----------
    pred_fn: function
        Function accepting an numpy input and returning prediction.
    examples: ndarray
        Numpy examples, first axis should be example axis.
    n_iterations: int
        Number of iterations to compute.
    perturb_fn: function, optional
        Function accepting amplitude array and random generator and returning
        perturbation. Default is Gaussian perturbation.
    batch_size: int, optional
        Batch size for computing predictions.
    seed: int, optional
        Random generator seed

    Returns
    -------
    amplitude_pred_corrs: ndarray
        Correlations between amplitude perturbations and prediction changes
        for all sensors and frequency bins.

    References
    ----------

    .. [EEGDeepLearning] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    pred_fn_new = lambda x: [pred_fn(x)]
    pred_corrs = spectral_perturbation_correlation(
        perturb_fn,
        mean_diff_feature_maps,
        pred_fn_new,
        1,
        examples,
        n_iterations,
        batch_size=batch_size,
        seed=seed,
    )

    return pred_corrs[0]
