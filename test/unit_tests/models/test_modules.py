# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
import platform

import numpy as np
import pytest
import torch

from scipy.signal import lfilter as lfilter_scipy
from scipy.signal import freqz
from mne.time_frequency import psd_array_welch
from mne.filter import create_filter
from scipy.signal import fftconvolve as fftconvolve_scipy


from torch import nn
import matplotlib.pyplot as plt
from braindecode.models.tidnet import _BatchNormZG, _DenseSpatialFilter
from braindecode.models.modules import CombinedConv, MLP, TimeDistributed, DropPath, SafeLog, FilterBankLayer
from braindecode.models.labram import _SegmentPatch

from braindecode.models.functions import drop_path


def _filfilt_in_torch_sytle(b, a, x_np):
    # Forward filtering
    forward_filtered = lfilter_scipy(b=b.astype(np.double),
                                     a=a.astype(np.double),
                                     x=x_np.astype(np.double), axis=-1)

    # Reverse the filtered signal a long time axis
    reversed_signal = np.flip(forward_filtered, axis=-1)

    # Backward filtering
    backward_filtered = lfilter_scipy(b=b, a=a,
                                      x=reversed_signal, axis=-1)

    # Reverse back to original order
    filtered_scipy = np.flip(backward_filtered, axis=-1)

    return filtered_scipy


def test_time_distributed():
    n_channels = 4
    n_times = 100
    feat_size = 5
    n_windows = 4
    batch_size = 2
    X = torch.rand((batch_size, n_windows, n_channels, n_times))

    feat_extractor = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(n_channels * n_times, feat_size)
    )
    model = TimeDistributed(feat_extractor)

    out = model(X)
    out2 = [model(X[:, [i]]) for i in range(X.shape[1])]
    out2 = torch.stack(out2, dim=1).flatten(start_dim=2)

    assert out.shape == (batch_size, n_windows, feat_size)
    assert torch.allclose(out, out2, atol=1E-4, rtol=1e-4)

def test_reset_parameters():
    num_channels = 3

    bn = _BatchNormZG(num_channels)
    bn.reset_parameters()

    # Check running stats
    assert bn.running_mean.size(0) == num_channels
    assert torch.allclose(bn.running_mean, torch.zeros(num_channels))

    assert bn.running_var.size(0) == num_channels
    assert torch.allclose(bn.running_var, torch.ones(num_channels))

    # Check weight and bias
    assert bn.weight.size(0) == num_channels
    assert torch.allclose(bn.weight, torch.zeros(num_channels))

    assert bn.bias.size(0) == num_channels
    assert torch.allclose(bn.bias, torch.zeros(num_channels))


def test_dense_spatial_filter_forward_collapse_true():
    in_chans = 3
    growth = 8
    depth = 4
    in_ch = 1
    bottleneck = 4
    drop_prob = 0.0
    activation = torch.nn.LeakyReLU
    collapse = True

    dense_spatial_filter = _DenseSpatialFilter(
        in_chans, growth, depth, in_ch, bottleneck, drop_prob, activation, collapse
    )

    x = torch.rand(5, 3, 10)  # 3-dimensional input
    output = dense_spatial_filter(x)
    assert output.shape[:2] == torch.Size([5, 33])


def test_dense_spatial_filter_forward_collapse_false():
    in_chans = 3
    growth = 8
    depth = 4
    in_ch = 1
    bottleneck = 4
    drop_prob = 0.0
    activation = torch.nn.LeakyReLU
    collapse = False

    dense_spatial_filter = _DenseSpatialFilter(
        in_chans, growth, depth, in_ch, bottleneck, drop_prob, activation, collapse
    )

    x = torch.rand(5, 3, 10)  # 3-dimensional input
    output = dense_spatial_filter(x)
    assert output.shape[:2] == torch.Size([5, 33])


# Issue with False, False option
@pytest.mark.skipif(platform.system() == 'Linux'
                    or platform.system() == 'Windows',
                    reason="Not supported on Linux")
@pytest.mark.parametrize(
    "bias_time,bias_spat", [(False, False), (False, True), (True, False), (True, True)]
)
def test_combined_conv(bias_time, bias_spat):
    batch_size = 64
    in_chans = 44
    timepoints = 1000

    data = torch.rand([batch_size, 1, timepoints, in_chans])
    conv = CombinedConv(in_chans=in_chans, bias_spat=bias_spat, bias_time=bias_time)

    combined_out = conv(data)
    sequential_out = conv.conv_spat(conv.conv_time(data))

    assert torch.isclose(combined_out, sequential_out, atol=1e-4).all()

    diff = combined_out - sequential_out
    assert ((diff**2).mean().sqrt() / sequential_out.std()) < 1e-5
    assert (diff.abs().median() / sequential_out.abs().median()) < 1e-5


@pytest.mark.parametrize("hidden_features", [None, (10, 10), (50, 50, 50), [10, 10, 10]])
def test_mlp_increase(hidden_features):
    model = MLP(in_features=40, hidden_features=hidden_features)
    if hidden_features is None:
        assert len(model) == 6
    else:
        # For each layer that we add, the model
        # increase with 2 layers + 2 initial layers (input, output layer)
        assert len(model) == 2 * (len(hidden_features)) + 2


def test_segm_patch_not_learning():
    n_chans = 64
    patch_size = 200
    embed_dim = 200
    n_segments = 5
    X = []
    for i in range(n_segments):
        if i % 2 == 0:
            X.append(torch.zeros((n_chans, patch_size)))
        else:
            X.append(torch.ones((n_chans, patch_size)))

    n_times = patch_size * n_segments
    X = torch.concat(X, dim=1)

    assert n_times == X.shape[-1]

    module = _SegmentPatch(
        n_times=n_times,
        n_chans=n_chans,
        patch_size=patch_size,
        emb_dim=embed_dim,
        learned_patcher=False,
    )

    with torch.no_grad():
        # Adding batch dimension
        X_split = module(X.unsqueeze(0))

        assert X_split.shape[1] == n_chans
        assert X_split.shape[2] == n_times // patch_size
        assert X_split.shape[3] == embed_dim

        assert torch.allclose(X_split[0, 0, 0].sum(), torch.zeros(1))
        assert torch.equal(X_split[0, 0, 1].unique(), torch.ones(1))


@pytest.fixture(scope="module")
def x_metainfo():
    return {
        "batch_size": 2,
        "n_chans": 64,
        "patch_size": 200,
        "n_segments": 5,
        "n_times": 1000,
        "X": torch.zeros((2, 64, 1000)),
    }


def test_segm_patch(x_metainfo):

    module = _SegmentPatch(
        n_times=x_metainfo["n_times"],
        n_chans=x_metainfo["n_chans"],
        patch_size=x_metainfo["patch_size"],
        emb_dim=x_metainfo["patch_size"],
    )

    with torch.no_grad():
        # Adding batch dimension
        X_split = module(x_metainfo["X"])
        assert X_split.shape[0] == x_metainfo["batch_size"]
        assert X_split.shape[1] == x_metainfo["n_chans"]
        assert X_split.shape[2] == x_metainfo["n_times"] // x_metainfo["patch_size"]
        assert X_split.shape[3] == x_metainfo["patch_size"]


def test_drop_path_prob_1(x_metainfo):
    """
    Test that the DropPath module sets the input tensor to zero.
    """

    module = DropPath(drop_prob=1)

    with torch.no_grad():
        # Adding batch dimension
        X_zeros = module(x_metainfo["X"])
        assert torch.allclose(X_zeros, torch.zeros_like(x_metainfo["X"]))


def test_drop_path_prob_0(x_metainfo):
    """
    Test that the DropPath module using prob equal 0
    """

    module = DropPath(drop_prob=0)

    with torch.no_grad():
        # Adding batch dimension
        X_original = module(x_metainfo["X"])
        assert torch.allclose(X_original, x_metainfo["X"])


def test_drop_path_representation():
    drop_prob = 0.5
    module = DropPath(drop_prob=0.5)
    expected_repr = f"p={drop_prob}"

    # Get the actual repr string
    actual_repr = repr(module)

    assert expected_repr in actual_repr


def test_drop_path_no_drop():
    x = torch.rand(3, 3)  # Example input tensor
    output = drop_path(x, drop_prob=0.0, training=False)
    assert torch.equal(
        output, x
    ), "Output should be equal to input when drop_prob is 0.0 or training is False."


def test_drop_path_with_dropout_shape():
    x = torch.rand(5, 4)  # Example input tensor
    output = drop_path(x, drop_prob=0.5, training=True)
    assert (
        output.shape == x.shape
    ), "Output tensor must have the same shape as the input tensor."


def test_drop_path_scale_by_keep():
    torch.manual_seed(0)
    x = torch.rand(1, 10)  # Single-dimension tensor for simplicity
    drop_prob = 0.2
    scaled_output = drop_path(x, drop_prob=drop_prob, training=True, scale_by_keep=True)
    unscaled_output = drop_path(
        x, drop_prob=drop_prob, training=True, scale_by_keep=False
    )
    # This test relies on statistical expectation and may need multiple runs or adjustments
    scale_factor = 1 / (1 - drop_prob)
    assert torch.allclose(
        scaled_output.mean(), unscaled_output.mean() * scale_factor, atol=0.1
    ), "Scaled output does not match expected scaling."


def test_drop_path_different_dimensions():
    x_2d = torch.rand(2, 2)  # 2D tensor
    x_3d = torch.rand(2, 2, 2)  # 3D tensor
    output_2d = drop_path(x_2d, drop_prob=0.5, training=True)
    output_3d = drop_path(x_3d, drop_prob=0.5, training=True)
    assert (
        output_2d.shape == x_2d.shape and output_3d.shape == x_3d.shape
    ), "Output tensor must maintain input shape across different dimensions."



@pytest.mark.parametrize(
    "eps, expected_repr",
    [
        (1e-6, "eps=1e-06"),
        (1e-4, "eps=0.0001"),
        (1e-8, "eps=1e-08"),
        (0.0, "eps=0.0"),
        (123.456, "eps=123.456"),
    ]
)
def test_safelog_extra_repr(eps, expected_repr):
    """
    Test the extra_repr method of the SafeLog class to ensure it returns
    the correct string representation based on the eps value.

    Parameters
    ----------
    eps : float
        The epsilon value to initialize SafeLog with.
    expected_repr : str
        The expected string output from extra_repr.
    """
    # Initialize the SafeLog module with the given eps
    module = SafeLog(eps=eps)

    # Get the extra representation
    repr_output = module.extra_repr()

    # Assert that the extra_repr output matches the expected string
    assert repr_output == expected_repr, f"Expected '{expected_repr}', got '{repr_output}'"

@pytest.mark.parametrize("ftype", ["butterworth", "cheby1", "cheby1", "cheby2", "butter"])
@pytest.mark.parametrize("phase", ["forward", "zero", "zero-double"])
@pytest.mark.parametrize("l_freq, h_freq", [(4, 8), (8, 12), (13, 30)])
def test_filter_bank_layer_matches_mne_iir(l_freq, h_freq, phase, ftype):
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Test parameters
    n_chans = 22
    batch_size = 1
    n_times = 1000
    x = torch.randn(batch_size, n_chans, n_times, dtype=torch.float64)
    if ftype in ["butter", "buttord", "butterworth"]:
        iir_params = dict(ftype=ftype, output="ba", order=4)
    else:
        iir_params = dict(ftype=ftype, output="ba", order=4, rs=1, rp=1)

    filter_parameters = dict(sfreq=256,
                      method="iir",
                      iir_params=iir_params,
                      phase=phase,
                      verbose=False)

    filts = create_filter(data=None,
        l_freq=l_freq,
        h_freq=h_freq,
        **filter_parameters
    )  # creating iir filter

    # Initialize your FilterBankLayer
    filter_bank_layer = FilterBankLayer(
        n_chans=n_chans,
        band_filters=[(l_freq, h_freq)],
        **filter_parameters
    )
    filtered_signal_torch = filter_bank_layer(x)

    # Simulating filtfilt from torch with scipy
    x_np = x.numpy().astype(np.float64)
    filtered_scipy = _filfilt_in_torch_sytle(b=filts["b"], a=filts["a"],
                                            x_np=x_np)
    # Compare the outputs
    np.testing.assert_array_almost_equal(
        filtered_signal_torch.numpy().flatten(),
        filtered_scipy.flatten(),
        err_msg=f"Filtered outputs do not match between FilterBankLayer "
                f"and MNE-Python for and band=({l_freq}-{h_freq})Hz"
    )


@pytest.mark.parametrize("phase", ["linear", "zero", "zero-double",
                                   "minimum", "minimum-half"])
@pytest.mark.parametrize("fir_window", ["hamming", "hann"])
@pytest.mark.parametrize("fir_design", ["firwin", "firwin2"])
@pytest.mark.parametrize("l_freq, h_freq", [(4, 8), (8, 12), (13, 30)])
def test_filter_bank_layer_fftconvolve_comparison_fir(l_freq, h_freq, fir_design, fir_window, phase):#, fir_design, fir_window)
    """
    Test that the FilterBankLayer applies FIR filters correctly across multiple channels
    by comparing its output to scipy's fftconvolve.
    """
    # ---------------------------
    # 1. Setup Test Parameters
    # ---------------------------

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Define parameters
    batch_size = 1
    n_chans = 22
    n_times = 1000
    sfreq = 256  # Sampling frequency in Hz
    tolerance = 1e5
    # Generate random input tensor: Shape (batch_size, n_chans, n_times)
    x = torch.randn(batch_size, n_chans, n_times).float()

    filter_parameters = dict(
        sfreq=sfreq,
        method='fir',
        phase=phase,
        filter_length=1024, # If the filter length is not this side, nothing works
        l_trans_bandwidth=1.0,  # Narrower transition bandwidth
        h_trans_bandwidth=1.0,
        fir_window=fir_window,
        fir_design=fir_design,
        verbose=True)

    # Create FIR filter using MNE:
    filt = create_filter(data=None, l_freq=l_freq, h_freq=h_freq, **filter_parameters)

    # Expand filter to match channels: Shape (1, n_chans, filter_length)
    filt_expanded = torch.from_numpy(filt).unsqueeze(0).repeat(n_chans,
                                                               1).unsqueeze(0).float()

    # ---------------------------
    # 2. Initialize FilterBankLayer
    # ---------------------------

    # Initialize the FilterBankLayer with one band
    filter_bank_layer = FilterBankLayer(
        n_chans=n_chans,
        band_filters=[(l_freq, h_freq)],
        # Increased filter length for better frequency resolution
        **filter_parameters
    )

    # ---------------------------
    # 3. Apply FilterBankLayer
    # ---------------------------

    # Apply the filter bank to the input tensor
    # Expected output shape: (batch_size, n_bands, n_chans, n_times)
    filtered_output = filter_bank_layer(x)

    # Convert the FilterBankLayer output to NumPy for comparison
    filtered_torch_np = filtered_output.numpy()

    # Convert input tensor and filter to NumPy arrays
    x_numpy = x.numpy()  # Shape: (1, n_chans, n_times)
    filt_numpy = filt_expanded.numpy()  # Shape: (1, n_chans, filter_length)

    # Apply scipy's fftconvolve per Channel
    filtered_scipy = fftconvolve_scipy(x_numpy, filt_numpy, mode='same', axes=2)

    # Assert that all differences are below the tolerance
    assert np.allclose(filtered_torch_np, filtered_scipy, atol=tolerance), (
        f"Filtered outputs differ beyond the tolerance of {tolerance}."
    )



@pytest.mark.parametrize("method", ["fir"]) # "iir" is not working to 4-8, 8-12, 12-16. I think "sos" solve, but not implemented in Torch.
@pytest.mark.parametrize("l_freq, h_freq", [
    (4, 8), (8, 12), (12, 16), (16, 20),
    (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)
])
def test_filter_bank_layer_psd(l_freq, h_freq, method):
    """
    Test the FilterBankLayer by analyzing the power spectral density (PSD) of
    the output using MNE.
    """
    # Test parameters
    sfreq = 256  # Sampling frequency in Hz
    duration = 5  # Duration in seconds
    t = np.arange(0, duration, 1 / sfreq)  # Time vector

    # Generate a composite signal with multiple sine waves
    freqs = [2, 4, 6, 10, 14, 18, 22, 26, 30, 34, 38]
    signals = [np.sin(2 * np.pi * freq * t) for freq in freqs]
    composite_signal = np.sum(signals, axis=0)

    # Convert the signal to a torch tensor
    composite_signal_torch = torch.tensor(composite_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    # Initialize the FilterBankLayer
    filter_bank_layer = FilterBankLayer(
        n_chans=1,
        sfreq=sfreq,
        method=method,
        filter_length=1024, # Making an huge filter
        l_trans_bandwidth=1.0,  # Narrower transition bands
        h_trans_bandwidth=1.0, # Narrower transition bands
        band_filters=[(l_freq, h_freq)],
        verbose=False
    )

    # Apply the FilterBankLayer to the composite signal
    filtered_signals = filter_bank_layer(composite_signal_torch)
    # Shape after squeezing: (n_times,)
    filtered_signal_np = filtered_signals.squeeze().detach().numpy()

    # Compute PSD using MNE
    psds, freqs = psd_array_welch(
        filtered_signal_np,
        sfreq=sfreq,
        fmin=0,
        fmax=sfreq / 2,
        n_fft=1024,
        average='mean',
        verbose=False
    )

    # Identify frequencies within and outside the band
    idx_in_band = np.where((freqs >= l_freq) & (freqs <= h_freq))[0]
    idx_out_band = np.where((freqs < l_freq) | (freqs > h_freq))[0]

    # Calculate power in and out of the band
    power_in_band = np.sum(psds[idx_in_band])
    power_out_band = np.sum(psds[idx_out_band])

    # Assert that power in the band is significantly higher than outside
    assert power_in_band > 10 * power_out_band, \
        (f"Power in band {l_freq}-{h_freq} Hz is not significantly higher "
         f"than outside the band.")


def test_filter_bank_layer_frequency_response():
    """
    Test the FilterBankLayer by analyzing the frequency responses of its filters.
    """
    # Test parameters
    sfreq = 256  # Sampling frequency in Hz

    # Define the frequency bands for the filter bank
    band_filters = [(4, 8), (8, 12), (12, 16), (16, 20),
                    (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]

    # Initialize the FilterBankLayer
    filter_bank_layer = FilterBankLayer(
        n_chans=1,
        sfreq=sfreq,
        band_filters=band_filters,
        filter_length=1024,
        l_trans_bandwidth=1.0,  # Narrower transition bands
        h_trans_bandwidth=1.0,
        method='fir',
        phase='zero',
        fir_window='hamming',
        fir_design='firwin',
        verbose=False
    )

    num_fft = 1024  # Increase for higher frequency resolution

    # Prepare plots
    num_filters = len(band_filters)
    fig, axes = plt.subplots(num_filters, 1,
                             figsize=(10, 2 * num_filters),
                             sharex=True)

    # Iterate over each filter in the filter bank
    for idx, ((l_freq, h_freq), filt_dict, ax) in enumerate(zip(
            band_filters, filter_bank_layer.filts.values(), axes)):

        # Extract filter coefficients
        b = filt_dict['b'].detach().numpy()
        a = np.array([1.0])  # FIR filter, so a is [1.0]

        # Compute frequency response
        w, h = freqz(b, a, worN=num_fft, fs=sfreq)

        # Compute magnitude in dB
        h_dB = 20 * np.log10(np.abs(h) + 1e-12)  # Add epsilon to avoid log(0)

        # Plot frequency response
        ax.plot(w, h_dB, label=f'Band {l_freq}-{h_freq} Hz')
        ax.axvspan(l_freq, h_freq, color='red', alpha=0.3, label='Passband')
        ax.set_title(f'Frequency Response of Filter {idx+1}')
        ax.set_ylabel('Amplitude (dB)')
        ax.set_xlim(0, sfreq / 2)
        ax.set_ylim(-100, 5)
        ax.grid(True)
        ax.legend()

        # Programmatic verification
        # Define frequency ranges for passband and stopbands
        passband = (w >= l_freq) & (w <= h_freq)
        # Allow 2 Hz margin for transition bands, adjusted for higher frequencies
        margin = min(2.0, l_freq * 0.25)
        stopband_lower = w < (l_freq - margin)
        stopband_upper = w > (h_freq + margin)

        # Check that the gain in the passband is close to 0 dB
        passband_gain = h_dB[passband]
        assert np.all(passband_gain > -3), \
            f"Passband gain for filter {idx+1} is not within acceptable range."

        # Check that the gain in the stopbands is significantly attenuated
        stopband_gain = h_dB[stopband_lower | stopband_upper]
        # Adjust acceptable attenuation for higher frequencies
        attenuation_threshold = -40 if h_freq <= 20 else -30
        assert np.all(stopband_gain < attenuation_threshold), \
            f"Stopband attenuation for filter {idx+1} is not sufficient."


    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
