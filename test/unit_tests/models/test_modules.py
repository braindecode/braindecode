# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)
import platform

import numpy as np
import pytest
import torch

from warnings import catch_warnings, simplefilter
from mne.filter import create_filter
from mne.time_frequency import psd_array_welch
from scipy.signal import fftconvolve as fftconvolve_scipy
from scipy.signal import freqz
from scipy.signal import lfilter as lfilter_scipy
from torch import nn

from braindecode.functional import drop_path
from braindecode.modules import (
    MLP,
    CombinedConv,
    DropPath,
    FilterBankLayer,
    LinearWithConstraint,
    SafeLog,
    TimeDistributed,
    GeneralizedGaussianFilter,
    CausalConv1d,
    MaxNormLinear,
)
from braindecode.models.labram import _SegmentPatch
from braindecode.models.tidnet import _BatchNormZG, _DenseSpatialFilter
from braindecode.models.ifnet import _SpatioTemporalFeatureBlock


def old_maxnorm(weight: torch.Tensor,
                max_norm_val: float = 2.0,
                eps: float = 1e-5) -> torch.Tensor:
    w = weight.clone()
    # clamp denominator ≥ max_norm_val/2, numerator ≤ max_norm_val
    denom  = w.norm(2, dim=0, keepdim=True).clamp(min=max_norm_val / 2)
    number  = denom.clamp(max=max_norm_val)
    return w * (number / (denom + eps))


class OldCausalConv1d(nn.Conv1d):
    """Causal 1-dimensional convolution
    Code modified from [1]_ and [2]_.
    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels (number of filters).
    kernel_size : int
        Kernel size.
    dilation : int, optional
        Dilation (number of elements to skip within kernel multiplication).
        Default to 1.
    **kwargs :
        Other keyword arguments to pass to torch.nn.Conv1d, except for
        `padding`!!
    References
    ----------
    .. [1] https://discuss.pytorch.org/t/causal-convolution/3456/4
    .. [2] https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    padding: torch.Tensor

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        **kwargs,
    ):
        assert "padding" not in kwargs, (
            "The padding parameter is controlled internally by "
            f"{type(self).__name__} class. You should not try to override this"
            " parameter."
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs,
        )

    def forward(self, X):
        out = super().forward(X)
        return out[..., : -self.padding[0]]


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
    "epilson, expected_repr",
    [
        (1e-6, "eps=1e-06"),
        (1e-4, "eps=0.0001"),
        (1e-8, "eps=1e-08"),
        (0.0, "eps=0.0"),
        (123.456, "eps=123.456"),
    ]
)
def test_safelog_extra_repr(epilson, expected_repr):
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
    module = SafeLog(epilson)

    # Get the extra representation
    repr_output = module.extra_repr()

    # Assert that the extra_repr output matches the expected string
    assert repr_output == expected_repr, f"Expected '{expected_repr}', got '{repr_output}'"



@pytest.fixture
def sample_input():
    """Create a sample input tensor."""
    batch_size = 2
    n_chans = 8
    time_points = 1000
    return torch.randn(batch_size, n_chans, time_points)


def test_default_band_filters(sample_input):
    """Test that default band_filters are set correctly when None is provided."""
    n_chans = 8
    sfreq = 100
    layer = FilterBankLayer(n_chans=n_chans, sfreq=sfreq, band_filters=None)

    expected_band_filters = [(low, low + 4) for low in range(4, 36 + 1, 4)]
    assert layer.band_filters == expected_band_filters, "Default band_filters not set correctly."
    assert layer.n_bands == 9, "Number of bands should be 9."

    output = layer(sample_input)
    assert output.shape[1] == layer.n_bands, "Output band dimension mismatch."
    assert output.shape[2] == n_chans, "Output channel dimension mismatch."


def test_band_filters_as_int_warning(sample_input):
    """Test that providing band_filters as int raises a warning and sets band_filters correctly."""
    n_chans = 8
    sfreq = 100
    band_filters_int = 9
    with pytest.warns(UserWarning,
                      match="Creating the filter banks equally divided"):
        layer = FilterBankLayer(n_chans=n_chans, sfreq=sfreq,
                                band_filters=band_filters_int)

    expected_intervals = torch.linspace(4, 40, steps=band_filters_int + 1)
    expected_band_filters = [(low.item(), high.item()) for low, high in
                             zip(expected_intervals[:-1],
                                 expected_intervals[1:])]

    assert layer.band_filters == expected_band_filters, "band_filters not correctly set from int."
    assert layer.n_bands == band_filters_int, "Number of bands should match the provided int."


def test_invalid_band_filters_raises_value_error():
    """Test that providing invalid band_filters raises a ValueError."""
    n_chans = 8
    sfreq = 100
    invalid_band_filters = "invalid_type"  # Not a list or int

    with pytest.raises(ValueError,
                       match="`band_filters` should be a list of tuples"):
        FilterBankLayer(n_chans=n_chans, sfreq=sfreq,
                        band_filters=invalid_band_filters)


def test_band_filters_none_defaults(sample_input):
    """
    Test that when band_filters is None and no n_bands or band_width are provided,
    the default 9 bands with 4Hz bandwidth are correctly initialized.
    """
    n_chans = 8
    sfreq = 100
    layer = FilterBankLayer(n_chans=n_chans, sfreq=sfreq, band_filters=None)

    # Define the expected default band_filters
    expected_band_filters = [(low, low + 4) for low in range(4, 36 + 1, 4)]

    # Assertions to verify band_filters and number of bands
    assert layer.band_filters == expected_band_filters, "Default band_filters not set correctly when band_filters=None."
    assert layer.n_bands == 9, "Number of bands should be 9 when band_filters=None."

    # Forward pass to ensure output shape is correct
    output = layer(sample_input)
    assert output.shape == (
        sample_input.shape[0], layer.n_bands, n_chans,
        sample_input.shape[2]
    ), "Output shape is incorrect when band_filters=None."


def test_band_filters_with_incorrect_tuple_length():
    """Test that providing band_filters with tuples not of length 2 raises a ValueError."""
    n_chans = 8
    sfreq = 100
    invalid_band_filters = [(4, 8), (12,)]  # Second tuple has only one element

    with pytest.raises(ValueError,
                       match="The band_filters items should be splitable in 2 values"):
        FilterBankLayer(n_chans=n_chans, sfreq=sfreq,
                        band_filters=invalid_band_filters)


def test_iir_params_output_sos_warning(sample_input):
    """Test that providing iir_params with output='sos' raises a warning and modifies the output."""
    n_chans = 8
    sfreq = 100
    iir_params = {"output": "sos"}

    with pytest.warns(UserWarning,
                      match="It is not possible to use second-order section"):
        layer = FilterBankLayer(
            n_chans=n_chans,
            sfreq=sfreq,
            band_filters=None,
            method="iir",
            iir_params=iir_params
        )

    assert layer.a_list is not None, "Filters should be initialized."
    assert layer.b_list is not None, "Filters should be initialized."

    assert layer.a_list[0].dtype == torch.float64, "Filter coefficients should be float64."


@pytest.mark.parametrize('method', ['iir', 'fir'])
def test_forward_pass_filter_bank(method, sample_input):
    """Test the forward pass of the FilterBankLayer with IIR filtering."""
    n_chans = 8
    sfreq = 100
    layer = FilterBankLayer(
        n_chans=n_chans,
        sfreq=sfreq,
        band_filters=None,
        method=method,
    )

    output = layer(sample_input)
    assert output.shape == (
        sample_input.shape[0], layer.n_bands, n_chans, sample_input.shape[2]
    ), f"Output shape is incorrect for {method} filtering."


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
def test_filter_bank_layer_fftconvolve_comparison_fir(l_freq, h_freq,
                                                      fir_design, fir_window,
                                                      phase):
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
    # Iterate over each filter in the filter bank
    for idx, ((l_freq, h_freq), b_value) in enumerate(zip(
            band_filters, filter_bank_layer.b_list)):
        # Extract filter coefficients
        b = b_value.detach().numpy()
        a = np.array([1.0])  # FIR filter, so a is [1.0]

        # Compute frequency response
        w, h = freqz(b, a, worN=num_fft, fs=sfreq)

        # Compute magnitude in dB
        h_dB = 20 * np.log10(np.abs(h) + 1e-12)  # Add epsilon to avoid log(0)

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


def test_initialization_valid_parameters():
    """
    Test that the GeneralizedGaussianFilter initializes correctly with valid parameters.
    """
    in_channels = 2
    sequence_length = 256
    sample_rate = 100.0  # Hz
    filter_layer = GeneralizedGaussianFilter(
        in_channels=in_channels,
        out_channels=in_channels,
        sequence_length=sequence_length,
        sample_rate=sample_rate,
        inverse_fourier=True,
    )
    assert isinstance(filter_layer, nn.Module), "Filter layer should be an instance of nn.Module"


def test_initialization_invalid_parameters():
    """
    Test that the GeneralizedGaussianFilter raises an assertion error when initialized with invalid parameters.
    """
    in_channels = 2
    out_channels = 5  # Not a multiple of in_channels
    sequence_length = 256
    sample_rate = 100.0  # Hz
    with pytest.raises(AssertionError):
        GeneralizedGaussianFilter(
            in_channels=in_channels,
            out_channels=out_channels,  # Should raise an error
            sequence_length=sequence_length,
            sample_rate=sample_rate,
            inverse_fourier=True,
            f_mean=(10.0, 20.0),
            bandwidth=(5.0, 10.0),
            shape=(2.0, 2.5)
        )

def test_filter_construction_clamping():
    """
    Test that the filter parameters are clamped correctly during filter construction.
    """
    in_channels = 1
    out_channels = 1
    sequence_length = 256
    sample_rate = 100.0  # Hz
    filter_layer = GeneralizedGaussianFilter(
        in_channels=in_channels,
        out_channels=out_channels,
        sequence_length=sequence_length,
        sample_rate=sample_rate,
        f_mean=(50.0,),  # Above the clamp maximum
        bandwidth=(0.5,),  # Below the clamp minimum
        shape=(1.5,)  # Below the clamp minimum
    )
    filter_layer.construct_filters()
    f_mean_clamped = filter_layer.f_mean.data.item() * (sample_rate / 2)
    bandwidth_clamped = filter_layer.bandwidth.data.item() * (sample_rate / 2)
    shape_clamped = filter_layer.shape.data.item()
    assert f_mean_clamped <= 45.0, "f_mean should be clamped to a maximum of 45.0 Hz"
    assert np.round(bandwidth_clamped) >= 1.0, "bandwidth should be clamped to a minimum of 1.0 Hz"
    assert shape_clamped >= 2.0, "shape should be clamped to a minimum of 2.0"

def test_forward_pass_output_shape():
    """
    Test that the forward pass returns the correct output shape.
    """
    batch_size = 3
    in_channels = 2
    out_channels = 4  # Must be a multiple of in_channels
    sequence_length = 256
    sample_rate = 100.0  # Hz
    filter_layer = GeneralizedGaussianFilter(
        in_channels=in_channels,
        out_channels=out_channels,
        sequence_length=sequence_length,
        sample_rate=sample_rate,
        inverse_fourier=True,
        f_mean=(10.0, 20.0),
        bandwidth=(5.0, 10.0),
        shape=(2.0, 2.5),
        group_delay = (10.0, 20.0)
    )
    input_tensor = torch.randn(batch_size, in_channels, sequence_length)
    output = filter_layer(input_tensor)
    expected_shape = (batch_size, out_channels, sequence_length)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"


def test_forward_pass_no_inverse_fourier():
    """
    Test the forward pass when inverse_fourier is set to False, ensuring the output is in the frequency domain.
    """
    batch_size = 2
    in_channels = 1
    out_channels = 2
    sequence_length = 128
    sample_rate = 100.0  # Hz
    filter_layer = GeneralizedGaussianFilter(
        in_channels=in_channels,
        out_channels=out_channels,
        sequence_length=sequence_length,
        sample_rate=sample_rate,
        inverse_fourier=False,
        f_mean=(15.0, 30.0),
        bandwidth=(5.0, 5.0),
        shape=(2.0, 2.0)
    )
    input_tensor = torch.randn(batch_size, in_channels, sequence_length)
    output = filter_layer(input_tensor)
    # Since inverse_fourier=False, output should be in frequency domain
    freq_bins = sequence_length // 2 + 1
    expected_shape = (batch_size, out_channels, freq_bins, 2)  # Last dimension is real and imaginary parts
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    # Verify that output is real-valued (since it's the real and imaginary parts)
    assert output.dtype == torch.float32 or output.dtype == torch.float64, "Output should be real-valued tensor"

@pytest.fixture
def input_linear_constraint():
    torch.manual_seed(0)
    return torch.randn(10, 5)  # Batch size of 10, input features of 5


@pytest.fixture
def layer_with_constraint():
    torch.manual_seed(0)
    return LinearWithConstraint(in_features=5, out_features=3, max_norm=1.0)


def test_weight_norm_constraint(input_linear_constraint, layer_with_constraint):
    """
    Test whether the weight norms do not exceed max_norm after forward pass.
    """
    layer_with_constraint(input_linear_constraint)
    # Calculate the L2 norm of each column (dim=0)
    weight_norms = layer_with_constraint.weight.data.norm(p=2, dim=1)
    assert torch.all(weight_norms <= layer_with_constraint.max_norm + 1e-6), (
        f"Weight norms {weight_norms} exceed max_norm {layer_with_constraint.max_norm}"
    )


def test_no_constraint_if_within_norm():
    """
    Test that weights within the max_norm are not altered after forward pass.
    """
    in_features = 3
    out_features = 2
    max_norm = 2.0
    layer = LinearWithConstraint(in_features, out_features, max_norm)

    # Initialize weights with norms less than or equal to max_norm
    with torch.no_grad():
        layer.weight.data = torch.tensor([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0]])

    input = torch.randn(1, in_features)
    original_weights = layer.weight.data.clone()
    layer(input)

    # Weights should remain unchanged
    assert torch.allclose(layer.weight.data, original_weights), "Weights were altered despite being within max_norm"


@pytest.mark.parametrize("max_norm", [0.5, 1.0, 2.0])
def test_max_norm_parameter(max_norm):
    """
    Test that different max_norm values are respected.
    """
    in_features = 3
    out_features = 2
    layer = LinearWithConstraint(in_features=in_features, out_features=out_features, max_norm=max_norm)
    with torch.no_grad():
        layer.weight.data = torch.tensor([[3.0, 0.0, 0.0],
                                         [0.0, 4.0, 0.0]])
    input = torch.randn(1, in_features)
    layer(input)
    weight_norms = layer.weight.data.norm(p=2, dim=1)
    assert torch.all(weight_norms <= max_norm + 1e-6), (
        f"For max_norm {max_norm}, weight norms {weight_norms} exceed max_norm"
    )



@pytest.mark.parametrize("out_in", [
    (4,  8),
    (8, 16),
    (16,32),
])
def test_new_vs_old_maxnorm_are_identical(out_in):
    out_features, in_features = out_in
    torch.manual_seed(0)
    W = torch.randn(out_features, in_features, requires_grad=True)

    W_ref = old_maxnorm(W, max_norm_val=2.0, eps=1e-5)

    layer = MaxNormLinear(
        in_features=in_features,
        out_features=out_features,
        max_norm_val=2.0,
        eps=1e-5,
        bias=True,
    )

    layer.weight = W.clone()

    W_new = layer.weight

    assert torch.allclose(W_ref, W_new, atol=1e-6), (
        f"MaxNorm mismatch: max|W_ref - W_new| = {(W_ref - W_new).abs().max():.3e}"
    )


@pytest.mark.parametrize("batch,in_ch,out_ch,length,kernel,dilation", [
    (1, 1, 1, 20, 3, 1),
    (2, 3, 4, 50, 5, 2),
    (4, 2, 2, 30, 7, 3),
])
def test_matches_padded_conv(batch, in_ch, out_ch, length, kernel, dilation):
    torch.manual_seed(0)
    # instantiate causal conv
    causal_new = CausalConv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, bias=True)
    # clone weights/bias for reference conv
    causal_old = OldCausalConv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, bias=True)

    causal_old.weight.data.copy_(causal_new.weight.data)
    causal_old.bias.data.copy_(causal_new.bias.data)

    # random input
    x = torch.randn(batch, in_ch, length)

    # causal output
    y_causal = causal_new(x)
    # reference: padded conv then trim right
    y_ref = causal_old(x)[..., : length]

    assert y_causal.shape == (batch, out_ch, length)
    assert torch.allclose(y_causal, y_ref, atol=1e-6), "CausalConv1d differs from trimmed padded Conv1d"




@pytest.mark.parametrize("batch,in_ch,out_ch,length,kernel,dilation", [
    (1, 1, 1, 20, 3, 1),
    (2, 3, 4, 50, 5, 2),
])
def test_gradients_match_casual_conv(batch, in_ch, out_ch, length, kernel, dilation):
    torch.manual_seed(0)
    # new implementation
    causal_new = CausalConv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, bias=True)
    # old implementation
    causal_old = OldCausalConv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, bias=True)

    # sync parameters
    causal_old.weight.data.copy_(causal_new.weight.data)
    causal_old.bias.data.copy_(causal_new.bias.data)

    # random input with gradient tracking
    x_new = torch.randn(batch, in_ch, length, requires_grad=True)
    x_old = x_new.clone().detach().requires_grad_(True)

    # forward + backward new
    y_new = causal_new(x_new)
    loss_new = y_new.sum()
    loss_new.backward()

    # forward + backward old
    y_old = causal_old(x_old)[..., :length]
    loss_old = y_old.sum()
    loss_old.backward()

    # compare input gradients
    assert torch.allclose(x_new.grad, x_old.grad, atol=1e-6), \
        "Input gradients differ between new and old implementations"

    # compare weight gradients
    assert torch.allclose(causal_new.weight.grad, causal_old.weight.grad, atol=1e-6), \
        "Weight gradients differ between new and old implementations"

    # compare bias gradients
    assert torch.allclose(causal_new.bias.grad, causal_old.bias.grad, atol=1e-6), \
        "Bias gradients differ between new and old implementations"


@pytest.mark.parametrize("n_bands,kernel_sizes,expected_warning", [
    (2, [63], "Reducing number of bands"),         # n_bands > len(kernel_sizes)
    (1, [63, 31], "Reducing number of kernels"),   # n_bands < len(kernel_sizes)
    (1, [63], "not divisible by"),                 # n_times % stride_factor != 0
])
def test_warning_conditions(n_bands, kernel_sizes, expected_warning):
    with catch_warnings(record=True) as w:
        simplefilter("always")
        block = _SpatioTemporalFeatureBlock(
            n_times=130,                  # not divisible by 16
            in_channels=4,
            out_channels=8,
            kernel_sizes=kernel_sizes,
            n_bands=n_bands,
            stride_factor=16
        )
        assert any(expected_warning in str(warn.message) for warn in w)


def test_forward_pass_ifnet_output_shape():
    block = _SpatioTemporalFeatureBlock(
        n_times=128,                     # divisible by stride_factor
        in_channels=4,
        out_channels=8,
        kernel_sizes=[63, 31],
        n_bands=2,
        stride_factor=16
    )
    x = torch.randn(2, 4, 128)           # batch_size=2
    out = block(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2            # batch_size preserved
