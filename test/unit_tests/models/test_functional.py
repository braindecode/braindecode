import numpy as np
import pytest
import torch

from braindecode.functional import hilbert_freq, plv_time


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(0)
    np.random.seed(0)


def test_hilbert_freq_shape_forward_fourier_true():
    """
    Test that hilbert_freq returns the correct shape when forward_fourier=True.
    Input shape: (batch, channels, seq_len)
    Output shape: (batch, channels, seq_len, 2)
    """
    batch, channels, seq_len = 2, 3, 100
    input_tensor = torch.randn(batch, channels, seq_len)
    output = hilbert_freq(input_tensor, forward_fourier=True)
    expected_shape = (batch, channels, seq_len, 2)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


def test_hilbert_freq_constant_signal():
    """
    Test hilbert_freq with a constant signal.
    The imaginary part of the Hilbert transform should be zero.
    """
    batch, channels, seq_len = 1, 2, 100
    input_tensor = torch.ones(batch, channels, seq_len)
    output = hilbert_freq(input_tensor, forward_fourier=True)
    # Imaginary parts should be close to zero
    assert torch.allclose(output[..., 1], torch.zeros_like(output[..., 1]), atol=1e-5), \
        "Imaginary part should be zero for constant input"


def test_plv_time_shape():
    """
    Test that plv_time returns the correct shape.
    Input shape: (batch, channels, time)
    Output shape: (batch, channels, channels)
    """
    batch, channels, time = 2, 4, 500
    input_tensor = torch.randn(batch, channels, time)
    plv_matrix = plv_time(input_tensor, forward_fourier=True)
    expected_shape = (batch, channels, channels)
    assert plv_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {plv_matrix.shape}"


def test_plv_time_perfect_synchronization():
    """
    Test plv_time with perfectly synchronized signals.
    The PLV matrix should be all ones.
    """
    batch, channels, time = 1, 3, 1000
    t = torch.linspace(0, 2 * np.pi, steps=time)
    signal = torch.sin(t)
    # Create identical signals across all channels
    input_tensor = signal.unsqueeze(0).repeat(batch, channels, 1)
    plv_matrix = plv_time(input_tensor, forward_fourier=True)
    expected = torch.ones(batch, channels, channels)
    assert torch.allclose(plv_matrix, expected, atol=1e-5), \
        "PLV should be 1 for perfectly synchronized signals"


def test_daubechies_filters_match_pywt():
    """daubechies_filters reproduces pywt's db-N decomposition filters to
    machine precision (skipped if PyWavelets is unavailable)."""
    pywt = pytest.importorskip("pywt")

    from braindecode.functional import daubechies_filters

    for n in (2, 3, 4, 6):
        filt = daubechies_filters(n)
        w = pywt.Wavelet(f"db{n}")
        assert filt.shape == (2, 2 * n)
        assert torch.allclose(filt[0], torch.tensor(w.dec_lo, dtype=torch.float32))
        assert torch.allclose(filt[1], torch.tensor(w.dec_hi, dtype=torch.float32))


def test_wavelet_decomposition_matches_pywt():
    """wavelet_decomposition is bit-identical to pywt/ptwt wavedec(mode='periodic')
    across sizes (skipped if the reference library is unavailable)."""
    ptwt = pytest.importorskip("ptwt")
    pywt = pytest.importorskip("pywt")

    from braindecode.functional import daubechies_filters, wavelet_decomposition

    w = pywt.Wavelet("db4")
    filt = daubechies_filters(4)
    for n in (64, 500, 2560, 5000):
        x = torch.randn(4, n)
        ref = torch.cat(
            ptwt.wavedec(x.unsqueeze(1), w, mode="periodic"), dim=-1
        ).squeeze(1)
        out = wavelet_decomposition(x, filt)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)
