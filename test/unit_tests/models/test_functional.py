import numpy as np
import pytest
import torch
from scipy.fft import next_fast_len
from scipy.signal import hilbert

from braindecode.functional import (
    _real_dft,
    fft_conv1d,
    hilbert_freq,
    plv_time,
    rotate_pairs,
    sinusoidal_positional_encoding,
)


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


@pytest.mark.parametrize("seq_len", [7, 8, 101, 128])
def test_hilbert_freq_matches_scipy(seq_len):
    """hilbert_freq keeps the input length and matches scipy.signal.hilbert
    for odd and even lengths."""
    t = np.arange(seq_len)
    x = np.stack([np.sin(0.3 * t) + 0.1 * t, np.cos(1.7 * t) - 0.05 * t])
    output = hilbert_freq(torch.from_numpy(x), forward_fourier=True)
    assert output.shape == (2, seq_len, 2)
    expected = hilbert(x, axis=-1)
    np.testing.assert_allclose(output[..., 0].numpy(), expected.real, atol=1e-10)
    np.testing.assert_allclose(output[..., 1].numpy(), expected.imag, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int16])
def test_hilbert_freq_dtype(dtype):
    """Floating input keeps its dtype; integer input returns float32, not truncated."""
    x = (torch.randn(2, 64) * 100).to(dtype)
    out = hilbert_freq(x)
    assert out.dtype == (dtype if dtype.is_floating_point else torch.float32)
    expected = hilbert_freq(x.float())
    torch.testing.assert_close(out.float(), expected, atol=0.5, rtol=1e-2)


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


@pytest.mark.parametrize("forward_fourier", [False, True])
def test_hilbert_freq_bfloat16_matches_float32(forward_fourier):
    """The complex-valued Hilbert path accepts BF16 coefficients.

    PyTorch does not support BF16 inputs to ``view_as_complex`` (and some
    backends also lack BF16 FFT kernels), so the implementation computes this
    narrow part in FP32 while retaining the real-valued input dtype contract.
    """
    if forward_fourier:
        input_tensor = torch.randn(2, 3, 32)
    else:
        input_tensor = torch.randn(2, 3, 17, 2)
    bfloat16_input = input_tensor.to(torch.bfloat16)

    output = hilbert_freq(bfloat16_input, forward_fourier=forward_fourier)
    reference = hilbert_freq(
        bfloat16_input.float(), forward_fourier=forward_fourier
    )

    assert output.dtype == torch.bfloat16
    assert torch.allclose(output.float(), reference, atol=2e-2, rtol=2e-2)


def test_hilbert_freq_bfloat16_backward():
    """The FP32 complex boundary remains differentiable for BF16 inputs."""
    input_tensor = torch.randn(2, 3, 17, 2, dtype=torch.bfloat16, requires_grad=True)

    output = hilbert_freq(input_tensor, forward_fourier=False)
    output.square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert input_tensor.grad is not None
    assert input_tensor.grad.dtype == torch.bfloat16
    assert torch.isfinite(input_tensor.grad).all()


def test_plv_time_bfloat16_matches_float32():
    """PLV can consume the BF16 Fourier coefficients used by EEGMiner."""
    input_tensor = torch.randn(2, 3, 17, 2)
    bfloat16_input = input_tensor.to(torch.bfloat16)

    output = plv_time(bfloat16_input, forward_fourier=False)
    reference = plv_time(bfloat16_input.float(), forward_fourier=False)

    assert output.dtype == torch.bfloat16
    assert torch.allclose(output.float(), reference, atol=2e-2, rtol=2e-2)


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


def test_sinusoidal_positional_encoding_even_dim():
    """Shape, contiguity, and the defining sin/cos values at position 0."""
    pe = sinusoidal_positional_encoding(50, 16)
    assert pe.shape == (50, 16)
    assert pe.is_contiguous()
    # position 0: sin(0)=0 on even channels, cos(0)=1 on odd channels.
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0] * 8))


def test_sinusoidal_positional_encoding_odd_dim_truncates_contiguously():
    """Odd dim is computed on the next even width, truncated, and contiguous
    (a non-contiguous view would break safetensors buffer saving)."""
    pe_odd = sinusoidal_positional_encoding(50, 15)
    assert pe_odd.shape == (50, 15)
    assert pe_odd.is_contiguous()
    assert torch.equal(pe_odd, sinusoidal_positional_encoding(50, 16)[:, :15])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8,), (2, 3, 8), (2, 0, 8)])
def test_rotate_pairs(dtype, shape):
    x = torch.randn(shape, dtype=dtype).transpose(0, -1).contiguous().transpose(0, -1)
    x.requires_grad_()
    expected = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    actual = rotate_pairs(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.jit.script(rotate_pairs)(x), expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    grad = torch.autograd.grad((actual * weights).sum(), x)[0]
    ref_grad = torch.autograd.grad((expected * weights).sum(), x)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


def test_rotate_pairs_rejects_odd_width():
    with pytest.raises(RuntimeError):
        rotate_pairs(torch.zeros(2, 3))


@pytest.mark.parametrize("n", [7, 8, 120, 121])
def test_real_rfft_and_irfft_match_torch_fft(n):
    """The matmul-based one-sided DFT (used on devices without complex
    tensors) matches ``torch.fft.rfft``/``irfft`` to float64 precision."""
    x = torch.randn(3, 5, n, dtype=torch.float64)
    real, imag = _real_dft.real_rfft(x)
    ref = torch.fft.rfft(x, dim=-1)
    torch.testing.assert_close(real, ref.real, rtol=0, atol=1e-10)
    torch.testing.assert_close(imag, ref.imag, rtol=0, atol=1e-10)
    back = _real_dft.real_irfft(real, imag, n=n)
    torch.testing.assert_close(
        back, torch.fft.irfft(ref, n=n, dim=-1), rtol=0, atol=1e-10
    )


@pytest.mark.parametrize("forward_fourier,n", [(True, 120), (True, 121), (False, 120)])
def test_hilbert_freq_real_dft_path_matches_complex_path(monkeypatch, forward_fourier, n):
    """Forcing the real-valued DFT path (as on HPU) reproduces the
    ``torch.fft``-based path exactly, for odd and even lengths and for both
    ``forward_fourier`` values."""
    x = torch.randn(4, 6, n, dtype=torch.float64)
    if not forward_fourier:
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1))
    expected = hilbert_freq(x, forward_fourier)
    monkeypatch.setattr(_real_dft, "needs_real_dft", lambda t: True)
    actual = hilbert_freq(x, forward_fourier)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def test_real_dft_path_ignores_bf16_autocast(monkeypatch):
    """The real-valued DFT path used on HPU must not be run in bfloat16 by an
    ambient autocast context: both the forward values and the gradients must
    match the same path computed with autocast disabled."""
    x = torch.randn(2, 4, 120, requires_grad=True)
    monkeypatch.setattr(_real_dft, "needs_real_dft", lambda t: True)

    fp32 = hilbert_freq(x, True)
    fp32.square().sum().backward()
    fp32_grad = x.grad.clone()
    x.grad = None

    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = hilbert_freq(x, True)
    mixed.square().sum().backward()

    torch.testing.assert_close(mixed.float(), fp32, rtol=0, atol=1e-6)
    torch.testing.assert_close(x.grad, fp32_grad, rtol=0, atol=1e-6)


def test_real_dft_basis_rejects_lengths_that_overflow_the_phase_index():
    # Check the accepted boundary without allocating its quadratic-size basis.
    cosine, sine = _real_dft.real_dft_basis(65536, device="meta", dtype=torch.float32)
    assert cosine.shape == sine.shape == (32769, 65536)
    with pytest.raises(ValueError, match="65536"):
        _real_dft.real_dft_basis(65537, device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("kernel_size", [7, 8])
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_fft_conv1d_matches_conv1d_same(kernel_size, dtype, tol):
    """Odd and even kernels, aligned like ``padding="same"``; gradients too."""
    x = torch.randn(3, 4, 50, dtype=dtype, requires_grad=True)
    w = torch.randn(5, 4, kernel_size, dtype=dtype, requires_grad=True)
    b = torch.randn(5, dtype=dtype, requires_grad=True)
    ref = torch.nn.functional.conv1d(x, w, b, padding="same")
    out = fft_conv1d(x, w, b)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=tol * ref.abs().max().item())
    g = torch.randn_like(ref)
    got = torch.autograd.grad(out, (x, w, b), g)
    want = torch.autograd.grad(ref, (x, w, b), g)
    for a, e in zip(got, want):
        torch.testing.assert_close(a, e, rtol=0, atol=tol * e.abs().max().item())
    half = fft_conv1d(x.detach().bfloat16(), w.detach().bfloat16())
    assert half.dtype == torch.bfloat16
    ref = fft_conv1d(x.detach().bfloat16().float(), w.detach().bfloat16().float())
    torch.testing.assert_close(half, ref.bfloat16(), rtol=0, atol=0)


def test_fft_len_matches_scipy():
    from braindecode.functional.functions import _next_fast_len

    assert all(_next_fast_len(n) == next_fast_len(n, real=True) for n in range(1, 5000))
