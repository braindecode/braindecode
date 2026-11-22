"""Real-valued DFT helpers for accelerators without complex tensors.

Intel Gaudi (HPU) has no complex dtype, so ``torch.fft`` with complex outputs
cannot run there. These helpers express the one-sided DFT, and the inverse
transforms built on top of it, as matmuls against a cosine/sine basis. Under
bfloat16 autocast those matmuls would run in bfloat16, whereas ``torch.fft`` is
never downcast, so callers wrap the numerically sensitive parts with
:func:`fp32_island`.
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import functools
import math
from typing import Optional, Tuple

import torch
from torch import nn


def needs_real_dft(x: torch.Tensor) -> bool:
    """Whether ``x`` lives on a device without complex-tensor FFT support."""
    return x.device.type == "hpu"


@torch.jit.unused
def _matmul_no_autocast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=a.device.type, enabled=False):
        return torch.matmul(a, b)


@torch.jit.unused
def _bmm_no_autocast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=a.device.type, enabled=False):
        return torch.bmm(a, b)


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix product with autocast disabled in eager/compiled execution."""
    if torch.jit.is_scripting():
        return torch.matmul(a, b)
    return _matmul_no_autocast(a, b)


def _bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matrix product with autocast disabled in eager/compiled execution."""
    if torch.jit.is_scripting():
        return torch.bmm(a, b)
    return _bmm_no_autocast(a, b)


def _upcast_low_precision(x: torch.Tensor) -> torch.Tensor:
    """Promote float16/bfloat16 inputs to float32 while keeping float32/float64."""
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        return x.to(torch.float32)
    return x


def fp32_island(fn):
    """Run ``fn`` with autocast disabled, in at least float32 precision.

    Any bfloat16/float16 tensor argument is upcast to float32 before ``fn``
    runs; float32 and float64 arguments are left untouched, so the same code
    path can also be checked against a float64 ``torch.fft`` reference.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        ref = next(a for a in args if isinstance(a, torch.Tensor))
        with torch.autocast(device_type=ref.device.type, enabled=False):
            args = tuple(
                a.float()
                if isinstance(a, torch.Tensor)
                and a.is_floating_point()
                and torch.finfo(a.dtype).bits < 32
                else a
                for a in args
            )
            return fn(*args, **kwargs)

    return wrapped


def real_dft_basis(
    n: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-sided cosine/sine basis of an ``n``-point DFT, shape ``(n // 2 + 1, n)``.

    The phase index is reduced modulo ``n`` in integers before scaling, so the
    DC row is exact (``sin`` is exactly zero there); the Nyquist row (present
    for even ``n``) is exact to floating-point precision.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if (n // 2) * (n - 1) >= 2**31:
        # The int32 phase index below would overflow and give wrong phases.
        raise ValueError(f"n={n} exceeds the 65536 samples of the real-valued DFT")
    freqs = torch.arange(n // 2 + 1, device=device, dtype=torch.int32)
    samples = torch.arange(n, device=device, dtype=torch.int32)
    phase_index = (freqs.unsqueeze(1) * samples.unsqueeze(0)) % n
    angles = 2.0 * math.pi * phase_index.to(dtype) / n
    return angles.cos(), angles.sin()


def real_rfft(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real and imaginary parts of ``torch.fft.rfft(x, dim=-1)`` without complex tensors."""
    work = _upcast_low_precision(x)
    n = work.shape[-1]
    cosine, sine = real_dft_basis(n, device=work.device, dtype=work.dtype)
    return _matmul(work, cosine.transpose(0, 1)), -_matmul(work, sine.transpose(0, 1))


def real_irfft(real: torch.Tensor, imag: torch.Tensor, n: int) -> torch.Tensor:
    """``torch.fft.irfft(torch.complex(real, imag), n=n, dim=-1)`` without complex tensors."""
    real_w = _upcast_low_precision(real)
    imag_w = _upcast_low_precision(imag)
    if real_w.shape[-1] != n // 2 + 1:
        raise ValueError(
            f"expected {n // 2 + 1} one-sided bins for n={n}, got {real_w.shape[-1]}"
        )
    cosine, sine = real_dft_basis(n, device=real_w.device, dtype=real_w.dtype)
    output = _matmul(real_w[..., :1], cosine[:1, :])
    if n % 2 == 0:
        stop = real_w.shape[-1] - 1
        output = output + _matmul(real_w[..., -1:], cosine[-1:, :])
    else:
        stop = real_w.shape[-1]
    if stop > 1:
        output = output + 2.0 * (
            _matmul(real_w[..., 1:stop], cosine[1:stop, :])
            - _matmul(imag_w[..., 1:stop], sine[1:stop, :])
        )
    return output / n


def real_analytic_ifft(
    real: torch.Tensor, imag: torch.Tensor, n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse ``n``-point FFT of a one-sided spectrum, without complex tensors.

    ``real``/``imag`` hold the bins ``0 .. n // 2`` of a length-``n`` spectrum
    whose remaining (negative-frequency) bins are zero, so this is a plain
    (non-Hermitian) inverse DFT, not :func:`real_irfft`. Returns the real and
    imaginary parts of the generally complex time-domain signal, each of shape
    ``(..., n)``. This is the last step of :func:`hilbert_freq_real`.
    """
    real_w = _upcast_low_precision(real)
    imag_w = _upcast_low_precision(imag)
    if real_w.shape[-1] != n // 2 + 1:
        raise ValueError(
            f"expected {n // 2 + 1} one-sided bins for n={n}, got {real_w.shape[-1]}"
        )
    cosine, sine = real_dft_basis(n, device=real_w.device, dtype=real_w.dtype)
    out_real = (_matmul(real_w, cosine) - _matmul(imag_w, sine)) / n
    out_imag = (_matmul(real_w, sine) + _matmul(imag_w, cosine)) / n
    return out_real, out_imag


def hilbert_freq_real(
    x: torch.Tensor, forward_fourier: bool, seq_len: int, keep_nyquist: bool
) -> torch.Tensor:
    """Real-valued equivalent of :func:`braindecode.functional.hilbert_freq`."""
    orig_dtype = x.dtype
    work = _upcast_low_precision(x)
    if not work.is_floating_point():
        work = work.to(torch.float32)
    if forward_fourier:
        real, imag = real_rfft(work)
    else:
        real, imag = work[..., 0], work[..., 1]
    scale = torch.full((real.shape[-1],), 2.0, dtype=real.dtype, device=real.device)
    scale[0] = 1.0  # Don't multiply the DC-term by 2
    if keep_nyquist:
        scale[-1] = 1.0  # Nor the Nyquist term, which only exists for even lengths
    out_real, out_imag = real_analytic_ifft(real * scale, imag * scale, n=seq_len)
    out = torch.stack((out_real, out_imag), dim=-1)
    return out.to(orig_dtype) if x.is_floating_point() else out


def fft_conv1d_real(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Real-valued equivalent of :func:`braindecode.functional.fft_conv1d`.

    Overlap-save: the input is cut into blocks of ``n`` samples (the power of
    two ``>= 2 * kernel_size``) that advance by ``n - kernel_size + 1`` outputs,
    each block's DFT is a matmul against the cosine/sine basis, and per
    frequency the complex product with the kernels, summed over the input
    channels, is one real matmul.
    """
    work_x = _upcast_low_precision(x)
    work_w = _upcast_low_precision(weight).to(device=work_x.device, dtype=work_x.dtype)
    n_times, kernel_size = work_x.shape[-1], work_w.shape[-1]
    n = 1
    while n < 2 * kernel_size:
        n = n * 2
    hop = n - kernel_size + 1
    n_blocks = -(-n_times // hop)
    n_freqs = n // 2 + 1
    # "same" alignment: torch pads the extra zero of an even kernel on the right
    left = (kernel_size - 1) // 2
    padded = nn.functional.pad(work_x, [left, (n_blocks + 1) * hop - n_times - left])
    hops = padded.unflatten(-1, (n_blocks + 1, hop))
    # block i = hop i + the first kernel_size - 1 samples of hop i + 1
    blocks = torch.cat([hops[:, :, :-1], hops[:, :, 1:, : kernel_size - 1]], dim=-1)
    cosine, sine = real_dft_basis(n, device=work_x.device, dtype=work_x.dtype)
    basis = torch.cat([cosine, -sine], dim=0).transpose(0, 1)  # -> [Re | Im] of rfft
    x_f = _matmul(blocks, basis)  # (batch, in, blocks, 2 * n_freqs)
    w_f = _matmul(nn.functional.pad(work_w, [0, n - kernel_size]), basis)
    w_re, w_im = w_f[..., :n_freqs], w_f[..., n_freqs:]
    # conv1d is a cross-correlation: x_f * conj(w_f), summed over the input
    # channels. Per frequency [Re x, Im x] @ [[Re w, -Im w], [Im w, Re w]].
    w_mat = torch.cat(
        [torch.cat([w_re, -w_im], dim=0), torch.cat([w_im, w_re], dim=0)], dim=1
    )
    x_f_perm = x_f.unflatten(-1, (2, n_freqs)).permute(4, 0, 2, 3, 1).flatten(1, 2)
    y_f = _bmm(x_f_perm.flatten(2), w_mat.permute(2, 1, 0))  # (freq, ., 2 * out)
    y_f = y_f.unflatten(1, (work_x.shape[0], n_blocks)).unflatten(-1, (2, -1))
    y_f = y_f.permute(1, 4, 2, 3, 0).flatten(-2)  # (batch, out, blocks, 2 * n_freqs)
    # inverse DFT of the first `hop` samples, the ones without wrap-around
    scale = torch.full((n_freqs, 1), 2.0 / n, dtype=work_x.dtype, device=work_x.device)
    scale[0] = 1.0 / n
    scale[-1] = 1.0 / n  # Nyquist (n is even)
    inverse = torch.cat([scale * cosine[:, :hop], -scale * sine[:, :hop]], dim=0)
    y = _matmul(y_f, inverse).flatten(-2)[..., :n_times]
    if bias is not None:
        y = y + bias.to(device=y.device, dtype=y.dtype).unsqueeze(-1)
    return y.to(device=x.device, dtype=x.dtype)
