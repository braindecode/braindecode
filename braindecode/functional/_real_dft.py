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
from typing import Optional

import torch
from torch import nn


def needs_real_dft(x: torch.Tensor) -> bool:
    """Whether ``x`` lives on a device without complex-tensor FFT support."""
    return x.device.type == "hpu"


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


def real_dft_basis(n, *, device, dtype):
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
    phase_index = (freqs[:, None] * samples[None, :]) % n
    angles = 2 * torch.pi * phase_index.to(dtype) / n
    return angles.cos(), angles.sin()


def real_rfft(x):
    """Real and imaginary parts of ``torch.fft.rfft(x, dim=-1)`` without complex tensors."""
    n = x.shape[-1]
    cosine, sine = real_dft_basis(n, device=x.device, dtype=x.dtype)
    return x @ cosine.transpose(0, 1), -(x @ sine.transpose(0, 1))


def real_irfft(real, imag, *, n):
    """``torch.fft.irfft(torch.complex(real, imag), n=n, dim=-1)`` without complex tensors."""
    if real.shape[-1] != n // 2 + 1:
        raise ValueError(
            f"expected {n // 2 + 1} one-sided bins for n={n}, got {real.shape[-1]}"
        )
    cosine, sine = real_dft_basis(n, device=real.device, dtype=real.dtype)
    output = real[..., :1] @ cosine[:1, :]
    if n % 2 == 0:
        interior = slice(1, -1)
        output = output + real[..., -1:] @ cosine[-1:, :]
    else:
        interior = slice(1, None)
    if real.shape[-1] > (2 if n % 2 == 0 else 1):
        output = output + 2 * (
            real[..., interior] @ cosine[interior, :]
            - imag[..., interior] @ sine[interior, :]
        )
    return output / n


def real_analytic_ifft(real, imag, *, n):
    """Inverse ``n``-point FFT of a one-sided spectrum, without complex tensors.

    ``real``/``imag`` hold the bins ``0 .. n // 2`` of a length-``n`` spectrum
    whose remaining (negative-frequency) bins are zero, so this is a plain
    (non-Hermitian) inverse DFT, not :func:`real_irfft`. Returns the real and
    imaginary parts of the generally complex time-domain signal, each of shape
    ``(..., n)``. This is the last step of :func:`hilbert_freq_real`.
    """
    if real.shape[-1] != n // 2 + 1:
        raise ValueError(
            f"expected {n // 2 + 1} one-sided bins for n={n}, got {real.shape[-1]}"
        )
    cosine, sine = real_dft_basis(n, device=real.device, dtype=real.dtype)
    out_real = (real @ cosine - imag @ sine) / n
    out_imag = (real @ sine + imag @ cosine) / n
    return out_real, out_imag


@torch.jit.unused
def hilbert_freq_real(x: torch.Tensor, forward_fourier: bool) -> torch.Tensor:
    """Real-valued equivalent of :func:`braindecode.functional.hilbert_freq`.

    Lives here rather than in ``functions.py`` because ``torch.jit.unused`` needs
    real annotations, and ``functions.py`` uses ``from __future__ import annotations``.
    """

    @fp32_island
    def compute(x):
        if forward_fourier:
            seq_len = x.shape[-1]
            real, imag = real_rfft(x)
        else:
            seq_len = 2 * (x.shape[-2] - 1)
            real, imag = x[..., 0], x[..., 1]
        scale = torch.full((real.shape[-1],), 2.0, dtype=real.dtype, device=real.device)
        scale[0] = 1.0  # Don't multiply the DC-term by 2
        if forward_fourier and seq_len % 2 == 0:
            scale[-1] = 1.0  # Nor the Nyquist term, which only exists for even lengths
        out_real, out_imag = real_analytic_ifft(real * scale, imag * scale, n=seq_len)
        return torch.stack((out_real, out_imag), dim=-1)

    return compute(x)


@torch.jit.unused
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

    @fp32_island
    def compute(x, weight):
        n_times, kernel_size = x.shape[-1], weight.shape[-1]
        n = 1 << (2 * kernel_size - 1).bit_length()
        hop = n - kernel_size + 1
        n_blocks = -(-n_times // hop)
        n_freqs = n // 2 + 1
        # "same" alignment: torch pads the extra zero of an even kernel on the right
        left = (kernel_size - 1) // 2
        x = nn.functional.pad(x, (left, (n_blocks + 1) * hop - n_times - left))
        hops = x.unflatten(-1, (n_blocks + 1, hop))
        # block i = hop i + the first kernel_size - 1 samples of hop i + 1
        blocks = torch.cat([hops[:, :, :-1], hops[:, :, 1:, : kernel_size - 1]], -1)
        cosine, sine = real_dft_basis(n, device=x.device, dtype=x.dtype)
        basis = torch.cat([cosine, -sine]).transpose(0, 1)  # -> [Re | Im] of rfft
        x_f = blocks @ basis  # (batch, in, blocks, 2 * n_freqs)
        w_f = nn.functional.pad(weight.to(x), (0, n - kernel_size)) @ basis
        w_re, w_im = w_f[..., :n_freqs], w_f[..., n_freqs:]
        # conv1d is a cross-correlation: x_f * conj(w_f), summed over the input
        # channels. Per frequency [Re x, Im x] @ [[Re w, -Im w], [Im w, Re w]].
        w_f = torch.cat([torch.cat([w_re, -w_im]), torch.cat([w_im, w_re])], 1)
        x_f = x_f.unflatten(-1, (2, n_freqs)).permute(4, 0, 2, 3, 1).flatten(1, 2)
        y_f = torch.bmm(x_f.flatten(2), w_f.permute(2, 1, 0))  # (freq, ., 2 * out)
        y_f = y_f.unflatten(1, (x.shape[0], n_blocks)).unflatten(-1, (2, -1))
        y_f = y_f.permute(1, 4, 2, 3, 0).flatten(
            -2
        )  # (batch, out, blocks, 2 * n_freqs)
        # inverse DFT of the first `hop` samples, the ones without wrap-around
        scale = torch.full((n_freqs, 1), 2.0 / n, dtype=x.dtype, device=x.device)
        scale[0] = 1.0 / n
        scale[-1] = 1.0 / n  # Nyquist (n is even)
        inverse = torch.cat([scale * cosine[:, :hop], -scale * sine[:, :hop]])
        return (y_f @ inverse).flatten(-2)[..., :n_times]

    y = compute(x, weight)
    if bias is not None:
        y = y + bias.to(y).unsqueeze(-1)
    return y.to(x)
