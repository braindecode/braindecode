# License: BSD-3
"""TorchScript-safe :func:`torchaudio.functional.lfilter` and ``filtfilt``.

PyTorch 2.15 added ``torch.autograd.Function.input_grad_buffers``, annotated
``tuple[Tensor | None, ...]``, which TorchScript cannot parse. Scripting a
caller of :func:`torchaudio.functional.lfilter` compiles torchaudio's
``DifferentiableFIR`` and ``DifferentiableIIR`` (``torch.autograd.Function``
subclasses) as TorchScript classes, so it fails with "Expression of type dots
cannot be used in a type expression".

Eager calls still go to torchaudio, with its autograd. Scripted calls run the
ops that torchaudio's own scripted path runs (its ``ts_apply`` calls
``forward`` directly when scripting): the FIR part as a ``conv1d``, the IIR
recursion through torchaudio's ``_lfilter_core_loop``. Scripted outputs are
unchanged.

Lives here rather than in ``modules/filter.py`` because ``torch.jit.unused``
needs real annotations, and ``filter.py`` uses ``from __future__ import
annotations``.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torchaudio.functional import filtfilt as _torchaudio_filtfilt
from torchaudio.functional import lfilter as _torchaudio_lfilter


def _generic_lfilter_core_loop(
    input_signal_windows: Tensor,
    a_coeffs_flipped: Tensor,
    padded_output_waveform: Tensor,
) -> None:
    """Run torchaudio's Python IIR recursion, in place on the padded output."""
    n_order = a_coeffs_flipped.size(1)
    a_coeffs_flipped = a_coeffs_flipped.unsqueeze(2)
    for i_sample, o0 in enumerate(input_signal_windows.permute(2, 0, 1)):
        windowed = padded_output_waveform[:, :, i_sample : i_sample + n_order]
        o0 -= (windowed.transpose(0, 1) @ a_coeffs_flipped)[..., 0].t()
        padded_output_waveform[:, :, i_sample + n_order - 1] = o0


try:
    # torchaudio's own choice: its C++ op when the extension is built, else the
    # Python loop above. Using it keeps the scripted recursion identical.
    from torchaudio.functional.filtering import _lfilter_core_loop
except ImportError:  # pragma: no cover - torchaudio versions without it
    _lfilter_core_loop = _generic_lfilter_core_loop


@torch.jit.unused
def _eager_lfilter(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor) -> Tensor:
    return _torchaudio_lfilter(
        waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs, clamp=False
    )


@torch.jit.unused
def _eager_filtfilt(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor) -> Tensor:
    return _torchaudio_filtfilt(
        waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs, clamp=False
    )


def _scripted_lfilter(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor) -> Tensor:
    """Run torchaudio's scripted ``lfilter(clamp=False)`` for 1-D coefficients."""
    a_coeffs = a_coeffs.unsqueeze(0)
    b_coeffs = b_coeffs.unsqueeze(0)
    shape = waveform.size()
    waveform = waveform.reshape(-1, 1, shape[-1])
    # DifferentiableFIR.forward
    b_normalized = b_coeffs / a_coeffs[:, 0:1]
    n_order = b_normalized.size(1)
    waveform = F.conv1d(
        F.pad(waveform, (n_order - 1, 0)),
        b_normalized.flip(1).contiguous().unsqueeze(1),
        groups=1,
    )
    # DifferentiableIIR.forward
    a_normalized = a_coeffs / a_coeffs[:, 0:1]
    n_order = a_normalized.size(1)
    n_batch, n_channel, n_sample = waveform.shape
    padded_output = torch.zeros(
        n_batch,
        n_channel,
        n_sample + n_order - 1,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    _lfilter_core_loop(waveform, a_normalized.flip(1).contiguous(), padded_output)
    output = padded_output[:, :, n_order - 1 :]
    return output.reshape(shape[:-1] + output.shape[-1:])


def _scripted_filtfilt(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor) -> Tensor:
    """Run torchaudio's scripted ``filtfilt(clamp=False)`` for 1-D coefficients."""
    forward_filtered = _scripted_lfilter(waveform, a_coeffs, b_coeffs)
    return _scripted_lfilter(forward_filtered.flip(-1), a_coeffs, b_coeffs).flip(-1)


def lfilter(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor) -> Tensor:
    """:func:`torchaudio.functional.lfilter` with ``clamp=False``, scriptable.

    ``a_coeffs`` and ``b_coeffs`` are 1-D and of the same length.
    """
    if torch.jit.is_scripting():
        return _scripted_lfilter(waveform, a_coeffs, b_coeffs)
    return _eager_lfilter(waveform, a_coeffs, b_coeffs)


def filtfilt(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor) -> Tensor:
    """:func:`torchaudio.functional.filtfilt` with ``clamp=False``, scriptable.

    ``a_coeffs`` and ``b_coeffs`` are 1-D and of the same length.
    """
    if torch.jit.is_scripting():
        return _scripted_filtfilt(waveform, a_coeffs, b_coeffs)
    return _eager_filtfilt(waveform, a_coeffs, b_coeffs)
