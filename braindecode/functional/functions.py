# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
import torch.nn.functional as F


def square(x):
    return x * x


def safe_log(x, eps: float = 1e-6) -> torch.Tensor:
    """Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


def identity(x):
    return x


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample.

    Notes: This implementation is taken from timm library.

    All credit goes to Ross Wightman.

    Parameters
    ----------
    x : torch.Tensor
        input tensor
    drop_prob : float, optional
        survival rate (i.e. probability of being kept), by default 0.0
    training : bool, optional
        whether the model is in training mode, by default False
    scale_by_keep : bool, optional
        whether to scale output by (1/keep_prob) during training, by default True

    Returns
    -------
    torch.Tensor
        output tensor

    Notes from Ross Wightman:
    (when applied in main path of residual blocks)
    This is the same as the DropConnect impl I created for EfficientNet,
    etc. networks, however,
    the original name is misleading as 'Drop Connect' is a different form
    of dropout in a separate paper...
    See discussion : https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    ... I've opted for changing the layer and argument names to 'drop path'
    rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Generates a 1-dimensional Gaussian kernel based on the specified kernel.

    size and standard deviation (sigma).
    This kernel is useful for Gaussian smoothing or filtering operations in
    image processing. The function calculates a range limit to ensure the kernel
    effectively covers the Gaussian distribution. It generates a tensor of
    specified size and type, filled with values distributed according to a
    Gaussian curve, normalized using a softmax function
    to ensure all weights sum to 1.

    Parameters
    ----------
    kernel_size : int
    sigma : float

    Returns
    -------
    kernel1d : torch.Tensor

    Notes
    -----
    Code copied and modified from TorchVision:
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L725-L732
    All rights reserved.

    LICENSE in https://github.com/pytorch/vision/blob/main/LICENSE
    """
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d


def hilbert_freq(x, forward_fourier=True):
    r"""
    Compute the Hilbert transform using PyTorch, separating the real and
    imaginary parts.

    The analytic signal :math:`x_a(t)` of a real-valued signal :math:`x(t)`
    is defined as:

    .. math::

        x_a(t) = x(t) + i y(t) = \mathcal{F}^{-1} \{ U(f) \mathcal{F}\{x(t)\} \}

    where:
    - :math:`\mathcal{F}` is the Fourier transform,
    - :math:`U(f)` is the unit step function,
    - :math:`y(t)` is the Hilbert transform of :math:`x(t)`.


    Parameters
    ----------
    input : torch.Tensor
        Input tensor. The expected shape depends on the `forward_fourier` parameter:

        - If `forward_fourier` is True:
            (..., seq_len)
        - If `forward_fourier` is False:
            (..., seq_len / 2 + 1, 2)

    forward_fourier : bool, optional
        Determines the format of the input tensor.
        - If True, the input is in the forward Fourier domain.
        - If False, the input contains separate real and imaginary parts.
        Default is True.

    Returns
    -------
    torch.Tensor
        Output tensor with shape (..., seq_len, 2), where the last dimension represents
        the real and imaginary parts of the Hilbert transform.

    Examples
    --------
    >>> import torch
    >>> input = torch.randn(10, 100)  # Example input tensor
    >>> output = hilbert_transform(input)
    >>> print(output.shape)
    torch.Size([10, 100, 2])

    Notes
    -----
    The implementation is matching scipy implementation, but using torch.
    https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/_signaltools.py#L2287-L2394

    """
    if forward_fourier:
        x = torch.fft.rfft(x, norm=None, dim=-1)
        x = torch.view_as_real(x)
    x = x * 2.0
    x[..., 0, :] = x[..., 0, :] / 2.0  # Don't multiply the DC-term by 2
    x = F.pad(
        x, [0, 0, 0, x.shape[-2] - 2]
    )  # Fill Fourier coefficients to retain shape
    x = torch.view_as_complex(x)
    x = torch.fft.ifft(x, norm=None, dim=-1)  # returns complex signal
    x = torch.view_as_real(x)

    return x


def plv_time(x, forward_fourier=True, epsilon: float = 1e-6):
    """Compute the Phase Locking Value (PLV) metric in the time domain.

    The Phase Locking Value (PLV) is a measure of the synchronization between
    different channels by evaluating the consistency of phase differences
    over time. It ranges from 0 (no synchronization) to 1 (perfect
    synchronization) [Lachaux1999]_.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor containing the signal data.

        - If `forward_fourier` is `True`, the shape should be `(..., channels, time)`.
        - If `forward_fourier` is `False`, the shape should be `(..., channels, freqs, 2)`,
          where the last dimension represents the real and imaginary parts.

    forward_fourier : bool, optional
        Specifies the format of the input tensor `x`.

        - If `True`, `x` is assumed to be in the time domain.
        - If `False`, `x` is assumed to be in the Fourier domain with separate real and
          imaginary components.

        Default is `True`.
    epsilon : float, default 1e-6
        Small numerical value to ensure positivity constraint on the complex part

    Returns
    -------
    plv : torch.Tensor
        The Phase Locking Value matrix with shape `(..., channels, channels)`. Each
        element `[i, j]` represents the PLV between channel `i` and channel `j`.

    References
    ----------
    .. [Lachaux1999] Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999).
        Measuring phase synchrony in brain signals. Human brain mapping,
        8(4), 194-208.
    """
    # Compute the analytic signal using the Hilbert transform.
    # x_a has separate real and imaginary parts.
    analytic_signal = hilbert_freq(x, forward_fourier)
    # Calculate the amplitude (magnitude) of the analytic signal.
    # Adding a small epsilon (1e-6) to avoid division by zero.
    amplitude = torch.sqrt(
        analytic_signal[..., 0] ** 2 + analytic_signal[..., 1] ** 2 + 1e-6
    )
    # Normalize the analytic signal to obtain unit vectors (phasors).
    unit_phasor = analytic_signal / amplitude.unsqueeze(-1)

    # Compute the real part of the outer product between phasors of
    # different channels.
    real_real = torch.matmul(unit_phasor[..., 0], unit_phasor[..., 0].transpose(-2, -1))

    # Compute the imaginary part of the outer product between phasors of
    # different channels.
    imag_imag = torch.matmul(unit_phasor[..., 1], unit_phasor[..., 1].transpose(-2, -1))

    # Compute the cross-terms for the real and imaginary parts.
    real_imag = torch.matmul(unit_phasor[..., 0], unit_phasor[..., 1].transpose(-2, -1))
    imag_real = torch.matmul(unit_phasor[..., 1], unit_phasor[..., 0].transpose(-2, -1))

    # Combine the real and imaginary parts to form the complex correlation.
    correlation_real = real_real + imag_imag
    correlation_imag = real_imag - imag_real

    # Determine the number of time points (or frequency bins if in Fourier domain).
    time = amplitude.shape[-1]

    # Calculate the PLV by averaging the magnitude of the complex correlation over time.
    # epsilon is small numerical value to ensure positivity constraint on the complex part
    plv_matrix = (
        1 / time * torch.sqrt(correlation_real**2 + correlation_imag**2 + epsilon)
    )

    return plv_matrix


# -----------------------------------------------------------------------------
# DANCE functional helpers
#
# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
#
# Ported from the DANCE event-detection model (facebookresearch/dance, MIT).
# -----------------------------------------------------------------------------


def iou_1d(s1, e1, s2, e2, eps: float = 1e-7):
    """ELEMENTWISE 1-D temporal IoU. All four inputs share the same shape S
    (e.g. ``(B, Q)``); returns IoU of shape S. Used by ``DanceLoss`` on the
    matched ``(B, Q)`` spans. For (Q,)x(T,) pairwise IoU use ``pairwise_iou_1d``.
    """
    inter = (torch.minimum(e1, e2) - torch.maximum(s1, s2)).clamp(min=0)
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / (union + eps)


def pairwise_iou_1d(s1, e1, s2, e2, eps: float = 1e-7):
    """PAIRWISE 1-D temporal IoU. ``s1,e1`` shape ``(Q,)``, ``s2,e2`` shape
    ``(T,)``; returns ``(Q, T)`` (broadcast ``[:, None]`` x ``[None, :]``).
    Transcribed from ``dance/matcher.py:33-39`` (``_pairwise_iou``). Used by
    ``HungarianMatcher`` to build the ``(Q, n_targets)`` cost matrix.
    """
    inter_start = torch.maximum(s1[:, None], s2[None, :])
    inter_end = torch.minimum(e1[:, None], e2[None, :])
    inter = (inter_end - inter_start).clamp(min=0)
    union = (e1 - s1)[:, None] + (e2 - s2)[None, :] - inter
    return inter / (union + eps)


def detr_to_dense_probs(preds, num_latents, n_classes):
    """Project DETR query predictions onto a dense per-token distribution.

    ``T == num_latents`` (passed directly; NOT derived from duration*frequency),
    so the output ``(B, num_latents, n_classes)`` can never silently mismatch
    the dense head. Training-side only (Python loop; scipy already breaks the
    graph upstream -- never on the export path).
    """
    # ponytail: per-query Python loop kept; training-only, never on export path.
    cls = preds["class"]
    b, q, _ = cls.shape
    t = int(num_latents)
    out = []
    for bi in range(b):
        mask = torch.zeros(t, n_classes, device=cls.device, dtype=cls.dtype)
        for qi in range(q):
            probs = torch.softmax(cls[bi, qi], dim=-1)
            s = float(preds["start"][bi, qi])
            e = float(preds["end"][bi, qi])
            start = int(max(0, s * t))
            end = int(min(t, e * t))
            if start < end:
                mask[start:end] += probs
        out.append(mask / (mask.sum(-1, keepdim=True) + 1e-8))
    return torch.stack(out)


def events_to_mask(events, n_classes, n_times):
    """(n_classes, n_times) multilabel mask; row 0 = background where idle."""
    mask = torch.zeros(n_classes, n_times)
    mask[0] = 1.0
    for s, e, c in events:
        s_i, e_i = int(s), int(e)
        if s_i < e_i and int(c) != 0:
            mask[int(c), s_i:e_i] = 1.0
            mask[0, s_i:e_i] = 0.0
    return mask


def extract_events_from_detr_batch(outputs, duration):
    """Decode a DETR batch dict into per-window ``(start_s, end_s, class, conf)``."""
    cls = outputs["class"]
    b, q, _ = cls.shape
    results = []
    for bi in range(b):
        events = []
        for qi in range(q):
            probs = torch.softmax(cls[bi, qi], dim=-1)
            label = int(probs.argmax())
            if label == 0:
                continue
            s = float(outputs["start"][bi, qi]) * duration
            e = float(outputs["end"][bi, qi]) * duration
            events.append((s, e, label, float(probs[label])))
        results.append(events)
    return results
