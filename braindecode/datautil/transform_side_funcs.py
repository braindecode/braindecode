from torchaudio.functional import sparse_image_warp
from torch import Tensor, tensor


def warp_along_axis(spec, points_to_warp, dist_to_warp, axis):

    # 3 dimensions dans le spectrogramme,la 3e = liste de spectrogrammes ? à essayer
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len
    point_to_warp = horizontal_line_at_ctr[points_to_warp]
    assert isinstance(point_to_warp, Tensor)

    src_pts, dest_pts = (tensor([[[y, point_to_warp]]], device=device),
                         tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def mask_along_axis(
        specgram: Tensor,
        mask_start: int,
        mask_end: int,
        mask_value: float,
        axis: int
) -> Tensor:
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram (channel, freq, time)
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)

    Returns:
        Tensor: Masked spectrogram of dimensions (channel, freq, time)
    """

    mask_start = (mask_start.long()).squeeze()
    mask_end = (mask_end.long()).squeeze()

    if axis == 1:
        specgram[:, mask_start:mask_end] = mask_value
    elif axis == 2:
        specgram[:, :, mask_start:mask_end] = mask_value
    else:
        raise ValueError('Only Frequency and Time masking are supported')

    specgram = specgram.reshape(specgram.shape[:-2] + specgram.shape[-2:])

    return specgram
