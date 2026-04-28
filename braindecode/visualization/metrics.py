# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from braindecode.visualization.topology import _build_topomap


def _import_ssim():
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "compute_metrics requires scikit-image; install with "
            "`pip install braindecode[viz]`."
        ) from exc
    return structural_similarity

METRIC_NAMES = [
    "SSIM_top5_abs",
    "SSIM_absnorm",
    "SSIM_norm",
    "SSIM_raw",
    "Cosine_top5_abs",
    "Cosine_absnorm",
    "Cosine_norm",
    "Cosine_raw",
    "RelevanceMassAccuracy_top5",
    "RelevanceRankAccuracy_topK",
    "RelevanceMassAccuracy_norm",
    "RelevanceMassAccuracy_absnorm",
    "Pearson_topK_vs_top5ref",
    "Pearson_absnorm_vs_refnorm",
    "Pearson_norm_vs_refnorm",
    "Pearson_raw_vs_refnorm",
]


def compute_metrics(
    explanations,
    reference,
    chs_info=None,
    gt_flag=True,
    abs_condition=False,
    prctile_val=95,
):
    """Compute 16 attribution quality metrics between explanations and reference.

    If ``chs_info`` is provided, attributions are first projected onto a 2D
    scalp topography via MNE before computing metrics (topographic mode).
    Otherwise metrics are computed directly on the raw attribution maps
    (channel-wise mode).

    Parameters
    ----------
    explanations : numpy.ndarray
        Attribution maps of shape ``(n_samples, n_chans, n_times)``.
    reference : numpy.ndarray
        Ground truth or baseline attribution maps, same shape as
        ``explanations``.
    chs_info : list of dict, optional
        Channel info list (braindecode ``chs_info`` format with ``ch_name``
        and position fields). If provided, enables topographic projection.
    gt_flag : bool, default=True
        If True, take absolute value of ``reference`` (ground truth mode).
        If False, use ``reference`` as-is (comparison mode, e.g. randomized
        weights).
    abs_condition : bool, default=False
        If True, take absolute value of ``explanations``. If False, zero
        negative values only.
    prctile_val : float, default=95
        Top-percentile threshold (e.g. 95 keeps the top 5%) used for
        ``*_top5_abs`` and top-percentile reference masks. The top-K mask
        used for ``RelevanceRankAccuracy`` and ``Pearson_topK`` is
        independently sized from the count of GT-positive entries.

    Returns
    -------
    metrics : numpy.ndarray
        Array of shape ``(n_samples, 16)`` with metric values per sample.
        See :data:`METRIC_NAMES` for the metric at each index.
    n_skipped : int
        Number of samples skipped due to all-zero or constant attributions
        or reference.
    """
    if chs_info is not None:
        return _compute_topo_metrics(
            explanations, reference, chs_info, gt_flag, abs_condition, prctile_val
        )
    return _compute_channel_metrics(
        explanations, reference, gt_flag, abs_condition, prctile_val
    )


def _compute_channel_metrics(
    explanations, reference, gt_flag, abs_condition, prctile_val
):
    if reference.shape != explanations.shape:
        raise ValueError(
            f"reference shape {reference.shape} does not match "
            f"explanations shape {explanations.shape}"
        )

    explanations = np.nan_to_num(explanations, nan=0.0)
    reference = np.nan_to_num(reference, nan=0.0)

    if gt_flag:
        reference = np.abs(reference)

    explanation_abs = (
        np.abs(explanations) if abs_condition else np.clip(explanations, 0, None)
    )

    return _compute_metrics_core(explanations, reference, explanation_abs, prctile_val)


def _compute_topo_metrics(
    explanations, reference, chs_info, gt_flag, abs_condition, prctile_val
):
    explanations = np.mean(explanations, axis=2)
    reference = np.mean(reference, axis=2)

    # Build the topomap interpolator once and reuse for every sample —
    # all the geometry (info, sphere, Delaunay triangulation) depends only
    # on chs_info, not on the per-sample values.
    Xi, Yi, interp = _build_topomap(chs_info)

    def _project(arr):
        return np.array(
            [interp.set_values(a).set_locations(Xi, Yi)() for a in arr]
        )

    explanations_topo = np.nan_to_num(_project(explanations), nan=0.0)
    reference_topo = np.nan_to_num(_project(reference), nan=0.0)

    if gt_flag:
        reference_topo = np.abs(reference_topo)

    explanation_abs = (
        np.abs(explanations_topo)
        if abs_condition
        else np.clip(explanations_topo, 0, None)
    )

    return _compute_metrics_core(
        explanations_topo, reference_topo, explanation_abs, prctile_val
    )


def _safe_minmax(x):
    """Min-max normalize, returning zeros if x is constant."""
    lo, hi = x.min(), x.max()
    rng = hi - lo
    if rng == 0:
        return np.zeros_like(x), 0.0
    return (x - lo) / rng, rng


def _compute_metrics_core(explanations, reference, explanation_abs, prctile_val):
    ssim = _import_ssim()
    n_samples = explanations.shape[0]
    scores = np.zeros((n_samples, 16))
    n_skipped = 0

    for idx in range(n_samples):
        attr = explanations[idx]
        attr_abs = explanation_abs[idx]
        gt = reference[idx]

        attr_norm, attr_range = _safe_minmax(attr)
        attr_absnorm, attr_abs_range = _safe_minmax(attr_abs)
        gt_norm, gt_range = _safe_minmax(gt)

        # Skip degenerate samples: all-zero attribution/reference, or constant
        # values that make min-max normalization undefined.
        if (
            np.all(attr_abs == 0)
            or np.all(gt == 0)
            or attr_range == 0
            or attr_abs_range == 0
            or gt_range == 0
        ):
            n_skipped += 1
            continue

        # Two masks with subtly different semantics, preserved from the
        # original implementation: count only positives for K, but use
        # non-zero as the GT indicator mask (matters when gt_flag=False
        # leaves negative reference values in place).
        n_gt_positive = int((gt > 0).sum())
        gt_bool = gt.astype(bool, copy=False)
        if n_gt_positive == 0:
            n_skipped += 1
            continue

        # Top-percentile masks (zero everything below the prctile_val threshold).
        attr_topperc = np.where(attr < np.percentile(attr, prctile_val), 0, attr)
        gt_topperc = np.where(gt_norm < np.percentile(gt_norm, prctile_val), 0, gt_norm)

        # Top-K mask: keep the K largest abs-normalized attribution entries
        # where K = number of GT-positive locations. np.partition is O(N)
        # vs O(N log N) for full sort.
        flat = attr_absnorm.ravel()
        topk_threshold = np.partition(flat, -n_gt_positive)[-n_gt_positive]
        attr_topk = np.where(attr_absnorm < topk_threshold, 0, attr_absnorm)

        # SSIM data ranges. The masked maps can become near-constant on
        # degenerate inputs; fall back to the unmasked range to keep SSIM
        # well-defined.
        topperc_range = attr_topperc.max() - attr_topperc.min()
        if topperc_range == 0:
            topperc_range = attr_range

        scores[idx, 0] = ssim(gt_topperc, attr_topperc, win_size=7, data_range=topperc_range)
        # min-max normalized arrays have data_range == 1 by construction.
        scores[idx, 1] = ssim(gt_norm, attr_absnorm, win_size=7, data_range=1.0)
        scores[idx, 2] = ssim(gt_norm, attr_norm, win_size=7, data_range=1.0)
        scores[idx, 3] = ssim(gt_norm, attr, win_size=7, data_range=attr_range)

        scores[idx, 4] = cosine_similarity(
            gt_topperc.reshape(1, -1), attr_topperc.reshape(1, -1)
        )[0, 0]
        scores[idx, 5] = cosine_similarity(
            gt_norm.reshape(1, -1), attr_absnorm.reshape(1, -1)
        )[0, 0]
        scores[idx, 6] = cosine_similarity(
            gt_norm.reshape(1, -1), attr_norm.reshape(1, -1)
        )[0, 0]
        scores[idx, 7] = cosine_similarity(
            gt_norm.reshape(1, -1), attr.reshape(1, -1)
        )[0, 0]

        # Relevance mass / rank accuracy: fraction of attribution mass that
        # falls within GT-positive locations. Guards against zero denominators
        # produced by adversarial inputs (e.g. all-negative attributions).
        scores[idx, 8] = _safe_div(np.abs(attr_topperc[gt_bool]).sum(), attr_topperc.sum())
        topk_count = np.count_nonzero(attr_topk)
        scores[idx, 9] = _safe_div(np.count_nonzero(attr_topk[gt_bool]), topk_count)
        scores[idx, 10] = _safe_div(np.abs(attr_norm[gt_bool]).sum(), attr_norm.sum())
        scores[idx, 11] = _safe_div(np.abs(attr_absnorm[gt_bool]).sum(), attr_absnorm.sum())

        scores[idx, 12], _ = stats.pearsonr(attr_topk.ravel(), gt_topperc.ravel())
        scores[idx, 13], _ = stats.pearsonr(attr_absnorm.ravel(), gt_norm.ravel())
        scores[idx, 14], _ = stats.pearsonr(attr_norm.ravel(), gt_norm.ravel())
        scores[idx, 15], _ = stats.pearsonr(attr.ravel(), gt_norm.ravel())

    return scores, n_skipped


def _safe_div(num, denom):
    return num / denom if denom != 0 else 0.0
