from copy import deepcopy

import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

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
    gt_flag=1,
    abs_condition=0,
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
    gt_flag : int, default=1
        If 1, take absolute value of ``reference`` (ground truth mode).
        If 0, use ``reference`` as-is (comparison mode, e.g. randomized
        weights).
    abs_condition : int, default=0
        If 1, take absolute value of ``explanations``. If 0, zero negative
        values only.
    prctile_val : int, default=95
        Percentile threshold for top-K masking.

    Returns
    -------
    metrics : numpy.ndarray
        Array of shape ``(n_samples, 16)`` with metric values per sample.
        See :data:`METRIC_NAMES` for the metric at each index.
    n_skipped : int
        Number of samples skipped due to all-zero attributions or reference.
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
    assert reference.shape == explanations.shape

    explanations = explanations.copy()
    reference = reference.copy()
    explanations[np.isnan(explanations)] = 0
    reference[np.isnan(reference)] = 0

    if gt_flag:
        reference = np.abs(reference)

    explanation_abs = deepcopy(explanations)
    explanation_abs = (
        np.abs(explanation_abs) if abs_condition else np.clip(explanation_abs, 0, None)
    )

    return _compute_metrics_core(explanations, reference, explanation_abs, prctile_val)


def _compute_topo_metrics(
    explanations, reference, chs_info, gt_flag, abs_condition, prctile_val
):
    from braindecode.visualization.topology import project_to_topomap

    explanations = np.mean(explanations, axis=2)
    reference = np.mean(reference, axis=2)

    explanations_topo = np.array(
        [project_to_topomap(e, chs_info) for e in explanations]
    )
    reference_topo = np.array([project_to_topomap(r, chs_info) for r in reference])

    explanations_topo[np.isnan(explanations_topo)] = 0
    reference_topo[np.isnan(reference_topo)] = 0

    assert explanations_topo.shape == reference_topo.shape

    if gt_flag:
        reference_topo = np.abs(reference_topo)

    explanation_abs = deepcopy(explanations_topo)
    explanation_abs = (
        np.abs(explanation_abs) if abs_condition else np.clip(explanation_abs, 0, None)
    )

    return _compute_metrics_core(
        explanations_topo, reference_topo, explanation_abs, prctile_val
    )


def _compute_metrics_core(explanations, reference, explanation_abs, prctile_val):
    n_samples = explanations.shape[0]
    scores = np.zeros((n_samples, 16))
    n_skipped = 0

    for idx in range(n_samples):
        # Skip degenerate samples where all values are zero
        if np.all(explanation_abs[idx] == 0) or np.all(reference[idx] == 0):
            n_skipped += 1
            continue

        # Raw attribution map for this sample
        attr = deepcopy(explanations[idx])
        # Min-max normalized attribution
        attr_norm = (attr - attr.min()) / (attr.max() - attr.min())

        # Absolute-value attribution (negatives zeroed or flipped depending on abs_condition)
        attr_abs = deepcopy(explanation_abs[idx])
        # Min-max normalized absolute attribution
        attr_absnorm = (attr_abs - attr_abs.min()) / (attr_abs.max() - attr_abs.min())

        # Ground truth / reference map for this sample
        gt = deepcopy(reference[idx])
        # Min-max normalized ground truth
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min())

        # Top-percentile masked attribution (only top prctile_val% retained)
        attr_topperc = deepcopy(attr)
        attr_topperc[attr < np.percentile(attr, prctile_val)] = 0

        # Top-percentile masked ground truth
        gt_topperc = deepcopy(gt_norm)
        gt_topperc[gt_norm < np.percentile(gt_norm, prctile_val)] = 0

        # Top-K mask: keep top-K attribution values where K = number of GT-positive locations
        n_gt_positive = len(np.where(gt > 0)[0])
        topk_threshold = np.sort(attr_absnorm.flatten())[-n_gt_positive]
        attr_topk = deepcopy(attr_absnorm)
        attr_topk[attr_absnorm < topk_threshold] = 0

        #  SSIM: structural similarity between attribution and ground truth
        # Top-percentile versions
        scores[idx, 0] = ssim(
            gt_topperc,
            attr_topperc,
            win_size=7,
            data_range=attr_topperc.max() - attr_topperc.min(),
        )
        # Absolute-normalized vs normalized GT
        scores[idx, 1] = ssim(
            gt_norm,
            attr_absnorm,
            win_size=7,
            data_range=attr_absnorm.max() - attr_absnorm.min(),
        )
        # Normalized vs normalized GT
        scores[idx, 2] = ssim(
            gt_norm, attr_norm, win_size=7, data_range=attr_norm.max() - attr_norm.min()
        )
        # Raw vs normalized GT
        scores[idx, 3] = ssim(
            gt_norm, attr, win_size=7, data_range=attr.max() - attr.min()
        )

        #  Cosine similarity: directional agreement between attribution and GT
        scores[idx, 4] = cosine_similarity(
            gt_topperc.reshape(1, -1), attr_topperc.reshape(1, -1)
        )[0][0]
        scores[idx, 5] = cosine_similarity(
            gt_norm.reshape(1, -1), attr_absnorm.reshape(1, -1)
        )[0][0]
        scores[idx, 6] = cosine_similarity(
            gt_norm.reshape(1, -1), attr_norm.reshape(1, -1)
        )[0][0]
        scores[idx, 7] = cosine_similarity(gt_norm.reshape(1, -1), attr.reshape(1, -1))[
            0
        ][0]

        #  Relevance mass accuracy (top-percentile): fraction of top attribution mass
        #     that falls within GT-positive locations
        total_mass = attr_topperc.sum()
        correct_mass = np.abs(attr_topperc[np.array(gt, dtype=bool)]).sum()
        scores[idx, 8] = correct_mass / total_mass

        #  Relevance rank accuracy (top-K): fraction of top-K attribution locations
        #     that overlap with GT-positive locations
        total_topk = len(np.where(attr_topk)[0])
        correct_topk = len(np.where(attr_topk[np.array(gt, dtype=bool)])[0])
        scores[idx, 9] = correct_topk / total_topk

        #  Relevance mass accuracy (normalized): same as index 8 but on normalized map
        total_mass = attr_norm.sum()
        correct_mass = np.abs(attr_norm[np.array(gt, dtype=bool)]).sum()
        scores[idx, 10] = correct_mass / total_mass

        #  Relevance mass accuracy (abs-normalized): same on abs-normalized map
        total_mass = attr_absnorm.sum()
        correct_mass = np.abs(attr_absnorm[np.array(gt, dtype=bool)]).sum()
        scores[idx, 11] = correct_mass / total_mass

        #  Pearson correlation: linear agreement between attribution and GT
        scores[idx, 12], _ = stats.pearsonr(
            attr_topk.reshape(-1), gt_topperc.reshape(-1)
        )
        scores[idx, 13], _ = stats.pearsonr(
            attr_absnorm.reshape(-1), gt_norm.reshape(-1)
        )
        scores[idx, 14], _ = stats.pearsonr(attr_norm.reshape(-1), gt_norm.reshape(-1))
        scores[idx, 15], _ = stats.pearsonr(attr.reshape(-1), gt_norm.reshape(-1))

    return scores, n_skipped
