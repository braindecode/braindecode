# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)


import numpy as np

from braindecode.visualization.topology import project_to_topomap

METRIC_NAMES = [
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
N_METRICS = len(METRIC_NAMES)


def compute_metrics(
    explanations,
    reference,
    chs_info=None,
    abs_reference=True,
    abs_explanation=False,
    prctile_val=95,
):
    """Compute attribution-quality metrics between explanations and reference.

    If ``chs_info`` is provided, attributions are first averaged over time
    and projected onto a 2-D scalp topography via MNE before computing
    metrics (topographic mode). Otherwise metrics are computed directly on
    the raw attribution maps (channel-wise mode).

    Parameters
    ----------
    explanations : numpy.ndarray
        Attribution maps of shape ``(n_samples, n_chans, n_times)``.
    reference : numpy.ndarray
        Ground truth or baseline attribution maps, same shape as
        ``explanations``.
    chs_info : list of dict, optional
        Channel info list (braindecode ``chs_info`` format). If provided,
        enables topographic projection.
    abs_reference : bool, default=True
        If True, take absolute value of ``reference`` (ground truth mode).
        If False, use ``reference`` as-is (comparison mode, e.g. randomized
        weights).
    abs_explanation : bool, default=False
        If True, take absolute value of ``explanations``. If False, clip
        negative values to zero.
    prctile_val : float, default=95
        Top-percentile threshold (e.g. 95 keeps the top 5%) used for
        ``*_top5`` masks.

    Returns
    -------
    metrics : numpy.ndarray
        Array of shape ``(n_samples, 12)`` with metric values per sample.
        See :data:`METRIC_NAMES` for the metric at each index. Skipped
        samples have an all-zero row.
    n_skipped : int
        Number of samples skipped due to all-zero or constant attributions
        or reference.
    """
    if explanations.shape != reference.shape:
        raise ValueError(
            f"reference shape {reference.shape} does not match "
            f"explanations shape {explanations.shape}"
        )

    explanations = np.nan_to_num(explanations, nan=0.0)
    reference = np.nan_to_num(reference, nan=0.0)

    if chs_info is not None:
        explanations = _project_batch(explanations.mean(axis=2), chs_info)
        reference = _project_batch(reference.mean(axis=2), chs_info)
        explanations = np.nan_to_num(explanations, nan=0.0)
        reference = np.nan_to_num(reference, nan=0.0)

    if abs_reference:
        reference = np.abs(reference)

    explanation_abs = (
        np.abs(explanations) if abs_explanation else np.clip(explanations, 0, None)
    )

    return _compute(explanations, reference, explanation_abs, prctile_val)


def _project_batch(arr, chs_info):
    return np.stack([project_to_topomap(a, chs_info) for a in arr])


def _compute(attr, gt, attr_abs, prctile_val):
    n = attr.shape[0]
    A = attr.reshape(n, -1)
    G = gt.reshape(n, -1)
    Aabs = attr_abs.reshape(n, -1)

    A_norm = _minmax(A)
    Aabs_norm = _minmax(Aabs)
    G_norm = _minmax(G)

    A_topperc = _topperc_mask(A, prctile_val)
    G_topperc = _topperc_mask(G_norm, prctile_val)

    # K = number of strictly positive entries in gt (per sample, varies).
    k_per_sample = (G > 0).sum(axis=1)
    A_topk = _topk_mask(Aabs_norm, k_per_sample)

    # Boolean GT mask used for indexing — non-zero counts as "positive"
    # (preserves original behaviour where negative gt entries contribute
    # under abs_reference=False).
    G_bool = G.astype(bool)

    skip = (
        (Aabs.sum(axis=1) == 0)
        | (G.sum(axis=1) == 0)
        | (A.max(axis=1) == A.min(axis=1))
        | (Aabs.max(axis=1) == Aabs.min(axis=1))
        | (G.max(axis=1) == G.min(axis=1))
        | (k_per_sample == 0)
    )

    scores = np.zeros((n, N_METRICS))
    scores[:, 0] = _cosine(G_topperc, A_topperc)
    scores[:, 1] = _cosine(G_norm, Aabs_norm)
    scores[:, 2] = _cosine(G_norm, A_norm)
    scores[:, 3] = _cosine(G_norm, A)
    scores[:, 4] = _mass_acc(A_topperc, G_bool)
    scores[:, 5] = _rank_acc(A_topk, G_bool)
    scores[:, 6] = _mass_acc(A_norm, G_bool)
    scores[:, 7] = _mass_acc(Aabs_norm, G_bool)
    scores[:, 8] = _pearson(A_topk, G_topperc)
    scores[:, 9] = _pearson(Aabs_norm, G_norm)
    scores[:, 10] = _pearson(A_norm, G_norm)
    scores[:, 11] = _pearson(A, G_norm)

    scores[skip] = 0.0
    return scores, int(skip.sum())


def _safe_div(num, den):
    return num / np.where(den == 0, 1.0, den)


def _minmax(X):
    """Per-sample min-max normalize. X: (n, F)."""
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    return _safe_div(X - lo, hi - lo)


def _topperc_mask(X, prctile_val):
    """Zero values strictly below the per-sample top-percentile threshold."""
    threshold = np.percentile(X, prctile_val, axis=1, keepdims=True)
    return np.where(X < threshold, 0.0, X)


def _topk_mask(X, k_per_sample):
    """Per-sample top-K mask. K varies per sample, so this loops over rows."""
    out = np.zeros_like(X)
    for i, k in enumerate(k_per_sample):
        if k <= 0:
            continue
        threshold = np.partition(X[i], -k)[-k]
        out[i] = np.where(X[i] < threshold, 0.0, X[i])
    return out


def _cosine(A, B):
    """Per-sample cosine similarity for batched (n, F) arrays."""
    return _safe_div((A * B).sum(axis=1), np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


def _pearson(A, B):
    """Per-sample Pearson correlation for batched (n, F) arrays."""
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    den = np.sqrt((Ac * Ac).sum(axis=1) * (Bc * Bc).sum(axis=1))
    return _safe_div((Ac * Bc).sum(axis=1), den)


def _mass_acc(attr, gt_bool):
    """Fraction of attribution mass landing within GT-positive locations."""
    correct = np.where(gt_bool, np.abs(attr), 0.0).sum(axis=1)
    return _safe_div(correct, attr.sum(axis=1))


def _rank_acc(attr_topk, gt_bool):
    """Fraction of top-K attribution locations that fall within GT-positive locations."""
    is_topk = attr_topk != 0
    return _safe_div((is_topk & gt_bool).sum(axis=1), is_topk.sum(axis=1))
