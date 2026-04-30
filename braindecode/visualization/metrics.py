# Authors: Vandit Shah <shahvanditt@gmail.com>
#          Akshay Sujatha Ravindran <asujatharavindran@uh.edu>  (12-metric
#          and 4-SSIM layout adapted from the author's research code for
#          Sujatha Ravindran & Contreras-Vidal, "An empirical comparison
#          of deep learning explainability approaches for EEG using
#          simulated ground truth," Scientific Reports 13, 2023.
#          DOI: 10.1038/s41598-023-43871-8)
#
# License: BSD (3-clause)


import numpy as np
import torch
import torch.nn.functional as F

from braindecode.visualization.topology import project_to_topomap

METRIC_NAMES = [
    "Cosine_topperc_abs",
    "Cosine_absnorm",
    "Cosine_norm",
    "Cosine_raw",
    "RelevanceMassAccuracy_topperc",
    "RelevanceRankAccuracy_topK",
    "RelevanceMassAccuracy_norm",
    "RelevanceMassAccuracy_absnorm",
    "Pearson_topK_vs_topperc",
    "Pearson_absnorm_vs_refnorm",
    "Pearson_norm_vs_refnorm",
    "Pearson_raw_vs_refnorm",
]
N_METRICS = len(METRIC_NAMES)

SSIM_METRIC_NAMES = [
    "SSIM_topperc",
    "SSIM_absnorm",
    "SSIM_norm",
    "SSIM_raw",
]


def _ssim_uniform(img_a, img_b, win_size, data_range):
    """Uniform-window SSIM (Wang et al., 2004), no extra dependencies.

    Pure-torch reimplementation of :func:`skimage.metrics.structural_similarity`
    with ``gaussian_weights=False`` and the default Bessel-corrected
    sample covariance. Matches skimage to machine epsilon (~2e-16)
    when both run in float64.
    """
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError(f"win_size must be odd and >= 3, got {win_size}")

    a = torch.as_tensor(img_a, dtype=torch.float64).reshape(1, 1, *img_a.shape)
    b = torch.as_tensor(img_b, dtype=torch.float64).reshape(1, 1, *img_b.shape)

    n_pixels_per_window = float(win_size * win_size)
    cov_norm = n_pixels_per_window / (n_pixels_per_window - 1.0)  # Bessel

    def pool(t):
        return F.avg_pool2d(t, kernel_size=win_size, stride=1)

    mu_a = pool(a)
    mu_b = pool(b)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cov_norm * (pool(a * a) - mu_a2)
    sigma_b2 = cov_norm * (pool(b * b) - mu_b2)
    sigma_ab = cov_norm * (pool(a * b) - mu_ab)

    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)

    return float((numerator / denominator).mean().item())


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
        ``*_topperc`` masks.

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
    explanations, reference, explanation_abs = _prepare_attributions(
        explanations, reference, chs_info, abs_reference, abs_explanation
    )
    return _compute(explanations, reference, explanation_abs, prctile_val)


def _prepare_attributions(
    explanations, reference, chs_info, abs_reference, abs_explanation
):
    """Shared preamble for :func:`compute_metrics` and :func:`compute_ssim_metrics`.

    Validates the shapes match, replaces NaNs with zero, optionally
    projects each sample (mean over time) onto a 2-D scalp topomap, and
    derives the abs/clipped explanation map.

    Returns ``(explanations, reference, explanation_abs)``.
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
    return explanations, reference, explanation_abs


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
    """Element-wise division returning 0 where the denominator is 0."""
    out = np.zeros_like(num, dtype=float)
    nonzero = den != 0
    np.divide(num, den, out=out, where=nonzero)
    return out


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
    return _safe_div(
        (A * B).sum(axis=1), np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    )


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


def compute_ssim_metrics(
    explanations,
    reference,
    chs_info=None,
    abs_reference=True,
    abs_explanation=False,
    prctile_val=95,
    win_size=7,
):
    """Compute four SSIM-based attribution-quality metrics.

    Companion to :func:`compute_metrics`. Returned column order matches
    :data:`SSIM_METRIC_NAMES`:

    1. ``SSIM_topperc``  : SSIM between top-percentile masks
       (each kept at its own scale).
    2. ``SSIM_absnorm``  : SSIM between min-max-normalized
       ``|explanation|`` and reference.
    3. ``SSIM_norm``     : SSIM between min-max-normalized raw
       explanation and reference.
    4. ``SSIM_raw``      : SSIM between raw explanation and
       normalized reference.

    No extra dependencies; SSIM is computed with a pure-PyTorch implementation.

    Parameters
    ----------
    explanations, reference : numpy.ndarray
        Same shape ``(n_samples, n_chans, n_times)``. When ``chs_info``
        is given, each sample is time-averaged and projected to a 2-D
        topomap before SSIM, matching :func:`compute_metrics`'
        topographic mode.
    chs_info : list of dict, optional
        Channel info; enables topographic projection.
    abs_reference, abs_explanation : bool
        Same semantics as in :func:`compute_metrics`.
    prctile_val : float
        Top-percentile threshold for the ``SSIM_topperc`` mask.
    win_size : int
        SSIM sliding-window size. ``7`` matches the benchmark in
        Sujatha Ravindran & Contreras-Vidal (2023).

    Returns
    -------
    scores : numpy.ndarray of shape ``(n_samples, 4)``
        SSIM values per sample. Skipped samples have an all-zero row.
    n_skipped : int
        Number of samples skipped due to all-zero or constant
        attributions or reference.
    """
    explanations, reference, explanation_abs = _prepare_attributions(
        explanations, reference, chs_info, abs_reference, abs_explanation
    )

    n_samples = explanations.shape[0]
    scores = np.zeros((n_samples, 4))
    n_skipped = 0
    for i in range(n_samples):
        exp = explanations[i]
        ref = reference[i]
        exp_abs = explanation_abs[i]

        # Skip degenerate samples; matches compute_metrics' policy.
        if (
            exp_abs.sum() == 0
            or ref.sum() == 0
            or exp.max() == exp.min()
            or exp_abs.max() == exp_abs.min()
            or ref.max() == ref.min()
        ):
            n_skipped += 1
            continue

        exp_norm = _minmax_2d(exp)
        exp_abs_norm = _minmax_2d(exp_abs)
        ref_norm = _minmax_2d(ref)

        exp_topperc = np.where(exp < np.percentile(exp, prctile_val), 0.0, exp)
        ref_topperc = np.where(
            ref_norm < np.percentile(ref_norm, prctile_val), 0.0, ref_norm
        )

        # Shrink win_size to fit the array if needed (must stay odd ≥ 3).
        effective_win = _fit_win_size(win_size, exp.shape)

        scores[i, 0] = _ssim_uniform(
            ref_topperc,
            exp_topperc,
            effective_win,
            _range_or_one(exp_topperc),
        )
        scores[i, 1] = _ssim_uniform(
            ref_norm,
            exp_abs_norm,
            effective_win,
            _range_or_one(exp_abs_norm),
        )
        scores[i, 2] = _ssim_uniform(
            ref_norm,
            exp_norm,
            effective_win,
            _range_or_one(exp_norm),
        )
        scores[i, 3] = _ssim_uniform(
            ref_norm,
            exp,
            effective_win,
            _range_or_one(exp),
        )

    return scores, n_skipped


def _minmax_2d(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _range_or_one(arr):
    span = float(np.nanmax(arr) - np.nanmin(arr))
    return span if span > 0 else 1.0


def _fit_win_size(requested, shape):
    """Largest odd window size that fits inside ``shape``, capped at ``requested``."""
    smallest_dim = min(shape)
    if smallest_dim < 3:
        raise ValueError(f"SSIM needs both image dims >= 3; got shape {shape}.")
    upper = min(requested, smallest_dim)
    if upper % 2 == 0:
        upper -= 1
    return max(upper, 3)
