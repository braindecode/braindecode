import mne
import numpy as np
import pytest

from braindecode.visualization.metrics import METRIC_NAMES, compute_metrics

N_SAMPLES = 5
N_CHANS = 8
N_TIMES = 64
_SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_data():
    rng = np.random.default_rng(_SEED)
    explanations = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    reference = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    return explanations, reference


@pytest.fixture
def positive_reference():
    """All-positive reference so gt_flag=1 abs is a no-op."""
    rng = np.random.default_rng(_SEED + 1)
    return (
        np.abs(rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES))).astype(np.float32) + 0.1
    )


@pytest.fixture
def chs_info():
    """Standard 10-20 EEG channel info for topo-mode tests."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:N_CHANS]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


# ---------------------------------------------------------------------------
# Basic shape / bookkeeping
# ---------------------------------------------------------------------------

def test_metric_names_length():
    assert len(METRIC_NAMES) == 16


def test_output_shape(random_data):
    explanations, reference = random_data
    scores, n_skipped = compute_metrics(explanations, reference)
    assert scores.shape == (N_SAMPLES, 16)
    assert n_skipped == 0


def test_skipped_zero_explanation(random_data):
    explanations, reference = random_data
    explanations[2] = 0.0
    _, n_skipped = compute_metrics(explanations, reference)
    assert n_skipped == 1


def test_skipped_zero_reference(random_data):
    explanations, reference = random_data
    reference[0] = 0.0
    reference[4] = 0.0
    _, n_skipped = compute_metrics(explanations, reference)
    assert n_skipped == 2


def test_scores_finite(random_data):
    explanations, reference = random_data
    scores, _ = compute_metrics(explanations, reference)
    assert np.all(np.isfinite(scores))


def test_skipped_sample_scores_are_zero(random_data):
    """Skipped samples must leave their score row as all zeros in the output array."""
    explanations, reference = random_data
    explanations[1] = 0.0
    scores, _ = compute_metrics(explanations, reference)
    assert np.all(scores[1] == 0.0), f"Skipped sample row is not zero: {scores[1]}"


# ---------------------------------------------------------------------------
# Perfect attribution — similarity metrics must be near 1
# ---------------------------------------------------------------------------

def test_similarity_metrics_perfect(positive_reference):
    """Identical explanation and reference must yield near-1 scores for all similarity metrics."""
    scores, _ = compute_metrics(positive_reference.copy(), positive_reference, gt_flag=1)
    assert np.all(scores[:, 1] > 0.99), f"SSIM_absnorm: {scores[:, 1]}"
    assert np.all(scores[:, 5] > 0.99), f"Cosine_absnorm: {scores[:, 5]}"
    assert np.all(scores[:, 13] > 0.99), f"Pearson_absnorm: {scores[:, 13]}"


# ---------------------------------------------------------------------------
# Relevance mass / rank accuracy semantics
# ---------------------------------------------------------------------------

def test_relevance_accuracy_perfect(positive_reference):
    """Explanations = reference → all relevance accuracy metrics must be near 1."""
    scores, _ = compute_metrics(positive_reference.copy(), positive_reference, gt_flag=1)
    assert np.all(scores[:, 8] > 0.99), f"MassAccuracy_top5: {scores[:, 8]}"
    assert np.all(scores[:, 9] > 0.99), f"RankAccuracy_topK: {scores[:, 9]}"
    assert np.all(scores[:, 10] > 0.99), f"MassAccuracy_norm: {scores[:, 10]}"
    assert np.all(scores[:, 11] > 0.99), f"MassAccuracy_absnorm: {scores[:, 11]}"


# ---------------------------------------------------------------------------
# Perfect > random for all metrics
# ---------------------------------------------------------------------------

def test_perfect_scores_higher_than_random(random_data):
    """Perfect attribution must outscore random across all four metric groups.

    Uses a sparse binary reference (only half the channels active) so that
    relevance mass and rank accuracy metrics are actually discriminative —
    with all-positive GT every explanation trivially scores 1 on mass accuracy.
    """
    rng = np.random.default_rng(_SEED + 6)
    # Sparse GT: positive only in the first half of channels, zero elsewhere
    reference = np.zeros((N_SAMPLES, N_CHANS, N_TIMES), dtype=np.float32)
    reference[:, :N_CHANS // 2, :] = rng.standard_normal(
        (N_SAMPLES, N_CHANS // 2, N_TIMES)
    ).astype(np.float32)
    reference = np.abs(reference) + 0.1 * (reference != 0)

    # Perfect: explanation matches GT exactly
    perfect = reference.copy()
    # Random: explanation is unrelated to GT
    random_exp, _ = random_data

    perfect_scores, _ = compute_metrics(perfect, reference, gt_flag=1)
    random_scores, _ = compute_metrics(random_exp, reference, gt_flag=1)

    # One representative from each group
    for idx, name in [(1, "SSIM_absnorm"), (5, "Cosine_absnorm"),
                      (11, "MassAccuracy_absnorm"), (13, "Pearson_absnorm")]:
        assert perfect_scores[:, idx].mean() > random_scores[:, idx].mean(), (
            f"{name}: perfect mean {perfect_scores[:, idx].mean():.3f} not > "
            f"random mean {random_scores[:, idx].mean():.3f}"
        )


# ---------------------------------------------------------------------------
# gt_flag and abs_condition flags
# ---------------------------------------------------------------------------

def test_gt_flag_zero_keeps_signed_reference():
    """gt_flag=0 must use reference as-is. Perfect match with signed reference → high scores."""
    reference = np.random.default_rng(_SEED + 2).standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    # With gt_flag=0 the reference is not abs'd, so identical explanations should match well
    scores, _ = compute_metrics(reference.copy(), reference.copy(), gt_flag=0)
    # Cosine_norm (6) and Pearson_norm_vs_refnorm (14) use the signed normalized
    # attribution — both should be 1 when explanations == reference
    assert np.all(scores[:, 6] > 0.99), f"Cosine_norm with gt_flag=0: {scores[:, 6]}"
    assert np.all(scores[:, 14] > 0.99), f"Pearson_norm with gt_flag=0: {scores[:, 14]}"


def test_abs_condition_makes_negatives_positive():
    """abs_condition=1 flips negatives — negative-only attributions become positive."""
    reference = np.abs(np.random.default_rng(_SEED + 3).standard_normal((N_SAMPLES, N_CHANS, N_TIMES))).astype(np.float32) + 0.1
    # All-negative explanations: with abs_condition=0 (clip negatives to 0) → zeroed
    # with abs_condition=1 (take abs) → matches reference
    explanations = -reference.copy()
    scores_clip, _ = compute_metrics(explanations.copy(), reference.copy(), gt_flag=1, abs_condition=0)
    scores_abs, _ = compute_metrics(explanations.copy(), reference.copy(), gt_flag=1, abs_condition=1)
    # abs_condition=1 recovers the signal; abs_condition=0 clips all to zero → lower scores
    assert scores_abs[:, 5].mean() > scores_clip[:, 5].mean()


def test_high_prctile_retains_fewer_values(random_data):
    """prctile_val=95 zeros out 95% of values; prctile_val=50 zeros out only 50%.
    The SSIM_top5_abs metric (index 0) is computed on the top-percentile masked map,
    so a stricter threshold must produce a sparser map and a different score."""
    explanations, reference = random_data
    scores_95, _ = compute_metrics(explanations.copy(), reference.copy(), prctile_val=95)
    scores_50, _ = compute_metrics(explanations.copy(), reference.copy(), prctile_val=50)
    # A stricter mask (95) keeps far fewer values than a loose mask (50),
    # so the SSIM computed on those masked maps must differ
    assert not np.allclose(scores_95[:, 0], scores_50[:, 0])


# ---------------------------------------------------------------------------
# Topo mode (with chs_info)
# ---------------------------------------------------------------------------

def test_topo_mode_output_shape(chs_info):
    rng = np.random.default_rng(_SEED + 4)
    explanations = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    reference = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    scores, n_skipped = compute_metrics(explanations, reference, chs_info=chs_info)
    assert scores.shape == (N_SAMPLES, 16)


def test_topo_mode_perfect_attribution(chs_info):
    """Topo mode: identical attribution → high similarity scores."""
    reference = (
        np.abs(np.random.default_rng(_SEED + 5).standard_normal((N_SAMPLES, N_CHANS, N_TIMES))).astype(np.float32) + 0.1
    )
    scores, _ = compute_metrics(reference.copy(), reference, chs_info=chs_info, gt_flag=1)
    assert np.all(scores[:, 5] > 0.99), f"Topo Cosine_absnorm: {scores[:, 5]}"
    assert np.all(scores[:, 13] > 0.99), f"Topo Pearson_absnorm: {scores[:, 13]}"
