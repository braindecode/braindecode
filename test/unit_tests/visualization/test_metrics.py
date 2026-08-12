import numpy as np
import pytest

from braindecode.visualization.metrics import METRIC_NAMES, compute_metrics

from .conftest import SEED

N_SAMPLES = 5
N_CHANS = 8
N_TIMES = 64


@pytest.fixture
def random_data():
    rng = np.random.default_rng(SEED + 42)
    explanations = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    reference = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    return explanations, reference


@pytest.fixture
def positive_reference():
    """All-positive reference so abs_reference=True is a no-op."""
    rng = np.random.default_rng(SEED + 43)
    return (
        np.abs(rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES))).astype(np.float32) + 0.1
    )


@pytest.fixture(scope="module")
def chs_info(chs_info_factory):
    return chs_info_factory(N_CHANS)


def test_metric_names_length():
    assert len(METRIC_NAMES) == 12


def test_output_shape(random_data):
    explanations, reference = random_data
    scores, n_skipped = compute_metrics(explanations, reference)
    assert scores.shape == (N_SAMPLES, 12)
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
    explanations, reference = random_data
    explanations[1] = 0.0
    scores, _ = compute_metrics(explanations, reference)
    assert np.all(scores[1] == 0.0)


def test_similarity_metrics_perfect(positive_reference):
    """Identical explanation and reference must yield near-1 cosine and Pearson scores."""
    scores, _ = compute_metrics(positive_reference.copy(), positive_reference)
    assert np.all(scores[:, 1] > 0.99), f"Cosine_absnorm: {scores[:, 1]}"
    assert np.all(scores[:, 9] > 0.99), f"Pearson_absnorm: {scores[:, 9]}"


def test_relevance_accuracy_perfect(positive_reference):
    scores, _ = compute_metrics(positive_reference.copy(), positive_reference)
    assert np.all(scores[:, 4] > 0.99), f"MassAccuracy_topperc: {scores[:, 4]}"
    assert np.all(scores[:, 5] > 0.99), f"RankAccuracy_topK: {scores[:, 5]}"
    assert np.all(scores[:, 6] > 0.99), f"MassAccuracy_norm: {scores[:, 6]}"
    assert np.all(scores[:, 7] > 0.99), f"MassAccuracy_absnorm: {scores[:, 7]}"


def test_perfect_scores_higher_than_random():
    """Sparse-GT case where mass and rank accuracy are discriminative."""
    rng = np.random.default_rng(SEED + 6)
    reference = np.zeros((N_SAMPLES, N_CHANS, N_TIMES), dtype=np.float32)
    reference[:, : N_CHANS // 2, :] = rng.standard_normal(
        (N_SAMPLES, N_CHANS // 2, N_TIMES)
    ).astype(np.float32)
    reference = np.abs(reference) + 0.1 * (reference != 0)

    perfect = reference.copy()
    random_exp = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)

    perfect_scores, _ = compute_metrics(perfect, reference)
    random_scores, _ = compute_metrics(random_exp, reference)

    for idx, name in [
        (1, "Cosine_absnorm"),
        (7, "MassAccuracy_absnorm"),
        (9, "Pearson_absnorm"),
    ]:
        assert perfect_scores[:, idx].mean() > random_scores[:, idx].mean(), (
            f"{name}: perfect mean {perfect_scores[:, idx].mean():.3f} not > "
            f"random mean {random_scores[:, idx].mean():.3f}"
        )


def test_abs_reference_false_keeps_signed_reference():
    """abs_reference=False uses reference as-is. Perfect match → high signed metrics."""
    reference = np.random.default_rng(SEED + 2).standard_normal(
        (N_SAMPLES, N_CHANS, N_TIMES)
    ).astype(np.float32)
    scores, _ = compute_metrics(reference.copy(), reference.copy(), abs_reference=False)
    assert np.all(scores[:, 2] > 0.99), f"Cosine_norm: {scores[:, 2]}"
    assert np.all(scores[:, 10] > 0.99), f"Pearson_norm: {scores[:, 10]}"


def test_abs_explanation_recovers_negatives():
    """abs_explanation=True recovers all-negative explanations; clipping zeros them."""
    reference = np.abs(
        np.random.default_rng(SEED + 3).standard_normal((N_SAMPLES, N_CHANS, N_TIMES))
    ).astype(np.float32) + 0.1
    explanations = -reference.copy()
    scores_clip, _ = compute_metrics(explanations.copy(), reference.copy(), abs_explanation=False)
    scores_abs, _ = compute_metrics(explanations.copy(), reference.copy(), abs_explanation=True)
    assert scores_abs[:, 1].mean() > scores_clip[:, 1].mean()


def test_high_prctile_retains_fewer_values(random_data):
    explanations, reference = random_data
    scores_95, _ = compute_metrics(explanations.copy(), reference.copy(), prctile_val=95)
    scores_50, _ = compute_metrics(explanations.copy(), reference.copy(), prctile_val=50)
    # Cosine_topperc_abs (idx 0) is computed on the top-percentile masked maps.
    assert not np.allclose(scores_95[:, 0], scores_50[:, 0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="reference shape"):
        compute_metrics(
            np.zeros((2, 8, 32), dtype=np.float32),
            np.zeros((2, 8, 16), dtype=np.float32),
        )


def test_topo_mode_output_shape(chs_info):
    rng = np.random.default_rng(SEED + 4)
    explanations = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    reference = rng.standard_normal((N_SAMPLES, N_CHANS, N_TIMES)).astype(np.float32)
    scores, _ = compute_metrics(explanations, reference, chs_info=chs_info)
    assert scores.shape == (N_SAMPLES, 12)


def test_topo_mode_perfect_attribution(chs_info):
    reference = np.abs(
        np.random.default_rng(SEED + 5).standard_normal((N_SAMPLES, N_CHANS, N_TIMES))
    ).astype(np.float32) + 0.1
    scores, _ = compute_metrics(reference.copy(), reference, chs_info=chs_info)
    assert np.all(scores[:, 1] > 0.99), f"Topo Cosine_absnorm: {scores[:, 1]}"
    assert np.all(scores[:, 9] > 0.99), f"Topo Pearson_absnorm: {scores[:, 9]}"
