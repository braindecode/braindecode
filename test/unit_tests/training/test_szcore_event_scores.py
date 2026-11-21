# Authors: Aayushya Patel <67443596+ptlnextdoor@users.noreply.github.com>
#
# License: BSD-3

from importlib import metadata

import numpy as np
import pytest

from braindecode.training import szcore_event_scores

NAN = float("nan")
# (reference, hypothesis, params, n_ref, tp, fp, sensitivity, precision, f1, fp_per_day)
# Expected values come from timescoring 0.0.7 on a 3600 s recording.
# fmt: off
FIXED_CASES = {
    "hit_in_pre_tolerance": ([(1800, 1860)], [(1790, 1800)], {}, 1, 1, 0, 1, 1, 1, 0),
    "miss_before_tolerance": ([(1800, 1860)], [(1700, 1760)], {}, 1, 0, 1, 0, 0, 0, 24),
    "hit_in_post_tolerance": ([(1800, 1860)], [(1910, 1915)], {}, 1, 1, 0, 1, 1, 1, 0),
    "merge_gap_under_90": ([], [(3000, 3002), (3050, 3052)], {}, 0, 0, 1, NAN, 0, 0, 24),
    "no_merge_gap_90": ([], [(3000, 3002), (3092, 3094)], {}, 0, 0, 2, NAN, 0, 0, 48),
    "merged_gap_counts_as_overlap": (
        [(1800, 1860)], [(1700, 1702), (1780, 1782)], {}, 1, 1, 0, 1, 1, 1, 0
    ),
    "split_long_reference": ([(100, 800)], [(150, 160)], {}, 3, 1, 0, 1 / 3, 1, 0.5, 0),
    "always_on_one_hour": (
        [(1800, 1860)], [(0, 3600)], {}, 1, 1, 10, 1, 1 / 11, 1 / 6, 240
    ),
    "no_reference_one_alarm": ([], [(100, 110)], {}, 0, 0, 1, NAN, 0, 0, 24),
    "empty_both": ([], [], {}, 0, 0, 0, NAN, NAN, NAN, 0),
    "clip_at_file_start": ([(10, 40)], [(0, 1)], {}, 1, 1, 0, 1, 1, 1, 0),
    "min_overlap_half": (
        [(1800, 1860)], [(1800, 1820)], {"min_overlap": 0.5}, 1, 0, 1, 0, 0, 0, 24
    ),
    "custom_no_tolerance": (
        [(1800, 1860)], [(1790, 1800)],
        {"tolerance_start": 0, "tolerance_end": 0}, 1, 0, 1, 0, 0, 0, 24,
    ),
}
# fmt: on
KEYS = ("n_ref_events", "tp", "fp", "sensitivity", "precision", "f1", "fp_per_day")


@pytest.mark.parametrize("case", FIXED_CASES.values(), ids=FIXED_CASES)
def test_fixed_cases(case):
    reference, hypothesis, params, *expected = case
    scores = szcore_event_scores(reference, hypothesis, 3600, **params)
    assert [scores[key] for key in KEYS] == pytest.approx(expected, nan_ok=True)
    assert scores["fp_per_hour"] == pytest.approx(scores["fp_per_day"] / 24)


def _random_events(rng, n_events, duration):
    """Sorted, non-overlapping events, the input timescoring is defined for."""
    cuts = np.sort(rng.uniform(0, duration, 2 * n_events).round(2))
    return cuts.reshape(-1, 2)


def test_matches_timescoring():
    pytest.importorskip("timescoring")
    if metadata.version("timescoring") != "0.0.7":
        pytest.skip("scores are only pinned to timescoring 0.0.7")
    from timescoring.annotations import Annotation
    from timescoring.scoring import EventScoring

    rng = np.random.default_rng(0)
    duration = 7200
    for _ in range(300):
        reference = _random_events(rng, rng.integers(0, 4), duration)
        hypothesis = _random_events(rng, rng.integers(0, 8), duration)
        params = {
            "tolerance_start": float(rng.choice([0, 30, 45])),
            "tolerance_end": float(rng.choice([0, 60, 90])),
            "min_overlap": float(rng.choice([0, 0.1, 0.5])),
            "max_event_duration": float(rng.choice([120, 300])),
            "min_duration_between_events": float(rng.choice([0, 30, 90])),
        }
        oracle = EventScoring(
            Annotation(list(map(tuple, reference)), 1, duration),
            Annotation(list(map(tuple, hypothesis)), 1, duration),
            EventScoring.Parameters(
                toleranceStart=params["tolerance_start"],
                toleranceEnd=params["tolerance_end"],
                minOverlap=params["min_overlap"],
                maxEventDuration=params["max_event_duration"],
                minDurationBetweenEvents=params["min_duration_between_events"],
            ),
        )
        scores = szcore_event_scores(reference, hypothesis, duration, **params)
        expected = [
            oracle.refTrue,
            oracle.tp,
            oracle.fp,
            oracle.sensitivity,
            oracle.precision,
            oracle.f1,
            oracle.fpRate,
        ]
        assert [scores[key] for key in KEYS] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize(
    "reference, duration",
    [([(10, 5)], 3600), ([(1, 2, 3)], 3600), ([(10, 20)], 0)],
    ids=["offset_before_onset", "three_columns", "zero_duration"],
)
def test_rejects_bad_input(reference, duration):
    with pytest.raises(ValueError):
        szcore_event_scores(reference, [], duration)


def test_reference_past_recording_end_is_missed():
    scores = szcore_event_scores([(4000, 4010)], [], 3600)
    assert (scores["n_ref_events"], scores["tp"], scores["fp"]) == (1, 0, 0)
    assert scores["sensitivity"] == 0
