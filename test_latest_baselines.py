import numpy as np

from score_latest_nait import nait_scores, oriented_first_pc
from score_latest_temp import _bracketed_search


def test_oriented_first_pc_points_with_mean_change():
    values = np.asarray([[1.0, 0.1], [2.0, -0.1], [3.0, 0.0]])
    direction = oriented_first_pc(values, seed=13)
    assert float(values.mean(0) @ direction) >= 0.0
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_nait_scores_sum_layer_projections():
    reference = [
        np.asarray([[1.0, 0.0], [2.0, 0.1], [3.0, -0.1]]),
        np.asarray([[0.0, 1.0], [0.1, 2.0], [-0.1, 3.0]]),
    ]
    candidates = [
        np.asarray([[2.0, 0.0], [0.0, 2.0]]),
        np.asarray([[0.0, 2.0], [2.0, 0.0]]),
    ]
    scores, directions, explained = nait_scores(reference, candidates, seed=23)
    expected = sum(candidate @ direction for candidate, direction in zip(candidates, directions))
    assert np.allclose(scores, expected)
    assert len(explained) == 2
    assert all(0.0 <= value <= 1.0 + 1e-8 for value in explained)


def test_temp_calibration_brackets_overshoot():
    scale, ratio = _bracketed_search(
        lambda value: 1.0 + 12.0 * value,
        initial=0.02,
        low_target=2.0,
        high_target=3.0,
        max_steps=10,
        label="unit_test_scale",
    )
    assert 2.0 <= ratio <= 3.0
    assert 0.08 <= scale <= 0.17
