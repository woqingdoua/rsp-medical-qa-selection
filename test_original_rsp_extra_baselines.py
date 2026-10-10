from __future__ import annotations

import numpy as np
import torch

from score_original_rsp_extra_baselines import (
    SENTINEL,
    bald_scores,
    batchbald_ranking_score,
    kcenter_ranking_score,
    sift_ranking_score,
)


def test_bald_is_zero_for_identical_passes() -> None:
    probabilities = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    values = np.stack([probabilities] * 8, axis=0)
    assert np.allclose(bald_scores(values), 0.0, atol=1e-12)


def test_bald_detects_epistemic_disagreement() -> None:
    values = np.asarray(
        [
            [[0.99, 0.01], [0.6, 0.4]],
            [[0.01, 0.99], [0.6, 0.4]],
        ],
        dtype=np.float64,
    )
    scores = bald_scores(values)
    assert scores[0] > scores[1]


def test_batchbald_returns_exact_distinct_ranked_batch() -> None:
    values = np.asarray(
        [
            [[0.95, 0.05], [0.05, 0.95], [0.6, 0.4], [0.5, 0.5]],
            [[0.05, 0.95], [0.95, 0.05], [0.6, 0.4], [0.5, 0.5]],
        ],
        dtype=np.float64,
    )
    first = batchbald_ranking_score(values, 2, 200, 7, device="cpu", chunk_size=2)
    second = batchbald_ranking_score(values, 2, 200, 7, device="cpu", chunk_size=2)
    assert np.array_equal(first, second)
    assert int(np.sum(first > SENTINEL / 2)) == 2
    assert sorted(first[first > SENTINEL / 2].tolist()) == [1.0, 2.0]


def test_sift_and_kcenter_select_exact_budget() -> None:
    embeddings = np.eye(4, dtype=np.float64)
    uncertainty = np.asarray([0.1, 0.4, 0.3, 0.2])
    sift = sift_ranking_score(embeddings, uncertainty, 2, 0.5)
    kcenter = kcenter_ranking_score(embeddings, 3, seed=11)
    assert int(np.sum(sift > SENTINEL / 2)) == 2
    assert int(np.sum(kcenter > SENTINEL / 2)) == 3
    assert torch.isfinite(torch.tensor(sift[sift > SENTINEL / 2])).all()
