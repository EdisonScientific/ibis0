"""Unit tests for the shared compute_reward scoring."""

import pytest
from ibis_env_utils import UNPARSEABLE_REWARD, compute_reward


def test_reward_likelihood_supported():
    assert compute_reward(0.9, 1) == pytest.approx(0.9)
    assert compute_reward(0.9, 0) == pytest.approx(0.1)


def test_reward_brier():
    assert compute_reward(1.0, 1, reward_type="brier") == pytest.approx(1.0)
    assert compute_reward(0.0, 1, reward_type="brier") == pytest.approx(0.0)
    assert compute_reward(0.5, 1, reward_type="brier") == pytest.approx(0.75)


def test_reward_accuracy_brier_range():
    # correct side of 0.5 + brier reward
    assert compute_reward(1.0, 1, reward_type="accuracy_brier") == pytest.approx(2.0)
    assert compute_reward(0.0, 1, reward_type="accuracy_brier") == pytest.approx(0.0)


def test_reward_whitened_is_finite_and_in_unit_range():
    r = compute_reward(0.0, 1, reward_type="log_likelihood_whitened")  # worst, clipped
    assert 0.0 <= r <= 1.0
    assert compute_reward(1.0, 1, reward_type="log_likelihood_whitened") == pytest.approx(1.0, abs=1e-3)


def test_reward_unparseable():
    assert compute_reward(None, 1) == UNPARSEABLE_REWARD
    # whitened has no room for a negative penalty -> 0.0
    assert compute_reward(None, 1, reward_type="log_likelihood_whitened") == 0.0
