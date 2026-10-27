"""Unit tests for per-episode context-length sampling (`context_size_for`)."""

import pytest
from ibis_env_utils import context_size_for, validate_context_size_range


def test_range_none_returns_fixed_context_size():
    assert context_size_for("ep1", context_size=16, context_size_range=None) == 16
    assert context_size_for("ep1", context_size=7) == 7


def test_draw_is_within_inclusive_range():
    ks = [context_size_for(f"ep{i}", context_size_range=(2, 16)) for i in range(2000)]
    assert all(2 <= k <= 16 for k in ks)


def test_full_range_is_reachable():
    ks = {context_size_for(f"ep{i}", context_size_range=(2, 16)) for i in range(2000)}
    assert ks == set(range(2, 17))


def test_deterministic_per_episode():
    # Same episode_id + seed -> same K, regardless of how many times / when it's drawn.
    assert context_size_for("epX", context_size_range=(2, 16)) == context_size_for("epX", context_size_range=(2, 16))


def test_seed_changes_the_assignment():
    a = [context_size_for(f"ep{i}", context_size_range=(2, 16), seed=0) for i in range(200)]
    b = [context_size_for(f"ep{i}", context_size_range=(2, 16), seed=1) for i in range(200)]
    assert a != b  # a different seed reshuffles the per-episode assignment


def test_roughly_uniform():
    n = 6000
    ks = [context_size_for(f"ep{i}", context_size_range=(2, 16)) for i in range(n)]
    expected = n / 15
    for value in range(2, 17):
        count = ks.count(value)
        assert 0.7 * expected < count < 1.3 * expected, (value, count)


def test_singleton_range():
    assert all(context_size_for(f"ep{i}", context_size_range=(5, 5)) == 5 for i in range(50))


def test_validate_context_size_range():
    assert validate_context_size_range(None) is None
    assert validate_context_size_range([2, 16]) == (2, 16)
    with pytest.raises(ValueError):
        validate_context_size_range((16, 2))
    with pytest.raises(ValueError):
        validate_context_size_range((2, 4, 8))
    with pytest.raises(ValueError):
        validate_context_size_range((-1, 4))
