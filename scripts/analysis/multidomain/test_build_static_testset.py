"""Selection tests for build_static_testset.select_domain.

`select_domain` decides *which* episodes are evaluated and at *what* context length. Every downstream
number depends on it, and the two properties that matter most are silent when they break: a k allocation
that quietly skews the context marginal, and a sensitivity subset that drifts out of the primary set,
which would invalidate the per-stub WoE attribution without any error.

`select_domain` reads its episode list through `load_dataset_split`, so these tests patch that with a
synthetic split — selection touches no episode content, which is what makes it cheap to test in
isolation.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_static_testset as B  # noqa: E402

K_LO, K_HI = 2, 16
N_K = K_HI - K_LO + 1  # 15


@pytest.fixture
def split(monkeypatch):
    """Patch the dataset reader with a synthetic TEST split of `n` episodes."""

    def _make(n):
        ids = [f"ep{i:05d}" for i in range(n)]
        monkeypatch.setattr(B, "load_dataset_split", lambda *a, **k: pd.DataFrame({"episode_id": ids}))
        return ids

    return _make


def _select(primary_n, sens_n=8, seed=0):
    return B.select_domain("tahoe", "unused", primary_n, K_LO, K_HI, sens_n, seed)


def test_k_allocation_is_balanced_when_divisible(split):
    """primary_n=900 divides 15 exactly: every k gets 60, no remainder logic engaged."""
    split(2000)
    primary, _ = _select(900)
    counts = pd.Series([k for _, k in primary]).value_counts()
    assert set(counts.index) == set(range(K_LO, K_HI + 1))
    assert set(counts) == {60}


def test_indivisible_remainder_goes_to_the_largest_k(split):
    """primary_n=2000 leaves r=5, which must land on k=12..16 — not k=2..6.

    A naive `i % n_k` round-robin puts the surplus at the *lowest* k
    values, which is the opposite of what we want: the extra samples belong at the long contexts.
    """
    split(2000)
    primary, _ = _select(2000)
    counts = pd.Series([k for _, k in primary]).value_counts().to_dict()
    assert len(primary) == 2000
    assert {k: counts[k] for k in range(12, 17)} == dict.fromkeys(range(12, 17), 134)
    assert {k: counts[k] for k in range(2, 12)} == dict.fromkeys(range(2, 12), 133)


@pytest.mark.parametrize("primary_n", [900, 1000, 1234, 2000])
def test_k_allocation_is_never_off_by_more_than_one(split, primary_n):
    """Whatever primary_n is, the per-k counts must stay within 1 of each other."""
    split(2000)
    primary, _ = _select(primary_n)
    counts = pd.Series([k for _, k in primary]).value_counts()
    assert len(primary) == primary_n
    assert counts.max() - counts.min() <= 1
    assert all(K_LO <= k <= K_HI for _, k in primary)


def test_a_single_extra_episode_lands_at_max_context(split):
    """r=1 must put the one surplus episode at k=16 exactly."""
    split(2000)
    primary, _ = _select(N_K + 1)  # 16 episodes over 15 k values -> r=1
    counts = pd.Series([k for _, k in primary]).value_counts().to_dict()
    assert counts[K_HI] == 2
    assert all(counts[k] == 1 for k in range(K_LO, K_HI))


def test_sensitivity_is_nested_within_primary(split):
    """Every sensitivity episode must also be a primary episode.

    This is what licenses per-stub WoE attribution against the primary aggregates. If it ever drifts,
    nothing errors — the figures just stop describing the same episodes.
    """
    split(2000)
    primary, sens = _select(2000, sens_n=256)
    assert {e for e, _ in sens} <= {e for e, _ in primary}
    assert len({e for e, _ in sens}) == 256


def test_sensitivity_is_the_dense_prefix_grid(split):
    """Each sensitivity episode appears at every k in 0..k_hi, once."""
    split(2000)
    _, sens = _select(2000, sens_n=256)
    assert len(sens) == 256 * (K_HI + 1)
    per_ep = pd.DataFrame(sens, columns=["e", "k"]).groupby("e")["k"].apply(sorted)
    assert all(ks == list(range(0, K_HI + 1)) for ks in per_ep)


def test_selection_is_deterministic_under_a_fixed_seed(split):
    """Same seed, same inputs -> identical pairs. The build must be reproducible."""
    split(2000)
    a_p, a_s = _select(2000, sens_n=64, seed=0)
    split(2000)
    b_p, b_s = _select(2000, sens_n=64, seed=0)
    assert a_p == b_p and a_s == b_s


def test_a_different_seed_selects_a_different_subsample(split):
    """The seed has to actually do something — guards against a silently ignored seed."""
    split(2000)
    a, _ = _select(900, seed=0)
    split(2000)
    b, _ = _select(900, seed=1)
    assert {e for e, _ in a} != {e for e, _ in b}


def test_primary_n_above_the_split_size_is_refused(split):
    """Asking for more episodes than exist must fail loudly, not silently truncate."""
    split(500)
    with pytest.raises(ValueError, match="< --primary-n"):
        _select(2000)


def test_duplicate_episode_ids_are_refused(split, monkeypatch):
    """Duplicate ids would make the sha order ambiguous and double-count episodes."""
    monkeypatch.setattr(B, "load_dataset_split", lambda *a, **k: pd.DataFrame({"episode_id": ["a", "b", "b"]}))
    with pytest.raises(ValueError, match="duplicate episode_ids"):
        _select(2)
