"""Tests for scripts/lib/calibration_metrics.py (run: uv run pytest scripts/lib/test_calibration_metrics.py)."""

import numpy as np
import pytest
from calibration_metrics import compute_calibration


def _binned_brier_from_bins(report) -> float:
    """Reconstruct the binned Brier purely from per-bin (p_k, o_k, cnt)."""
    return sum((cnt / report.n) * (p_k**2 - 2 * p_k * o_k + o_k) for p_k, o_k, cnt in report.bins)


@pytest.mark.parametrize("adaptive", [False, True])
def test_murphy_identity(adaptive):
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, size=2000)
    labels = (rng.uniform(0, 1, size=2000) < probs).astype(int)  # mildly calibrated
    r = compute_calibration(probs, labels, n_bins=10, adaptive=adaptive)
    identity = r.reliability - r.resolution + r.uncertainty
    assert abs(identity - _binned_brier_from_bins(r)) < 1e-9


def test_perfect_calibration_zero_reliability():
    # 30% positives at forecast 0.3, 70% positives at forecast 0.7 → reliability 0
    probs = [0.3] * 100 + [0.7] * 100
    labels = [1] * 30 + [0] * 70 + [1] * 70 + [0] * 30
    r = compute_calibration(probs, labels, n_bins=10)
    assert r.reliability < 1e-9
    assert r.resolution > 0.0  # the two groups separate the base rate


def test_constant_base_rate_forecaster_bss_zero():
    probs = [0.4] * 100
    labels = [1] * 40 + [0] * 60
    r = compute_calibration(probs, labels, n_bins=10)
    assert abs(r.brier - r.uncertainty) < 1e-9
    assert abs(r.brier_skill_score) < 1e-9
    assert r.resolution < 1e-9


def test_brier_extremes():
    assert compute_calibration([0.0, 1.0], [0, 1]).brier == pytest.approx(0.0)
    assert compute_calibration([1.0, 0.0], [0, 1]).brier == pytest.approx(1.0)


def test_ece_hand_computed():
    r = compute_calibration([0.2, 0.2, 0.8, 0.8], [0, 0, 1, 1], n_bins=10)
    assert r.ece == pytest.approx(0.2)
    assert r.mce == pytest.approx(0.2)


def test_single_class_auc_none():
    r = compute_calibration([0.3, 0.6, 0.9], [1, 1, 1])
    assert r.auroc is None and r.auprc is None


def test_perfectly_separating_auroc_one():
    r = compute_calibration([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert r.auroc == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_calibration([0.1, 0.2], [1])
