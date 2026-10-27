"""Model-agnostic calibration metrics for binary probabilistic forecasts.

Single source of truth for the calibration analysis in the Tahoe Evidence
Deliberation study. Pure numpy (+ lazy sklearn for AUROC/AUPRC). Consumed by
the cross-model report renderer and the per-run training dashboard so every
figure and table reports the same numbers.

The Murphy (1973) decomposition splits the Brier score over forecast bins into
    binned_brier = reliability - resolution + uncertainty
where reliability is calibration error (lower is better), resolution is how far
bin outcomes move from the base rate (higher is better), and uncertainty is the
irreducible base-rate variance. The identity is exact for the binned forecasts
(see test_calibration_metrics.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CalibrationReport:
    n: int
    base_rate: float
    brier: float
    brier_skill_score: float | None  # 1 - brier/uncertainty; None when base rate ∈ {0,1}
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    mce: float
    auroc: float | None
    auprc: float | None
    sharpness: float  # variance of the forecasts
    bins: list[tuple[float, float, int]]  # (mean_pred, empirical_freq, count) per non-empty bin
    smece: float | None = None  # binning-free kernel-smoothed calibration error

    def to_dict(self) -> dict:
        return asdict(self)


def _smece(probs: np.ndarray, labels: np.ndarray, bandwidth: float | None = None) -> float:
    """Binning-free calibration error: a kernel-smoothed mean |E[y|p] - p|.

    Nadaraya-Watson estimate of the empirical frequency r(p)=E[y|p] using a
    Gaussian kernel reflected at the [0,1] boundaries (so probability mass is not
    lost at the edges), then the density-weighted mean absolute residual. This is
    a practical estimator in the smooth-ECE family (Błasiok & Nakkiran, 2023): no
    bins, so no sparse-bin or bin-count artifacts. O(n^2); intended for full-data
    calls, not inner bootstrap loops.
    """
    n = len(probs)
    if n < 2:
        return 0.0
    if bandwidth is None:
        sd = float(probs.std())
        bandwidth = max(0.02, 1.06 * (sd if sd > 0 else 0.1) * n ** (-0.2))  # Silverman, floored
    h = bandwidth
    d = (probs[:, None] - probs[None, :]) / h
    k = np.exp(-0.5 * d * d)
    # reflect across 0 and 1 to remove boundary bias
    dlo = (probs[:, None] + probs[None, :]) / h
    dhi = ((2.0 - probs[:, None]) - probs[None, :]) / h
    k = k + np.exp(-0.5 * dlo * dlo) + np.exp(-0.5 * dhi * dhi)
    r_hat = (k * labels[None, :]).sum(axis=1) / k.sum(axis=1)
    return float(np.mean(np.abs(r_hat - probs)))


def _bin_edges(probs: np.ndarray, n_bins: int, adaptive: bool) -> np.ndarray:
    """Equal-width edges, or equal-count (quantile) edges when ``adaptive``."""
    if adaptive:
        edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)  # dedup collapses degenerate (tied) quantiles
    return np.linspace(0.0, 1.0, n_bins + 1)


def compute_calibration(
    probs,
    labels,
    n_bins: int = 10,
    adaptive: bool = False,
    with_smece: bool = True,
) -> CalibrationReport:
    """Compute calibration + discrimination metrics for binary forecasts.

    ``probs`` are predicted P(label=1) in [0,1]; ``labels`` are 0/1. Bins are
    right-closed (``(lo, hi]``) with the first bin including 0.0, matching the
    env's ECE convention.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.shape != labels.shape:
        raise ValueError(f"probs/labels shape mismatch: {probs.shape} vs {labels.shape}")
    n = len(probs)
    if n == 0:
        raise ValueError("need at least one forecast")

    base_rate = float(labels.mean())
    brier = float(np.mean((probs - labels) ** 2))
    uncertainty = base_rate * (1.0 - base_rate)
    bss = None if uncertainty == 0.0 else float(1.0 - brier / uncertainty)

    edges = _bin_edges(probs, n_bins, adaptive)
    # interior edges → right-closed bins; first bin includes 0.0, last includes 1.0
    idx = np.digitize(probs, edges[1:-1], right=True)

    reliability = resolution = ece = mce = 0.0
    bins: list[tuple[float, float, int]] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        p_k = float(probs[mask].mean())
        o_k = float(labels[mask].mean())
        w = cnt / n
        reliability += w * (p_k - o_k) ** 2
        resolution += w * (o_k - base_rate) ** 2
        gap = abs(p_k - o_k)
        ece += w * gap
        mce = max(mce, gap)
        bins.append((p_k, o_k, cnt))

    auroc, auprc = _auc(probs, labels)
    smece = _smece(probs, labels) if with_smece else None

    return CalibrationReport(
        n=n,
        base_rate=base_rate,
        brier=brier,
        brier_skill_score=bss,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        ece=ece,
        mce=mce,
        auroc=auroc,
        auprc=auprc,
        sharpness=float(np.var(probs)),
        bins=bins,
        smece=smece,
    )


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of ``x`` (1-based) with ties assigned their average rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied ranks
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _auc(probs: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    """AUROC (Mann-Whitney, tie-correct) and AUPRC (sklearn step-wise average precision).

    Pure numpy so the metrics module stays dependency-free.
    """
    if len(np.unique(labels)) < 2:
        return None, None

    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    ranks = _average_ranks(probs)
    auroc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    order = np.argsort(-probs, kind="mergesort")
    y = labels[order]
    s = probs[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    threshold_idxs = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp_t, fp_t = tp[threshold_idxs], fp[threshold_idxs]
    precision = tp_t / (tp_t + fp_t)
    recall = tp_t / tp_t[-1]
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    auprc = float(np.sum(np.diff(recall) * precision[1:]))

    return float(auroc), auprc
