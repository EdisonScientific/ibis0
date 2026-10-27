"""Property tests for the macro-averaged stratified/clustered bootstrap (macro_bootstrap.py)."""

import numpy as np
from macro_bootstrap import macro_ci, macro_point, paired_macro_ci

MEAN = lambda rows: float(np.mean([r["v"] for r in rows])) if rows else float("nan")  # noqa: E731


def _domains(spec, seed=1):
    rng = np.random.default_rng(seed)
    return [
        {"domain": d, "episode_id": f"{d}_{i}", "v": rng.normal(mu, 0.1)}
        for d, (mu, n) in spec.items()
        for i in range(n)
    ]


def test_macro_is_not_pooled_when_domain_sizes_differ():
    recs = _domains({"gwas": (0.2, 300), "depmap": (0.5, 600), "tahoe": (0.8, 100)})
    macro = macro_point(recs, MEAN)
    pooled = float(np.mean([r["v"] for r in recs]))
    assert abs(macro - 0.5) < 0.03  # mean of the three domain means
    assert abs(macro - pooled) > 0.02  # size-weighted pooled differs


def test_macro_ci_brackets_point_and_is_deterministic():
    recs = _domains({"gwas": (0.2, 300), "depmap": (0.5, 600), "tahoe": (0.8, 100)})
    p, lo, hi = macro_ci(recs, MEAN, n_boot=2000, seed=0)
    assert lo < p < hi
    assert (lo, hi) == macro_ci(recs, MEAN, n_boot=2000, seed=0)[1:]


def test_macro_ci_tighter_than_single_noisy_domain():
    recs = _domains({"gwas": (0.2, 300), "depmap": (0.5, 600), "tahoe": (0.8, 100)})
    _, lo, hi = macro_ci(recs, MEAN, n_boot=2000, seed=0)
    tah = [r for r in recs if r["domain"] == "tahoe"]
    _, tlo, thi = macro_ci(tah, MEAN, n_boot=2000, seed=0)
    assert (hi - lo) < (thi - tlo)  # averaging 3 independent domains shrinks the interval


def test_episode_cluster_bootstrap_is_wider_than_ignoring_clusters():
    rng = np.random.default_rng(2)
    recs = []
    for d in ("gwas", "depmap", "tahoe"):
        for e in range(40):
            base = rng.normal(0.5, 0.3)  # strong per-episode effect; 16 correlated rows each
            recs += [{"domain": d, "episode_id": f"{d}_{e}", "v": base + rng.normal(0, 0.01)} for _ in range(16)]
    _, clo, chi = macro_ci(recs, MEAN, cluster_key="episode_id", n_boot=1500, seed=0)
    _, nlo, nhi = macro_ci(recs, MEAN, cluster_key=None, n_boot=1500, seed=0)
    assert (chi - clo) > (nhi - nlo)  # ignoring within-episode correlation underestimates the CI


def test_paired_shared_resample_cancels_shared_variance():
    rng = np.random.default_rng(3)
    a = [
        {"domain": d, "episode_id": f"{d}_{e}", "v": rng.normal(0.5, 0.2)}
        for d in ("gwas", "depmap", "tahoe")
        for e in range(200)
    ]
    b = [{**r, "v": r["v"] + 0.10} for r in a]  # constant +0.10 per shared episode
    pt, lo, hi = paired_macro_ci({"A": a, "B": b}, MEAN, lambda m: m["B"] - m["A"], n_boot=1500, seed=0)
    assert abs(pt - 0.10) < 1e-9
    assert (hi - lo) < 0.01 and lo < 0.10 < hi  # shared-episode variance cancels -> near-zero width
