"""Macro-averaged metrics with clustered-bootstrap CIs for the multi-domain roster.

**This module is the cache key.** `_macro_ci_all` memoises on disk, and its `impl_files` is exactly this
file plus the modules whose behaviour the numbers depend on — so any edit here correctly invalidates a
cached interval, and an edit to a table emitter, a plot colour or a `savefig` call correctly does not.
That separation is why this is its own module: if it shared a file with the table and figure code, a
cosmetic change would throw away a multi-minute bootstrap.

Macro-averaging is the statistic itself (per-domain metric, then mean over domains) with the bootstrap
resampling episodes *within* domain — not an average of per-domain CIs, which would understate width. A
row is macro-averaged only over the FULL domain set; a partial row is reported per-domain and left out of
the macro column unless `allow_partial`, because a 2-of-3 macro is not comparable to a 3-of-3 one.

**A constant-`p = base_rate` reference row.** Every domain is balanced to ~50:50 by construction, so a
model that ignores the evidence and always answers the base rate scores Brier 0.25. Reported as a
reference level rather than as a competitor.

**Recalibration is a diagnostic, not a condition.** A two-parameter cross-fit Platt fit on the logit
removes most of the Brier gap for base and frontier models while barely moving the trained ones. Callers
ask for it explicitly; it never enters the primary comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_LIB = next(p for p in Path(__file__).resolve().parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))

# NOTE: macro_bootstrap is deliberately not imported. Its macro_ci takes one scalar statistic per call,
# and this module needs several metrics per row; _macro_ci_all reuses its clustering + percentile scheme
# in a single pass -- see that docstring for why one pass is faster AND better inference here.
import boot_cache  # noqa: E402
import calibration_metrics  # noqa: E402
from calibration_metrics import compute_calibration  # noqa: E402
from multidomain_cells import DOMAINS, load_attempts, load_cells  # noqa: E402
from roster_palette import ROSTER  # noqa: E402

# Every metric the table can report, in column order. The figure shows a subset (--figure-metrics).
METRICS: tuple[str, ...] = ("brier", "smece", "auroc", "reliability", "resolution")


# sMECE gets its own, smaller draw count. It is an O(n^2) kernel estimator (an n x n Gaussian over the
# episodes per call) that profiling put at ~91% of compute_calibration's runtime — ~35x the cost per draw of
# every other metric combined. Bootstrapping it at the same depth took the table from ~90 s to ~27 min.
# At 200 draws the interval is already stable (width ~0.054 vs ~0.054 at 100) and costs ~2.5 min, which
# is worth paying: a bare point estimate sitting between two CI'd columns reads as more precise than it
# is. Set --smece-boot 0 to skip it and report sMECE as a point estimate.
CHEAP_METRICS: tuple[str, ...] = tuple(m for m in METRICS if m != "smece")


def _platt_cv(p: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """2-fold cross-fit Platt scaling (intercept + slope on the logit). Out-of-fold probabilities.

    Cross-fit rather than in-sample: an in-sample fit would flatter every row, and most of all the badly
    calibrated ones, which is exactly the comparison this diagnostic exists to make.
    """
    from sklearn.linear_model import LogisticRegression

    pc = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), 2)
    out = np.empty(len(y), dtype=float)
    for i, te in enumerate(folds):
        tr = folds[1 - i]
        if len(np.unique(y[tr])) < 2:  # degenerate fold: leave those episodes untouched
            out[te] = p[te]
            continue
        out[te] = LogisticRegression(C=1e6).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
    return out


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([r["parsed_prob"] for r in rows], dtype=float),
        np.array([r["label"] for r in rows], dtype=int),
    )


def _all_metrics(
    rows: list[dict], recalibrate: bool, seed: int = 0, names: tuple[str, ...] = METRICS
) -> dict[str, float]:
    """Reported metrics for one domain's rows, from a single compute_calibration pass.

    `names` restricts what is computed; pass CHEAP_METRICS inside the main resampling loop so the O(n^2)
    sMECE kernel is skipped there.
    """
    if not rows:
        return {m: float("nan") for m in names}
    p, y = _arrays(rows)
    if recalibrate:
        if len(np.unique(y)) < 2:
            return {m: float("nan") for m in names}
        p = _platt_cv(p, y, seed=seed)
    rep = compute_calibration(p, y, with_smece=("smece" in names))
    return {m: float(getattr(rep, m)) for m in names}


def _clusters(cells: dict[str, list[dict]]) -> dict[str, list[list[dict]]]:
    """Episodes grouped by episode_id within each domain — the bootstrap's resampling unit."""
    out: dict[str, list[list[dict]]] = {}
    for d, rows in cells.items():
        by_ep: dict = {}
        for r in rows:
            by_ep.setdefault(r["episode_id"], []).append(r)
        out[d] = list(by_ep.values())
    return out


def _boot_macro(
    clusters: dict[str, list[list[dict]]], recalibrate: bool, names: tuple[str, ...], n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    """Bootstrap distribution of the macro-average for each metric in `names`."""
    rng = np.random.default_rng(seed)
    boot = {m: np.empty(n_boot) for m in names}
    domains = list(clusters)
    for b in range(n_boot):
        per: dict[str, list[float]] = {m: [] for m in names}
        for d in domains:
            cl = clusters[d]
            idx = rng.integers(0, len(cl), len(cl))
            rows = [r for i in idx for r in cl[i]]
            got = _all_metrics(rows, recalibrate, seed=b, names=names)
            for m in names:
                per[m].append(got[m])
        for m in names:
            boot[m][b] = np.nanmean(per[m])
    return boot


def _macro_ci_all(
    cells: dict[str, list[dict]], recalibrate: bool, n_boot: int, smece_boot: int, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """(point, lo, hi) for every metric, from clustered bootstrap passes.

    Two passes, because sMECE costs ~35x what the others do per draw: the cheap metrics run at `n_boot`,
    sMECE at `smece_boot` (0 = point estimate only, NaN bounds). Within a pass all metrics share the same
    draws, so their intervals are mutually consistent rather than independently resampled.

    Clustering: resample episodes (by episode_id) WITHIN domain, macro-average across domains per draw,
    then take percentiles of the macro distribution.

    Memoised on disk (scripts/lib/boot_cache.py). This is the slowest computation in the pipeline — the
    sMECE pass alone is ~35x the others per draw — and the figures get revised far more often than the
    cells change. The key covers the cell contents, both bootstrap sizes, the seed, the recalibrate flag,
    the metric list, and the source of this file and of the bootstrap helpers, so any change that could
    move the numbers misses the cache rather than serving a stale interval.
    """
    return boot_cache.cached(
        "master_table.macro_ci_all",
        {
            "data": boot_cache.fingerprint_records(cells),
            "recalibrate": recalibrate,
            "n_boot": n_boot,
            "smece_boot": smece_boot,
            "seed": seed,
            "metrics": list(METRICS),
            "cheap": list(CHEAP_METRICS),
        },
        lambda: {k: list(v) for k, v in _macro_ci_all_uncached(cells, recalibrate, n_boot, smece_boot, seed).items()},
        # This file, the cache machinery, and the metric definitions themselves: without calibration_metrics,
        # a change to compute_calibration would serve stale intervals. Notably absent, and correctly so: the table and figure modules.
        impl_files=[__file__, boot_cache.__file__, calibration_metrics.__file__],
    )


def _macro_ci_all_uncached(
    cells: dict[str, list[dict]], recalibrate: bool, n_boot: int, smece_boot: int, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """The actual two-pass bootstrap; see `_macro_ci_all` for the contract."""
    clusters = _clusters(cells)
    full = [_all_metrics(rows, recalibrate, names=METRICS) for rows in cells.values()]
    point = {m: float(np.nanmean([f[m] for f in full])) for m in METRICS}

    boot = _boot_macro(clusters, recalibrate, CHEAP_METRICS, n_boot, seed)
    if smece_boot > 0:
        boot |= _boot_macro(clusters, recalibrate, ("smece",), smece_boot, seed + 1)

    out = {}
    for m in METRICS:
        if m in boot:
            lo, hi = np.nanpercentile(boot[m], [2.5, 97.5])
            out[m] = (point[m], float(lo), float(hi))
        else:
            out[m] = (point[m], float("nan"), float("nan"))
    return out


def _row(key, label, entry, cells, attempts, missing, *, recal, allow_partial, n_boot, smece_boot) -> dict:
    rec: dict = {
        "key": key,
        "label": label,
        "entry": entry,
        "recalibrated": recal,
        "missing": missing,
        "n": sum(len(v) for v in cells.values()),
        "attempted": sum(a for a, _ in attempts.values()),
        "parsed": sum(p for _, p in attempts.values()),
        # Macro (per-domain rate, then mean over domains) to match every other column: a pooled rate
        # would let one domain's episode count dominate the reported failure rate.
        "parse_fail": float(np.mean([1.0 - (p / a) for a, p in attempts.values() if a])) if attempts else float("nan"),
        "per_domain": {d: _all_metrics(rws, recal) for d, rws in cells.items()},
    }
    if missing and not allow_partial:
        for m in METRICS:
            rec[m] = (float("nan"),) * 3
    else:
        rec |= _macro_ci_all(cells, recal, n_boot, smece_boot)
    return rec


def build_rows(
    testset_slug: str,
    keys: list[str],
    *,
    recalibrate: bool,
    allow_partial: bool,
    n_boot: int,
    smece_boot: int,
    root: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """(primary rows, recalibrated rows). The second list is empty unless `recalibrate`."""
    rows: list[dict] = []
    recal_rows: list[dict] = []
    for key in keys:
        e = ROSTER[key]
        cells = load_cells(testset_slug, e.eval_key, root=root)
        if not cells:
            print(f"[master] skip {key}: no cells under {e.eval_key}", file=sys.stderr)
            continue
        attempts = load_attempts(testset_slug, e.eval_key, root=root)
        missing = [d for d in DOMAINS if d not in cells]
        kw = dict(allow_partial=allow_partial, n_boot=n_boot, smece_boot=smece_boot)
        rows.append(_row(key, e.label, e, cells, attempts, missing, recal=False, **kw))
        # Recalibrate only where there is headroom: trained rows are already calibrated, so a near-
        # duplicate row for them would pad the diagnostic without informing anything.
        if recalibrate and (e.group == "frontier" or key.startswith("base_")):
            recal_rows.append(_row(f"{key}__recal", e.label, e, cells, attempts, missing, recal=True, **kw))
    return rows, recal_rows


def constant_reference(
    testset_slug: str, keys: list[str], n_boot: int, smece_boot: int, root: Path | None = None
) -> dict | None:
    """Constant-`p = observed base rate` row, built on whichever roster row has the full domain set.

    Uses the real episodes so `n`, the domain set, and the bootstrap clustering match every other row.
    """
    for key in keys:
        cells = load_cells(testset_slug, ROSTER[key].eval_key, root=root)
        if all(d in cells for d in DOMAINS):
            const = {
                d: [{**r, "parsed_prob": float(np.mean([x["label"] for x in rws]))} for r in rws]
                for d, rws in cells.items()
            }
            n = sum(len(v) for v in const.values())
            rec: dict = {
                "key": "_const",
                "label": "constant p = base rate",
                "entry": None,
                "recalibrated": False,
                "missing": [],
                "n": n,
                "attempted": n,
                "parsed": n,
                "parse_fail": 0.0,  # a constant predictor cannot fail to answer
                "per_domain": {d: _all_metrics(rws, False) for d, rws in const.items()},
            }
            rec |= _macro_ci_all(const, False, n_boot, smece_boot)
            return rec
    return None


def _outside_ci(pt: float, lo: float, hi: float) -> bool:
    """True when the point estimate falls outside its own bootstrap interval.

    Not a bug: `reliability` and `smece` are positively-biased statistics. Resampling injects noise that
    shows up as extra apparent miscalibration, so the bootstrap distribution sits ABOVE the full-sample
    value — most visibly for an already-well-calibrated row, where the true value is near the zero floor
    (the constant-p row has reliability exactly 0, and every resample is worse). Those intervals are
    useful for comparing rows but should not be read as covering the point.
    """
    return not np.isnan(pt) and not np.isnan(lo) and (pt < lo or pt > hi)
