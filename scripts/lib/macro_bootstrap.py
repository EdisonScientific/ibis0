"""Macro-averaged metrics with stratified (optionally episode-clustered) bootstrap CIs.

The multi-domain eval reports every metric **macro-averaged** over domains (equal weight per domain),
and its CI must come from a *single* bootstrap in which the macro-average is the statistic -- NOT from
post-hoc combining three per-domain CIs. This module is the
shared implementation every figure/table uses.

Key points:
- **Stratify within domain**: resample each domain's units separately, keeping the 1/N_domain weights
  fixed. (Pooling episodes and resampling would bootstrap the *micro*-average.)
- **Cluster by episode when rows are sub-episode** (e.g. per-stub WoE increments): pass
  `cluster_key="episode_id"` so the resample draws whole episodes, not individual correlated rows.
- **Paired comparisons**: `paired_macro_ci` resamples the shared episodes ONCE per iteration and
  applies the same draw to every model, so the CI on a between-model difference cancels shared-episode
  variance and is much tighter.

`statistic` is any callable `list[record] -> float` (mean, Brier, AUROC, ...). A domain whose statistic
is NaN in a given draw is dropped from that draw's macro-average (`nanmean`), so a degenerate resample
does not sink the whole iteration.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boot_cache  # noqa: E402

Record = dict
Statistic = Callable[[list], float]


def _clusters_by_domain(records: list, domain_key: str, cluster_key: str | None):
    """dict[domain -> list[cluster]] where a cluster is a list of records.

    With `cluster_key`, records sharing (domain, cluster_key) form one cluster (resampled as a unit);
    without it, each record is its own singleton cluster.
    """
    by_dom: dict = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        ckey = r[cluster_key] if cluster_key else i
        by_dom[r[domain_key]][ckey].append(r)
    return {d: list(cmap.values()) for d, cmap in by_dom.items()}


def macro_point(records: list, statistic: Statistic, domain_key: str = "domain") -> float:
    """Macro-average: statistic per domain, then nanmean over domains."""
    by_dom: dict = defaultdict(list)
    for r in records:
        by_dom[r[domain_key]].append(r)
    per = [statistic(rows) for rows in by_dom.values()]
    return float(np.nanmean(per)) if per else float("nan")


def macro_ci(
    records: list,
    statistic: Statistic,
    domain_key: str = "domain",
    cluster_key: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    stat_id: str | None = None,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for the macro-average, via a within-domain (clustered) bootstrap.

    `stat_id` names the statistic for the on-disk cache (see scripts/lib/boot_cache.py). It is required
    for caching because `statistic` is usually a lambda: reflection cannot tell `lambda rs: brier(rs)`
    from `lambda rs: auroc(rs)`, so an inferred key would serve one metric's CI for another. Omit it and
    the bootstrap simply runs uncached.
    """
    if stat_id is not None:
        return tuple(  # type: ignore[return-value]
            boot_cache.cached(
                "macro_ci",
                {
                    "data": boot_cache.fingerprint_records(records),
                    "stat": stat_id,
                    "domain_key": domain_key,
                    "cluster_key": cluster_key,
                    "n_boot": n_boot,
                    "seed": seed,
                },
                lambda: list(macro_ci(records, statistic, domain_key, cluster_key, n_boot, seed, stat_id=None)),
                impl_files=[__file__, boot_cache.__file__],
            )
        )
    point = macro_point(records, statistic, domain_key)
    grouped = _clusters_by_domain(records, domain_key, cluster_key)
    domains = list(grouped)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        per = []
        for d in domains:
            clusters = grouped[d]
            idx = rng.integers(0, len(clusters), len(clusters))  # resample clusters with replacement
            rows = [r for i in idx for r in clusters[i]]
            per.append(statistic(rows))
        boot[b] = np.nanmean(per) if per else np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_macro_ci(
    records_by_model: dict[str, list],
    macro_stat: Statistic,
    combine: Callable[[dict[str, float]], float],
    domain_key: str = "domain",
    cluster_key: str = "episode_id",
    n_boot: int = 2000,
    seed: int = 0,
    stat_id: str | None = None,
) -> tuple[float, float, float]:
    """CI on a between-model quantity with a shared per-iteration resample.

    `macro_stat(records)` computes one model's macro-averaged metric; `combine({model: macro})` reduces
    the per-model macros to the reported scalar (e.g. `lambda m: m["direct_rl"] - m["base_direct"]`).
    All models must share the same (domain, cluster_key) episodes (guaranteed by the shared manifest);
    each iteration resamples episodes within domain once and applies that draw to every model.

    `stat_id` must identify BOTH callables (`macro_stat` and `combine`) to enable caching — a shared
    resample paired with a different `combine` is a different quantity. Omit it to run uncached.
    """
    if stat_id is not None:
        return tuple(  # type: ignore[return-value]
            boot_cache.cached(
                "paired_macro_ci",
                {
                    "data": boot_cache.fingerprint_records(records_by_model),
                    "stat": stat_id,
                    "domain_key": domain_key,
                    "cluster_key": cluster_key,
                    "n_boot": n_boot,
                    "seed": seed,
                },
                lambda: list(
                    paired_macro_ci(records_by_model, macro_stat, combine, domain_key, cluster_key, n_boot, seed, None)
                ),
                impl_files=[__file__, boot_cache.__file__],
            )
        )
    models = list(records_by_model)
    # Per model: dict[domain -> dict[cluster_key -> rows]]; reference episode keys from the first model.
    idxed: dict[str, dict] = {}
    for m in models:
        by_dom: dict = defaultdict(lambda: defaultdict(list))
        for r in records_by_model[m]:
            by_dom[r[domain_key]][r[cluster_key]].append(r)
        idxed[m] = {d: dict(cmap) for d, cmap in by_dom.items()}
    ref = idxed[models[0]]
    domain_keys = {d: list(cmap) for d, cmap in ref.items()}  # episode keys per domain (shared)

    def macros_for(draw: dict[str, list]) -> dict[str, float]:
        out = {}
        for m in models:
            rows = [r for d, keys in draw.items() for key in keys for r in idxed[m][d].get(key, [])]
            out[m] = macro_stat(rows)
        return out

    point = combine(macros_for(domain_keys))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        draw = {d: [keys[i] for i in rng.integers(0, len(keys), len(keys))] for d, keys in domain_keys.items()}
        boot[b] = combine(macros_for(draw))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)
