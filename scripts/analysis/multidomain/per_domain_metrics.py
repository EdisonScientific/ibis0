"""Per-domain metrics with clustered-bootstrap CIs — the breakdown behind each macro row.

Deliberately NOT part of `macro_metrics.py`. That module's source hash is the macro bootstrap's cache
key, so putting this computation there would invalidate every cached macro interval and force a full
recompute of the figure set. Here it gets its own cache namespace and its own `impl_files`, and the
macro entries stay warm.

The statistic is the same clustered bootstrap the macro column uses, minus the average over domains:
`macro_bootstrap.macro_ci` fed one domain's records reduces to exactly a within-domain resample of that
domain's episodes, so the resampler is shared rather than reimplemented. Each entry is keyed on its own
`(domain, metric)` so it cannot collide with a macro entry or with another domain's.

Point estimates are checked against the ones `macro_metrics` already reports in `row["per_domain"]`.
That check is what makes the constant-`p` cell reconstruction below safe to keep in a second place.

Recalibrated rows are out of scope: the Platt fit is refitted inside every bootstrap draw against a
per-draw seed that `macro_ci` has no way to supply, and the diagnostic table is macro-only anyway.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import boot_cache  # noqa: E402
import calibration_metrics  # noqa: E402
import display_names as dn  # noqa: E402
import macro_bootstrap  # noqa: E402
from calibration_metrics import compute_calibration  # noqa: E402
from macro_metrics import METRICS  # noqa: E402
from multidomain_cells import DOMAINS, load_attempts, load_cells  # noqa: E402

CLUSTER_KEY = "episode_id"


def _statistic(metric: str):
    """`list[record] -> float` for one metric, computed exactly as `macro_metrics._all_metrics` does."""

    def stat(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        probs = np.array([r["parsed_prob"] for r in rows], dtype=float)
        labels = np.array([r["label"] for r in rows], dtype=int)
        report = compute_calibration(probs, labels, with_smece=(metric == "smece"))
        return float(getattr(report, metric))

    return stat


def _ci(records: list[dict], domain: str, metric: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    """(point, lo, hi) for one metric on one domain. `n_boot <= 0` reports the point estimate only."""
    statistic = _statistic(metric)
    if n_boot <= 0:
        return (statistic(records), float("nan"), float("nan"))
    return tuple(  # type: ignore[return-value]
        boot_cache.cached(
            "master_table.per_domain_ci",
            {
                "data": boot_cache.fingerprint_records(records),
                "domain": domain,
                "metric": metric,
                "n_boot": n_boot,
                "seed": seed,
                "cluster_key": CLUSTER_KEY,
            },
            lambda: list(
                macro_bootstrap.macro_ci(records, statistic, cluster_key=CLUSTER_KEY, n_boot=n_boot, seed=seed)
            ),
            # `macro_bootstrap`'s own cache omits calibration_metrics (its statistic is an opaque
            # callable); naming it here is what keeps a change to compute_calibration from serving a
            # stale interval.
            impl_files=[__file__, boot_cache.__file__, calibration_metrics.__file__, macro_bootstrap.__file__],
        )
    )


def per_domain_rows(
    cells: dict[str, list[dict]], attempts: dict[str, tuple[int, int]], *, n_boot: int, smece_boot: int, seed: int = 0
) -> list[dict]:
    """One dict per domain, shaped like a table row so a renderer can format it with the macro path."""
    out = []
    for domain in DOMAINS:
        if domain not in cells:
            continue
        records = cells[domain]
        attempted, parsed = attempts.get(domain, (len(records), len(records)))
        row: dict = {
            "key": domain,
            "label": dn.domain(domain),
            "n": len(records),
            "attempted": attempted,
            "parsed": parsed,
            "parse_fail": (1.0 - parsed / attempted) if attempted else float("nan"),
        }
        for metric in METRICS:
            # sMECE gets the smaller draw count and its own seed, mirroring the two-pass macro bootstrap.
            draws, metric_seed = (smece_boot, seed + 1) if metric == "smece" else (n_boot, seed)
            row[metric] = _ci(records, domain, metric, draws, metric_seed)
        out.append(row)
    return out


def _constant_cells(cells: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """The constant-`p = observed base rate` predictor's records, per domain.

    Mirrors `macro_metrics.constant_reference`, which builds these inline and returns a finished row
    rather than the cells. `_check_points` catches any drift between the two constructions.
    """
    return {
        d: [{**r, "parsed_prob": float(np.mean([x["label"] for x in rws]))} for r in rws] for d, rws in cells.items()
    }


def _check_points(row: dict) -> None:
    """Fail loudly when a per-domain point estimate disagrees with the one `macro_metrics` reported."""
    for sub in row["per_domain_rows"]:
        for metric in METRICS:
            want = row["per_domain"][sub["key"]][metric]
            got = sub[metric][0]
            if np.isnan(want) and np.isnan(got):
                continue
            if not abs(want - got) <= 1e-12:
                raise ValueError(
                    f"{row['key']}/{sub['key']}/{metric}: per-domain point {got!r} disagrees with "
                    f"macro_metrics' {want!r} — the cells the two modules scored are not the same"
                )


def attach(testset_slug: str, rows: list[dict], *, n_boot: int, smece_boot: int, root: Path | None = None) -> None:
    """Set `row["per_domain_rows"]` on every row, in `DOMAINS` order. Not for recalibrated rows."""
    loaded = {
        r["key"]: (
            load_cells(testset_slug, r["entry"].eval_key, root=root),
            load_attempts(testset_slug, r["entry"].eval_key, root=root),
        )
        for r in rows
        if r["entry"] is not None
    }
    for row in rows:
        if row["entry"] is not None:
            cells, attempts = loaded[row["key"]]
        else:
            # The constant reference is built on the first roster row with the full domain set, which is
            # the rule `macro_metrics.constant_reference` follows.
            source = next(c for c, _ in loaded.values() if all(d in c for d in DOMAINS))
            cells = _constant_cells(source)
            attempts = {d: (len(v), len(v)) for d, v in cells.items()}
        row["per_domain_rows"] = per_domain_rows(cells, attempts, n_boot=n_boot, smece_boot=smece_boot)
        _check_points(row)
