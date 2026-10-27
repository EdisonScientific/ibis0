"""Cross-fit Platt recalibration for the FULL roster, broken out by domain.

**Why this is its own module, and not three edits to the modules it calls.** `macro_metrics.py` and
`per_domain_metrics.py` are bootstrap *cache keys*: each passes its own `__file__` to
`boot_cache.cached(impl_files=...)`, which hashes the file's bytes. Changing either by one byte
correctly invalidates every interval it has ever cached — hours of sMECE bootstrap — for a change that
moves no primary number. The
table and figure modules are deliberately absent from those lists (see `macro_metrics`'s docstring:
"Notably absent, and correctly so"), so anything that only *consumes* the metrics can live outside the
cache boundary. This module is on the outside, and must stay there.

The corollary is the unusual thing here: it reaches into `macro_metrics._row`,
`macro_metrics._macro_ci_all` and `per_domain_metrics._check_points` rather than being given public
entry points. Adding those entry points is what this module exists to avoid. Each call site says so.

**Every system is recalibrated, including the trained ones.** `macro_metrics.build_rows` recalibrates
only the untrained rows, on the grounds that "trained rows are already calibrated, so a near-duplicate
row for them would pad the diagnostic without informing anything". That is true of the *diagnostic*
read on its own, and false of the comparison the paper draws from it: scoring recalibrated untrained
systems against unrecalibrated trained ones compares two different treatments. The near-duplicate row
is the control. `build_rows`'s gate is left alone — it still describes what `build_rows` does, and
`render_headline` and `render_grouped_metrics` both call it with `recalibrate=False`.

**Per-domain intervals.** `per_domain_metrics` excludes recalibrated rows because its resampler,
`macro_bootstrap.macro_ci`, takes a `list[record] -> float` statistic and has nowhere to put the
per-draw seed that a refitted Platt map needs. Rather than widen that signature — it is
`per_domain_ci`'s cache key, shared with every unrecalibrated row — this calls `_macro_ci_all` with a
single-domain cell dict. A macro-average over one domain is exactly a within-domain resample of that
domain's episodes, so the statistic is identical, and it arrives already seeded per draw, already
two-pass for sMECE, and already cached under a fingerprint no macro entry can collide with.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import display_names as dn  # noqa: E402
import macro_metrics  # noqa: E402
import per_domain_metrics  # noqa: E402
from macro_metrics import METRICS  # noqa: E402
from multidomain_cells import DOMAINS, load_attempts, load_cells  # noqa: E402
from roster_palette import ROSTER  # noqa: E402


def per_domain_recal_rows(
    cells: dict[str, list[dict]], attempts: dict[str, tuple[int, int]], *, n_boot: int, smece_boot: int
) -> list[dict]:
    """One recalibrated sub-row per domain, shaped like `per_domain_metrics.per_domain_rows` output."""
    out: list[dict] = []
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
        # Private on purpose: see the module docstring. A single-domain dict makes the macro average a
        # no-op, which is what turns this into a within-domain bootstrap of one domain's episodes.
        ci = macro_metrics._macro_ci_all({domain: records}, True, n_boot, smece_boot)
        # boot_cache round-trips through JSON, so a warm entry returns lists where a cold one returns
        # tuples. Normalise here rather than letting the renderer see a type that depends on cache state.
        row |= {m: tuple(ci[m]) for m in METRICS}
        out.append(row)
    return out


def build_recal_rows(
    testset_slug: str,
    keys: list[str],
    *,
    n_boot: int,
    smece_boot: int,
    allow_partial: bool = False,
    root: Path | None = None,
) -> list[dict]:
    """A recalibrated row per roster key, each with its per-domain breakdown attached.

    Rows are shaped exactly like `macro_metrics.build_rows`'s, so the table renderers format them
    through the same path.
    """
    rows: list[dict] = []
    for key in keys:
        e = ROSTER[key]
        cells = load_cells(testset_slug, e.eval_key, root=root)
        if not cells:
            print(f"[recal] skip {key}: no cells under {e.eval_key}", file=sys.stderr)
            continue
        attempts = load_attempts(testset_slug, e.eval_key, root=root)
        missing = [d for d in DOMAINS if d not in cells]
        # Private on purpose: see the module docstring. Calling `_row` directly is what bypasses
        # `build_rows`'s untrained-only gate without editing the file that gate lives in.
        row = macro_metrics._row(
            f"{key}__recal",
            e.label,
            e,
            cells,
            attempts,
            missing,
            recal=True,
            allow_partial=allow_partial,
            n_boot=n_boot,
            smece_boot=smece_boot,
        )
        row["per_domain_rows"] = per_domain_recal_rows(cells, attempts, n_boot=n_boot, smece_boot=smece_boot)
        # The same agreement check the unrecalibrated path runs: both sides compute the point estimate
        # as `_all_metrics(rows, recal)` at the default seed, so they must match to 1e-12. This is the
        # invariant that makes reconstructing the per-domain CI in a second place safe.
        per_domain_metrics._check_points(row)
        rows.append(row)
    return rows
