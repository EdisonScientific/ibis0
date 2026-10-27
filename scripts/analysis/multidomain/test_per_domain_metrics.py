"""The per-domain breakdown must agree with the macro table and must not share its bootstrap cache.

Both properties are silent when broken. A per-domain point estimate computed from slightly different
records still renders; and a per-domain entry written under the macro namespace would invalidate the
macro bootstrap on the next edit to this module, costing tens of minutes with nothing naming why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import per_domain_metrics as pdm  # noqa: E402
from macro_metrics import METRICS, _all_metrics  # noqa: E402
from multidomain_cells import DOMAINS  # noqa: E402


@pytest.fixture
def cells():
    """Three domains of well-separated forecasts, two rows per episode so clustering has something to do."""
    rng = np.random.default_rng(0)
    out = {}
    for i, d in enumerate(DOMAINS):
        rows = []
        for e in range(60):
            label = int(rng.random() < 0.5)
            p = float(np.clip(rng.normal(0.5 + (0.15 + 0.05 * i) * (2 * label - 1), 0.15), 0.01, 0.99))
            rows += [{"domain": d, "episode_id": f"{d}_{e}", "label": label, "parsed_prob": p} for _ in range(2)]
        out[d] = rows
    return out


@pytest.fixture
def attempts(cells):
    return {d: (len(rows), len(rows)) for d, rows in cells.items()}


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IBIS_BOOT_CACHE_DIR", str(tmp_path / "boot"))
    return tmp_path / "boot"


def test_point_estimates_match_macro_metrics(cells, attempts):
    rows = pdm.per_domain_rows(cells, attempts, n_boot=20, smece_boot=5)
    assert [r["key"] for r in rows] == list(DOMAINS)
    for row in rows:
        want = _all_metrics(cells[row["key"]], recalibrate=False)
        for m in METRICS:
            assert row[m][0] == want[m]


def test_intervals_bracket_the_point_for_unbiased_metrics(cells, attempts):
    rows = pdm.per_domain_rows(cells, attempts, n_boot=200, smece_boot=20)
    for row in rows:
        for m in ("brier", "auroc"):
            point, lo, hi = row[m]
            assert lo < point < hi


def test_zero_draws_reports_a_point_only(cells, attempts):
    row = pdm.per_domain_rows(cells, attempts, n_boot=20, smece_boot=0)[0]
    assert not np.isnan(row["smece"][0])
    assert np.isnan(row["smece"][1]) and np.isnan(row["smece"][2])


def test_cache_stays_out_of_the_macro_namespace(cells, attempts, cache_dir):
    pdm.per_domain_rows(cells, attempts, n_boot=20, smece_boot=5)
    assert sorted(p.name for p in cache_dir.iterdir()) == ["master_table.per_domain_ci"]
    # One entry per (domain, metric): distinct keys, so no domain serves another's interval.
    assert len(list((cache_dir / "master_table.per_domain_ci").glob("*.json"))) == len(DOMAINS) * len(METRICS)


def test_constant_cells_predict_each_domains_own_base_rate(cells):
    const = pdm._constant_cells(cells)
    for d, rows in const.items():
        expected = float(np.mean([r["label"] for r in cells[d]]))
        assert {r["parsed_prob"] for r in rows} == {expected}
        assert [r["episode_id"] for r in rows] == [r["episode_id"] for r in cells[d]]


def test_check_points_rejects_a_disagreeing_breakdown(cells, attempts):
    rows = pdm.per_domain_rows(cells, attempts, n_boot=20, smece_boot=5)
    row = {
        "key": "fake",
        "per_domain_rows": rows,
        "per_domain": {r["key"]: {m: r[m][0] for m in METRICS} for r in rows},
    }
    pdm._check_points(row)
    row["per_domain"][DOMAINS[1]]["brier"] += 1e-6
    with pytest.raises(ValueError, match="disagrees with"):
        pdm._check_points(row)
