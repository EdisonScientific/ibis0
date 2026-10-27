"""The recalibration diagnostic must cover every system, break out by domain, and stay cache-neutral.

The first two fail silently: a table
missing its trained rows still renders, and so does one with no per-domain sub-rows. The third is the
reason the code lives outside `macro_metrics` at all — see `recal_diagnostic`'s docstring.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import recal_diagnostic as rd  # noqa: E402
from macro_metrics import METRICS, _all_metrics, _macro_ci_all  # noqa: E402
from multidomain_cells import DOMAINS  # noqa: E402
from roster_palette import ORDER  # noqa: E402


@pytest.fixture
def cells():
    """Three domains of separable forecasts, two rows per episode so the clustering has something to do.

    Deliberately miscalibrated (probabilities pushed toward the extremes) so recalibration has work to
    do and a fit that silently no-ops would show up.
    """
    rng = np.random.default_rng(0)
    out = {}
    for i, d in enumerate(DOMAINS):
        rows = []
        for e in range(60):
            label = int(rng.random() < 0.5)
            p = float(np.clip(rng.normal(0.5 + (0.30 + 0.05 * i) * (2 * label - 1), 0.12), 0.01, 0.99))
            rows += [{"domain": d, "episode_id": f"{d}_{e}", "label": label, "parsed_prob": p} for _ in range(2)]
        out[d] = rows
    return out


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("IBIS_BOOT_CACHE_DIR", str(tmp_path / "boot"))
    return tmp_path / "boot"


@pytest.fixture(autouse=True)
def stub_loaders(cells, monkeypatch):
    """Serve the same synthetic cells for every roster key; the eval tree is not under test here."""
    monkeypatch.setattr(rd, "load_cells", lambda slug, key, root=None: cells)
    monkeypatch.setattr(
        rd, "load_attempts", lambda slug, key, root=None: {d: (len(v), len(v)) for d, v in cells.items()}
    )


def _rows(**kw):
    return rd.build_recal_rows("slug", list(ORDER), n_boot=20, smece_boot=5, **kw)


def test_every_roster_key_is_recalibrated():
    """Every system is recalibrated, trained ones included, so the trained rows serve as the control."""
    rows = _rows()
    assert [r["key"] for r in rows] == [f"{k}__recal" for k in ORDER]
    assert all(r["recalibrated"] for r in rows)
    trained = {"direct_rl__recal", "thinking_rl__recal", "classifier__recal"}
    assert trained <= {r["key"] for r in rows}


def test_every_row_is_broken_out_by_domain(cells):
    """Every recalibrated row carries a per-domain breakdown."""
    for row in _rows():
        assert [sub["key"] for sub in row["per_domain_rows"]] == list(DOMAINS)
        for sub in row["per_domain_rows"]:
            assert sub["n"] == len(cells[sub["key"]])
            assert all(len(sub[m]) == 3 for m in METRICS)


def test_per_domain_points_match_the_recalibrated_macro_breakdown():
    """The invariant the whole design rests on, and what `_check_points` enforces inside the builder."""
    for row in _rows():
        for sub in row["per_domain_rows"]:
            for m in METRICS:
                assert sub[m][0] == row["per_domain"][sub["key"]][m]


def test_single_domain_macro_ci_is_a_within_domain_bootstrap(cells):
    """`_macro_ci_all` on one domain must reduce to that domain's own resample, not a three-domain mean."""
    d = DOMAINS[0]
    got = _macro_ci_all({d: cells[d]}, True, 20, 5)
    want = _all_metrics(cells[d], recalibrate=True)
    for m in METRICS:
        assert got[m][0] == want[m]


def test_recalibration_actually_moves_the_probabilities(cells):
    """A no-op fit would leave every metric unchanged and every assertion above still passing."""
    plain = _all_metrics(cells[DOMAINS[0]], recalibrate=False)
    recal = _all_metrics(cells[DOMAINS[0]], recalibrate=True)
    assert recal["brier"] < plain["brier"]
    assert recal["reliability"] < plain["reliability"]


def test_cache_stays_out_of_the_per_domain_namespace(cache_dir):
    """Per-domain recal intervals are keyed by a single-domain fingerprint under the macro namespace.

    They must not land in `master_table.per_domain_ci`: that namespace's cache key includes
    `per_domain_metrics.py`, and writing there would couple the two.
    """
    _rows()
    assert sorted(p.name for p in cache_dir.iterdir()) == ["master_table.macro_ci_all"]


# Files whose bytes are hashed into a `boot_cache` key. Editing one is a decision to discard every
# interval it has cached (hours of sMECE bootstrap) -- which is
# sometimes right, but should never be a side effect of a docstring fix. Update the digest in the same
# commit as the edit, deliberately.
_CACHE_KEY_DIGESTS = {
    "macro_metrics.py": "aa014cda2344c9f686d4729b6220ad19",
    "per_domain_metrics.py": "5ec394d658eaffa064ab677ca7eef002",
}


@pytest.mark.parametrize("name,digest", sorted(_CACHE_KEY_DIGESTS.items()))
def test_bootstrap_cache_key_files_are_unchanged(name, digest):
    got = hashlib.blake2b((_HERE / name).read_bytes(), digest_size=16).hexdigest()
    assert got == digest, (
        f"{name} changed, which invalidates every bootstrap interval cached under its namespace. "
        f"If that is intended, set its digest to {got} and expect a full cold re-render."
    )
