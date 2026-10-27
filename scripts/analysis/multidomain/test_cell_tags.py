"""Tests for tag-based cell selection — the part that silently corrupts a curve when it is wrong.

`primary` and `sensitivity` share a result directory (the testset slug keys on repo + revision, not
config), so the tag is the ONLY thing separating them. Two failure modes are worth pinning down because
neither raises: reading `primary` cells while believing they are the sensitivity sweep, and ordering
`ctx*` lexically so `ctx10` sorts before `ctx2` and every k is plotted against the wrong x.
"""

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in _HERE.parents if p.name == "scripts") / "lib"))
sys.path.insert(0, str(_HERE))

import eval_paths as ep  # noqa: E402
from multidomain_cells import cell_paths_for_tag, load_attempts, load_cells  # noqa: E402

HUB = "hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}"
SENS = "hf://EdisonScientific/EDB-Bio/testsets/sensitivity/{domain}"
REV = "8cc5c6030194b2212e98fa9d5aa5a7b1fda7c3"
KEY = "gpt-5.6-sol"


def _cell(root, slug, domain, tag, probs):
    p = ep.cell_path(slug, KEY, domain, tag, root=root)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"episode_id": f"e{i}", "domain": domain, "label": i % 2, "parsed_prob": v, "n_stubs": 1}
        for i, v in enumerate(probs)
    ]
    p.write_text(json.dumps({"domain": domain, "results": rows}))
    return p


def test_primary_and_sensitivity_share_a_slug(tmp_path):
    # The premise of every other test here: the tag, not the testset path, is the discriminator.
    assert ep.testset_slug(HUB, REV) == ep.testset_slug(SENS, REV)


def test_ctx_star_orders_by_k_not_lexically(tmp_path):
    slug = ep.testset_slug(SENS, REV)
    for k in range(17):
        _cell(tmp_path, slug, "tahoe", f"ctx{k}", [0.5])
    got = cell_paths_for_tag(slug, KEY, "tahoe", "ctx*", root=tmp_path)
    ks = [int(p.stem.rsplit("_ctx", 1)[1]) for p in got]
    assert ks == list(range(17)), "ctx10 must not sort before ctx2"


def test_ctx_star_ignores_primary_and_vice_versa(tmp_path):
    slug = ep.testset_slug(SENS, REV)
    _cell(tmp_path, slug, "tahoe", "primary", [0.1] * 5)
    _cell(tmp_path, slug, "tahoe", "ctx0", [0.9] * 3)
    _cell(tmp_path, slug, "tahoe", "ctx1", [0.9] * 3)
    assert len(load_cells(slug, KEY, root=tmp_path, tag="primary")["tahoe"]) == 5
    assert len(load_cells(slug, KEY, root=tmp_path, tag="ctx*")["tahoe"]) == 6
    assert len(load_cells(slug, KEY, root=tmp_path, tag="ctx1")["tahoe"]) == 3


def test_a_tag_with_no_cells_yields_no_domain(tmp_path):
    # A missing tag must be absent, not silently fall back to whatever else is in the directory.
    slug = ep.testset_slug(SENS, REV)
    _cell(tmp_path, slug, "tahoe", "primary", [0.4])
    assert load_cells(slug, KEY, root=tmp_path, tag="ctx*") == {}


def test_attempts_sum_over_pooled_cells_and_count_unparsed(tmp_path):
    slug = ep.testset_slug(SENS, REV)
    _cell(tmp_path, slug, "gwas", "ctx0", [0.5, None, 0.5])
    _cell(tmp_path, slug, "gwas", "ctx1", [0.5, 0.5, 0.5])
    # 6 attempted, 5 parsed: the denominator must survive pooling, or the parse rate is unrecoverable.
    assert load_attempts(slug, KEY, root=tmp_path, tag="ctx*")["gwas"] == (6, 5)
    assert len(load_cells(slug, KEY, root=tmp_path, tag="ctx*")["gwas"]) == 5
