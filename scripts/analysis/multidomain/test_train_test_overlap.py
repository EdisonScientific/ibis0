"""The overlap measurement must see a test measurement inside a training pool only where a model could
have seen it, and the filtered metrics must partition the full cell exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import render_train_test_overlap as rto  # noqa: E402
import train_test_overlap as tto  # noqa: E402


def _tahoe_stub(drug, cell, gene, lfc=1.0):
    return {
        "kind": "evidence",
        "drug_metadata": {"drug": drug},
        "cell_metadata": {"cell_name": cell},
        "y": {"gene": gene, "log2FoldChange": lfc},
    }


def _tahoe_ep(eid, drug, cell, gene, pool):
    return {"episode_id": eid, "target_drug": drug, "target_cell": cell, "target_gene": gene, "context_pool": pool}


def test_stub_key_ignores_non_measurement_stubs():
    assert tto.stub_key("tahoe", {"kind": "aggregate", "y": {"gene": "G"}}) is None
    assert tto.stub_key("tahoe", _tahoe_stub("D", "C", "G")) == ("D", "C", "G")
    assert (
        tto.stub_key("depmap", {"modality": "expression", "gene": "G", "cell_metadata": {"model_id": "ACH-1"}}) is None
    )
    assert tto.stub_key("depmap", {"modality": "crispr", "gene": "G", "cell_metadata": {"model_id": "ACH-1"}}) == (
        "G",
        "ACH-1",
    )
    assert tto.stub_key("gwas", {"modality": "association", "snp": "rs1", "trait": "height"}) == ("rs1", "height")
    assert tto.stub_key("gwas", {"modality": "association", "snp": None, "trait": "height"}) is None


def test_overlap_counts_only_rendered_positions():
    # The test target (D1, C1, G) sits at position 0 of one training pool and at position 20 of another.
    hit = _tahoe_ep("tr1", "D2", "C2", "G", [_tahoe_stub("D1", "C1", "G")])
    deep = _tahoe_ep(
        "tr2", "D3", "C3", "G", [_tahoe_stub("Dx", f"C{i}", "G") for i in range(20)] + [_tahoe_stub("D1", "C1", "G")]
    )
    test = [_tahoe_ep("te1", "D1", "C1", "G", []), _tahoe_ep("te2", "D9", "C9", "G", [])]

    r = tto.overlap("tahoe", [hit], [], test, k=16)
    assert r["leaked_episode_ids"] == ["te1"]
    assert r["test_measurement_leaked"] == 1 and r["test_measurement_leaked_any_position"] == 1

    r = tto.overlap("tahoe", [deep], [], test, k=16)
    assert r["test_measurement_leaked"] == 0, "position 20 is never rendered, so it is not an overlap"
    assert r["test_measurement_leaked_any_position"] == 1, "but it is still in the stored pool"


def test_overlap_reports_duplicate_hypotheses_and_product_space():
    # Same (drug, cell, gene) is a target in train and in test; the product space counts TARGET axes only,
    # so a drug that appears only as a stub (D2) does not enlarge it.
    train = [_tahoe_ep("tr", "D1", "C1", "G", [_tahoe_stub("D2", "C1", "G")])]
    test = [_tahoe_ep("te", "D1", "C1", "G", []), _tahoe_ep("te2", "D3", "C1", "G", [])]
    r = tto.overlap("tahoe", train, [], test)
    assert r["test_hypothesis_also_train_target"] == 1
    assert r["measurement_space"] == 2 * 1 * 1  # drugs {D1, D3} x cells {C1} x genes {G}
    assert r["measurement_space_note"] == "2 drugs x 1 cell lines x 1 genes"
    assert r["exposed_as_stubs"] == 1 and abs(r["coverage"] - 0.5) < 1e-12


def test_gwas_space_is_observed_universe():
    train = [
        {
            "episode_id": "a",
            "target_snp": "rs1",
            "target_trait": "T",
            "context_pool": [{"modality": "association", "snp": "rs2", "trait": "T"}],
        }
    ]
    test = [{"episode_id": "b", "target_snp": "rs2", "target_trait": "T", "context_pool": []}]
    r = tto.overlap("gwas", train, [], test)
    assert r["measurement_space"] == 2  # (rs1,T) target and (rs2,T) stub/target
    assert r["exposed_as_stubs"] == 1 and r["leaked_episode_ids"] == ["b"]


def test_subset_metrics_partition_the_cell():
    rng = np.random.default_rng(0)
    recs = [
        {
            "episode_id": f"e{i}",
            "label": int(i % 2),
            "parsed_prob": float(np.clip(0.5 + 0.3 * (i % 2) - 0.15 + rng.normal(0, 0.1), 0.01, 0.99)),
        }
        for i in range(200)
    ]
    leaked = {f"e{i}" for i in range(0, 200, 5)}
    m = rto.subset_metrics(recs, leaked)
    assert m["full"]["n"] == 200 and m["leaked"]["n"] == 40 and m["filtered"]["n"] == 160
    # Brier is a mean, so the full value is the size-weighted mean of the two parts.
    recon = (m["leaked"]["brier"] * 40 + m["filtered"]["brier"] * 160) / 200
    assert abs(recon - m["full"]["brier"]) < 1e-12
    assert 0.5 < m["full"]["auroc"] <= 1.0


def test_subset_metrics_degenerate_subset_reports_n_only():
    recs = [
        {"episode_id": "a", "label": 1, "parsed_prob": 0.9},
        {"episode_id": "b", "label": 0, "parsed_prob": 0.1},
        {"episode_id": "c", "label": 1, "parsed_prob": 0.8},
    ]
    m = rto.subset_metrics(recs, {"a"})
    assert m["leaked"] == {"n": 1}
    assert "brier" in m["filtered"]


def test_macro_is_equal_weight_and_requires_every_domain():
    pd_ = {d: {"full": {"brier": v}} for d, v in zip(rto.DOMAINS, (0.1, 0.2, 0.3))}
    assert abs(rto.macro(pd_, "full", "brier") - 0.2) < 1e-12
    assert rto.macro({k: v for k, v in pd_.items() if k != "gwas"}, "full", "brier") is None
