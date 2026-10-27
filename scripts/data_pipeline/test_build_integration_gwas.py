"""Hermetic integration test for the GWAS builder.

Runs the real `build_gwas_dataset.py` end-to-end over the committed `gwas_mini` fixture (a small
synthetic-but-schema-faithful GWAS Catalog associations TSV + Reactome/GTEx/EFO side files) and
asserts the output adheres to the shared dataset convention: per-split parquet files with no `split`
column, an `ood_group` metadata column, spec-literal labels, schema-valid + leak-free episodes. No
real ~1M-row source, no network, no SLURM.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from validate import validate_episodes

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures" / "gwas_mini"
BUILD = HERE / "build_gwas_dataset.py"
SPLITS = ("train", "val", "test")
N_EPISODES, TEST_SIZE, VAL_SIZE = 120, 16, 10


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the builder once over the fixture; return the output dir."""
    out = tmp_path_factory.mktemp("gwas_mini_out")
    proc = subprocess.run(
        [
            sys.executable,
            str(BUILD),
            "--gwas-dir",
            str(FIXTURE),
            "--out-dir",
            str(out),
            "--n-episodes",
            str(N_EPISODES),
            "--test-size",
            str(TEST_SIZE),
            "--val-size",
            str(VAL_SIZE),
            # fixture has only 6 EFO parents; a small holdout keeps all three ood_group values present
            "--holdout-traits",
            "2",
            "--holdout-gene-frac",
            "0.1",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"build failed:\n{proc.stdout}\n{proc.stderr}"
    return out


def test_missing_trait_mappings_fails_loudly(tmp_path):
    """EFO-parent grouping is a required, reproducible input: no trait_mappings -> the build errors,
    rather than silently switching to a different (fallback) behavior."""
    (tmp_path / "gwas_catalog_associations.tsv").write_bytes(
        (FIXTURE / "gwas_catalog_associations.tsv").read_bytes()
    )  # associations present, trait_mappings deliberately absent
    proc = subprocess.run(
        [sys.executable, str(BUILD), "--gwas-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "trait-parent map not found" in proc.stderr


def test_writes_all_splits_and_manifest(built):
    for s in SPLITS:
        assert (built / f"{s}.parquet").exists(), f"missing {s}.parquet"
    manifest = json.loads((built / "manifest.json").read_text())
    # test/val sizes are exact (stratified holdout counts); train is the remainder.
    assert manifest["split_counts"]["test"] == TEST_SIZE
    assert manifest["split_counts"]["val"] == VAL_SIZE
    assert manifest["split_counts"]["train"] >= 1
    assert sum(manifest["split_counts"].values()) == manifest["n_records"]
    assert set(manifest["ood_group_counts"]) <= {"gene", "trait", "iid"}


def test_no_split_column_but_ood_group_present(built):
    for s in SPLITS:
        df = pd.read_parquet(built / f"{s}.parquet")
        assert "split" not in df.columns  # the split is the file, not a column
        assert "ood_group" in df.columns
        assert set(df["ood_group"]) <= {"gene", "trait", "iid"}


def test_each_split_is_schema_valid(built):
    for s in SPLITS:
        df = pd.read_parquet(built / f"{s}.parquet")
        report = validate_episodes(df, strict=False)
        assert report["ok"], f"{s}: {report['errors']}"


def test_episode_ids_disjoint_across_splits(built):
    ids = {s: set(pd.read_parquet(built / f"{s}.parquet")["episode_id"]) for s in SPLITS}
    total = sum(len(v) for v in ids.values())
    union = len(set().union(*ids.values()))
    assert total == union, "episode_id appears in more than one split"


def test_labels_are_spec_literal(built):
    """Positives replicate at the genome-wide bar; negatives are natural held-out failures (P1)."""
    for s in SPLITS:
        for row in pd.read_parquet(built / f"{s}.parquet").to_dict("records"):
            gt = row["ground_truth"]
            if row["label"] == 1:
                assert gt["p_value"] < 5e-8, f"supported episode with non-significant held-out p={gt['p_value']}"
            else:
                assert row["negative_type"] in ("sub_threshold", "direction_inconsistent")


def test_no_mhc_region_targets(built):
    """P1: MHC/HLA-region SNPs (chr6:25-34 Mb) are excluded from the target set (kept as evidence)."""
    for s in SPLITS:
        for row in pd.read_parquet(built / f"{s}.parquet").to_dict("records"):
            m = row["target_snp_meta"]
            pos = m.get("pos")
            in_mhc = str(m.get("chr")) == "6" and pos is not None and 25_000_000 <= int(pos) <= 34_000_000
            assert not in_mhc, f"MHC-region SNP {row['target_snp']} used as a target"
            assert row["target_snp"] != "rs8000001"  # the fixture's planted MHC pair


def test_sufficiency_tiers_are_engineered(built):
    """P4: sufficiency is a designed variable — pools span rich/moderate/poor rather than collapsing
    to one tier."""
    tiers = set()
    for s in SPLITS:
        tiers |= set(pd.read_parquet(built / f"{s}.parquet")["sufficiency_tier"])
    assert tiers <= {"rich", "moderate", "poor"}
    assert len(tiers) >= 2, f"sufficiency did not vary: {tiers}"


def test_no_target_leakage_in_pools(built):
    """P5: the target (SNP, trait) replication measurement must never appear as a stub in its own pool."""
    leaks = 0
    for s in SPLITS:
        for row in pd.read_parquet(built / f"{s}.parquet").to_dict("records"):
            for stub in row["context_pool"]:
                leaks += stub["snp"] == row["target_snp"] and stub["trait"] == row["target_trait"]
    assert leaks == 0


def test_pools_respect_size_and_relevance_tiers(built):
    """Pools are bounded by pool_size and every stub carries a known relevance tier."""
    known = {
        "same_gene_same_trait",
        "same_gene_related_trait",
        "same_snp_related_trait",
        "same_snp_unrelated_trait",
        "pathway_gene_same_trait",
        "control",
    }
    manifest = json.loads((built / "manifest.json").read_text())
    pool_size = manifest["pool_size"]
    for s in SPLITS:
        for row in pd.read_parquet(built / f"{s}.parquet").to_dict("records"):
            pool = row["context_pool"]
            assert 1 <= len(pool) <= pool_size
            assert all(stub["context_type"] in known for stub in pool)
