"""Hermetic integration test for the DepMap builder.

Runs the real `build_depmap_dataset.py` end-to-end over the committed `depmap_mini`
fixture (a ~1 MB slice of real DepMap/PRISM CSVs) and asserts the output adheres to the
shared dataset convention: per-split parquet files with no `split` column, an `ood_group`
metadata column, schema-valid + leak-free episodes. No 3 GB source, no network, no SLURM.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from validate import validate_episodes

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures" / "depmap_mini"
BUILD = HERE / "build_depmap_dataset.py"
SPLITS = ("train", "val", "test")
N_EPISODES, TEST_SIZE, VAL_SIZE = 80, 12, 8


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the builder once over the fixture; return the output dir."""
    out = tmp_path_factory.mktemp("depmap_mini_out")
    proc = subprocess.run(
        [
            sys.executable,
            str(BUILD),
            "--depmap-dir",
            str(FIXTURE / "depmap"),
            "--prism-dir",
            str(FIXTURE / "prism"),
            "--out-dir",
            str(out),
            "--n-episodes",
            str(N_EPISODES),
            "--test-size",
            str(TEST_SIZE),
            "--val-size",
            str(VAL_SIZE),
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"build failed:\n{proc.stdout}\n{proc.stderr}"
    return out


def test_writes_all_splits_and_manifest(built):
    for s in SPLITS:
        assert (built / f"{s}.parquet").exists(), f"missing {s}.parquet"
    manifest = json.loads((built / "manifest.json").read_text())
    # test/val sizes are exact (train_test_split counts); train is the remainder.
    assert manifest["split_counts"]["test"] == TEST_SIZE
    assert manifest["split_counts"]["val"] == VAL_SIZE
    assert manifest["split_counts"]["train"] >= 1
    assert sum(manifest["split_counts"].values()) == N_EPISODES
    assert set(manifest["ood_group_counts"]) <= {"gene", "lineage", "iid"}


def test_no_split_column_but_ood_group_present(built):
    for s in SPLITS:
        df = pd.read_parquet(built / f"{s}.parquet")
        assert "split" not in df.columns  # the split is the file, not a column
        assert "ood_group" in df.columns
        assert set(df["ood_group"]) <= {"gene", "lineage", "iid"}


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


def test_no_target_leakage_in_pools(built):
    """P5: the target (gene, cell) CRISPR measurement must never appear as a stub in its own pool."""
    leaks = 0
    for s in SPLITS:
        for row in pd.read_parquet(built / f"{s}.parquet").to_dict("records"):
            for stub in row["context_pool"]:
                same_gene = stub.get("modality") == "crispr" and stub.get("gene") == row["target_gene"]
                same_cell = (stub.get("cell_metadata") or {}).get("model_id") == row["target_cell_id"]
                leaks += same_gene and same_cell
    assert leaks == 0
