"""Unit tests for the dataset-agnostic per-split loader (`load_dataset_split`)."""

import pandas as pd
import pytest
from ibis_env_utils import Split, load_dataset_split


def _frame(ids):
    return pd.DataFrame({"episode_id": list(ids), "context_pool": [None] * len(ids)})


def _write(tmp_path, by_split):
    """Write one ``<split>.parquet`` per split and return the dir."""
    for split, frame in by_split.items():
        frame.to_parquet(tmp_path / f"{split}.parquet")
    return str(tmp_path)


def test_loads_only_the_requested_split(tmp_path):
    path = _write(tmp_path, {"train": _frame(["a", "b"]), "val": _frame(["c"]), "test": _frame(["d"])})
    assert list(load_dataset_split(path, Split.TRAIN)["episode_id"]) == ["a", "b"]
    assert list(load_dataset_split(path, Split.VAL)["episode_id"]) == ["c"]
    assert list(load_dataset_split(path, Split.TEST)["episode_id"]) == ["d"]
    # the loaded frame carries no `split` column — the split is which file you asked for
    assert "split" not in load_dataset_split(path, Split.TRAIN).columns


def test_accepts_split_as_string(tmp_path):
    path = _write(tmp_path, {"train": _frame(["a"]), "val": _frame(["b"])})
    assert list(load_dataset_split(path, "val")["episode_id"]) == ["b"]


def test_missing_requested_split_raises(tmp_path):
    path = _write(tmp_path, {"train": _frame(["a"])})  # no val/test
    with pytest.raises(FileNotFoundError, match="val"):
        load_dataset_split(path, Split.VAL)


def test_unknown_split_raises(tmp_path):
    path = _write(tmp_path, {"train": _frame(["a"])})
    with pytest.raises(ValueError):
        load_dataset_split(path, "trian")  # typo -> fails before touching the filesystem


def test_columns_projection(tmp_path):
    path = _write(tmp_path, {"train": _frame(["a", "b"])})
    df = load_dataset_split(path, Split.TRAIN, columns=["context_pool"])
    assert list(df.columns) == ["context_pool"]
    assert len(df) == 2


def test_non_directory_path_raises(tmp_path):
    with pytest.raises(ValueError, match="directory"):
        load_dataset_split(str(tmp_path / "dataset.parquet"), Split.TRAIN)


def test_hf_path_requires_config():
    with pytest.raises(ValueError, match="hf://"):
        load_dataset_split("hf://org/repo", Split.TRAIN)
