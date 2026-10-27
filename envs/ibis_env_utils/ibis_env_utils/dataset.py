"""Dataset-agnostic per-split loader for env training.

`load_dataset_split` is the single loader every ibis collection uses. It is
deliberately content-agnostic: its only contract is the on-disk layout — one parquet
file per split (`train`/`val`/`test`) under an `hf://` repo or a local directory, with
no `split` column in the data. It makes no assumptions about row schema, so it works
for any collection (Tahoe, DepMap, GWAS).

A call loads exactly one split and returns just those rows. The split is therefore a
property of *which frame you asked for*, not a column to filter on: callers treat the
returned frame as that split unconditionally (train on `train`, validate/select on
`val` throughout optimization, hold out `test` for final reporting), which makes
cross-split leakage unrepresentable.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd


class Split(StrEnum):
    """Canonical dataset splits.

    `train` and `val` are the working splits (the env trains on one and selects on the
    other); `test` is held out for final reporting. Shared vocabulary so the loader and
    every consumer name splits the same way.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def load_dataset_split(
    data_path: str, split: Split, columns: list[str] | None = None, revision: str | None = None
) -> pd.DataFrame:
    """Load one split's parquet file into a DataFrame.

    ``data_path`` is either ``hf://<org>/<repo>/<config>`` (downloads
    ``<config>/<split>.parquet`` from a Hugging Face dataset repo) or a local directory
    holding ``<split>.parquet`` files. The returned frame is exactly the requested
    ``split`` — every row belongs to it, so callers treat the whole frame as that split
    and never filter on a ``split`` column (there isn't one). ``columns`` optionally
    restricts the returned columns.

    ``revision`` pins the HF load to a specific commit (branch, tag or SHA); the default
    is the repo's default branch. **Pass it whenever the caller reports results keyed to a
    revision.** An analysis that files its outputs under one revision while reading the
    dataset from a moving ``main`` compares two different datasets, and because episodes
    are matched by ``episode_id`` the mismatch drops rows silently rather than raising.
    Ignored for a local ``data_path``, which is already a fixed set of files.

    Raises ``FileNotFoundError`` if the requested split's file is absent: a split is
    "required" precisely by being asked for, so a caller that needs ``train`` and
    ``val`` simply loads both and each load fails loudly if its file is missing.
    """
    split = Split(split)  # accept a plain string too; fail loudly on an unknown split

    if data_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        parts = data_path[len("hf://") :].strip("/").split("/")
        if len(parts) < 3:
            raise ValueError(f"hf:// path must be hf://<org>/<repo>/<config>, got {data_path!r}")
        repo_id, config = "/".join(parts[:2]), "/".join(parts[2:])
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=f"{config}/{split}.parquet",
                repo_type="dataset",
                revision=revision,
            )
        except EntryNotFoundError as e:
            at = f" at revision {revision}" if revision else ""
            raise FileNotFoundError(
                f"{data_path}: split {split.value!r} not published{at} ({config}/{split}.parquet)"
            ) from e
        return pd.read_parquet(local, columns=columns)

    # A local directory carries no revision; ignore any passed one (the render pipeline always threads
    # a revision through --common args) rather than erroring, so a local materialization renders cleanly.
    root = Path(data_path)
    if not root.is_dir():
        raise ValueError(
            f"data_path must be hf://<org>/<repo>/<config> or a directory of <split>.parquet files; got {data_path!r}"
        )
    p = root / f"{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{data_path}: missing required split file {split}.parquet")
    return pd.read_parquet(p, columns=columns)
