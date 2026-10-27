#!/usr/bin/env python
"""Materialize a published multi-domain test set at a PINNED revision into local parquet.

Optional. The test-set evals already load an `hf://` TESTSET at the pinned `TESTSET_REV`, so this is only
needed to score from local files: offline compute nodes, or a Hugging Face cache the job cannot write to.
Download the revision once, then point the eval's TESTSET at the local `<out>/testsets/<set>/{domain}` dirs.
Setting HF_HUB_CACHE to a writable directory keeps the one-off download independent of any shared cache:

    uv run --no-sync python scripts/analysis/multidomain/materialize_testset.py --out ibis_data/edb_bio_v1.0.0

Layout written (the shape `load_dataset_split` expects for a local data_path):
    <out>/<config>/<domain>/test.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ibis_env_utils import Split, load_dataset_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="EdisonScientific/EDB-Bio", help="hf dataset repo")
    ap.add_argument("--revision", default="v1.0.0", help="tag or commit SHA to pin (not a branch)")
    ap.add_argument("--configs", nargs="+", default=["testsets/primary", "testsets/sensitivity"])
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    ap.add_argument("--split", default="test", choices=[s.value for s in Split])
    ap.add_argument(
        "--out", required=True, type=Path, help="local root; writes <out>/<config>/<domain>/<split>.parquet"
    )
    args = ap.parse_args()

    split = Split(args.split)
    total = 0
    for cfg in args.configs:
        for dom in args.domains:
            df = load_dataset_split(f"hf://{args.repo}/{cfg}/{dom}", split, revision=args.revision)
            out = args.out / cfg / dom
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / f"{split.value}.parquet", index=False)
            total += len(df)
            print(
                f"  {cfg:11s} {dom:6s} rows={len(df):6d} episodes={df['episode_id'].nunique():5d} -> {out / f'{split.value}.parquet'}"
            )
    print(f"materialized {total:,} rows at {args.out} (revision {args.revision[:8]})")


if __name__ == "__main__":
    main()
