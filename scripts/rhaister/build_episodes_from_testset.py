"""Materialize the episodes parquet the Rhaister baselines read, from the published TEST set.

`evaluate_tahoe_meta_baselines.py` predates the frozen multi-domain test set. It wants one parquet
carrying every episode with a `split` column to filter on, because its pooled fit modes need the fit rows
and the eval target rows in the *same* frame. The published test set is the opposite shape: one file per
(config, domain), no `split` column, because `load_dataset_split`'s contract is that a split is which file
you asked for (see `ibis_env_utils/dataset.py`). This script is the one place those two shapes are joined,
so the evaluator keeps working unchanged and nothing else has to learn about a `split` column.

It reads:

  - the published primary test set (`hf://EdisonScientific/EDB-Bio/testsets/primary/tahoe`, pinned
    by revision) as `split="test"` -- these are the 2000 episodes being scored, with `context_pool` already
    truncated to each episode's `k`;
  - optionally the source dataset's train split (`hf://EdisonScientific/EDB-Bio/tahoe`)
    as `split="train"`, which the pooled/all-context fit modes need for train target rows and for the full
    pool of prompt evidence. Prompt-local needs only the test rows, so `--no-train` skips it.

Nested fields are reduced to exactly what `utils.py` reads. That is not just a size win: the two sources
are separate parquet files whose `context_pool` structs need not agree field-for-field, and a struct
mismatch would either fail the write or silently null a column. Projecting both onto one explicit set of
keys makes the combined schema well-defined.

The `k`-truncation is the substantive difference from the original run, and it is not something this
script can or should undo. The old tahoe-only holdout showed a full 32-stub pool (mean 31.0 evidence
stubs); the primary test set shows `k in [2, 16]` (mean 8.7 evidence stubs). Rhaister can only answer when
the target drug
appears in its fit data, so prompt-local coverage falls from ~99% to ~85% purely from having fewer stubs
to fit on. That is a property of the evaluation, not a defect -- but it means the new prompt-local number
is not comparable to the published one.

Usage (from `scripts/rhaister`, in the repo's main venv -- this reads HF, it does not need Rhaister):

    python build_episodes_from_testset.py --out ibis_data/rhaister/episodes-testset.parquet

    # prompt-local only, no train split needed
    python build_episodes_from_testset.py --no-train --out ibis_data/rhaister/episodes-test-only.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "envs" / "ibis_env_utils"))
import eval_paths  # noqa: E402
from ibis_env_utils import Split, load_dataset_split  # noqa: E402

# Exactly the fields utils.py touches, per nested column. Anything else is dropped so that the test and
# train frames share one struct schema.
CELL_META_KEYS = ("cell_name", "Cell_ID_Cellosaur")
# `pvalue` is deliberately absent from the target DE: the source `ground_truth_de` has no pvalue, and
# `make_measurement_row` already falls back to padj via `de.get("pvalue", padj)`. Emitting an all-null
# column instead would type it as parquet null and buy nothing.
TARGET_DE_KEYS = ("gene", "cell_line", "drug", "log2FoldChange", "padj", "baseMean")
STUB_DE_KEYS = ("gene", "log2FoldChange", "padj", "baseMean", "pvalue")


def _project(d, keys):
    """Keep `keys` from a mapping, preserving absence rather than inventing nulls."""
    if d is None:
        return {}
    return {k: d[k] for k in keys if k in d}


def slim_episode(row: dict, split: str) -> dict:
    """Reduce one episode to the columns and nested keys the Rhaister evaluator reads."""
    stubs = []
    for stub in row["context_pool"]:
        if stub.get("kind") != "evidence":
            # Only evidence stubs carry a DE result; get_context_measurements skips the rest anyway.
            continue
        stubs.append(
            {
                "kind": "evidence",
                "y": _project(stub["y"], STUB_DE_KEYS),
                "cell_metadata": _project(stub["cell_metadata"], CELL_META_KEYS),
                "drug_metadata": {"drug": str(stub["drug_metadata"]["drug"])},
                "gene_metadata": {"gene_symbol": str(stub["gene_metadata"]["gene_symbol"])},
            }
        )
    return {
        "episode_id": str(row["episode_id"]),
        "split": split,
        "target_cell": str(row["target_cell"]),
        "target_cell_meta": _project(row["target_cell_meta"], CELL_META_KEYS),
        "target_drug": str(row["target_drug"]),
        "target_gene": str(row["target_gene"]),
        "target_direction": str(row["target_direction"]),
        "label": int(row["label"]),
        "ground_truth_de": _project(row["ground_truth_de"], TARGET_DE_KEYS),
        "context_pool": stubs,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--testset",
        default="hf://EdisonScientific/EDB-Bio/testsets/primary/tahoe",
        help="Published test-set config to score (its TEST split becomes split='test').",
    )
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION, help="Pin the test set by commit.")
    ap.add_argument(
        "--source",
        default="hf://EdisonScientific/EDB-Bio/tahoe",
        help="Source dataset supplying the TRAIN split for pooled/all-context fits.",
    )
    ap.add_argument(
        "--no-train",
        dest="include_train",
        action="store_false",
        help="Emit only the test episodes (enough for --fit-mode prompt-local).",
    )
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    test = load_dataset_split(args.testset, Split.TEST, revision=args.revision)
    rows = [slim_episode(r, "test") for r in test.to_dict("records")]
    n_test = len(rows)
    stub_counts = [len(r["context_pool"]) for r in rows]
    print(f"[test ] {n_test} episodes from {args.testset} @ {args.revision[:8]}")
    print(f"[test ] evidence stubs: total={sum(stub_counts)} mean={sum(stub_counts) / max(n_test, 1):.2f}")
    if "k" in test.columns:
        print(f"[test ] k range: {int(test['k'].min())}..{int(test['k'].max())}")

    n_train = 0
    if args.include_train:
        train = load_dataset_split(args.source, Split.TRAIN)
        train_rows = [slim_episode(r, "train") for r in train.to_dict("records")]
        n_train = len(train_rows)
        tc = [len(r["context_pool"]) for r in train_rows]
        print(f"[train] {n_train} episodes from {args.source}")
        print(f"[train] evidence stubs: total={sum(tc)} mean={sum(tc) / max(n_train, 1):.2f}")
        rows.extend(train_rows)

        overlap = {r["episode_id"] for r in rows if r["split"] == "test"} & {r["episode_id"] for r in train_rows}
        if overlap:
            # The two frames come from different repos; an id in both would mean the test episodes are not
            # actually held out of the fit data, which would invalidate the whole baseline.
            raise SystemExit(f"{len(overlap)} episode_id(s) appear in both splits, e.g. {sorted(overlap)[:3]}")

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    meta = {
        "testset": args.testset,
        "revision": args.revision,
        "source": args.source if args.include_train else None,
        "n_test": n_test,
        "n_train": n_train,
        "columns": list(frame.columns),
    }
    args.out.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[out  ] {args.out} ({len(frame)} rows, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
