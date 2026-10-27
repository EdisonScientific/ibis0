"""Publish a built episode collection (train/val/test splits) to an HF dataset repo.

The repo is created private unless ``--public`` is passed. One dataset repo holds every
collection, one config per biological dataset, each exposing native `train` / `val` /
`test` HuggingFace splits laid out as
``<config>/{train,val,test}.parquet`` (+ ``<config>/manifest.json``).
The config is registered in the dataset card's ``configs:`` metadata so
``load_dataset(repo, "<config>")`` returns every split; the env loads the parquets
directly (preserving the nested stub structure) via ``hf://<repo>/<config>``.

    .venv/bin/python scripts/data_pipeline/publish.py \
        --collection-dir ibis_data/raw/tahoe --config tahoe [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import EntryNotFoundError

sys.path.insert(0, str(Path(__file__).parent))
from validate import validate_episodes

DEFAULT_REPO = "EdisonScientific/EDB-Bio"
SPLITS = ("train", "val", "test")


def _update_card_configs(repo_id: str, config: str) -> None:
    """Register/refresh this config's split files in the dataset card YAML."""
    try:
        card = DatasetCard.load(repo_id, repo_type="dataset")
    except EntryNotFoundError:
        # Only on a genuinely missing card; any other error propagates rather than
        # overwriting an existing card and de-registering other configs.
        card = DatasetCard("# ibis Evidence Deliberation\n\nEpisode datasets for the ibis project.\n")
        card.data = DatasetCardData()
    configs = [c for c in (card.data.get("configs") or []) if c.get("config_name") != config]
    configs.append(
        {
            "config_name": config,
            "data_files": [{"split": s, "path": f"{config}/{s}.parquet"} for s in SPLITS],
        }
    )
    card.data["configs"] = sorted(configs, key=lambda c: c["config_name"])
    card.push_to_hub(repo_id, repo_type="dataset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-dir", required=True, help="dir with train/val/test.parquet (+ manifest.json)")
    ap.add_argument("--config", required=True, help="config name, e.g. tahoe / depmap / chembl")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--public", action="store_true", help="create as public (default: private)")
    ap.add_argument("--tier-field", default="context_type")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--dry-run", action="store_true", help="validate only; do not upload")
    args = ap.parse_args()

    d = Path(args.collection_dir)
    ids: dict[str, set] = {}
    for split in SPLITS:
        pq = d / f"{split}.parquet"
        assert pq.exists(), f"missing {pq} (expected {', '.join(s + '.parquet' for s in SPLITS)})"
        df = pd.read_parquet(pq)
        report = validate_episodes(df, tier_field=args.tier_field, label_col=args.label_col)
        ids[split] = set(df["episode_id"]) if "episode_id" in df.columns else set()
        print(f"[{args.config}/{split}] n={len(df)} -> {json.dumps(report['stats'])}")
        if report["warnings"]:
            print(f"  warnings: {report['warnings']}")

    leaked = sum(len(s) for s in ids.values()) - len(set().union(*ids.values()))
    if leaked:
        raise ValueError(f"{leaked} episode_id(s) appear in more than one split — refusing to publish")

    if args.dry_run:
        print("\n[dry-run] both splits validated; nothing uploaded.")
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    for fname in [f"{s}.parquet" for s in SPLITS] + ["manifest.json"]:
        src = d / fname
        if not src.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=f"{args.config}/{fname}",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"{args.config}: add/update {fname}",
        )
        print(f"uploaded {args.config}/{fname}")
    _update_card_configs(args.repo_id, args.config)
    print(f'\npublished. In configs use:  data_path = "hf://{args.repo_id}/{args.config}"')


if __name__ == "__main__":
    main()
