# Dataset pipeline: build → validate → publish

How evidence-deliberation episode datasets are created, split, and hosted on
HuggingFace. Every collection (GWAS, DepMap, Tahoe) is one **config** in a single
dataset repo (`EdisonScientific/EDB-Bio`), each exposing
`train` / `val` / `test` splits. (Envs load a config from
`hf://EdisonScientific/EDB-Bio/<config>`; the loader is separate.)

**Reproducibility.** Each builder is seeded: rebuilding a domain from the same raw release with the
parameters recorded in its `manifest.json` reproduces every episode's content, label and split, in order.
The one exception is `episode_id`, which is a random UUID prefix and differs between builds. Rebuilding
DepMap from the 24Q4 / PRISM 24Q2 files reproduces all 31,997 published episodes this way. The GWAS
Catalog is updated continuously, so a GWAS rebuild matches only against the same catalog download.

## The shared fields

Stub *payloads* are domain-specific, but every episode has the same required fields,
which the publish-time validator relies on:

- `episode_id` — unique.
- `label` (0/1) — supported / not-supported (balanced ~50:50).
- `context_pool` — list of ≤32 evidence stubs, each tagged with a relevance tier
  (`context_type`) and a verdict; the domain payload is free-form per dataset.
- The rendered prompt is **not** stored — the env renders it from the episode.

The split is the **file** (`train`/`val`/`test`.parquet), not a column — `load_dataset_split`
loads one file and treats every row as that split. There is no `split` column in the data.

`test` is a fixed ~2k stratified-iid held-out slice; `val` is a 256-episode monitoring
slice; `train` is the rest. Non-iid generalization tests (cell lineage, MOA, gene, …)
are their own dataset configs whose split is the disjoint axis.

## The files

- `build_<source>.py` — **one per data source.** Pulls the raw source, builds episodes
  with the shared fields, makes a stratified-iid `train`/`val`/`test` split, and writes
  `<out-dir>/{train,val,test}.parquet` (+ `manifest.json`). Bespoke per source.
- `validate.py` — `validate_episodes(df)` (label balance, per-stub relevance tier, pool
  bounds, and an optional per-dataset leakage check — does the target's own held-out
  measurement appear in its own evidence pool?) plus a CLI to validate collections,
  which also checks no `episode_id` appears in more than one split:
  `python validate.py <dir1> <dir2> …`.
- `publish.py` — uploads a collection's `train`/`val`/`test` parquets to the HF dataset
  repo and registers the config in the dataset card. Validates first.

## Adding a new dataset

```bash
# 1. build (writes train/val/test.parquet + manifest.json)
.venv/bin/python scripts/data_pipeline/build_<source>.py --out-dir .../<config>

# 2. validate
.venv/bin/python scripts/data_pipeline/validate.py .../<config>

# 3. publish (uploads + registers the config; validates again)
.venv/bin/python scripts/data_pipeline/publish.py --collection-dir .../<config> --config <config>
```

Run scripts via SLURM with `.venv/bin/python` (not on the login node). Publishing needs
the HF token readable (note: `.env` redirects `HF_HOME`).
