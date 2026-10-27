# Tahoe evidence deliberation (`tahoe_meta`)

Reward/evaluation environment for the Tahoe-100M domain of EDB-Bio. Each episode shows a **target
hypothesis** ("drug *D* up/down-regulates gene *G* in cell line *C*") and a pool of evidence stubs —
differential-expression results for the same gene under *other* drug × cell-line contexts — and asks
for the probability that the hypothesis is supported by the held-out measurement. The measurement that
defines the label never appears in the pool.

## Data

Load the released dataset by Hub path (`hf://EdisonScientific/EDB-Bio/tahoe`) or point `data_path` at a
local directory of `{train,val,test}.parquet`. Stubs are embedded per episode in the `context_pool`
column, pre-shuffled, so rendering the first *k* stubs gives nested prefixes.

To rebuild from Tahoe-100M: `sbatch scripts/data_pipeline/build_tahoe_meta_dataset.sbatch` (see
`scripts/data_pipeline/build_tahoe_meta_dataset.py --help` and `scripts/data_pipeline/README.md`).

## Environment

`load_environment(data_path, context_size=16, context_size_range=None, context_seed=0,
prompt_style="reasoning", reward_type="likelihood", ...)`:

- `context_size_range=(lo, hi)` draws *k* per episode uniformly in `{lo..hi}`, keyed by `episode_id`
  and `context_seed` (the paper's runs used `(2, 16)`).
- `prompt_style`: `direct` (answer only) or `reasoning` (think first).
- Prompts are rendered by the shared `ibis_env_utils` skeleton and scored by the shared rubric, as in
  `gwas_meta` and `depmap_meta`.

The run configs under `train/multi_meta/runs/` show the exact arguments used for the paper.
