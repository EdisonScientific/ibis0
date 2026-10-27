# Tahoe-meta Rhaister Baselines

This directory runs `tahoebio/Rhaister` on Tahoe-meta-derived structured matrices and converts Rhaister's predicted LFCs into the binary Brier reward used for the LLM evaluations.

## What Rhaister Can Predict

Rhaister predicts a matrix where rows are `cell + treatment` and columns are genes. Here `treatment` means `drug + dose`.

Tahoe-meta prompts do not state a dose, so we use one dose policy: for each dose-free question, ask Rhaister to predict the highest same-unit dose of the target drug available in the fit matrix. This favors Rhaister by maximizing coverage without adding the held-out target answer.

A held-out question is covered only when Rhaister can emit a prediction for the selected `cell + treatment + gene`.

### Dose (`--dose-mode ignore`)

The published episode datasets carry **no dose metadata**, so the dose policy above cannot run on them.
`build_tahoe_meta_dataset.py` only emits `concentration`/`concentration_unit` when the raw Tahoe row has
them (see its `dose_fields`), and neither `EdisonScientific/EDB-Bio/tahoe` nor the
test set built from it has those keys in `ground_truth_de` or in a stub's `y`.

`--dose-mode ignore` therefore defines a treatment as the **drug alone**. What that costs:

- **prompt-local: nothing.** Every stub in an episode is about the target gene, and no episode in the
  N=2000 test set repeats a `(cell, drug)` among its stubs (0/2000), so `(cell, drug)` already identifies
  a Rhaister row uniquely. Dose-free is exactly equivalent here.
- **pooled / all-context: 1.4% of fit rows.** That fraction shares a `(cell, drug)` within a gene and gets
  averaged together rather than held apart as separate dose rows.

Recovering dose would mean matching every episode DE row back to the raw Tahoe shards (~190 shards,
~16.5 GB) for the small effect above. `--dose-mode require` (the default) treats a treatment as
(drug, dose, unit) and errors on data without dose, so a dataset that carries dose runs unchanged under it.

## From LFC To Probability

Rhaister does not natively answer the Tahoe-meta binary question with a probability. Its main prediction is an effect size, LFC, with p/FDR-style outputs also available in the Rhaister pipeline.

To compare against the LLM Brier reward, we sign the predicted LFC by the question direction and report two conversions:

- `soft`: `sigmoid(signed_lfc)`
- `hard`: `1` when `signed_lfc > 0.5`, otherwise `0`

Questions without an observed Rhaister prediction are excluded from Brier scoring and counted through coverage.

## Baselines

The Rhaister fit rows are observed one-gene fits: no eval-gene union and no default-filled dense gene
values. We intentionally read the episode parquet instead of rejoining raw source data, so the split,
held-out targets, and prompt stubs stay aligned with the LLM benchmark.

### Current: frozen N=2000 TEST set

Scored on the published test set `hf://EdisonScientific/EDB-Bio/testsets/primary/tahoe` at revision
`v1.0.0`, run with `--dose-mode ignore`. Both runs are deterministic (re-running reproduces every metric
bit-for-bit).

| Baseline | Fit data | Coverage | Soft reward | Soft AUROC | Hard reward | Hard AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rhaister prompt-local | Evidence stubs shown in each prompt | 85.1% (1702/2000) | 0.7690 | 0.6715 | 0.5870 | 0.5902 |
| Rhaister all-context, no eval targets | Train target rows plus all structured context evidence, excluding held-out target rows | 88.5% (1769/2000) | 0.7705 | 0.6779 | 0.6032 | 0.6058 |
| Rhaister pooled, train split | Tahoe-meta train target rows only | 3.1% (61/2000) | 0.8192 | 0.7744 | 0.8033 | 0.8028 |

The train-split row is reported for continuity only: at 61 scored questions out of 2000 it is not a usable
comparison, and its higher scores are a selection effect of the tiny subset it can answer, not better
prediction. The paper's Tahoe structured-baselines appendix reports the prompt-local and all-context rows.

For the LLM roster on this same test set, read the master table produced by
`scripts/analysis/multidomain/render_master_table.py` rather than copying a number in here — Rhaister has
no roster slot in `scripts/lib/roster_palette.py` (it is Tahoe-only and coverage-limited, so it is not
macro-averaged across domains).

**Coverage is limited by context length.** Rhaister can only answer when the target drug appears in its
fit data, and the primary test set shows `k in [2, 16]` evidence stubs (mean 8.7). Coverage tracks `k`
directly:

| `k` | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt-local coverage | 55.6% | 67.7% | 86.5% | 91.7% | 95.5% | 97.0% | 99.3% | 99.3% |

**The unscored episodes are not a random subset.** As in the master table's parse-failure caveat, uncovered
questions are *excluded* from Brier scoring rather than scored as wrong, and they skew low-context: mean
`k` 5.05 among unscored vs 9.71 among scored (prompt-local), with a slightly lower positive rate (0.463 vs
0.505). Soft reward rises with `k` (0.7476 at `k<=4` to 0.7784 at `k>12`), so the headline number is
computed on the easier, higher-context part of the set and reads optimistically against a model evaluated
at 100% coverage.

## Usage

Install the local Rhaister baseline environment (its own venv — it pins an upstream Rhaister commit and
must not share the repo's):

```bash
cd scripts/rhaister
uv sync --python 3.12
```

### 1. Materialize the episodes parquet

`evaluate_tahoe_meta_baselines.py` wants one parquet with a `split` column, because its pooled fit modes
need fit rows and eval target rows in the same frame. The published test set is one file per
(config, domain) with no `split` column. `build_episodes_from_testset.py` bridges the two. Run it with the
**repo's main venv** — it reads HF and does not need Rhaister:

```bash
OUT=ibis_data/rhaister
# test episodes only -- enough for prompt-local
python build_episodes_from_testset.py --no-train --out $OUT/episodes-test-only.parquet
# test + train -- required by the pooled fit modes
python build_episodes_from_testset.py --out $OUT/episodes-testset.parquet
```

Both write a `.provenance.json` alongside recording the test-set path, revision, and row counts.

### 2. Run the baselines

Use the Rhaister venv. `--eval-split test` because the published set is the TEST split;
`--dose-mode ignore` because it carries no dose (see above).

Prompt-local baseline:

```bash
RUN=$OUT/testset-v1.0.0
./.venv/bin/python -B evaluate_tahoe_meta_baselines.py \
  --episodes-parquet $OUT/episodes-test-only.parquet \
  --eval-split test --dose-mode ignore \
  --fit-mode prompt-local \
  --work-dir $RUN/prompt-local \
  --out $RUN/prompt-local/report.json
```

Pooled all-context baseline:

```bash
./.venv/bin/python -B evaluate_tahoe_meta_baselines.py \
  --episodes-parquet $OUT/episodes-testset.parquet \
  --eval-split test --train-split train --dose-mode ignore \
  --fit-mode pooled \
  --include-context-evidence \
  --exclude-eval-targets \
  --work-dir $RUN/all-context-no-eval-targets \
  --out $RUN/all-context-no-eval-targets/report.json
```

Train-split-only baseline (very low coverage; not in the paper table):

```bash
./.venv/bin/python -B evaluate_tahoe_meta_baselines.py \
  --episodes-parquet $OUT/episodes-testset.parquet \
  --eval-split test --train-split train --dose-mode ignore \
  --fit-mode pooled \
  --work-dir $RUN/train-split \
  --out $RUN/train-split/report.json
```

Each run writes `report.json` (metrics + provenance, including `dose_mode`) and
`report.predictions.csv` (one row per scored question). Both baselines take ~3 min on 16 CPU cores; keep
the run directory keyed on the test-set revision so two publishes cannot share one.
