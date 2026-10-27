# multidomain — the published-test-set results pipeline

Build the frozen multi-domain TEST set, score every model in the roster on it, render the tables and
figures. This is the **primary** results pipeline: the same per-episode cells feed the headline
calibration/discrimination comparison, the cross-domain transfer question, and the reliability curves,
so it sits beside the other analysis directories rather than inside any one of them.

## Pipeline order

```
build_static_testset.py          -> local parquet tree  (+ Hub publish with --publish, off by default)
  |
  +-- eval_classification_testset.py   (torchrun, one node)   \
  +-- eval_generative_testset.py       (served vLLM endpoint)  }-> canonical eval tree (eval_paths.py)
  +-- eval_frontier_testset.py         (provider API)         /
        |
        +-- verify_cells.py            (coverage + per-(domain,k) parse rate)
        +-- render_headline_compact.py (paper headline figure) / render_headline.py (two-row + per-domain) -> outputs/
        +-- render_master_table.py / render_reliability.py / render_transfer_matrix.py -> outputs/
        +-- render_roc.py / render_prob_hist.py                                        -> outputs/
        +-- render_dose_response.py / render_woe_structure.py   (sensitivity only)     -> outputs/
  +-- train_test_overlap.py -> render_train_test_overlap.py  (+ source dataset)  -> outputs/
```

## `--tag` selects the config, not `--testset`

`primary` and `sensitivity` resolve to the **same** `<testset>` slug — it keys on repo + revision, not
config (see `eval_paths.testset_slug`) — so they share one model directory and are told apart only by each
cell's `tag` (`primary` vs `ctx<k>`). Consequences worth knowing before pointing a renderer anywhere:

- Passing `--testset .../sensitivity/{domain}` while leaving the tag at its default **re-reads the
  `primary` cells** and labels them as the sensitivity result. That is indistinguishable from a genuine
  null result, so every renderer that can read either config takes an explicit `--tag`.
- `--tag 'ctx*'` pools every k. `render_dose_response.py` and `render_woe_structure.py` **refuse**
  `--tag primary`, because a per-k curve and a per-stub WoE increment are both undefined on a config that
  scores each episode at one k.
- `run.json` records a `testsets` **list**, since one directory legitimately spans both configs; the scalar
  `testset` is kept only while that list has one entry, rather than left naming half the directory.

## 1. Test set

[`build_static_testset.py`](build_static_testset.py) materializes the **full episode content** — not an
`(episode_id, k)` manifest — so a regenerated upstream dataset cannot silently resolve the same
selection to different evidence. Two configs per domain:

| Config | Rows | Built by |
|---|---|---|
| `primary` | one per episode, `k` balanced over `--primary-k-range` (default `2,16`) | `--primary-n` (default 2000) episodes/domain |
| `sensitivity` | one per (episode, `k`) over the dense grid `0..--sensitivity-k-max` | `--sensitivity-n` (default 256) episodes, nested in `primary` |

Writes `<out-dir>/<config>/<domain>/test.parquet`, `<out-dir>/<config>/index.jsonl` (selection only, no
content) and `<out-dir>/provenance.json`; `--out-dir` defaults to `outputs/static_testset`. Selection is
seeded per `(--seed, domain)` and `k` is keyed off `sha256(episode_id)`, so a rebuild is byte-identical.

The published copy is `testsets/` in `EdisonScientific/EDB-Bio` at tag `v1.0.0`, built at default arguments.
Every downstream script pins that revision through **`eval_paths.DEFAULT_TESTSET_REVISION`, the single
definition** every renderer's `--revision` default reads. Cells scored against two revisions must never be
pooled, and a renderer reading a stale revision still renders a figure that looks complete, so do not
reintroduce per-script copies.

## 2. Eval scripts — one per model family

| Script | Model family | How it runs | Launcher |
|---|---|---|---|
| [`eval_classification_testset.py`](eval_classification_testset.py) | trained BCE classification head | `torchrun` under training parallelism; loads the checkpoint once and loops `SWEEP_DOMAINS` internally | [`eval_classification_testset.sbatch`](../../post_training/eval_classification_testset.sbatch) |
| [`eval_generative_testset.py`](eval_generative_testset.py) | local weights (base or RL) | OpenAI-compatible vLLM endpoint via `--base-url`; `--domain` per invocation | [`eval_generative_testset.sbatch`](../../post_training/eval_generative_testset.sbatch) (serves, polls, evals, tears down in one allocation) |
| [`eval_frontier_testset.py`](eval_frontier_testset.py) | frontier API (`--provider openai\|anthropic`) | direct API calls, `--reasoning-effort` / `--max-output-tokens` set symmetrically on both providers | [`eval_frontier_testset.sh`](../../post_training/eval_frontier_testset.sh) (plain shell — no GPU) |

The generative and frontier scripts take `--domain gwas|depmap|tahoe` plus `--data-path` and `--out-dir`;
the classifier takes its config through the prime-rl CLI and its targets through `SWEEP_DOMAINS`,
`SWEEP_TESTSET`, `SWEEP_OUT_DIR` (the launcher resolves all three).

All three write the **same per-episode JSON cell** into the tree defined by
[`scripts/lib/eval_paths.py`](../../lib/eval_paths.py):

```
<root>/<testset>/<model_key>/<domain>_<tag>.json     tag = primary | ctx<k>
<root>/<testset>/<model_key>/run.json                provenance, written by the launcher
```

`<root>` defaults to `ibis_data/eval` (`IBIS_EVAL_ROOT` overrides), `<testset>` carries the
Hub revision, and every record holds `episode_id`, `domain`, `label`, `parsed_prob`, `n_stubs`. That
shared format is what makes the renderers model-family-agnostic: they read cells, not model families.

## 3. Renderers

| Script | Shows | Emits (under `--out-dir`) |
|---|---|---|
| [`render_master_table.py`](render_master_table.py) | full roster plus a constant-`p = base rate` reference row, in two forms: **compact** (macro point estimates only, best per column in bold — the body table) and **full** (`n`, parse-fail rate, 95% clustered-bootstrap CIs and a per-domain breakdown under every row — the appendix table) | `master_table.{md,tex}`, `master_table_full.{md,tex}`, `master_table.json`; `recalibration.md`/`.tex` only with `--recalibrate` |
| [`render_reliability.py`](render_reliability.py) | reliability diagrams — a macro panel across the top, per-domain panels beneath | `reliability.pdf`, `reliability.png`, `reliability_curves.json` |
| [`render_transfer_matrix.py`](render_transfer_matrix.py) | 4x4 leave-one-out transfer matrix (rows = training mix, cols = test domain + `all` macro), with paired `LOO − all` gaps | `transfer_matrix_<metric>.md`, `transfer_matrix_<metric>.csv`, `transfer_matrix_<metric>.pdf`, `cell_metrics.json` |
| [`render_roc.py`](render_roc.py) | ROC curves — the discrimination counterpart of the reliability diagram, same layout (macro panel on top, per-domain beneath, all panels square, dashed diagonal = chance) | `roc.{pdf,png}`, `roc_curves.json`, `roc_auc.md` |
| [`render_prob_hist.py`](render_prob_hist.py) | predicted-probability distributions; `--tag primary` = final beliefs, `--tag 'ctx*'` = pooled over deliberation | `prob_hist_pooled{,_deliberation}.{pdf,png}` + `_stats.json` |
| [`render_dose_response.py`](render_dose_response.py) | **sensitivity only** — metric vs. k, macro-averaged, with a parse-rate survivorship panel and the pre-training-range k shaded | `dose_response.{pdf,png}`, `dose_response.json` |
| [`render_woe_structure.py`](render_woe_structure.py) | **sensitivity only** — per-stub weight of evidence in decibans. One figure per question, matching the source scripts: `woe_by_tier` (per domain, with Spearman ρ), `woe_by_verdict` (violins), `woe_directional` (signed WoE — does support push up and refutation down), `woe_by_position` (primacy/saturation, macro) | `woe_by_{tier,verdict,position}.{pdf,png}`, `woe_directional.{pdf,png}`, `woe_structure.{json,md}` |
| [`train_test_overlap.py`](train_test_overlap.py) + [`render_train_test_overlap.py`](render_train_test_overlap.py) | **train/test overlap** (paper Appendix A.7) — the first reads the SOURCE dataset at `--dataset-revision` and flags every test episode whose held-out measurement appears as a stub in the first 16 positions of a train/val pool, with each domain's measurement-space size and pool coverage; the second scores the roster on the full primary cells, the overlapping episodes, and the remainder, and draws the full-vs-filtered bar grid | `overlap_ids.json`, `overlap_table.{csv,md}`; `overlap_metrics.{csv,json}`, `train_test_overlap.{pdf,png}` |

And one checker, which is not a renderer but belongs between the evals and them:

| Script | Checks |
|---|---|
| [`verify_cells.py`](verify_cells.py) | every model has its full cell set (51 at `ctx*`, 3 at `primary`), listing what is missing per domain, and the **parse rate per (domain, k)**. `--fix-run-json` reconciles `cells_written` against the directory |

Defaults: `--out-dir` is `outputs/master_table` / `outputs/reliability` / `outputs/transfer_matrix`;
`--n-boot` 1000 (master, with `--smece-boot` 200 for the O(n²) sMECE kernel), 500 (reliability per-bin
CI), 2000 (transfer matrix). Master figure panels come from `--figure-metrics`, default
`brier,smece,auroc` out of `brier,smece,auroc,reliability,resolution`. Reliability uses 15 equal-mass
bins per curve (`--n-bins`) and 10 shared bins for the macro panel (`--macro-bins`, 0 disables it).
The transfer matrix selects one metric per run (`--metric`, default `brier`) and its heatmap is PDF only.
Dose-response defaults to `--metrics ptrue,brier,auroc` plus the parse-rate panel out of
`ptrue,brier,auroc,smece,mean_p,sharpness`; pass `--metrics ptrue --no-parse-panel` for the single-panel
form the paper uses. WoE and dose-response CIs are episode-clustered (`--n-boot` 1000).

`render_roc.py` reports AUROC from `calibration_metrics.compute_calibration` — the same tie-corrected
Mann-Whitney statistic as the master table, verified identical to 12 decimal places macro and per-domain —
rather than integrating the plotted curve. Its macro *curve* is per-domain TPR averaged on a shared FPR
grid, which is a different object from the mean of per-domain AUROCs; the table's number is the one to
quote. `--ci` shades the TPR band, off by default because seven overlaid bands occlude each other.

**All model curves are solid.** `roster_palette` declares a dashed `linestyle` for the one same-hue variant
(base-thinking) and for the two frontier rows, but `plot_color` already separates the shared hue by shade,
so dashing on top of it double-encodes one distinction — and the frontier rows have unique colours, so
their dash encodes nothing.

**Per-domain WoE taxonomies are the curated per-domain hierarchies** (`render_woe_structure.HIER`);
depmap's `molecular_mutation`/`molecular_cn`/`molecular_expression` context types are **collapsed** into
a single `molecular` tier, because ρ is computed against tier rank and splitting one source
tier into three would both change the rank vector and require inventing an order among them. Verdict rules
are likewise per domain (tahoe's `verdict` field, depmap's modality rules, GWAS significance at p<5e-8),
and **GWAS is omitted from `woe_directional`**: its stubs' direction is their own allele's sign, not
consistency with the target hypothesis, so the domain has no genuine refuting category.

`outputs/` is gitignored — regenerate artifacts, do not commit them.

## Shared libs (`scripts/lib/`)

| Module | Contributes |
|---|---|
| [`eval_paths.py`](../../lib/eval_paths.py) | the canonical result layout: testset slug, model keys, cell paths, `run.json` read/write (which refuses to mix provenance in one directory). Also a CLI, so bash launchers resolve paths through the same code the readers use |
| [`testset_io.py`](../../lib/testset_io.py) | `testset_passes` — the one definition of how published rows group into passes (`primary` -> one pass, `sensitivity` -> one per `k`), imported by all three evals |
| [`roster_palette.py`](../../lib/roster_palette.py) | the 7-row primary roster: order, labels, colours, linestyles, and each row's `eval_key`. LOO rows live in `render_transfer_matrix.py` instead |
| [`display_names.py`](../../lib/display_names.py) | domain/metric/model display names and each metric's direction (`lower_is_better`, `direction_arrow`) |
| [`macro_bootstrap.py`](../../lib/macro_bootstrap.py) | `macro_ci` / `paired_macro_ci` — within-domain (optionally episode-clustered) bootstrap where the macro-average is the statistic; the paired version shares one resample across models |
| [`calibration_metrics.py`](../../lib/calibration_metrics.py) | `compute_calibration`: Brier, the Murphy reliability/resolution/uncertainty decomposition, ECE/MCE, AUROC/AUPRC, and the binning-free sMECE |

## Conventions

- **Unparsed episodes are excluded from metrics, never scored as wrong.** The evals keep the row with
  `parsed_prob: null` (plus `api_status` on the frontier path), so the denominator survives; the master
  table reports a **parse-failure rate** alongside every row, because a row graded on a subset must not
  look identical to one that answered everything.
  - On a curve **over k** this exclusion is not just a caveat but a potential confound: if the parse rate
    falls with k, the metric is computed on survivors and the survivorship is a function of the x axis,
    biasing the curve the flattering way. So `render_dose_response.py` plots parse rate as a panel of the
    same figure, and `verify_cells.py` reports it per (domain, k) — read both before believing a slope.
- **Macro-averaging is the statistic**: compute the metric per domain, then take an equal-weight mean
  over domains. CIs come from a single bootstrap that resamples episodes *within* domain and
  macro-averages each draw — not from combining per-domain intervals, which would understate the width.
  A row missing a domain is left out of the macro column unless `--allow-partial` is passed.
- **PDF is the paper artifact; PNG is written alongside for quick viewing** (the transfer-matrix heatmap
  is PDF only).

