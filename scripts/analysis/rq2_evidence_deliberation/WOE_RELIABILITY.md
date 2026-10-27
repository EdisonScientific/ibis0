# Weight-of-Evidence reliability diagram

A belief-state-invariant analogue of a standard calibration/reliability diagram, applied to the
*incremental weight of evidence* each system assigns as it reads one more evidence stub. Where a
calibration diagram asks "when a system says P, is it right P of the time?", this asks "when a system
claims a stub is worth `w` decibans of evidence, do the odds of the truth actually move by `w`?" — and
it does so in a way that does not depend on where the system's belief started (its incoming state).

Macro-averaged over the three domains, plus one panel per domain, one curve per system — matching the
paper's standard calibration diagrams.

## Quantities

For system `m`, domain `d`, episode `i`, stub position `j` (context length `j-1 -> j` on the nested
sensitivity sweep):

- stated probability `p_{m,i,j}` = belief that the held-out measurement supports the target, after `j`
  stubs (the context-size sweep value at `k=j`).
- **claimed weight of evidence** `w = (logit(p_j) - logit(p_{j-1})) * 10/ln(10)` decibans — the log
  Bayes factor an ideal Bayesian attaches to stub `j`.
- **incoming belief** `ell = logit(p_{j-1})` (nats) — the nuisance state we make the diagram invariant to.
- **outcome** `theta_i in {0,1}` — the episode's ground-truth label. Fixed per episode, shared by all
  its stubs.

`logit` clips `p` to `[1e-3, 1-1e-3]`, so `ell` and `w` saturate near +-30 dB.

## Construction (`analyze_woe_reliability.py`)

1. **Rows** — one per `(m, d, i, j)`, `j = 1..16`, carrying `w`, `ell`, `theta`.
2. **Strata `S`** = quantile bins of `ell` (incoming belief), `S=5`, per system/domain. The nuisance we
   condition on; pooling over strata is what makes the exchange rate belief-state invariant.
3. **Tiers `K`** = per-system quantile bins of `w`, `K=5`.
4. **Realized gap** — for each tier vs the **near-neutral (`w~=0`) reference tier**, the Mantel-Haenszel
   pooled log odds-ratio of `theta` across the `ell` strata (hand-rolled; a Haldane-Anscombe 0.5
   correction keeps a perfectly separable saturated tier finite). **Implied gap** = the tier's mean `w`
   (nats) minus the reference tier's. Anchoring to the neutral tier (rather than the weakest one) puts
   the origin at "no net evidence", so both axes are **signed**: tiers of negative claimed WoE (evidence
   *against* the hypothesis) fall to negative x, and if that claimed disbelief is realized they land at
   negative y (third quadrant). The reference tier rank is picked once per system across domains so the
   macro average-by-rank keeps a stable origin.
5. **Diagram** — realized gap (y) vs implied gap (x), per system, through the origin; dashed identity =
   ideal exchange rate. A curve shallower than the diagonal (in either direction) overclaims.
6. **Slopes** — two summary exchange rates, both `<1` = overclaims. **`fit_slope`** is the
   through-origin least-squares slope of the reliability points: this is the paper's beta and the
   figure's slope bars. **beta_WoE** is a stratum-adjusted logistic slope of `theta` on `w` (the
   unconditional analogue of conditional logistic; agrees with it at these row counts, and is
   reference-invariant), kept as a cross-check.

**Uncertainty** — episode-cluster bootstrap over the 256 episodes/domain (macro = equal-weight over
domains, never pooling episodes across domains). **Zero atom** — stubs with
exactly `w=0` (belief unchanged) are a point mass; `P(w=0)` is reported per system (the `z=` annotation)
and excluded from the curve.

**Single-draw caveat (no EIV).** The inputs are one draw per `(episode, k)`, so `ell` carries
finite-sample noise that attenuates beta_WoE toward 0 (regression dilution): the reported beta_WoE is a
*conservative* estimate of overclaiming, and the ranking/shape of curves is what to read, not the
absolute value. The deterministic classifier has noise-free `ell` and needs no such correction. Replicate
sweeps (`w` from one replicate, `ell` from another) would remove the attenuation.

## Input data — reuse, no regeneration

The test-set evaluation already contains complete nested `k=0..16` sensitivity sweeps for the full
7-system roster, so there is **no model inference in this analysis**:

    ibis_data/eval/EDB-Bio@df43070d/<model>/<domain>_ctx{k}.json

- `<model>` == the `roster_palette.py` `eval_key` for each of `base_direct, base_thinking, direct_rl,
  thinking_rl, classifier, gpt56sol, opus5`.
- N = 256 episodes/domain, `k = 0..16`, 100% parsed, episode ids stable across consecutive `k`.
- schema: `{episode_id, domain, label, parsed_prob, n_stubs}`.

## Results (macro, N=256/domain, 300-bootstrap)

Values from the published `woe_reliability_beta.csv` (macro rows). `fit_slope` is the paper's β; beta_WoE is
the logistic cross-check. Point estimates are below 1 (over-reaction to evidence) for every system; RL-trained
models weigh evidence best and base models worst, with frontier models in between.

| System | fit_slope (paper β) [95% CI] | beta_WoE [95% CI] | P(w=0) |
|---|---|---|---|
| direct RL | 0.89 [0.74, 1.11] | 0.79 [0.67, 0.96] | 0.68 |
| thinking RL | 0.70 [0.57, 0.83] | 0.58 [0.49, 0.67] | 0.28 |
| Claude Opus 5 | 0.51 [0.40, 0.62] | 0.56 [0.46, 0.68] | 0.21 |
| classifier | 0.41 [0.36, 0.48] | 0.45 [0.40, 0.52] | 0.01 |
| GPT-5.6-sol | 0.37 [0.31, 0.45] | 0.38 [0.31, 0.46] | 0.17 |
| Qwen base (thinking) | 0.36 [0.28, 0.45] | 0.25 [0.19, 0.33] | 0.39 |
| Qwen base (direct) | 0.13 [0.07, 0.18] | 0.19 [0.15, 0.24] | 0.66 |

The direct (non-thinking) models also carry a large zero-atom: two-thirds of Qwen-base-direct's belief
updates are exact no-ops (coarse, often-identical probabilities), vs ~1% for the near-continuous
classifier — a separate reliability defect the `z=` annotation surfaces.

## Run

    D=scripts/analysis/rq2_evidence_deliberation
    uv run --no-sync python $D/analyze_woe_reliability.py --out outputs/bf/woe_reliability --n-boot 300
    uv run --no-sync python $D/render_woe_reliability_v2.py --stem outputs/bf/woe_reliability \
        --outdir outputs/woe_reliability_v2
    uv run --no-sync python $D/render_woe_reliability.py --stem outputs/bf/woe_reliability \
        --outdir outputs/figures

Outputs: `outputs/bf/woe_reliability_{points,beta}.csv`; `render_woe_reliability_v2.py` draws the paper's
figure (`woe_reliability_v2.{pdf,png}`: slope bars, four reliability panels and the Tahoe evidence
hierarchy) and is what `scripts/analysis/multidomain/render_all.sh` runs; `render_woe_reliability.py`
draws the full-roster diagnostics (`woe_reliability_headline`, `woe_reliability_grid`,
`woe_reliability_by_domain`). Both draw through `scripts/lib/figure_io` + `roster_palette`; `outputs/` is
gitignored.
