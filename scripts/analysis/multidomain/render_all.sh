#!/bin/bash
# Render the complete multi-domain results set from the eval tree, in one command.
#
# Every renderer here reads the same cells and the same roster; running them by hand means retyping the
# test set and revision eight times and remembering which ones need `--tag ctx*`. Getting one of those
# wrong is silent: a renderer pointed at the wrong tag produces a plausible figure from the wrong data.
#
# Coverage is reported FIRST and never inferred from success. A model whose cells are missing does not make
# a renderer fail — the row simply does not appear — so `verify_cells.py` runs up front and its output is
# the thing to read before trusting any figure. Individual renderers are allowed to fail without aborting
# the rest; the summary at the end says which did.
#
# Usage:
#   bash scripts/analysis/multidomain/render_all.sh                    # the paper results, into outputs/
#   ONLY=primary   bash .../render_all.sh   # skip the sensitivity figures
#   ONLY=sensitivity bash .../render_all.sh
#
# Env:
#   TESTSET  published primary config template. The sensitivity CELLS live in the SAME model dirs — the
#            eval-tree slug keys on repo + revision, not config — so this stays the primary path and
#            `--tag` selects which cells to read. That holds for every renderer that only reads cells.
#   SENS_TESTSET  the sensitivity config path, derived from TESTSET. Needed because render_woe_structure
#            additionally loads the DATASET (each episode's stub pool), and there the config is not
#            interchangeable at all: primary has one k per episode, sensitivity the dense sweep.
#   REV      test-set revision, pinned — results scored against different revisions are not comparable
#   OUT      output root (default outputs/); each renderer gets its own subdirectory
#   ONLY     primary | sensitivity | all (default all)
#   *_BOOT   bootstrap draw counts -- see the pinned block below, which is the ONLY place they are
#            defined. Every renderer's flag is required with no Python default, so this script is the
#            single statement of what the paper's intervals were computed at. Override one for a
#            throwaway check (e.g. PROBE_BOOT=200), never for a render you intend to publish.
#   DATASET_REV  SOURCE-dataset revision for the train/test overlap check (default: the paper's tag v1.0.0)
#   DOSE_BY_DOMAIN  comma list of per-domain dose statistics (default "ptrue,logbf" — the paper uses both)
#   EXTRA_*  pass-through, e.g. EXTRA_MASTER="--allow-partial"
#   WOE_STEM  CSV stem (<stem>_points.csv / <stem>_beta.csv) of the WoE reliability analysis behind the
#            main-text WoE reliability figure. Defaults to the CSVs the paper's figure was rendered from,
#            which carry the published intervals; re-running analyze_woe_reliability.py reproduces their point estimates.
#   IBIS_FIG_STYLE  edison (default: the paper's house style) | classic (pre-house-style look). Read by
#            scripts/lib/figure_io.py, so it applies to every renderer below at once.
#
# BOOT/SMECE_BOOT go to both macro-CI renderers from one variable on purpose. render_headline's bar
# panels come from render_master_table's `build_rows`, so a mismatched draw count is not merely slower:
# it is a different boot_cache key, and the table and the figure would then report different CI widths
# for the same metric. One variable makes that impossible to get wrong from here.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

TESTSET="${TESTSET:-hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}}"
# Derived, not a second knob: it must name the same repo at the same revision, differing only in config.
SENS_TESTSET="${SENS_TESTSET:-${TESTSET/\/primary\//\/sensitivity\/}}"
REV="${REV:-df43070d4a1338210ed33c368d49e58da60471cb}"
# SOURCE-dataset revision the test set was built from. The overlap check reads training POOLS, which live in
# the source dataset, not the published test set; an unpinned read would compare the test set against pools it
# was never split against. v1.0.0 is the dataset tag the paper pins (Appendix A.6).
DATASET_REV="${DATASET_REV:-v1.0.0}"
OUT="${OUT:-outputs}"
ONLY="${ONLY:-all}"
DOSE_BY_DOMAIN="${DOSE_BY_DOMAIN:-ptrue,logbf}"
WOE_STEM="${WOE_STEM:-${IBIS_DATA:-ibis_data}/woe_reliability/woe_reliability}"
R=scripts/analysis/multidomain
# Overridable: the renderers need only the analysis deps, not the training stack, so PY can point at any
# interpreter that has them, e.g. PY=/path/to/ibis0/.venv/bin/python. Default is the project uv environment.
PY="${PY:-uv run --no-sync python}"

# --- bootstrap draw counts: THE single source of truth -------------------------------------------
#
# Every renderer's draw-count flag is `required=True` with NO Python default, so these values exist in
# exactly one place: here. Omitting one is an argparse error, not a plausible wrong figure.
#
# Why it is built this way: a draw count that differs from the paper's is a different `boot_cache` key, so
# it recomputes every sMECE bootstrap from scratch AND produces CI widths that do not match the paper --
# both invisible, since the figures still render perfectly.
#
# The counts are NOT uniform, and deliberately so: they scale inversely with per-draw cost. Do not
# "tidy" them to one number -- that would change every published interval. Changing any value here is a
# change to the paper's numbers and requires a full re-render.
BOOT="${BOOT:-2000}"                        # scalar macro CIs: headline(_compact) + master_table + grouped_metrics
SMECE_BOOT="${SMECE_BOOT:-200}"             # sMECE only, ~35x costlier per draw than the other metrics
CURVE_BOOT="${CURVE_BOOT:-0}"               # headline curve bands; 0 unless --ci, which does not draw them
TRANSFER_BOOT="${TRANSFER_BOOT:-2000}"      # leave-one-out transfer matrix
DOSE_BOOT="${DOSE_BOOT:-1000}"              # dose-response, per point
WOE_BOOT="${WOE_BOOT:-1000}"                # WoE structure, episode-clustered
ROC_BOOT="${ROC_BOOT:-500}"                 # ROC TPR band, per point
ROC_AUC_BOOT="${ROC_AUC_BOOT:-2000}"        # macro-AUROC CI in the roc_auc table
RELIABILITY_BOOT="${RELIABILITY_BOOT:-500}" # reliability, per bin
PROBHIST_BOOT="${PROBHIST_BOOT:-2000}"      # prob-hist grouped row, mean-forecast panel
PROBE_BOOT="${PROBE_BOOT:-4000}"            # all three belief-revision probes (cheap per-draw scalars)

# Built once so both macro-CI renderers get byte-identical flags -- they share `build_rows`, so a
# mismatch is a different `boot_cache` key and two different intervals for the same metric.
BOOT_ARGS="--n-boot $BOOT --smece-boot $SMECE_BOOT"

mkdir -p "$OUT"
PASS=(); FAIL=(); SKIP=()

# Freshness reference for the paper-figure check at the bottom. Existence is NOT the property that
# matters: a figure this script fails to produce is still sitting on disk from whenever it was last made
# by hand, so an existence test passes while the paper carries a figure built from a different palette or
# a different revision. Anything older than this stamp did not come from this run.
STAMP="$OUT/.render_stamp"
: > "$STAMP"

run() { # run <label> <subdir> <script> [args...]   — set RUN_TESTSET to override $TESTSET for one call
  local label="$1" sub="$2" script="$3"; shift 3
  local ts="${RUN_TESTSET:-$TESTSET}"
  printf '\n\033[1m>>> %s\033[0m\n' "$label"
  if $PY "$R/$script" --testset "$ts" "$@" --out-dir "$OUT/$sub"; then
    PASS+=("$label")
  else
    FAIL+=("$label")
    echo "!! $label FAILED (continuing)" >&2
  fi
}

echo "======================================================================"
echo " testset : $TESTSET"
echo " revision: $REV"
echo " out     : $OUT"
echo " scope   : $ONLY"
echo "======================================================================"

# --- coverage first, per config. Read this before trusting anything below. ---
#
# The tag must match the scope: verify_cells defaults to `ctx*`, so a primary-only render would otherwise
# print sensitivity coverage. And `--include-loo` belongs only to `primary`: the 3 leave-one-out
# classifiers deliberately do not run the dense sweep (they exist to fill the transfer matrix, which needs
# nothing from it), so including them in the sensitivity check reports a design decision as a shortfall.
if [ "$ONLY" != sensitivity ]; then
  printf '\n\033[1m>>> coverage: primary\033[0m\n'
  $PY "$R/verify_cells.py" --testset "$TESTSET" --revision "$REV" --tag primary --include-loo || true
fi
if [ "$ONLY" != primary ]; then
  printf '\n\033[1m>>> coverage: sensitivity (ctx*)\033[0m\n'
  $PY "$R/verify_cells.py" --testset "$TESTSET" --revision "$REV" --tag 'ctx*' --no-include-loo || true
fi

if [ "$ONLY" != sensitivity ]; then
  # Primary set: aggregate figures.
  #
  # `headline_compact` first — it is the paper's main-text headline figure (one row,
  # 3.1 in tall, scalars as insets). The two-row `headline` is the full-size diagnostic it was cut from
  # and still supplies the appendix per-domain companion; both share one loader and one set of drawing
  # primitives, so they cannot disagree on a number. grouped_metrics/reliability/roc are rendered further
  # down: they carry options the combined figures do not (per-key subsets, --ci bands, the roc_auc.md
  # table, --tag ctx*), and they are the diagnostics to reach for when a headline panel looks wrong.
  run "headline (compact, paper)"      headline_compact render_headline_compact.py --revision "$REV" \
      $BOOT_ARGS --curve-boot "$CURVE_BOOT" ${EXTRA_HEADLINE:-}
  run "headline (two-row) + per-domain" headline      render_headline.py      --revision "$REV" \
      $BOOT_ARGS --curve-boot "$CURVE_BOOT" ${EXTRA_HEADLINE:-}
  run "master table"                   master_table render_master_table.py \
      --revision "$REV" --recalibrate $BOOT_ARGS ${EXTRA_MASTER:-}
  run "grouped metrics"                master_table render_grouped_metrics.py \
      --revision "$REV" $BOOT_ARGS ${EXTRA_GROUPED:-}
  run "reliability"                    reliability   render_reliability.py   --revision "$REV" \
      --n-boot "$RELIABILITY_BOOT" ${EXTRA_RELIABILITY:-}
  run "ROC"                            roc           render_roc.py           --revision "$REV" --tag primary \
      --n-boot "$ROC_BOOT" --auc-boot "$ROC_AUC_BOOT" ${EXTRA_ROC:-}
  run "prob-hist (primary)"            prob_hist     render_prob_hist.py     --revision "$REV" --tag primary ${EXTRA_PROBHIST:-}
  # Train/test overlap (Appendix A.7). Two steps that must agree on the revision pair: the ids come from the
  # source dataset at DATASET_REV, the metrics from the cells scored against REV.
  printf '\n\033[1m>>> train/test overlap\033[0m\n'
  if $PY "$R/train_test_overlap.py" --dataset-revision "$DATASET_REV" --out-dir "$OUT/train_test_overlap" ${EXTRA_OVERLAP:-} &&
     $PY "$R/render_train_test_overlap.py" --overlap-ids "$OUT/train_test_overlap/overlap_ids.json" \
       --testset "$TESTSET" --revision "$REV" --out-dir "$OUT/train_test_overlap"; then PASS+=("train/test overlap"); else
       FAIL+=("train/test overlap"); echo "!! train/test overlap FAILED (continuing)" >&2; fi
  # NOTE: --testset-revision, not --revision. The odd one out; do not "fix" it here by guessing.
  printf '\n\033[1m>>> transfer matrix\033[0m\n'
  # --combined-figure is opt-in on the renderer but NOT optional here: transfer_combined.pdf is the
  # figure the paper carries; without the flag the renderer emits only the standalone figures.
  if $PY "$R/render_transfer_matrix.py" --testset "$TESTSET" --testset-revision "$REV" \
       --combined-figure --n-boot "$TRANSFER_BOOT" --out-dir "$OUT/transfer_matrix" ${EXTRA_TRANSFER:-}; then PASS+=("transfer matrix"); else
       FAIL+=("transfer matrix"); echo "!! transfer matrix FAILED (continuing)" >&2; fi
else
  SKIP+=("primary figures")
fi

if [ "$ONLY" != primary ]; then
  # Sensitivity set: everything driven by the dense nested sweep.
  # Emits BOTH dose_response_by_metric (macro-average, metric varies) and, via --by-domain-metric,
  # dose_response_by_domain_<metric> (one statistic, one panel per domain + macro).
  # `--by-domain-metric` takes ONE statistic per invocation: each (metric, domain) pair is its own
  # bootstrap, so the renderer prices them one at a time rather than accepting a list. The paper uses
  # both `ptrue` and `logbf`, so loop. Each pass also redraws the identical dose_response_by_metric
  # panel — wasted work, not a correctness problem, and the cost of the renderer's interface.
  for dose_metric in ${DOSE_BY_DOMAIN//,/ }; do
    run "dose-response ($dose_metric)" dose_response render_dose_response.py --revision "$REV" --tag 'ctx*' \
        --by-domain-metric "$dose_metric" --n-boot "$DOSE_BOOT" ${EXTRA_DOSE:-}
  done
  # WoE is the ONE renderer that needs the sensitivity path rather than the primary one. Every other
  # script here uses `--testset` only to compute the eval-tree slug, which keys on repo + revision and so
  # is the same for both configs; this one ALSO loads the parquet, to read each episode's stub pool, and
  # there the config is the whole difference — primary carries one k per episode, sensitivity the dense
  # sweep; the renderer refuses the primary path.
  RUN_TESTSET="$SENS_TESTSET" \
  run "WoE structure"        woe_structure  render_woe_structure.py  --revision "$REV" --tag 'ctx*' \
      --n-boot "$WOE_BOOT" ${EXTRA_WOE:-}
  run "prob-hist (pooled k)" prob_hist      render_prob_hist.py      --revision "$REV" --tag 'ctx*' ${EXTRA_PROBHIST:-}
  # The published distribution figure. Same cells and same tag as the line above, which stays as the
  # four-panel diagnostic to reach for when a panel looks wrong.
  run "prob-hist (grouped row)" prob_hist   render_prob_hist_grouped.py --revision "$REV" --tag 'ctx*' \
      --boot "$PROBHIST_BOOT" ${EXTRA_PROBHIST_GROUPED:-}
  # Main-text WoE reliability figure. Outside `run`: the script lives in rq2_evidence_deliberation/ and takes
  # --outdir. Its 2x2 panels and beta bars come from the analysis CSVs (WOE_STEM); its Tahoe panel reads the
  # sensitivity cells, hence the sensitivity config.
  printf '\n\033[1m>>> WoE reliability\033[0m\n'
  if [ ! -f "${WOE_STEM}_points.csv" ] || [ ! -f "${WOE_STEM}_beta.csv" ]; then
    FAIL+=("WoE reliability"); echo "!! WoE reliability: no CSVs at WOE_STEM=$WOE_STEM (continuing)" >&2
  elif $PY scripts/analysis/rq2_evidence_deliberation/render_woe_reliability_v2.py --stem "$WOE_STEM" \
       --testset "$SENS_TESTSET" --revision "$REV" --tag 'ctx*' --outdir "$OUT/woe_reliability_v2" ${EXTRA_WOE_REL:-}; then
    PASS+=("WoE reliability")
  else
    FAIL+=("WoE reliability"); echo "!! WoE reliability FAILED (continuing)" >&2
  fi
else
  SKIP+=("sensitivity figures")
fi

# --- belief-revision probes -------------------------------------------------------------------------
#
# Outside the `run` helper on purpose. These read the conflict and position probe sets, which are built
# locally by `build_probe_sets.py` and land under `local-conflict` / `local-position` in the eval tree, so
# they take neither `--testset` nor `--revision`: `eval_paths.testset_slug` REQUIRES a revision for an
# hf:// path and REFUSES one for a local directory.
for probe in render_conflict_additivity render_woe_position render_belief_revision; do
  printf '\n\033[1m>>> %s\033[0m\n' "$probe"
  if $PY "$R/$probe.py" --boot "$PROBE_BOOT" --out-dir "$OUT/belief_revision" ${EXTRA_PROBE:-}; then
    PASS+=("$probe")
  else
    FAIL+=("$probe"); echo "!! $probe FAILED (continuing)" >&2
  fi
done

# --- does this actually cover the paper? -------------------------------------------------------------
#
# Checked by NAME, because the failure mode is silent. A renderer that stops emitting one of its outputs
# still exits 0, so the summary above says "ok" while the figure on disk is whatever a stale invocation
# last wrote.
#
# NOT from here, and deliberately unchecked -- they read different data entirely: bed_*.pdf
# (scripts/analysis/rq5_bed/, filenames set by --out so they cannot be checked statically). The
# illustrative task schematic is not generated by code in this repository.
PAPER_FIGURES=(belief_revision/belief_revision.pdf
               belief_revision/conflict_additivity_panels.pdf
               belief_revision/woe_position_panels.pdf)
if [ "$ONLY" != sensitivity ]; then
  PAPER_FIGURES+=(headline_compact/headline_compact.pdf transfer_matrix/transfer_combined.pdf
                  train_test_overlap/train_test_overlap.pdf)
fi
if [ "$ONLY" != primary ]; then
  PAPER_FIGURES+=(prob_hist/prob_hist_grouped_row.pdf woe_reliability_v2/woe_reliability_v2.pdf
                  woe_structure/woe_directional.pdf woe_structure/woe_staircase.pdf)
  for dose_metric in ${DOSE_BY_DOMAIN//,/ }; do
    PAPER_FIGURES+=("dose_response/dose_response_by_domain_$dose_metric.pdf")
  done
fi
MISSING=(); STALE=()
for fig in "${PAPER_FIGURES[@]}"; do
  if [ ! -f "$OUT/$fig" ]; then MISSING+=("$fig")
  elif [ ! "$OUT/$fig" -nt "$STAMP" ]; then STALE+=("$fig")  # on disk, but not from this run
  fi
done
rm -f "$STAMP"

echo
echo "======================================================================"
printf ' ok      : %s\n' "${PASS[*]:-none}"
[ ${#SKIP[@]} -gt 0 ] && printf ' skipped : %s\n' "${SKIP[*]}"
[ ${#FAIL[@]} -gt 0 ] && printf ' FAILED  : %s\n' "${FAIL[*]}"
[ ${#MISSING[@]} -gt 0 ] && printf ' PAPER FIGURES MISSING: %s\n' "${MISSING[*]}"
[ ${#STALE[@]}   -gt 0 ] && printf ' PAPER FIGURES STALE (on disk, not from this run): %s\n' "${STALE[*]}"
[ ${#MISSING[@]} -eq 0 ] && [ ${#STALE[@]} -eq 0 ] &&
  printf ' paper    : all %d figures this script owns were rebuilt into %s\n' "${#PAPER_FIGURES[@]}" "$OUT"
echo "======================================================================"
echo "Re-read the coverage block above: a missing model is a MISSING ROW, not a failure."
[ ${#FAIL[@]} -eq 0 ] && [ ${#MISSING[@]} -eq 0 ] && [ ${#STALE[@]} -eq 0 ]
