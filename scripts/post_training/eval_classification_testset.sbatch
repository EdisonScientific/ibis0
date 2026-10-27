#!/bin/bash
#SBATCH --job-name=mm-cls-testset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out

# Published-test-set eval for a trained classification-head checkpoint, scoring EVERY eval domain in
# ONE allocation. Launcher for eval_classification_testset.py.
#
# The classifier loads the model + scalar head under TORCHRUN training parallelism (same as the
# trainer/test-eval), so it needs the full node rather than a single served GPU. Because the checkpoint
# is domain-independent, the sweep loads it once and loops DOMAINS internally: a leave-one-out row of
# the transfer matrix — one model on all three domains — is a single job, not three.
#
# Usage:
#   RUN=train/multi_meta/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0
#   STEP=2400 OUTPUT_DIR=${IBIS_DATA:-ibis_data}/runs/$(basename "$RUN") \
#     sbatch --job-name="$(basename "$RUN")-testset" scripts/post_training/eval_classification_testset.sbatch "$RUN"
#
# Required:
#   $1        RUN_DIR — the cls run config dir (supplies train.toml: model, parallelism, prompt_style).
#   STEP      checkpoint step to load; OUTPUT_DIR/checkpoints/step_<STEP> must exist.
# Optional:
#   OUTPUT_DIR   run root holding checkpoints/step_<STEP> (default ${IBIS_DATA:-ibis_data}/runs/<run-id>).
#                Set it explicitly when checkpoints live outside the default runs root.
#   DOMAINS      comma list of eval domains scored under one model load (default gwas,depmap,tahoe).
#   TESTSET      published test-set path template containing {domain} (default: the frozen primary set).
#                Point at the sensitivity config for the dense nested sweep, or at a local directory of
#                the same shape for a cheap smoke.
#   GPUS_PER_NODE  GPUs for torchrun (default 8; override + `sbatch --gres=gpu:b200:<n>` for a smoke).
#   TESTSET_REV  test-set revision, required for an hf:// set (default: df43070d, EDB-Bio tag v1.0.0).
#   Results land at ${IBIS_DATA:-ibis_data}/eval/<testset>/<run-name>__step_<N>/<domain>_<tag>.json,
#   with a run.json recording the checkpoint path, step and mtime.
set -euo pipefail

RUN_DIR="${1:?usage: STEP=<N> sbatch eval_classification_testset.sbatch <run-config-dir>}"
RUN_DIR="${RUN_DIR%/}"
RUN_ID="$(basename "$RUN_DIR")"
STEP="${STEP:?set STEP=<checkpoint step to load>}"

PROJECT_DIR="${PROJECT_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)}"
OUTPUT_DIR="${OUTPUT_DIR:-${IBIS_DATA:-ibis_data}/runs/$RUN_ID}"

# Fail before burning an allocation: a wrong OUTPUT_DIR would otherwise die deep inside the
# checkpoint load.
if [ ! -d "$OUTPUT_DIR/checkpoints/step_$STEP" ]; then
  echo "error: no checkpoint at $OUTPUT_DIR/checkpoints/step_$STEP" >&2
  echo "       (set OUTPUT_DIR explicitly if checkpoints are elsewhere, e.g. ${IBIS_DATA:-ibis_data}/runs/$RUN_ID)" >&2
  exit 1
fi

DOMAINS="${DOMAINS:-gwas,depmap,tahoe}"
# NOTE: keep the default in its own variable. Inlining it as ${TESTSET:-...primary/{domain}} looks
# fine and even works when TESTSET is unset, but when TESTSET *is* set bash mis-parses the `}` inside
# `{domain}` and appends a stray `}` to the value — so only overrides break, and only at run time.
DEFAULT_TESTSET='hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}'
TESTSET="${TESTSET:-$DEFAULT_TESTSET}"

# The sweep imports data_adapters / model_utils / train_classification_head from the shared trainer
# dir. Unlike the trainer (whose own dir Python auto-adds to sys.path), this script lives elsewhere,
# so that dir must be on PYTHONPATH explicitly — plus every eval domain's env module + ibis_env_utils.
ENV_PYTHONPATH="$PROJECT_DIR/train/_common/classification"
for m in ${DOMAINS//,/ }; do ENV_PYTHONPATH="${ENV_PYTHONPATH}:$PROJECT_DIR/envs/${m}_meta"; done
ENV_PYTHONPATH="${ENV_PYTHONPATH}:$PROJECT_DIR/envs/ibis_env_utils"

BASE_CONFIG="train/_common/classification-base.toml"
RUN_CONFIG="$RUN_DIR/train.toml"

# Results go to the canonical eval tree (scripts/lib/eval_paths.py), keyed by run name + step, NOT into
# the training run directory: eval artifacts do not belong beside checkpoints, and one tree keeps every
# model family in one place for the figures. TESTSET_REV is required for an hf:// set so results from two
# publishes can never share a directory.
# Only an hf:// set carries a revision; a local directory must not record one. Note ${VAR-default}
# (no colon) defaults just when VAR is UNSET, so an explicit TESTSET_REV="" is honored -- with
# ${VAR:-default} an empty override silently became the Hub revision, mislabelling local results.
case "$TESTSET" in
  hf://*) TESTSET_REV="${TESTSET_REV-df43070d4a1338210ed33c368d49e58da60471cb}" ;;
  *)      TESTSET_REV="" ;;
esac
eval "$(uv run --no-sync python scripts/lib/eval_paths.py resolve \
          --testset "$TESTSET" --revision "$TESTSET_REV" --run-name "$RUN_ID" --step "$STEP")"
export SWEEP_OUT_DIR="$IBIS_RESULT_DIR"
export SWEEP_TESTSET="$TESTSET"
export SWEEP_TESTSET_REV="$TESTSET_REV"
export SWEEP_DOMAINS="$DOMAINS"

cd "$PROJECT_DIR"
[ -f .env ] && { set -a; source .env; set +a; }
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${SLURM_NNODES}
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Log the checkpoint's mtime alongside its path: the path alone does not uniquely identify which
# checkpoint generation loaded (run-ids and steps can repeat across roots), and the mtime is free.
echo "[$(date +%T)] run=$RUN_ID step=$STEP domains=$DOMAINS"
echo "[$(date +%T)] ckpt=$OUTPUT_DIR/checkpoints/step_$STEP mtime=$(stat -c %y "$OUTPUT_DIR/checkpoints/step_$STEP" 2>/dev/null)"
echo "[$(date +%T)] testset=$TESTSET"
echo "[$(date +%T)] output_dir=$OUTPUT_DIR results=$SWEEP_OUT_DIR"

export PROJECT_DIR OUTPUT_DIR ENV_PYTHONPATH BASE_CONFIG RUN_CONFIG STEP MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  cd "$PROJECT_DIR"
  [ -f .env ] && { set -a; source .env; set +a; }
  # Overwrite PYTHONPATH (do not prepend): a stale entry can shadow the venv torch and break imports.
  export PYTHONPATH="$ENV_PYTHONPATH"
  uv run --no-sync torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --node-rank="$SLURM_NODEID" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    scripts/analysis/multidomain/eval_classification_testset.py \
    @ "$BASE_CONFIG" @ "$RUN_CONFIG" --output-dir "$OUTPUT_DIR" --ckpt.resume-step "$STEP"
'
# Provenance, written after the cells so cells_written reflects what actually landed. Globbed rather than
# hardcoded: the eval decides how the config splits into passes (primary -> one cell/domain, sensitivity
# -> ctx0..ctx16), so only the directory knows the real cell names. Quote before joining so the no-cells
# case stays an empty list instead of a bogus [""].
CELLS=$(cd "$SWEEP_OUT_DIR" 2>/dev/null && ls *_*.json 2>/dev/null | grep -v '^run\.json$' \
          | sed 's/\.json$//' | sed 's/.*/"&"/' | paste -sd, -)
uv run --no-sync python scripts/lib/eval_paths.py record \
  --testset "$TESTSET" --revision "$TESTSET_REV" --run-name "$RUN_ID" --step "$STEP" \
  --json "{\"model_family\":\"classifier\",\"model_source\":\"$OUTPUT_DIR/checkpoints/step_$STEP\",\"checkpoint_mtime\":\"$(stat -c %y "$OUTPUT_DIR/checkpoints/step_$STEP" 2>/dev/null)\",\"run_name\":\"$RUN_ID\",\"prompt_style\":\"direct\",\"slurm_job_id\":${SLURM_JOB_ID:-0},\"cells_written\":[$CELLS]}" >/dev/null || true
echo "[$(date +%T)] classifier test-set eval DONE — results under $SWEEP_OUT_DIR"
