#!/bin/bash
#SBATCH --job-name=classifier
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH -o logs/%x_%j.log
#SBATCH -e logs/%x_%j.log
#
# Unified classification-head launcher for every domain. Single-node 8-GPU torchrun of the shared trainer
# (train/_common/classification/train_classification_head.py), composing the shared recipe
# (train/_common/classification-base.toml) with the run's own train.toml.
#
# Submit with the run directory as the first argument, and name the job after the run:
#   RUN=train/multi_meta/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0
#   sbatch --job-name="$(basename "$RUN")" train/_common/launch-classification.sbatch "$RUN"
#
# The domain (for PYTHONPATH) is derived from the run path (train/<domain>_meta/...); override
# OUTPUT_DIR=... to redirect checkpoints/logs.
set -euo pipefail

RUN_DIR="${1:?usage: sbatch [--job-name=<run-id>] launch-classification.sbatch <run-config-dir>}"
RUN_DIR="${RUN_DIR%/}"
RUN_ID="$(basename "$RUN_DIR")"

# Domain from the path: train/<domain>_meta/... -> <domain>. Drives PYTHONPATH (the env module).
# For the cross-domain runs under train/multi_meta/... this yields the literal "multi": the run's
# config may pull records from any of gwas/depmap/tahoe, so all three env modules go on PYTHONPATH.
DOMAIN="$(printf '%s\n' "$RUN_DIR" | sed -n 's#.*train/\([a-z0-9]*\)_meta/.*#\1#p')"
[ -n "$DOMAIN" ] || { echo "error: could not derive domain from run dir '$RUN_DIR'" >&2; exit 1; }

if [ "$DOMAIN" = "multi" ]; then
  ENV_MODULES=(gwas_meta depmap_meta tahoe_meta)   # envs/tahoe_meta also supplies the `episodes` module
else
  ENV_MODULES=("${DOMAIN}_meta")
fi

# Worktree-aware: resolve the repo root of wherever `sbatch` was invoked (not a pinned checkout);
# override by exporting PROJECT_DIR.
PROJECT_DIR="${PROJECT_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)}"
OUTPUT_DIR="${OUTPUT_DIR:-${IBIS_DATA:-ibis_data}/runs/$RUN_ID}"

# Absolute PYTHONPATH for the selected domain env module(s) + ibis_env_utils (resolved here where
# PROJECT_DIR is known, then exported into the srun shell).
ENV_PYTHONPATH=""
for m in "${ENV_MODULES[@]}"; do ENV_PYTHONPATH="${ENV_PYTHONPATH}$PROJECT_DIR/envs/$m:"; done
ENV_PYTHONPATH="${ENV_PYTHONPATH}$PROJECT_DIR/envs/ibis_env_utils"

BASE_CONFIG="train/_common/classification-base.toml"
RUN_CONFIG="$RUN_DIR/train.toml"

cd "$PROJECT_DIR"
[ -f .env ] && { set -a; source .env; set +a; }
# .env (optional, gitignored; see .env.example) carries HF_TOKEN, WANDB_* and cache locations.

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=8

export PROJECT_DIR OUTPUT_DIR DOMAIN ENV_PYTHONPATH BASE_CONFIG RUN_CONFIG MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE
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
    train/_common/classification/train_classification_head.py \
    @ "$BASE_CONFIG" @ "$RUN_CONFIG" --output-dir "$OUTPUT_DIR"
'
