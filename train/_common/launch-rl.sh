#!/usr/bin/env bash
# Unified RL launcher for every domain/experiment.
#
# Usage:
#   train/_common/launch-rl.sh <run-config-dir>
# e.g.
#   train/_common/launch-rl.sh train/multi_meta/runs/gwas-depmap-tahoe__qwen3-30b-a3b__rl-thinking__r2.1.0
#
# Composes the shared recipe (train/_common/rl-base.toml) with the run's own rl.toml + slurm.toml
# via prime-rl's left-to-right config merge, then works around prime-rl's multi-node sbatch
# generator not emitting `#SBATCH --mem` (jobs otherwise OOM at the ~16GB cgroup default): dry-run
# to generate the sbatch, then submit it with --mem=0.
#
# SMOKE TESTS: if you add `--max-steps N` to shorten a run, use N >= 5. With N <= 3 the
# orchestrator hangs on the last step waiting for a final weight broadcast the trainer skips.
#
# Env overrides:
#   OUTPUT_DIR=...   where checkpoints/logs go (default: ${IBIS_DATA:-ibis_data}/runs/<run-id>)
#   SLURM_EXCLUDE=... comma-separated nodes to exclude (transient drains/zombies — deliberately NOT
#                     committed to slurm.toml). Appended to the generated sbatch before submission.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

RUN_DIR="${1:?usage: launch-rl.sh <run-config-dir> (e.g. train/gwas_meta/runs/<run-id>)}"
RUN_DIR="${RUN_DIR%/}"
RUN_ID="$(basename "$RUN_DIR")"

BASE_CONFIG="train/_common/rl-base.toml"
RL_CONFIG="$RUN_DIR/rl.toml"
SLURM_CONFIG="$RUN_DIR/slurm.toml"
for f in "$BASE_CONFIG" "$RL_CONFIG" "$SLURM_CONFIG"; do
  [ -f "$f" ] || { echo "error: missing config $f" >&2; exit 1; }
done

OUTPUT_DIR="${OUTPUT_DIR:-${IBIS_DATA:-ibis_data}/runs/$RUN_ID}"

# Generate the sbatch (dry-run) from base + per-run overrides.
uv run rl @ "$BASE_CONFIG" @ "$RL_CONFIG" @ "$SLURM_CONFIG" \
  --dry-run True --output-dir "$OUTPUT_DIR"

# The generated sbatch inherits this shell's environment (sbatch --export=ALL), so load .env here:
# that is how HF_TOKEN, WANDB_*, cache locations and TORCH_BLAS_PREFER_CUBLASLT reach every node.
[ -f .env ] && { set -a; source .env; set +a; }
# Required on the paper's stack (cuBLAS 12.9 + PyTorch 2.10): bf16 matmul crashes without it.
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"

# Submit with --mem=0 (full node); optionally exclude transiently-bad nodes.
SBATCH_ARGS=(--mem=0)
[ -n "${SLURM_EXCLUDE:-}" ] && SBATCH_ARGS+=(--exclude "$SLURM_EXCLUDE")
sbatch "${SBATCH_ARGS[@]}" "$OUTPUT_DIR/rl.sbatch"
