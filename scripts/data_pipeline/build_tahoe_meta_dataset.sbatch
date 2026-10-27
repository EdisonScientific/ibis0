#!/bin/bash
#SBATCH --job-name=tahoe-meta-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_dataset.%j.log
#SBATCH --error=logs/build_dataset.%j.log
#SBATCH --partition=main

set -euo pipefail

WORKTREE="$(git rev-parse --show-toplevel)"
cd "$WORKTREE"
[ -f .env ] && { set -a; source .env; set +a; }
echo "[sbatch] host=$(hostname)  job=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK"
echo "[sbatch] worktree=$(pwd)  python=$(which python)"
echo "[sbatch] free memory at start: $(awk '/MemAvailable/{printf "%.1f GB", $2/1024/1024}' /proc/meminfo)"
echo

uv run python -u scripts/data_pipeline/build_tahoe_meta_dataset.py

echo
echo "[sbatch] done. output:"
ls -la ${IBIS_DATA:-ibis_data}/raw/tahoe-meta-data/v1/
