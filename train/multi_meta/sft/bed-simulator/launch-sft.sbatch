#!/bin/bash
#SBATCH --job-name=bed-simulator-sft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH -o logs/bed-simulator-sft-%j.log
#SBATCH -e logs/bed-simulator-sft-%j.log
#
# SFT of the BED conditional evidence simulator (Qwen3-30B-A3B), single node 8-GPU.
#   sbatch train/multi_meta/sft/bed-simulator/launch-sft.sbatch   # the paper's theta' simulator
#   CONFIG="<sft.toml @ overrides.toml>" sbatch ...                 # any other layering
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] && { set -a; source .env; set +a; }
# A stale PYTHONPATH can shadow the venv's packages.
unset PYTHONPATH
# Logs offline by default (sync later with `wandb sync`); set WANDB_MODE=online after `wandb login`.
export WANDB_MODE="${WANDB_MODE:-offline}"
# prime-rl converts the HF weights to its own format on first load and writes the result into the
# model's snapshot directory, so HF_HUB_CACHE must be writable by the job.

CONFIG="${CONFIG:-train/multi_meta/sft/bed-simulator/sft.toml @ train/multi_meta/sft/bed-simulator/sft-thetaprime.toml}"
uv run torchrun --standalone --nproc-per-node 8 \
  -m prime_rl.trainer.sft.train @ $CONFIG ${EXTRA:-}
echo "DONE bed-simulator sft"
