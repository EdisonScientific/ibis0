#!/bin/bash
#SBATCH --job-name=mm-frontier-testset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out

# SLURM wrapper for eval_frontier_testset.sh. The API path needs no GPU, so this exists purely for
# durability: a full frontier config is ~19k requests and a couple of hours, and run from an interactive
# shell it dies with that shell. Under SLURM it survives the submitting session, gets sacct history and a
# log path for free, and can be requeued. The resume journal makes a restart cost minutes, not the run.
#
# Deliberately a THIN wrapper. eval_frontier_testset.sh stays the single source of truth for how a frontier
# model is scored (effort/budget symmetry, refusal handling, result layout); this file only adds an
# allocation and the credentials. Every knob is passed straight through.
#
# **No --partition directive on purpose.** This needs no GPU, so it should land on the cluster default
# rather than camp on a B200 node. Override at submit time if you have a CPU partition:
#   sbatch -p <cpu-partition> scripts/post_training/eval_frontier_testset.sbatch anthropic claude-opus-5
#
# **API keys are sourced inside the job, not inherited.** sbatch propagates the submitting environment by
# default, so inheriting *looks* fine — until someone submits from a shell that never sourced the secrets
# and the job 401s an hour in, or the keys end up in `scontrol show job` output. Sourcing here makes the
# failure immediate and keeps the values out of the job record.
#
# Usage:
#   # primary set (the default), one provider:
#   sbatch scripts/post_training/eval_frontier_testset.sbatch anthropic claude-opus-5
#   # dense sensitivity sweep:
#   TESTSET='hf://EdisonScientific/EDB-Bio/testsets/sensitivity/{domain}' \
#     sbatch scripts/post_training/eval_frontier_testset.sbatch openai gpt-5.6-sol
#   # cost pilot before committing:
#   LIMIT=2 DOMAINS=tahoe sbatch scripts/post_training/eval_frontier_testset.sbatch anthropic claude-opus-5
#
# Args / env: $1 provider, $2 model id; EFFORT / MAXTOK / CONC / DOMAINS / LIMIT / TESTSET / TESTSET_REV
# all behave exactly as documented in eval_frontier_testset.sh.

set -uo pipefail

PROV="${1:?usage: sbatch eval_frontier_testset.sbatch <openai|anthropic> <model-id>}"
MODEL="${2:?usage: sbatch eval_frontier_testset.sbatch <openai|anthropic> <model-id>}"

# Worktree-aware: resolve the repo root of wherever `sbatch` was invoked (not a pinned checkout);
# override by exporting PROJECT_DIR. Same convention as the other testset launchers.
PROJECT_DIR="${PROJECT_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)}"
cd "$PROJECT_DIR" || { echo "error: cannot cd to PROJECT_DIR='$PROJECT_DIR'" >&2; exit 1; }

# shellcheck disable=SC1091
[ -f .env ] && { set -a; source .env; set +a; }

# Fail before the allocation does any work rather than after the first API 401.
case "$PROV" in
  anthropic) [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "error: ANTHROPIC_API_KEY is not set (export it or put it in .env)" >&2; exit 1; } ;;
  openai)    [ -n "${OPENAI_API_KEY:-}"    ] || { echo "error: OPENAI_API_KEY is not set (export it or put it in .env)" >&2; exit 1; } ;;
  *) echo "error: provider must be 'anthropic' or 'openai', got '$PROV'" >&2; exit 1 ;;
esac

echo "[$(date +%T)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) repo=$PROJECT_DIR"
exec bash scripts/post_training/eval_frontier_testset.sh "$PROV" "$MODEL"
