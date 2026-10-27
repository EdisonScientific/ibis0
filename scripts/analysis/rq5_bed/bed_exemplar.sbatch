#!/bin/bash
#SBATCH --job-name=bed-exemplar
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH -o logs/bed-exemplar-%j.log
#SBATCH -e logs/bed-exemplar-%j.log
#
# Exemplar-EIG baseline: same p'-weighted 2-pole EIG as theta_2pole, but y+/y- are REAL supporting/
# refuting exemplars from the training corpus (no simulator). Belief-only pool: 6 belief-CLS replicas
# on this node's GPUs 0-5 (exemplar_eig makes only belief calls). Matches the 512-ep run's episodes.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] && { set -a; source .env; set +a; }
BELIEF_RUN="${BELIEF_RUN:-${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0}"
BELIEF_TOML="train/multi_meta/sft/bed-simulator/bed_belief_serve.toml"
POOLS="${POOLS:-${IBIS_DATA:-ibis_data}/bed_sim/exemplar_pools.json}"
OUT_DIR="${OUT_DIR:-${IBIS_DATA:-ibis_data}/bed_sim/bed_choosers512}"; mkdir -p "$OUT_DIR"
DOMAIN="${DOMAIN:-gwas}"; MAX_EP="${MAX_EP:-512}"; BUDGET="${BUDGET:-16}"; CANDS="${CANDS:-16}"
ROOT="$(git rev-parse --show-toplevel)"
CLS_PYPATH="$ROOT/train/_common/classification:$ROOT/envs/ibis_env_utils:$ROOT/envs/gwas_meta:$ROOT/envs/depmap_meta:$ROOT/envs/tahoe_meta"

for i in 0 1 2 3 4 5; do   # 6 belief-CLS replicas
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$CLS_PYPATH" uv run --no-sync torchrun --nnodes=1 --nproc-per-node=1 \
    --master-port=$((29500+i)) scripts/classification/eval_classification_head.py \
    @ "$BELIEF_TOML" --run-dir "$BELIEF_RUN" --ckpt.resume-step 2400 --serve --serve-port $((8000+i)) \
    > "$OUT_DIR/exsrv_${DOMAIN}_$((8000+i)).log" 2>&1 &
done
BELIEF_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1,http://localhost:8005/v1"

hcb(){ curl -sf -m 25 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"belief","messages":[{"role":"system","content":"c"},{"role":"user","content":"q"}],"max_tokens":4}' 2>/dev/null | grep -q probability; }
echo "waiting for 6 belief replicas..."
for a in $(seq 1 120); do
  ok=1; for p in 8000 8001 8002 8003 8004 8005; do hcb "http://localhost:$p/v1" || ok=0; done
  [ "$ok" = 1 ] && { echo "BELIEF POOL UP (attempt $a)"; break; }; sleep 10
done

echo "======== exemplar-EIG: $DOMAIN ($MAX_EP ep, budget $BUDGET) ========"
PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/bed_thetaprime.py \
  --domain "$DOMAIN" --belief-url "$BELIEF_URLS" --sim-url "http://localhost:8000/v1" --sim-model dummy \
  --exemplar-pools "$POOLS" --policies exemplar_eig \
  --max-episodes "$MAX_EP" --budget "$BUDGET" --max-candidates "$CANDS" --concurrency 128 --no-faith \
  --out "$OUT_DIR/exemplar_$DOMAIN.jsonl"
echo "DONE bed-exemplar $DOMAIN"
