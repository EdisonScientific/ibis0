#!/bin/bash
#SBATCH --job-name=bed-frontier
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o logs/bed-frontier-%j.log
#SBATCH -e logs/bed-frontier-%j.log
#
# Frontier-chooser BED re-run with the paper's frontier models: Claude Opus 5 + GPT-5.6-sol.
# Only the belief-CLS tracker runs locally (8 replicas on 8 GPUs, ports 8000-8007) -> the y-axis is
# the SAME classifier as every other arm; the two frontier models only *choose* the next experiment.
# Replays the exact 512-episode set per domain (seed 0, same max-candidates) so lines are comparable.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] && { set -a; source .env; set +a; }
BELIEF_RUN="${BELIEF_RUN:-${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0}"
BELIEF_TOML="train/multi_meta/sft/bed-simulator/bed_belief_serve.toml"
OUT_DIR="${OUT_DIR:-${IBIS_DATA:-ibis_data}/bed_sim/bed_choosers512}"; mkdir -p "$OUT_DIR"
DOMAIN="${DOMAIN:-gwas}"; MAX_EP="${MAX_EP:-512}"; BUDGET="${BUDGET:-16}"; CANDS="${CANDS:-16}"
OPUS_MODEL="${OPUS_MODEL:-claude-opus-5}"; GPT_MODEL="${GPT_MODEL:-gpt-5.6-sol}"
ROOT="$(git rev-parse --show-toplevel)"
CLS_PYPATH="$ROOT/train/_common/classification:$ROOT/envs/ibis_env_utils:$ROOT/envs/gwas_meta:$ROOT/envs/depmap_meta:$ROOT/envs/tahoe_meta"

for i in 0 1 2 3 4 5 6 7; do                                   # 8 belief-CLS replicas
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$CLS_PYPATH" uv run --no-sync torchrun --nnodes=1 --nproc-per-node=1 \
    --master-port=$((29500+i)) scripts/classification/eval_classification_head.py \
    @ "$BELIEF_TOML" --run-dir "$BELIEF_RUN" --ckpt.resume-step 2400 --serve --serve-port $((8000+i)) \
    > "$OUT_DIR/srvf_${DOMAIN}_$((8000+i)).log" 2>&1 &
done

BELIEF_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1"
BELP='{"model":"belief","messages":[{"role":"system","content":"c"},{"role":"user","content":"q"}],"max_tokens":4}'
hcb(){ curl -sf -m 25 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d "$BELP" 2>/dev/null | grep -q probability; }
echo "waiting for 8 belief servers..."
for a in $(seq 1 120); do
  ok=1; for p in 8000 8001 8002 8003 8004 8005 8006 8007; do hcb "http://localhost:$p/v1" || ok=0; done
  [ "$ok" = 1 ] && { echo "BELIEF SERVERS UP (attempt $a)"; break; }
  sleep 10
done

[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "FATAL: no ANTHROPIC_API_KEY"; exit 1; }
[ -n "${OPENAI_API_KEY:-}" ]    || { echo "FATAL: no OPENAI_API_KEY"; exit 1; }
echo "======== BED frontier: $DOMAIN ($MAX_EP ep, budget $BUDGET) Opus5=$OPUS_MODEL GPT=$GPT_MODEL ========"
PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/bed_thetaprime.py \
  --domain "$DOMAIN" --belief-url "$BELIEF_URLS" --sim-url "$BELIEF_URLS" --sim-model belief \
  --policies "naive_opus5,naive_gpt56" \
  --opus-model "$OPUS_MODEL" --opus-key "$ANTHROPIC_API_KEY" \
  --gpt-model "$GPT_MODEL" --gpt-key "$OPENAI_API_KEY" \
  --max-episodes "$MAX_EP" --budget "$BUDGET" --max-candidates "$CANDS" \
  --concurrency "${CONC:-128}" --ext-concurrency "${EXT:-20}" --seed 0 \
  --out "$OUT_DIR/frontier_$DOMAIN.jsonl"
echo "DONE bed-frontier $DOMAIN"
