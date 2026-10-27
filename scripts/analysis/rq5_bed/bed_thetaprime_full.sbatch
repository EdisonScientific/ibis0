#!/bin/bash
#SBATCH --job-name=bed-tp-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH -o logs/bed-tp-full-%j.log
#SBATCH -e logs/bed-tp-full-%j.log
#
# Self-contained BED experiment: start a localhost pool of theta'-simulator + belief replicas on this
# node's 8 GPUs (6 sim on 8000-8005, 2 belief on 8006-8007), wait for health, then run the 5-arm BED
# driver (random/oracle/theta_2pole/mc1/mc16) over all 3 domains. Everything on localhost -> no
# cross-node URL wiring, no orphaned servers (they die with the job).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] && { set -a; source .env; set +a; }
SIMCKPT="${SIMCKPT:-${IBIS_DATA:-ibis_data}/sft/bed-simulator-thetaprime/weights/step_1200}"
BELIEF_RUN="${BELIEF_RUN:-${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0}"
BELIEF_TOML="train/multi_meta/sft/bed-simulator/bed_belief_serve.toml"
OUT_DIR="${OUT_DIR:-${IBIS_DATA:-ibis_data}/bed_sim/bed_tp}"
MAX_EP="${MAX_EP:-150}"; BUDGET="${BUDGET:-16}"; CANDS="${CANDS:-16}"
CLS_PYPATH="$(git rev-parse --show-toplevel)/train/_common/classification:$(git rev-parse --show-toplevel)/envs/ibis_env_utils:$(git rev-parse --show-toplevel)/envs/gwas_meta:$(git rev-parse --show-toplevel)/envs/depmap_meta:$(git rev-parse --show-toplevel)/envs/tahoe_meta"
mkdir -p "$OUT_DIR"

# --- start 6 theta'-sim replicas (GPUs 0-5, ports 8000-8005) ---
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/hf_openai_serve.py \
    --ckpt "$SIMCKPT" --port $((8000+i)) > "$OUT_DIR/sim_$i.log" 2>&1 &
done
# --- start 2 belief replicas (GPUs 6-7, ports 8006-8007) ---
for i in 6 7; do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$CLS_PYPATH" uv run --no-sync torchrun --nnodes=1 --nproc-per-node=1 \
    --master-port=$((29500+i)) scripts/classification/eval_classification_head.py \
    @ "$BELIEF_TOML" --run-dir "$BELIEF_RUN" --ckpt.resume-step 2400 \
    --serve --serve-port $((8000+i)) > "$OUT_DIR/belief_$i.log" 2>&1 &
done

SIM_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1,http://localhost:8005/v1"
BELIEF_URLS="http://localhost:8006/v1,http://localhost:8007/v1"

echo "waiting for all 8 replicas to load + answer (~10 min)..."
hc_sim(){ curl -sf -m 20 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' >/dev/null 2>&1; }
hc_bel(){ curl -sf -m 25 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"belief","messages":[{"role":"system","content":"c"},{"role":"user","content":"q"}],"max_tokens":4}' 2>/dev/null | grep -q probability; }
for attempt in $(seq 1 120); do
  ok=1
  for p in 0 1 2 3 4 5; do hc_sim "http://localhost:$((8000+p))/v1" || ok=0; done
  for p in 6 7; do hc_bel "http://localhost:$((8000+p))/v1" || ok=0; done
  [ "$ok" = "1" ] && { echo "ALL REPLICAS HEALTHY (attempt $attempt)"; break; }
  sleep 10
done

for DOMAIN in gwas depmap tahoe; do
  echo "======== BED run: $DOMAIN ($MAX_EP ep, budget $BUDGET, $CANDS cand, 5 arms) ========"
  PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/bed_thetaprime.py \
    --domain "$DOMAIN" --belief-url "$BELIEF_URLS" --sim-url "$SIM_URLS" --sim-model "$SIMCKPT" \
    --max-episodes "$MAX_EP" --budget "$BUDGET" --max-candidates "$CANDS" --concurrency 256 \
    --out "$OUT_DIR/full_$DOMAIN.jsonl"
done
echo "==== ALL DOMAINS DONE ===="
PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/bed_thetaprime_report.py \
  --results "$OUT_DIR/full_*.jsonl" || true
echo "DONE bed-tp-full"
