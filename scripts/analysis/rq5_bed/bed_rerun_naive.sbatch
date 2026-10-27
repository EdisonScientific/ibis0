#!/bin/bash
#SBATCH --job-name=bed-rerun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o logs/bed-rerun-%j.log
#SBATCH -e logs/bed-rerun-%j.log
#
# Naive-CHOOSER arms with the selection system prompt (CHOOSER_SYS): naive_base, naive_rld (local
# servers) + naive_gpt56 (GPT-5.6-sol via API). The belief-CLS tracker (6 replicas) is the y-axis for
# every arm. Replays the 512-episode set (seed 0, same max-candidates).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] && { set -a; source .env; set +a; }
BASE_CKPT="${BASE_CKPT:-Qwen/Qwen3-30B-A3B}"
RLD_CKPT="${RLD_CKPT:-${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__rl-direct__r2.1.0/weights/step_4200}"
BELIEF_RUN="${BELIEF_RUN:-${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0}"
BELIEF_TOML="train/multi_meta/sft/bed-simulator/bed_belief_serve.toml"
OUT_DIR="${OUT_DIR:-${IBIS_DATA:-ibis_data}/bed_sim/bed_choosers512}"; mkdir -p "$OUT_DIR"
DOMAIN="${DOMAIN:-gwas}"; MAX_EP="${MAX_EP:-512}"; BUDGET="${BUDGET:-16}"; CANDS="${CANDS:-16}"
GPT_MODEL="${GPT_MODEL:-gpt-5.6-sol}"
ROOT="$(git rev-parse --show-toplevel)"
CLS_PYPATH="$ROOT/train/_common/classification:$ROOT/envs/ibis_env_utils:$ROOT/envs/gwas_meta:$ROOT/envs/depmap_meta:$ROOT/envs/tahoe_meta"

hf_serve(){ CUDA_VISIBLE_DEVICES=$1 PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/hf_openai_serve.py --ckpt "$2" --port $3 > "$OUT_DIR/srvr_${DOMAIN}_$3.log" 2>&1 & }
hf_serve 6 "$BASE_CKPT" 8006                                    # base LM
hf_serve 7 "$RLD_CKPT" 8007                                     # RL'd LM
for i in 0 1 2 3 4 5; do                                        # 6 belief-CLS replicas
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$CLS_PYPATH" uv run --no-sync torchrun --nnodes=1 --nproc-per-node=1 \
    --master-port=$((29500+i)) scripts/classification/eval_classification_head.py \
    @ "$BELIEF_TOML" --run-dir "$BELIEF_RUN" --ckpt.resume-step 2400 --serve --serve-port $((8000+i)) \
    > "$OUT_DIR/srvr_${DOMAIN}_$((8000+i)).log" 2>&1 &
done

BELIEF_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1,http://localhost:8005/v1"
BASE_URL="http://localhost:8006/v1"; RLD_URL="http://localhost:8007/v1"
hc(){ curl -sf -m 20 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d "$SIMP" >/dev/null 2>&1; }
hcb(){ curl -sf -m 25 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d "$BELP" 2>/dev/null | grep -q probability; }
SIMP='{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":2}'
BELP='{"model":"belief","messages":[{"role":"system","content":"c"},{"role":"user","content":"q"}],"max_tokens":4}'
echo "waiting for CRITICAL servers (6 belief + base + rld)..."
for a in $(seq 1 120); do
  ok=1; for p in 8000 8001 8002 8003 8004 8005; do hcb "http://localhost:$p/v1" || ok=0; done
  hc "$BASE_URL" || ok=0; hc "$RLD_URL" || ok=0
  [ "$ok" = 1 ] && { echo "SERVERS UP (attempt $a)"; break; }
  sleep 10
done

[ -n "${OPENAI_API_KEY:-}" ] || { echo "FATAL: no OPENAI_API_KEY"; exit 1; }
echo "======== BED re-run naive: $DOMAIN ($MAX_EP ep, budget $BUDGET) base+rld+$GPT_MODEL, FIXED prompt ========"
PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/bed_thetaprime.py \
  --domain "$DOMAIN" --belief-url "$BELIEF_URLS" --sim-url "$BELIEF_URLS" --sim-model belief \
  --base-url "$BASE_URL" --base-model "$BASE_CKPT" --rld-url "$RLD_URL" --rld-model "$RLD_CKPT" \
  --gpt-model "$GPT_MODEL" --gpt-key "$OPENAI_API_KEY" \
  --policies "naive_base,naive_rld,naive_gpt56" \
  --max-episodes "$MAX_EP" --budget "$BUDGET" --max-candidates "$CANDS" \
  --concurrency "${CONC:-128}" --ext-concurrency "${EXT:-24}" --seed 0 \
  --out "$OUT_DIR/rerun_$DOMAIN.jsonl"
echo "DONE bed-rerun $DOMAIN"
