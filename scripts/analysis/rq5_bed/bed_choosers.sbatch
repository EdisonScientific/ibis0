#!/bin/bash
#SBATCH --job-name=bed-choosers
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH -o logs/bed-choosers-%j.log
#SBATCH -e logs/bed-choosers-%j.log
#
# 128-episode BED policy comparison with model_select choosers. localhost pool on 8 GPUs:
#   4 theta'-sim (8000-8003), 2 belief-CLS (8004-8005), 1 base LM (8006), 1 RL'd LM (8007).
# Arms: random, naive_base, naive_rld, mc16, theta_2pole (+ naive_gpt55 if OPENAI_API_KEY is set).
# Belief TRACKER (y-axis) is the CLS classifier for ALL arms -> isolates selection quality.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] && { set -a; source .env; set +a; }
SIMCKPT="${SIMCKPT:-${IBIS_DATA:-ibis_data}/sft/bed-simulator-thetaprime/weights/step_1200}"
BASE_CKPT="${BASE_CKPT:-Qwen/Qwen3-30B-A3B}"
RLD_CKPT="${RLD_CKPT:-${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__rl-direct__r2.1.0/weights/step_4200}"
BELIEF_RUN="${BELIEF_RUN:-${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0}"
BELIEF_TOML="train/multi_meta/sft/bed-simulator/bed_belief_serve.toml"
OUT_DIR="${OUT_DIR:-${IBIS_DATA:-ibis_data}/bed_sim/bed_choosers}"; mkdir -p "$OUT_DIR"
DOMAIN="${DOMAIN:-gwas}"; MAX_EP="${MAX_EP:-128}"; BUDGET="${BUDGET:-16}"; CANDS="${CANDS:-16}"
ROOT="$(git rev-parse --show-toplevel)"
CLS_PYPATH="$ROOT/train/_common/classification:$ROOT/envs/ibis_env_utils:$ROOT/envs/gwas_meta:$ROOT/envs/depmap_meta:$ROOT/envs/tahoe_meta"

hf_serve(){ CUDA_VISIBLE_DEVICES=$1 PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/hf_openai_serve.py --ckpt "$2" --port $3 > "$OUT_DIR/srv_${DOMAIN}_$3.log" 2>&1 & }
for i in 0 1 2 3; do hf_serve $i "$SIMCKPT" $((8000+i)); done   # 4 sim
hf_serve 6 "$BASE_CKPT" 8006                                    # base LM
hf_serve 7 "$RLD_CKPT" 8007                                     # RL'd LM
for i in 4 5; do                                               # 2 belief-CLS
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$CLS_PYPATH" uv run --no-sync torchrun --nnodes=1 --nproc-per-node=1 \
    --master-port=$((29500+i)) scripts/classification/eval_classification_head.py \
    @ "$BELIEF_TOML" --run-dir "$BELIEF_RUN" --ckpt.resume-step 2400 --serve --serve-port $((8000+i)) \
    > "$OUT_DIR/srv_${DOMAIN}_$((8000+i)).log" 2>&1 &
done

SIM_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1"
BELIEF_URLS="http://localhost:8004/v1,http://localhost:8005/v1"
BASE_URL="http://localhost:8006/v1"; RLD_URL="http://localhost:8007/v1"

hc(){ curl -sf -m 20 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d "$SIMP" >/dev/null 2>&1; }
hcb(){ curl -sf -m 25 -X POST "$1/chat/completions" -H 'Content-Type: application/json' -d "$BELP" 2>/dev/null | grep -q probability; }
SIMP='{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":2}'
BELP='{"model":"belief","messages":[{"role":"system","content":"c"},{"role":"user","content":"q"}],"max_tokens":4}'
# Health-gate ONLY the critical shared path (sim 8000-8003 + belief 8004-8005). Chooser servers
# (base/rld) are optional: probe once, include their arm only if up (a dead chooser must NOT stall the run).
echo "waiting for CRITICAL servers (4 sim + 2 belief)..."
for a in $(seq 1 120); do
  ok=1
  for p in 8000 8001 8002 8003; do hc "http://localhost:$p/v1" || ok=0; done
  hcb "http://localhost:8004/v1" || ok=0; hcb "http://localhost:8005/v1" || ok=0
  [ "$ok" = 1 ] && { echo "CRITICAL SERVERS UP (attempt $a)"; break; }
  sleep 10
done

POLICIES="random,theta_2pole"; [ "${MC16:-0}" = "1" ] && POLICIES="$POLICIES,mc16"  # MC16=1 to include the heavy MC-16 arm
BASE_ARGS=""; RLD_ARGS=""; GPT_ARGS=""
for a in $(seq 1 18); do hc "http://localhost:8006/v1" && break || sleep 10; done
if hc "http://localhost:8006/v1"; then POLICIES="$POLICIES,naive_base"; BASE_ARGS="--base-url $BASE_URL --base-model $BASE_CKPT"; echo "base chooser UP"; else echo "base chooser DOWN -> excluded"; fi
for a in $(seq 1 18); do hc "http://localhost:8007/v1" && break || sleep 10; done
if hc "http://localhost:8007/v1"; then POLICIES="$POLICIES,naive_rld"; RLD_ARGS="--rld-url $RLD_URL --rld-model $RLD_CKPT"; echo "rld chooser UP"; else echo "rld chooser DOWN -> excluded"; fi
if [ -n "${OPENAI_API_KEY:-}" ]; then POLICIES="$POLICIES,naive_gpt55"; GPT_ARGS="--gpt-model gpt-5.5-2026-04-23 --gpt-key $OPENAI_API_KEY"; echo "frontier arm ENABLED"; else echo "frontier arm SKIPPED (no key)"; fi

echo "======== BED choosers: $DOMAIN ($MAX_EP ep, budget $BUDGET, arms=$POLICIES) ========"
PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/bed_thetaprime.py \
  --domain "$DOMAIN" --belief-url "$BELIEF_URLS" --sim-url "$SIM_URLS" --sim-model "$SIMCKPT" \
  $BASE_ARGS $RLD_ARGS $GPT_ARGS \
  --policies "$POLICIES" --max-episodes "$MAX_EP" --budget "$BUDGET" --max-candidates "$CANDS" --concurrency "${CONC:-96}" ${FAITH_ARG:-} \
  --out "$OUT_DIR/choosers_$DOMAIN.jsonl"
echo "==== driver done; rendering report + figure ===="
PYTHONPATH="" uv run --no-sync python scripts/analysis/rq5_bed/bed_thetaprime_report.py --results "$OUT_DIR/choosers_$DOMAIN.jsonl" || true
PYTHONPATH="" uv run --no-sync --with matplotlib python scripts/analysis/rq5_bed/bed_thetaprime_figure.py --results "$OUT_DIR/choosers_$DOMAIN.jsonl" --out "$OUT_DIR/bed_choosers_$DOMAIN.png" || true
echo "DONE bed-choosers"
