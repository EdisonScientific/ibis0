#!/bin/bash
#SBATCH --job-name=mm-gen-testset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -o logs/%x_%j.out

# Sized for Qwen3-30B-A3B: bf16 weights ~60GB fit on ONE 180GB B200 (tp=1), leaving ~90GB for KV
# cache at gpu_mem_util=0.9. NOT exclusive / not --mem=0, so eval coexists with training jobs. For a
# larger model or higher throughput, bump TP and override --gres at submit: `sbatch --gres=gpu:b200:2`.

# Serve ONE generative model on the allocated node and evaluate it on a published multi-domain TEST set,
# in the requested prompt style(s). Self-contained: serve -> poll -> eval -> teardown, all in a single
# allocation (localhost endpoint, no cross-job networking). Launcher for eval_generative_testset.py.
#
# The same driver serves every generative condition in the multi-domain reproduction:
#   - base Qwen3-30B-A3B  -> the transfer FLOOR, run in BOTH modes (direct + reasoning)
#   - a trained rl-direct    weights/ dir -> MODES=direct
#   - a trained rl-thinking  weights/ dir -> MODES=reasoning
# (The classifier condition is NOT served — it runs under torchrun via eval_classification_testset.py.)
#
# Parameterized via env vars (all optional):
#   MODEL        HF id or local weights/ path          (default: Qwen/Qwen3-30B-A3B)
#   MODEL_TAG    short label used in output paths       (default: run-id+step for a weights/ path,
#                                                         else basename of MODEL — kept unique per run)
#   MODES        subset of {direct reasoning}           (default: "direct reasoning")
#   DOMAINS      subset of {gwas depmap tahoe}          (default: all three)
#   TESTSET      published test-set path template containing {domain}
#                (default: the frozen primary set; point at sensitivity/ for the dense nested sweep,
#                 or at a local directory of the same shape for a cheap smoke)
#   MAX_LEN      served max_model_len                   (default: 32768)
#   TP           tensor-parallel size                   (default: 1)
#   PORT         server port                            (default: 10000 + JOBID%20000, per-job unique)
#   CONCURRENCY  in-flight requests                     (default: 32)
#   TESTSET_REV  test-set revision, required for an hf:// set   (default: df43070d, EDB-Bio tag v1.0.0)
#   RUN_NAME     trained run name, e.g. gwas-depmap-tahoe__qwen3-30b-a3b__rl-direct__r2.1.0
#   STEP         checkpoint step; required with RUN_NAME
#   (omit RUN_NAME/STEP for an untrained served model: its key is <model>__<prompt_style>)
#
# Examples:
#   # Full base-Qwen floors (both modes, all domains) on the primary set:
#   sbatch scripts/post_training/eval_generative_testset.sbatch
#   # A trained direct-RL checkpoint, direct prompt only:
#   MODEL=${IBIS_DATA:-ibis_data}/runs/<run>/weights/step_4200 MODES=direct \
#     sbatch scripts/post_training/eval_generative_testset.sbatch

set -uo pipefail
# Worktree-aware: resolve the repo root of wherever `sbatch` was invoked (not a pinned checkout);
# override by exporting PROJECT_DIR.
PROJECT_DIR="${PROJECT_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)}"
cd "$PROJECT_DIR" || { echo "error: cannot cd to PROJECT_DIR='$PROJECT_DIR'" >&2; exit 1; }
[ -f .env ] && { set -a; source .env; set +a; }
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
# Unique output tag. A weights path (.../<run-id>/weights/step_N) collapses to just "step_N" under
# basename, so different runs' step_N would overwrite each other — key it by <run-id>__step_N instead.
if [ -z "${MODEL_TAG:-}" ]; then
  case "$MODEL" in
    */weights/step_*) MODEL_TAG="$(basename "$(dirname "$(dirname "$MODEL")")")__$(basename "$MODEL")" ;;
    *) MODEL_TAG="$(basename "$MODEL")" ;;
  esac
fi
MODES="${MODES:-direct reasoning}"
DOMAINS="${DOMAINS:-gwas depmap tahoe}"
# NOTE: keep the default in its own variable. Inlining it as ${TESTSET:-...primary/{domain}} looks
# fine and even works when TESTSET is unset, but when TESTSET *is* set bash mis-parses the `}` inside
# `{domain}` and appends a stray `}` to the value — so only overrides break, and only at run time.
DEFAULT_TESTSET='hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}'
TESTSET="${TESTSET:-$DEFAULT_TESTSET}"
MAX_LEN="${MAX_LEN:-32768}"
TP="${TP:-1}"
# Per-job port so co-scheduled serves on a (non-exclusive) shared node don't collide on 8000 — a
# stale/foreign server on a fixed port would pass the readiness check and get swept against by mistake.
PORT="${PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 20000))}"
CONCURRENCY="${CONCURRENCY:-32}"
# Decoding temperature. 0.7 matches what the RL policies were trained and validated at, so the
# benchmark scores the policy rather than a greedy decode of it, and it is held equal across every
# generative row so the untrained base is not measured under a different sampler than the trained ones.
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-0}"
# Draws per episode; the reported probability is their mean (see the eval script).
N_SAMPLES="${N_SAMPLES:-4}"
EVAL=scripts/analysis/multidomain/eval_generative_testset.py
# Results go to the canonical eval tree (scripts/lib/eval_paths.py), NOT a repo-relative outputs/ dir:
# they must outlive the worktree. TESTSET_REV is required for an hf:// test set so results collected
# against two publishes can never share a directory.
# Only an hf:// set carries a revision; a local directory must not record one. Note ${VAR-default}
# (no colon) defaults just when VAR is UNSET, so an explicit TESTSET_REV="" is honored -- with
# ${VAR:-default} an empty override silently became the Hub revision, mislabelling local results.
case "$TESTSET" in
  hf://*) TESTSET_REV="${TESTSET_REV-df43070d4a1338210ed33c368d49e58da60471cb}" ;;
  *)      TESTSET_REV="" ;;
esac
# RUN_NAME/STEP identify a trained checkpoint; leave them unset for an untrained served model, whose key
# is <model>__<prompt_style> instead (the base model has no run name, and its two modes are two rows).
#
# A trained checkpoint keyed as a served model (e.g. `step_4200__direct`) would lose the run identity,
# collide across runs at the same step, and miss the roster's `eval_key`. A `*/weights/step_N` path
# already contains both fields, so derive them rather than requiring them.
IDENT=()
if [ -z "${RUN_NAME:-}" ]; then
  case "$MODEL" in
    */weights/step_*)
      RUN_NAME="$(basename "$(dirname "$(dirname "$MODEL")")")"
      STEP="${STEP:-$(basename "$MODEL" | sed 's/^step_//')}"
      echo "[$(date +%T)] derived RUN_NAME=$RUN_NAME STEP=$STEP from the weights path"
      ;;
  esac
fi
if [ -n "${RUN_NAME:-}" ]; then
  IDENT=(--run-name "$RUN_NAME" --step "${STEP:?RUN_NAME requires STEP}")
fi

# One serve can cover both prompt styles, and each style is its own roster row with its own result
# directory, so the key is resolved per (domain, mode) inside the loop rather than once here.
LOGDIR="${IBIS_DATA:-ibis_data}/eval/_logs"
mkdir -p "$LOGDIR"
SERVER_LOG="$LOGDIR/serve_${MODEL_TAG}_${SLURM_JOB_ID:-local}.log"
echo "[$(date +%T)] serving MODEL=$MODEL (tp=$TP max_len=$MAX_LEN) on port $PORT; modes='$MODES' domains='$DOMAINS'"
echo "[$(date +%T)] testset=$TESTSET"

# --- serve via the blessed prime-rl entrypoint (NOT bare `vllm serve`; local mode = no [slurm] block
#     so it serves in THIS allocation). enable_thinking is a per-request flag the eval sends, so one
#     served model handles both the direct and reasoning passes. ---
uv run inference \
  --model.name "$MODEL" \
  --model.max_model_len "$MAX_LEN" \
  --parallel.tp "$TP" \
  --server.port "$PORT" \
  --gpu_memory_utilization 0.90 \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# --- wait for readiness. A COLD serve on a fresh B200 also JIT-compiles the FlashInfer TRTLLM MoE
#     kernel (SM100) via ptxas on the first forward pass — a single kernel can take 10-20 min. The
#     compiled kernel is cached under ~/.cache/flashinfer, so on a shared home directory this cost is
#     paid ~once; later serves come up in ~2-3 min. Window is generous so a cold compile is
#     never killed mid-ptxas (which would waste it — the JIT cache only writes on completion). ---
ready() { curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; }
up=0
for i in $(seq 1 300); do
  if ready; then up=1; echo "[$(date +%T)] server up after ~$((i * 10))s"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!! SERVER PROCESS DIED — last 40 log lines:"; tail -40 "$SERVER_LOG"; exit 1; fi
  sleep 10
done
if [ "$up" != 1 ]; then echo "!! SERVER READINESS TIMEOUT — last 40 log lines:"; tail -40 "$SERVER_LOG"; kill "$SERVER_PID" 2>/dev/null; exit 1; fi

# Use the id vLLM actually advertises (a local weights/ path is served under its own path string).
SERVED_ID=$(curl -s "http://localhost:$PORT/v1/models" | uv run --no-sync python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
if [ -z "$SERVED_ID" ]; then echo "!! could not read served model id from :$PORT"; kill "$SERVER_PID" 2>/dev/null; exit 1; fi
echo "[$(date +%T)] served model id: $SERVED_ID"

# --- evaluate each domain x mode against the served endpoint ---
rc=0
for domain in $DOMAINS; do
  DATA="${TESTSET//\{domain\}/$domain}"
  for mode in $MODES; do
    think=(); [ "$mode" = reasoning ] && think=(--enable-thinking)
    # Resolve this (model, mode) row's directory through eval_paths so the layout has one definition.
    ident=("${IDENT[@]}")
    [ ${#ident[@]} -eq 0 ] && ident=(--served "$MODEL" --style "$mode")
    eval "$(uv run --no-sync python scripts/lib/eval_paths.py resolve \
              --testset "$TESTSET" --revision "$TESTSET_REV" "${ident[@]}")"
    echo "[$(date +%T)] eval domain=$domain mode=$mode data=$DATA -> $IBIS_RESULT_DIR"
    uv run python "$EVAL" \
      --base-url "http://localhost:$PORT/v1" \
      --model "$SERVED_ID" --domain "$domain" --data-path "$DATA" --revision "$TESTSET_REV" \
      --prompt-style "$mode" "${think[@]}" --concurrency "$CONCURRENCY" \
      --temperature "$TEMPERATURE" --seed "$SEED" --n-samples "$N_SAMPLES" \
      ${PRIOR_NOTE:+--prior-note} \
      --out-dir "$IBIS_RESULT_DIR" || rc=$?
    # Provenance per row; write_run_json merges cells across domains and refuses a conflicting model.
    # Read the cell names off disk rather than assuming them: only the eval knows how the config split
    # into passes, so a hardcoded "${domain}_primary" silently mislabels every sensitivity run (which
    # writes ${domain}_ctx0..ctx16 instead). Globbing also means cells_written reflects what landed.
    # Quote each name BEFORE joining: quoting the joined string instead turns the no-cells case into a
    # bogus [""] entry, because the substitution still matches the empty line.
    CELLS=$(cd "$IBIS_RESULT_DIR" 2>/dev/null && ls "${domain}"_*.json 2>/dev/null \
              | sed 's/\.json$//' | sed 's/.*/"&"/' | paste -sd, -)
    uv run --no-sync python scripts/lib/eval_paths.py record \
      --testset "$TESTSET" --revision "$TESTSET_REV" "${ident[@]}" \
      --json "{\"model_family\":\"generative\",\"model_source\":\"$MODEL\",\"served_id\":\"$SERVED_ID\",\"slurm_job_id\":${SLURM_JOB_ID:-0},\"temperature\":$TEMPERATURE,\"seed\":$SEED,\"n_samples\":$N_SAMPLES,\"cells_written\":[$CELLS]}" >/dev/null || true
  done
done

kill "$SERVER_PID" 2>/dev/null
echo "[$(date +%T)] DONE rc=$rc — results under ${IBIS_DATA:-ibis_data}/eval/"
exit $rc
