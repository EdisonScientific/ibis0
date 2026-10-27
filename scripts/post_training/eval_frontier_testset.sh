#!/bin/bash
# Frontier API model on the published multi-domain TEST set: every domain, one model, one process.
#
# A plain shell script rather than an sbatch job on purpose — this path needs no GPU. It is the
# counterpart to eval_{classification,generative}_testset.sbatch, which do, and it resolves output paths
# through the same scripts/lib/eval_paths.py so all three model families land in one tree.
#
# Runs the domains SEQUENTIALLY. Two providers in parallel is fine (launch this twice); parallelising
# domains within a provider mostly buys rate-limit errors.
#
# Usage:
#   scripts/post_training/eval_frontier_testset.sh anthropic claude-opus-5
#   scripts/post_training/eval_frontier_testset.sh openai    gpt-5.6-sol
#
# Required: $1 provider (openai|anthropic), $2 API model id. The key comes from the environment
# or .env (ANTHROPIC_API_KEY / OPENAI_API_KEY) — never pass it on the command line, where it lands in shell
# history and the process table.
#
# Optional env:
#   EFFORT     reasoning depth, applied symmetrically to both providers (default high).
#   MAXTOK     output budget, the SAME number for both providers (default 24000). On Anthropic this
#              covers thinking AND the response together, so it must be generous.
#   CONC       in-flight requests (default 16).
#   DOMAINS    space-separated domain list (default "tahoe depmap gwas").
#   LIMIT      score only the first N episodes per domain — for a cost pilot before committing.
#   TESTSET / TESTSET_REV   published set + revision (defaults: the frozen primary set at the paper revision).
#
# Results: ${IBIS_DATA:-ibis_data}/eval/<testset>/<model-id>/<domain>_primary.json, plus a run.json
# recording provider, effort and budgets. Per-domain API token usage is written INTO each cell, so the
# billing record survives without this script's stdout.
set -uo pipefail

PROV="${1:?usage: eval_frontier_testset.sh <openai|anthropic> <model-id>}"
MODEL="${2:?usage: eval_frontier_testset.sh <openai|anthropic> <model-id>}"

PROJECT_DIR="${PROJECT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
[ -f .env ] && { set -a; source .env; set +a; }
EFFORT="${EFFORT:-high}"
MAXTOK="${MAXTOK:-24000}"
CONC="${CONC:-16}"
DOMAINS="${DOMAINS:-tahoe depmap gwas}"
# Keep the default in its own variable: inlining ${TESTSET:-...{domain}} makes bash mis-parse the `}`
# inside `{domain}` and append a stray one, but only when TESTSET is actually overridden.
DEFAULT_TESTSET='hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}'
TESTSET="${TESTSET:-$DEFAULT_TESTSET}"
# Only an hf:// set carries a revision. ${VAR-default} (no colon) so an explicit TESTSET_REV="" is
# honoured -- with ${VAR:-default} an empty override silently became the Hub revision.
case "$TESTSET" in
  hf://*) TESTSET_REV="${TESTSET_REV-df43070d4a1338210ed33c368d49e58da60471cb}" ;;
  *)      TESTSET_REV="" ;;
esac

eval "$(uv run --no-sync python scripts/lib/eval_paths.py resolve \
          --testset "$TESTSET" --revision "$TESTSET_REV" --api "$MODEL" | grep -v dotenv)"

echo "[$(date +%T)] $PROV:$MODEL effort=$EFFORT max_tok=$MAXTOK conc=$CONC -> $IBIS_RESULT_DIR"
# Echo the resolved template before spending anything: overriding TESTSET is the code path that used to
# pick up a stray `}` (see DEFAULT_TESTSET above), and on this path the only other check is a fail-fast
# in the Python eval. One line here makes a bad override visible in the log instead of inferred.
echo "[$(date +%T)] testset=$TESTSET rev=${TESTSET_REV:-<none>}"
rc=0
for d in $DOMAINS; do
  # Expand {domain} per iteration so a template and a plain path both work.
  DATA="${TESTSET//\{domain\}/$d}"
  uv run --no-sync python scripts/analysis/multidomain/eval_frontier_testset.py \
    --domain "$d" --data-path "$DATA" --revision "$TESTSET_REV" \
    --provider "$PROV" --model "$MODEL" --out-dir "$IBIS_RESULT_DIR" \
    --reasoning-effort "$EFFORT" --max-output-tokens "$MAXTOK" --concurrency "$CONC" \
    ${PRIOR_NOTE:+--prior-note} \
    ${LIMIT:+--limit "$LIMIT"} || rc=$?
done

# Provenance last, so cells_written reflects what actually landed. Read the cell names off disk rather
# than assuming them: only the eval knows how the config split into passes (primary -> one cell/domain,
# sensitivity -> ctx0..ctx16), so any hardcoded name mislabels one of the two configs. Quote each name
# BEFORE joining, or the no-cells case becomes a bogus [""] entry.
CELLS=$(cd "$IBIS_RESULT_DIR" 2>/dev/null && ls *_*.json 2>/dev/null | grep -v '^run\.json$' \
          | sed 's/\.json$//' | sed 's/.*/"&"/' | paste -sd, -)
uv run --no-sync python scripts/lib/eval_paths.py record \
  --testset "$TESTSET" --revision "$TESTSET_REV" --api "$MODEL" \
  --json "{\"model_family\":\"frontier\",\"provider\":\"$PROV\",\"model_source\":\"$MODEL\",\"prompt_style\":\"reasoning\",\"reasoning_effort\":\"$EFFORT\",\"max_output_tokens\":$MAXTOK,\"concurrency\":$CONC,\"cells_written\":[$CELLS]}" \
  >/dev/null || true
echo "[$(date +%T)] frontier eval DONE rc=$rc — results under $IBIS_RESULT_DIR"
exit "$rc"
