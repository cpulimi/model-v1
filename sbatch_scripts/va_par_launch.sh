#!/bin/bash
# One-shot launcher for the VA split/solve/merge pipeline.
#
# Run this ON THE LOGIN NODE -- it submits jobs, it is not itself an sbatch job:
#
#     bash sbatch_scripts/va_par_launch.sh                       # instances_low
#     VA_DATASET=instances_100hubs VA_RUN_NAME=va_100 \
#         bash sbatch_scripts/va_par_launch.sh
#
# Three stages:
#   1) split  -- load data, build batches, COMPILE each to learn its true
#                variable count, check the VA ceiling, write a manifest.
#                Runs here on the login node. No card, nothing sampled.
#   2) solve  -- SLURM array 1..N on sfpga01n, THROTTLED TO ONE AT A TIME.
#                sfpga01n has a single Vector Engine (/dev/veslot0 and /dev/ve0
#                are two device nodes for one card), so concurrent tasks would
#                contend. The point of the array is not speed -- it is that
#                every finished batch is checkpointed, so a walltime kill costs
#                one batch instead of the whole run.
#   3) merge  -- one job, afterok on the array, on a NORMAL partition. Merge is
#                pure CPU, so it does not hold the scarce VE node.
#
# All tuning lives in the VA_* environment variables below, read identically by
# va_par_solve.sh and va_par_merge.sh, so the three stages cannot drift apart.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
mkdir -p logs

# --- configuration (exported so solve/merge read the SAME values) ---------
export VA_DATASET="${VA_DATASET:-instances_low}"
export VA_RUN_NAME="${VA_RUN_NAME:-va_par_$(basename "$VA_DATASET")}"
export VA_OUTDIR="${VA_OUTDIR:-results/va_parallel}"
export VA_PART_BATCH_SIZE="${VA_PART_BATCH_SIZE:-1000}"
# 40000, not 50000: at 50000 the largest instances_low batch compiles to 61,635
# total vars and trips the 60,000 ceiling. Y and X do not shrink with Z, so a
# Z cap does not imply a total cap.
export VA_MAX_Z="${VA_MAX_Z:-40000}"
export VA_MAX_VARS="${VA_MAX_VARS:-60000}"
export VA_NUM_READS="${VA_NUM_READS:-100}"
export VA_NUM_SWEEPS="${VA_NUM_SWEEPS:-3000}"
export VA_MIN_PENALTY="${VA_MIN_PENALTY:-50000.0}"
export VA_ADAPTIVE_ITERS="${VA_ADAPTIVE_ITERS:-8}"
export VA_REPEATS="${VA_REPEATS:-1}"
export VA_VECTOR_MODE="${VA_VECTOR_MODE:-ACCURACY}"
export VA_SEED="${VA_SEED-42}"          # VA_SEED= (empty) for an unseeded run
export VA_PRECISION="${VA_PRECISION-}"  # SINGLE | DOUBLE | empty for VA default
export VA_TIME_LIMIT="${VA_TIME_LIMIT:-13000}"
export VA_HUB_PRUNE_ITERS="${VA_HUB_PRUNE_ITERS:-500}"
# Concurrent solve tasks. 1 because there is one card; raise only if a probe
# ever shows genuinely independent Vector Engines.
CONCURRENCY="${VA_CONCURRENCY:-1}"

echo ">>> project   $PROJECT"
echo ">>> dataset   $VA_DATASET"
echo ">>> run name  $VA_RUN_NAME"
echo ">>> outdir    $VA_OUTDIR"
echo ">>> seed      ${VA_SEED:-<unseeded>}"

# --- 1) split, here on the login node -------------------------------------
# No card needed, so this does not queue. It also fails fast: if a batch is over
# the VA ceiling, check_ceiling aborts here rather than after a queue wait.
VA_REQUIRE_CARD=0
source "$PROJECT/sbatch_scripts/va_env.sh"

echo ""
echo ">>> [1/3] splitting into batches ..."
python3 run_va_parallel_batches.py \
  --mode split \
  --dataset-dir "$VA_DATASET" \
  --run-name "$VA_RUN_NAME" \
  --output-dir "$VA_OUTDIR" \
  --part-batch-size "$VA_PART_BATCH_SIZE" \
  --max-z-vars-per-batch "$VA_MAX_Z" \
  --va-max-vars-per-batch "$VA_MAX_VARS"

MANIFEST="$VA_OUTDIR/$VA_RUN_NAME/parallel_work/batches_manifest.json"
N=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_batches'])" "$MANIFEST")
if [[ -z "${N:-}" || "$N" -lt 1 ]]; then
  echo "ERROR: could not read batch count from $MANIFEST" >&2
  exit 1
fi
echo ">>> $N batch(es) to solve"

# --- 2) solve array on the card -------------------------------------------
SOLVE_ID=$(sbatch --parsable --array=1-"$N"%"$CONCURRENCY" \
  "$PROJECT/sbatch_scripts/va_par_solve.sh")
echo ">>> [2/3] solve array submitted: job $SOLVE_ID (tasks 1-$N, $CONCURRENCY at a time)"

# --- 3) dependent merge, off the card -------------------------------------
MERGE_ID=$(sbatch --parsable --dependency=afterok:"$SOLVE_ID" \
  "$PROJECT/sbatch_scripts/va_par_merge.sh")
echo ">>> [3/3] merge submitted: job $MERGE_ID (runs after $SOLVE_ID succeeds)"

cat <<EOF

Queue:   squeue -u \$USER
Logs:    logs/va_par_solve_${SOLVE_ID}_<batch>.out
         logs/va_par_merge_${MERGE_ID}.out
Final:   $VA_OUTDIR/$VA_RUN_NAME/va/
Read:    python3 va_results.py --run-root $VA_OUTDIR

If some batches fail, the finished ones are kept. Resubmit only the missing
ids (merge names them), then run the merge again:
    sbatch --array=<ids>%$CONCURRENCY sbatch_scripts/va_par_solve.sh
    sbatch sbatch_scripts/va_par_merge.sh
EOF
