#!/bin/bash
# One-shot launcher for the SQA variant of the adaptive parallel batch pipeline
# on instances_low. Same split -> solve array -> dependent merge flow as
# low_par_launch.sh, but targeting the low_par_sqa_{solve,merge}.sh scripts.
#
# Run this ON THE LOGIN NODE (it submits jobs, it is not itself an sbatch job):
#     bash sbatch_scripts/low_par_sqa_launch.sh
#
# It does three things:
#   1) split  -> loads data + builds batches to discover N (cheap, no solving)
#   2) solve  -> submits a SLURM array (1..N): ALL batches run SIDE BY SIDE
#   3) merge  -> submits one merge job that runs AFTER all batches succeed
#                (afterok dependency), doing the same hub-prune/post-process and
#                writing the final qubo/ outputs + summary.json
#
# All solve/merge flags live in low_par_solve.sh and low_par_merge.sh; keep
# them identical there so batch ids, seeds, and the decomposition match.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DATASET=instances_low
RUN_NAME=low_par_sqa
OUTDIR=results/low_adaptive_parallel

# These split flags MUST match the batching flags inside low_par_sqa_solve.sh.
PART_BATCH_SIZE=1000
MAX_Z=50000

cd "$PROJECT"
mkdir -p "$PROJECT/logs"
echo ">>> project root: $PROJECT"

# --- 1) split (login node) ------------------------------------------------
module purge
module load mamba/latest gurobi/13.0.1
set +u
if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc"; fi
source activate qubo 2>/dev/null || conda activate qubo 2>/dev/null || mamba activate qubo
set -u
export PYTHONHASHSEED=0

echo ">>> splitting into batches ..."
python run_parallel_batches.py --mode split \
  --dataset-dir "$DATASET" --run-name "$RUN_NAME" --output-dir "$OUTDIR" \
  --part-batch-size "$PART_BATCH_SIZE" --max-z-vars-per-batch "$MAX_Z"

MANIFEST="$OUTDIR/$RUN_NAME/parallel_work/batches_manifest.json"
N=$(python -c "import json; print(json.load(open('$MANIFEST'))['total_batches'])")
if [[ -z "$N" || "$N" -lt 1 ]]; then
  echo "ERROR: could not determine batch count from $MANIFEST" >&2
  exit 1
fi
echo ">>> discovered $N batches (will run side by side as array 1-$N)"

# --- 2) submit solve array ------------------------------------------------
# NOTE: low_par_sqa_solve.sh pins -w sfpga01n and requests -c 128 / --mem 0, so
# array tasks are scheduled as that single node's cores/memory free up rather than
# fully in parallel. Drop -w / lower -c in the solve script for more concurrency.
SOLVE_ID=$(sbatch --parsable --array=1-"$N" "$PROJECT/sbatch_scripts/low_par_sqa_solve.sh")
echo ">>> SQA solve array submitted: job $SOLVE_ID (tasks 1-$N)"

# --- 3) submit dependent merge -------------------------------------------
MERGE_ID=$(sbatch --parsable --dependency=afterok:"$SOLVE_ID" "$PROJECT/sbatch_scripts/low_par_sqa_merge.sh")
echo ">>> SQA merge submitted: job $MERGE_ID (runs after $SOLVE_ID succeeds)"

echo
echo "Queue:  squeue -u \$USER"
echo "Logs:   logs/low_par_sqa_solve_${SOLVE_ID}_<batchid>.out  and  logs/low_par_sqa_merge_${MERGE_ID}.out"
echo "Final outputs land in: $OUTDIR/$RUN_NAME/qubo/  (summary.json, adaptive_iteration_log.csv, timing_memory_summary.csv, ...)"
