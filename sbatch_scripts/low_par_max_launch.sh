#!/bin/bash
# MAX-RESOURCE one-shot launcher (Option 3) for instances_low.
# Run ON THE LOGIN NODE:  bash sbatch_scripts/low_par_max_launch.sh
# Splits to discover N, submits a solve array 1-N (each task on a full node,
# --exclusive --mem=0, no time limit), then a dependent merge.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DATASET=instances_low
RUN_NAME=low_par_max
OUTDIR=results/low_adaptive_parallel_max

# MUST match the batching flags inside low_par_max_solve.sh.
PART_BATCH_SIZE=1000
MAX_Z=50000

cd "$PROJECT"
mkdir -p "$PROJECT/logs"
echo ">>> project root: $PROJECT"

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

SOLVE_ID=$(sbatch --parsable --array=1-"$N" "$PROJECT/sbatch_scripts/low_par_max_solve.sh")
echo ">>> solve array submitted: job $SOLVE_ID (tasks 1-$N, side by side)"

MERGE_ID=$(sbatch --parsable --dependency=afterok:"$SOLVE_ID" "$PROJECT/sbatch_scripts/low_par_max_merge.sh")
echo ">>> merge submitted: job $MERGE_ID (runs after $SOLVE_ID succeeds)"

echo
echo "Queue:  squeue -u \$USER"
echo "Logs:   logs/low_par_max_solve_${SOLVE_ID}_<batchid>.out  and  logs/low_par_max_merge_${MERGE_ID}.out"
echo "Final outputs: $OUTDIR/$RUN_NAME/qubo/  (summary.json, batch_summary.csv, ...)"
