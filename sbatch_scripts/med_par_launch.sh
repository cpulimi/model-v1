#!/bin/bash
# One-shot launcher for ADAPTIVE PARALLEL BATCH (Option 3) on SOL.
#
# Run this ON THE LOGIN NODE (it submits jobs, it is not itself an sbatch job):
#     bash sbatch_scripts/med_par_launch.sh
#
# It does three things:
#   1) split  -> loads data + builds batches to discover N (cheap, no solving)
#   2) solve  -> submits a SLURM array (1..N), one task per batch, adaptive QUBO
#   3) merge  -> submits a single merge job that runs AFTER all batches succeed
#                (afterok dependency), doing the same hub-prune/post-process and
#                writing the final qubo/ outputs + summary.json
#
# COMMON_FLAGS is defined once and reused by both solve and merge, so batch ids,
# seeds, and the batch decomposition are guaranteed identical (no drift).
#
# To adapt to instances_high: change DATASET/RUN_NAME/OUTDIR and the QUBO flags.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DATASET=instances_medium
RUN_NAME=med_par
OUTDIR=results/medium_adaptive_parallel

# SLURM resources
SOLVE_PART="general"; SOLVE_QOS="private"; SOLVE_CPUS=32; SOLVE_MEM="100G"; SOLVE_TIME="0-06:00:00"
MERGE_PART="general"; MERGE_QOS="private"; MERGE_CPUS=32;  MERGE_MEM="100G"; MERGE_TIME="0-03:00:00"

# Flags shared IDENTICALLY by solve and merge. Do not duplicate elsewhere.
COMMON_FLAGS="--dataset-dir $DATASET --run-name $RUN_NAME --output-dir $OUTDIR \
--seed 42 \
--part-batch-size 1000 --max-z-vars-per-batch 600000 \
--num-reads 30 --num-sweeps 500 --max-stages 2 --retry-reads-boost 2.0 \
--penalty-mode adaptive --min-penalty 50000.0 --constraint-multiplier 5.0 \
--constraint-multiplier-c2 3.0 --constraint-multiplier-c3 2.0 --c4-mode auto \
--adaptive-penalty-mode within-batch --adaptive-penalty-iterations 5 --adaptive-penalty-growth 1.5"

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
python run_parallel_batches.py --mode split $COMMON_FLAGS

MANIFEST="$OUTDIR/$RUN_NAME/parallel_work/batches_manifest.json"
N=$(python -c "import json; print(json.load(open('$MANIFEST'))['total_batches'])")
if [[ -z "$N" || "$N" -lt 1 ]]; then
  echo "ERROR: could not determine batch count from $MANIFEST" >&2
  exit 1
fi
echo ">>> discovered $N batches"

# --- 2) submit solve array (use script file; --wrap is fragile on SOL) ----
SOLVE_ID=$(sbatch --parsable \
  -p "$SOLVE_PART" -q "$SOLVE_QOS" -c "$SOLVE_CPUS" --mem "$SOLVE_MEM" -t "$SOLVE_TIME" \
  -J "${RUN_NAME}_solve" --array=1-"$N" \
  -o "$PROJECT/logs/%x_%A_%a.out" -e "$PROJECT/logs/%x_%A_%a.err" \
  "$PROJECT/sbatch_scripts/med_par_solve.sh")
echo ">>> solve array submitted: job $SOLVE_ID (tasks 1-$N)"

# --- 3) submit dependent merge -------------------------------------------
MERGE_ID=$(sbatch --parsable --dependency=afterok:"$SOLVE_ID" \
  -p "$MERGE_PART" -q "$MERGE_QOS" -c "$MERGE_CPUS" --mem "$MERGE_MEM" -t "$MERGE_TIME" \
  -J "${RUN_NAME}_merge" \
  -o "$PROJECT/logs/%x_%j.out" -e "$PROJECT/logs/%x_%j.err" \
  "$PROJECT/sbatch_scripts/med_par_merge.sh")
echo ">>> merge submitted: job $MERGE_ID (runs after $SOLVE_ID succeeds)"

echo
echo "Queue:  squeue -u \$USER"
echo "Final outputs land in: $OUTDIR/$RUN_NAME/qubo/  (summary.json, batch_summary.csv, ...)"
