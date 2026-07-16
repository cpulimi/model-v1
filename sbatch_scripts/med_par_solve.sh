#!/bin/bash
# Parallel-batch SOLVE job (Option 3). One array task per batch.
#
# STEP 1 (login node, once): discover how many batches you have, then set --array below.
#   python run_parallel_batches.py --mode split \
#     --dataset-dir instances_medium --run-name med_par \
#     --part-batch-size 1000 --max-z-vars-per-batch 600000
#
# STEP 2: submit this array, then submit med_par_merge.sh with a dependency:
#   sid=$(sbatch --parsable sbatch_scripts/med_par_solve.sh)
#   sbatch --dependency=afterok:$sid sbatch_scripts/med_par_merge.sh
#
# --array is set by med_par_launch.sh (or pass -t on sbatch command line).
#SBATCH -p lightwork
#SBATCH -q public
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-06:00:00
#SBATCH -J med_par_solve
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs
module purge
module load mamba/latest gurobi/13.0.1
set +u
if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc"; fi
source activate qubo 2>/dev/null || conda activate qubo 2>/dev/null || mamba activate qubo
set -u
export PYTHONHASHSEED=0
echo ">>> host=$(hostname) project=$PROJECT python=$(which python)"

# IMPORTANT: --run-name, --output-dir, and all batching/QUBO flags MUST be
# identical here and in med_par_merge.sh so batch ids and seeds line up.
python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_medium \
  --run-name med_par \
  --output-dir results/medium_adaptive_parallel \
  --seed 42 \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 600000 \
  --num-reads 30 \
  --num-sweeps 500 \
  --max-stages 2 \
  --retry-reads-boost 2.0 \
  --penalty-mode adaptive \
  --min-penalty 50000.0 \
  --constraint-multiplier 5.0 \
  --constraint-multiplier-c2 3.0 \
  --constraint-multiplier-c3 2.0 \
  --c4-mode auto \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 5 \
  --adaptive-penalty-growth 1.5

# --- additive SLURM memory accounting (does NOT affect the solve above) ---
# Captures SLURM's authoritative MaxRSS as an outside cross-check against the
# rss_peak_mb recorded in each batch pkl. Instrumentation only; joined by merge on
# (jobid, taskid) where jobid=SLURM_ARRAY_JOB_ID and taskid=SLURM_ARRAY_TASK_ID.
RUN_DIR="results/medium_adaptive_parallel/med_par"
SCALE="med"
mkdir -p "$RUN_DIR"
JOBID_FULL="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
sleep 10  # give SLURM a moment to flush accounting for this step
if command -v seff >/dev/null 2>&1; then
  seff "$JOBID_FULL" > "$RUN_DIR/seff_${SCALE}_${JOBID_FULL}.txt" 2>&1 || true
fi
MEM_LINE=$(sacct -j "$JOBID_FULL" \
  --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
  --units=M --noheader --parsable2 2>/dev/null \
  | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1)
MAXRSS=$(printf '%s' "$MEM_LINE" | awk -F'|' '{print $3}')
MAXVM=$(printf '%s' "$MEM_LINE" | awk -F'|' '{print $4}')
REQMEM=$(printf '%s' "$MEM_LINE" | awk -F'|' '{print $5}')
ELAPSED=$(printf '%s' "$MEM_LINE" | awk -F'|' '{print $6}')
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$MAXRSS" "$MAXVM" "$REQMEM" "$ELAPSED" \
  >> "$RUN_DIR/slurm_mem_${SCALE}.tsv"
echo ">>> slurm mem appended to $RUN_DIR/slurm_mem_${SCALE}.tsv: jobid=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} MaxRSS=$MAXRSS MaxVMSize=$MAXVM ReqMem=$REQMEM Elapsed=$ELAPSED"
