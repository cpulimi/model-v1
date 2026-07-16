#!/bin/bash
# Parallel-batch SOLVE job for instances_low (Option 3), HEAVY iteration budget.
# One array task per batch. --array is set by low_par_launch.sh.
#
# Purpose: test whether increasing the SA sampling budget (reads/sweeps/stages)
# lowers QUBO cost further on the small instance, toward Gurobi's $74.70M.
# Batching + penalties are kept IDENTICAL to the prior $77.25M low baseline so
# the only variable is the iteration count.
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 128
#SBATCH --mem 200G
#SBATCH -t 0-08:00:00
#SBATCH -J low_par_solve
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
echo ">>> host=$(hostname) project=$PROJECT python=$(which python) batch=${SLURM_ARRAY_TASK_ID}"

# IMPORTANT: --run-name, --output-dir, and all batching/QUBO flags MUST be
# identical here and in low_par_merge.sh so batch ids and seeds line up.
# HEAVY budget: num-reads 500, num-sweeps 12000, max-stages 5 (vs baseline
# 100/3000/3). Batching (part-batch 1000 / max-z 50000) and penalties match
# the prior low baseline so the comparison isolates the iteration effect.
python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_low \
  --run-name low_par_heavy \
  --output-dir results/low_adaptive_parallel \
  --seed 42 \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 50000 \
  --num-reads 500 \
  --num-sweeps 12000 \
  --max-stages 5 \
  --retry-reads-boost 2.0 \
  --penalty-mode adaptive \
  --min-penalty 50000.0 \
  --constraint-multiplier 5.0 \
  --c4-mode auto \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 5 \
  --adaptive-penalty-growth 1.5

# --- additive SLURM memory accounting (does NOT affect the solve above) ---
# Captures SLURM's authoritative MaxRSS as an outside cross-check against the
# rss_peak_mb recorded in each batch pkl. Instrumentation only; joined by merge on
# (jobid, taskid) where jobid=SLURM_ARRAY_JOB_ID and taskid=SLURM_ARRAY_TASK_ID.
RUN_DIR="results/low_adaptive_parallel/low_par_heavy"
SCALE="low"
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
