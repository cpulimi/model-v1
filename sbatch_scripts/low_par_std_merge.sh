#!/bin/bash
# Parallel-batch MERGE job for instances_low, SA sampler, STANDARD budget.
# Runs ONCE after all solve array tasks finish (afterok dependency). Combines
# every solved batch, runs the SAME hub-prune/post-process as the sequential
# solver, and writes the final qubo/ outputs + summary.json.
#
# Every flag below MUST match low_par_std_solve.sh (same run-name, output-dir,
# dataset, and batching flags) so it loads the right batch checkpoints.
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 128
#SBATCH --mem 200G
#SBATCH -t 7-00:00:00
#SBATCH -J low_par_std_merge
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

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

python run_parallel_batches.py \
  --mode merge \
  --dataset-dir instances_low \
  --run-name low_par_sa30 \
  --output-dir results/low_sa30 \
  --seed 42 \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 50000 \
  --sampler sa \
  --num-reads 30 \
  --num-sweeps 500 \
  --max-stages 2 \
  --retry-reads-boost 2.0 \
  --penalty-mode adaptive \
  --min-penalty 50000.0 \
  --constraint-multiplier 5.0 \
  --c4-mode auto \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 5 \
  --adaptive-penalty-growth 1.5 \
  --hub-prune-max-iterations 500

# --- additive SLURM memory accounting (does NOT affect the merge above) ---
# Captures SLURM's authoritative MaxRSS for the merge job as an outside
# cross-check against merge_rss_peak_mb. Instrumentation only.
RUN_DIR="results/low_std_comparison/low_par_std_sa"
SCALE="low"
mkdir -p "$RUN_DIR"
JOBID="${SLURM_JOB_ID}"
sleep 10  # let SLURM register the running step
if command -v seff >/dev/null 2>&1; then
  seff "$JOBID" > "$RUN_DIR/seff_${SCALE}_merge_${JOBID}.txt" 2>&1 || true
fi
# sstat reads the LIVE batch step (MaxRSS populated mid-job); sacct MaxRSS is blank
# until the step ends -- use sstat first for MaxRSS/MaxVMSize and fall back to sacct.
SSTAT_LINE=$(sstat -j "${SLURM_JOB_ID}.batch" --format=MaxRSS,MaxVMSize \
  --units=M --noheader --parsable2 2>/dev/null | head -n1)
MAXRSS=$(printf '%s' "$SSTAT_LINE" | awk -F'|' '{print $1}')
MAXVM=$(printf '%s' "$SSTAT_LINE" | awk -F'|' '{print $2}')
# sacct still supplies ReqMem/Elapsed for a running job (and a MaxRSS fallback).
SACCT_LINE=$(sacct -j "$JOBID" \
  --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
  --units=M --noheader --parsable2 2>/dev/null \
  | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1)
if [[ -z "$MAXRSS" ]]; then MAXRSS=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $3}'); fi
if [[ -z "$MAXVM" ]]; then MAXVM=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $4}'); fi
REQMEM=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $5}')
ELAPSED=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $6}')
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_JOB_ID}" "merge" "$MAXRSS" "$MAXVM" "$REQMEM" "$ELAPSED" \
  >> "$RUN_DIR/slurm_mem_${SCALE}.tsv"
echo ">>> slurm mem appended to $RUN_DIR/slurm_mem_${SCALE}.tsv: jobid=${SLURM_JOB_ID} task=merge MaxRSS=$MAXRSS MaxVMSize=$MAXVM ReqMem=$REQMEM Elapsed=$ELAPSED"
