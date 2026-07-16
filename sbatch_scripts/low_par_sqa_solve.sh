#!/bin/bash
# Parallel-batch SOLVE job for instances_low using the SQA sampler.
# One array task per batch. --array is set by low_par_sqa_launch.sh.
#
# SQA counterpart of low_par_solve.sh: batching + penalties match the SA scripts
# so the only difference is the sampler backend (SQASampler vs SASampler).
# SQA replicates the spins across --sqa-trotter slices, so memory is ~trotter x the
# SA footprint and each read is slower; we grab the whole node (--mem 0) and give a
# longer wall clock. TEST ON LOW FIRST before attempting med/high (likely OOM/timeout).
#SBATCH -p fpga
#SBATCH -q public
#SBATCH -w sfpga01n
#SBATCH -c 128
#SBATCH --mem 0
#SBATCH -t 0-12:00:00
#SBATCH -J low_par_sqa_solve
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
# identical here and in low_par_sqa_merge.sh so batch ids and seeds line up.
# SQA: --sampler sqa with an EXPLICIT --sqa-gamma (transverse field) so the
# quantum-inspired parameter is reproducible/reportable rather than the library
# default. num-reads/num-sweeps are intentionally conservative for a first SQA run.
python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_low \
  --run-name low_par_sqa \
  --output-dir results/low_adaptive_parallel \
  --seed 42 \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 50000 \
  --sampler sqa \
  --sqa-beta 5.0 \
  --sqa-trotter 8 \
  --sqa-gamma 1.0 \
  --num-reads 30 \
  --num-sweeps 1000 \
  --max-stages 2 \
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
RUN_DIR="results/low_adaptive_parallel/low_par_sqa"
SCALE="low"
mkdir -p "$RUN_DIR"
JOBID_FULL="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
sleep 10  # let SLURM register the running step
if command -v seff >/dev/null 2>&1; then
  seff "$JOBID_FULL" > "$RUN_DIR/seff_${SCALE}_${JOBID_FULL}.txt" 2>&1 || true
fi
# sstat reads the LIVE batch step, so MaxRSS is populated mid-job. sacct only
# flushes MaxRSS after the step ends, so for a still-running job it returns blank
# here -- use sstat first for MaxRSS/MaxVMSize and fall back to sacct otherwise.
SSTAT_LINE=$(sstat -j "${SLURM_JOB_ID}.batch" --format=MaxRSS,MaxVMSize \
  --units=M --noheader --parsable2 2>/dev/null | head -n1)
MAXRSS=$(printf '%s' "$SSTAT_LINE" | awk -F'|' '{print $1}')
MAXVM=$(printf '%s' "$SSTAT_LINE" | awk -F'|' '{print $2}')
# sacct still supplies ReqMem/Elapsed for a running job (and a MaxRSS fallback).
SACCT_LINE=$(sacct -j "$JOBID_FULL" \
  --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
  --units=M --noheader --parsable2 2>/dev/null \
  | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1)
if [[ -z "$MAXRSS" ]]; then MAXRSS=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $3}'); fi
if [[ -z "$MAXVM" ]]; then MAXVM=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $4}'); fi
REQMEM=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $5}')
ELAPSED=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $6}')
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$MAXRSS" "$MAXVM" "$REQMEM" "$ELAPSED" \
  >> "$RUN_DIR/slurm_mem_${SCALE}.tsv"
echo ">>> slurm mem appended to $RUN_DIR/slurm_mem_${SCALE}.tsv: jobid=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} MaxRSS=$MAXRSS MaxVMSize=$MAXVM ReqMem=$REQMEM Elapsed=$ELAPSED"
