#!/bin/bash
# Gurobi proven-optimum solve for the hub-scaling ladder (10/20/50/100 hubs).
#
# One SLURM array task per instance size, all four resident concurrently.
# --array is set by gurobi_optima_launch.sh; do not sbatch this file directly
# unless you pass --array yourself.
#
# Purpose: establish the TRUE optimum (and, on a timeout, a valid optimality
# bracket) for each ladder size so the annealing results can be reported as a
# gap to optimal rather than a gap to a 0.1%-gap incumbent. Hence --mip-gap 0.
#
# ---------------------------------------------------------------------------
# WHY THE #SBATCH -c / --mem BELOW ARE UNIFORM
# ---------------------------------------------------------------------------
# A SLURM job array shares ONE resource specification across every task: -c,
# --mem and -t cannot be varied per array index. Each task still gets its own
# INDEPENDENT allocation (this is not one shared fat allocation running four
# solves inside it), but all four allocations are the same shape.
#
# So the header is sized for the WORST case (100 hubs: 16h walltime, which must
# exceed the LARGEST GUROBI_TIME_LIMIT in the case block) and the per-size case
# block below controls what actually can vary per task: --threads and
# --gurobi-time-limit.
#
# If you want each size right-sized instead (4/8/16/32 cores), run the launcher
# with --per-size. That submits four independently-sized jobs against this same
# script and honours the CORES/MEM/WALLTIME columns of the case block. The
# tradeoff is that %N throttling (--max-concurrent) only exists for arrays.
# ---------------------------------------------------------------------------
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-16:00:00
#SBATCH -J gurobi_optima
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Size mapping. Single source of truth: the launcher sources this file with
# GUROBI_OPTIMA_MAP_ONLY=1 to read the same numbers, so the two scripts can
# never drift apart.
#
# GUROBI_TIME_LIMIT is deliberately set BELOW WALLTIME on every row. A task that
# is killed by SLURM at the walltime writes NOTHING -- no summary.json, no
# ObjBound, no bracket. A task that hits its own Gurobi time limit returns from
# model.optimize() with TIME_LIMIT status, still holds ObjBound, and walks
# through the normal summary-writing path. The margin has to cover CSV load,
# model build, solution extraction and output writing, not just the solve.
# ---------------------------------------------------------------------------
gurobi_optima_map() {
  local idx="$1"
  case "$idx" in
    0)
      SIZE=10;  DATASET=instances_10hubs
      CORES=4;   MEM=16G;   WALLTIME=0-02:00:00;  GUROBI_TIME_LIMIT=6000
      SINGLE_NODE_THREADS=8
      ;;
    1)
      SIZE=20;  DATASET=instances_20hubs
      CORES=8;   MEM=32G;   WALLTIME=0-04:00:00;  GUROBI_TIME_LIMIT=13200
      SINGLE_NODE_THREADS=16
      ;;
    2)
      SIZE=50;  DATASET=instances_50hubs
      CORES=16;  MEM=64G;   WALLTIME=0-08:00:00;  GUROBI_TIME_LIMIT=26400
      SINGLE_NODE_THREADS=32
      ;;
    3)
      SIZE=100; DATASET=instances_100hubs
      CORES=32;  MEM=128G;  WALLTIME=0-16:00:00;  GUROBI_TIME_LIMIT=54000
      SINGLE_NODE_THREADS=64
      ;;
    *)
      echo "ERROR: no size mapping for array task id '$idx' (expected 0-3)" >&2
      return 1
      ;;
  esac
  RUN_NAME="gurobi_optima_${SIZE}hubs"
  OUTDIR="results/gurobi_optima"
  return 0
}

# Array-mode header values. In array mode every task really does get -c 16 /
# --mem 64G / -t 12:00:00 regardless of the CORES/MEM/WALLTIME column, so
# --threads must follow what SLURM actually granted, not the ideal column.
GUROBI_OPTIMA_ARRAY_CORES=16
GUROBI_OPTIMA_ARRAY_MEM=64G
GUROBI_OPTIMA_ARRAY_WALLTIME=0-16:00:00

# Single-node fallback budget (used only with --single-node). One allocation,
# four concurrent python processes inside it. The SINGLE_NODE_THREADS column
# above must sum to <= GUROBI_OPTIMA_NODE_CORES, with a few cores left over for
# the OS and for the four parent processes' I/O.
GUROBI_OPTIMA_NODE_CORES=128
GUROBI_OPTIMA_NODE_MEM="${GUROBI_OPTIMA_NODE_MEM:-0}"   # 0 = all memory on the node
GUROBI_OPTIMA_NODE_WALLTIME=0-16:00:00

# Sourced by the launcher purely to read the mapping: stop here.
if [[ -n "${GUROBI_OPTIMA_MAP_ONLY:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs

# ---------------------------------------------------------------------------
# SINGLE-NODE FALLBACK MODE
# ---------------------------------------------------------------------------
# Only for clusters that cap the user at one node, where a 4-task array would
# serialize instead of running side by side. One allocation, four CONCURRENT
# python processes inside it, each pinned to its own Gurobi thread count.
#
# This is a shared allocation and is therefore NOT the default: a single task
# that dies takes nothing else with it, but the four solves do contend for the
# node's memory bandwidth, so the runtime numbers are slightly pessimistic
# compared with array mode. Say so if you report timings from this path.
# ---------------------------------------------------------------------------
if [[ "${GUROBI_OPTIMA_SINGLE_NODE:-0}" == "1" ]]; then
  module purge
  module load mamba/latest gurobi/13.0.1
  set +u
  if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc"; fi
  source activate qubo 2>/dev/null || conda activate qubo 2>/dev/null || mamba activate qubo
  set -u
  export PYTHONHASHSEED=0

  JOBID="${SLURM_JOB_ID:-manual}"
  GRANTED="${SLURM_CPUS_PER_TASK:-${GUROBI_OPTIMA_NODE_CORES}}"

  echo "============================================================================"
  echo "GUROBI OPTIMA - SINGLE-NODE MODE - RESOLVED MAPPING"
  echo "============================================================================"
  echo "  host:               $(hostname)"
  echo "  project:            $PROJECT"
  echo "  python:             $(which python)"
  echo "  slurm job id:       ${JOBID}"
  echo "  cores granted:      ${GRANTED}"
  echo "  mem/node:           ${SLURM_MEM_PER_NODE:-n/a} MB"
  echo "  PYTHONHASHSEED:     ${PYTHONHASHSEED}"
  echo "  ---- four CONCURRENT solves in this one allocation ----"
  THREAD_SUM=0
  for idx in 0 1 2 3; do
    gurobi_optima_map "$idx"
    THREAD_SUM=$(( THREAD_SUM + SINGLE_NODE_THREADS ))
    printf "    task %s: %3s hubs  %-20s threads=%-3s grb_limit=%ss\n" \
      "$idx" "$SIZE" "$DATASET" "$SINGLE_NODE_THREADS" "$GUROBI_TIME_LIMIT"
  done
  echo "  thread sum:         ${THREAD_SUM} of ${GRANTED} granted cores"
  if [[ "$THREAD_SUM" -gt "$GRANTED" ]]; then
    echo "  !! FATAL: thread sum ${THREAD_SUM} exceeds granted cores ${GRANTED}." >&2
    echo "            The four solves would oversubscribe the node and the runtime" >&2
    echo "            numbers would be meaningless. Lower SINGLE_NODE_THREADS." >&2
    exit 1
  fi
  echo "============================================================================"

  PIDS=(); IDXS=(); SIZES=(); LOGS=()
  for idx in 0 1 2 3; do
    gurobi_optima_map "$idx"
    if [[ ! -d "$DATASET" ]]; then
      echo "ERROR: dataset dir '$DATASET' not found under $PROJECT" >&2
      exit 1
    fi
    LOG="logs/gurobi_optima_singlenode_${JOBID}_${SIZE}hubs.out"
    echo ">>> launching ${SIZE} hubs  threads=${SINGLE_NODE_THREADS}  -> $LOG"
    python run_aligned_fsl_comparison.py \
      --solver gurobi \
      --dataset-dir "$DATASET" \
      --run-name "$RUN_NAME" \
      --output-dir "$OUTDIR" \
      --seed 42 \
      --mip-gap 0 \
      --threads "$SINGLE_NODE_THREADS" \
      --gurobi-time-limit "$GUROBI_TIME_LIMIT" \
      --console-log > "$LOG" 2>&1 &
    PIDS+=("$!"); IDXS+=("$idx"); SIZES+=("$SIZE"); LOGS+=("$LOG")
  done

  # Wait on every child regardless of failures, so one bad size cannot abandon
  # the other three mid-solve.
  FAILED=0
  for n in "${!PIDS[@]}"; do
    if wait "${PIDS[$n]}"; then
      echo ">>> ${SIZES[$n]} hubs: OK   (${LOGS[$n]})"
    else
      RC_N=$?
      echo ">>> ${SIZES[$n]} hubs: FAILED rc=$RC_N   (${LOGS[$n]})" >&2
      FAILED=1
    fi
  done

  echo
  echo ">>> all four solves finished. Collect with:"
  echo "        python collect_gurobi_optima.py"
  exit $FAILED
fi

# Index comes from the array in array mode, or from --export in --per-size mode.
TASK_IDX="${SLURM_ARRAY_TASK_ID:-${GUROBI_OPTIMA_SIZE_IDX:-}}"
if [[ -z "$TASK_IDX" ]]; then
  echo "ERROR: neither SLURM_ARRAY_TASK_ID nor GUROBI_OPTIMA_SIZE_IDX is set." >&2
  echo "       Submit via sbatch_scripts/gurobi_optima_launch.sh." >&2
  exit 1
fi
gurobi_optima_map "$TASK_IDX"

# Threads MUST equal the cores SLURM actually gave this task. Leaving Gurobi at
# Threads=0 makes it size its thread pool from the NODE's core count, not the
# cgroup's, which oversubscribes the allocation and makes the runtime numbers
# meaningless for a fair comparison against annealing.
GRANTED_CORES="${SLURM_CPUS_PER_TASK:-}"
if [[ -z "$GRANTED_CORES" ]]; then
  GRANTED_CORES="$CORES"
  CORES_SOURCE="case-block fallback (SLURM_CPUS_PER_TASK unset)"
else
  CORES_SOURCE="SLURM_CPUS_PER_TASK"
fi
THREADS="$GRANTED_CORES"

module purge
module load mamba/latest gurobi/13.0.1
set +u
if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc"; fi
source activate qubo 2>/dev/null || conda activate qubo 2>/dev/null || mamba activate qubo
set -u
export PYTHONHASHSEED=0

echo "============================================================================"
echo "GUROBI OPTIMA - RESOLVED MAPPING"
echo "============================================================================"
echo "  host:                 $(hostname)"
echo "  project:              $PROJECT"
echo "  python:               $(which python)"
echo "  array job / task:     ${SLURM_ARRAY_JOB_ID:-n/a} / ${TASK_IDX}"
echo "  slurm job id:         ${SLURM_JOB_ID:-n/a}"
echo "  ---- mapping ----"
echo "  size:                 ${SIZE} hubs"
echo "  dataset dir:          ${DATASET}"
echo "  run name:             ${RUN_NAME}"
echo "  output dir:           ${OUTDIR}"
echo "  ideal cores (case):   ${CORES}"
echo "  granted cores:        ${GRANTED_CORES}   [from ${CORES_SOURCE}]"
echo "  gurobi --threads:     ${THREADS}"
echo "  ideal mem (case):     ${MEM}"
echo "  slurm mem/node:       ${SLURM_MEM_PER_NODE:-n/a} MB"
echo "  ideal walltime:       ${WALLTIME}"
echo "  gurobi time limit:    ${GUROBI_TIME_LIMIT}s  (must be < walltime so the"
echo "                        process exits cleanly and writes ObjBound)"
echo "  mip gap requested:    0  (prove optimality, do not stop at 0.1%)"
echo "  PYTHONHASHSEED:       ${PYTHONHASHSEED}"
echo "  final summary:        ${OUTDIR}/${RUN_NAME}/gurobi/summary.json"
echo "============================================================================"

if [[ ! -d "$DATASET" ]]; then
  echo "ERROR: dataset dir '$DATASET' not found under $PROJECT" >&2
  exit 1
fi

# Model formulation is untouched: no --enforce-hub-capacity, no --top-hubs-per-zip,
# no --max-parts-total, no --max-service-miles override. Those change the
# instance, not just the search.
# set +e so a Gurobi failure still falls through to the accounting block below
# and to an explicit exit code, instead of set -e killing the task silently.
set +e
python run_aligned_fsl_comparison.py \
  --solver gurobi \
  --dataset-dir "$DATASET" \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTDIR" \
  --seed 42 \
  --mip-gap 0 \
  --threads "$THREADS" \
  --gurobi-time-limit "$GUROBI_TIME_LIMIT" \
  --console-log

RC=$?
set -e
echo ">>> python exit code: $RC"

# --- additive SLURM memory accounting (does NOT affect the solve above) -----
# Outside cross-check on the VmHWM peak_memory_mb now recorded in summary.json
# by memory_accounting_version 2. Instrumentation only.
RUN_DIR="${OUTDIR}/${RUN_NAME}"
mkdir -p "$RUN_DIR"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOBID_FULL="${SLURM_ARRAY_JOB_ID}_${TASK_IDX}"
else
  JOBID_FULL="${SLURM_JOB_ID:-unknown}"
fi
if command -v seff >/dev/null 2>&1; then
  seff "$JOBID_FULL" > "$RUN_DIR/seff_${JOBID_FULL}.txt" 2>&1 || true
fi
SACCT_LINE=$(sacct -j "$JOBID_FULL" \
  --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
  --units=M --noheader --parsable2 2>/dev/null \
  | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1)
MAXRSS=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $3}')
MAXVM=$(printf '%s'  "$SACCT_LINE" | awk -F'|' '{print $4}')
REQMEM=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $5}')
ELAPSED=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $6}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$JOBID_FULL" "$SIZE" "$THREADS" "$MAXRSS" "$MAXVM" "$REQMEM" "$ELAPSED" \
  >> "$RUN_DIR/slurm_mem_${SIZE}hubs.tsv"
echo ">>> slurm mem: job=$JOBID_FULL size=${SIZE} MaxRSS=$MAXRSS MaxVMSize=$MAXVM ReqMem=$REQMEM Elapsed=$ELAPSED"

exit $RC
