#!/bin/bash
# One-shot launcher for the Gurobi proven-optimum ladder (10/20/50/100 hubs).
#
# Run this ON THE LOGIN NODE (it submits jobs; it is not itself an sbatch job):
#
#     bash sbatch_scripts/gurobi_optima_launch.sh --dry-run     # show, submit nothing
#     bash sbatch_scripts/gurobi_optima_launch.sh               # submit and walk away
#
# Submit once. All four sizes run CONCURRENTLY. Nothing is sequenced by hand and
# there is no merge dependency -- each size writes its own summary.json, and
# collect_gurobi_optima.py gathers them whenever you come back.
#
# OPTIONS
#   --dry-run              Print the resolved mapping and the exact sbatch
#                          command, submit nothing.
#   --max-concurrent N     Throttle the array to at most N tasks running at once
#                          (--array=0-3%N). Use this if the Gurobi license has
#                          fewer tokens than 4. Array mode only.
#   --per-size             Submit four INDEPENDENTLY SIZED jobs (4/8/16/32 cores)
#                          instead of one uniform array. Right-sizes each task at
#                          the cost of losing %N throttling.
#   --single-node          FALLBACK for a cluster that caps you at ONE node, where
#                          a 4-task array would serialize. Submits ONE -N 1 -c 128
#                          job that runs all four solves CONCURRENTLY inside it.
#                          Shared allocation: the solves contend for node memory
#                          bandwidth, so timings are slightly pessimistic.
#   --collect              After submitting, print the collector command to run
#                          once the jobs finish.
#
# ARRAY MODE vs --per-size
#   A SLURM array shares one -c/--mem/-t across all tasks; they cannot vary per
#   index. Array mode therefore gives each task its OWN allocation of the same
#   shape (16 cores / 64G / 12h -- sized for 100 hubs). --per-size gives each
#   task its own right-sized allocation but submits four separate job ids, so
#   %N throttling does not apply.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SOLVE="$PROJECT/sbatch_scripts/gurobi_optima_solve.sh"

DRY_RUN=0
MAX_CONCURRENT=""
PER_SIZE=0
SINGLE_NODE=0
SHOW_COLLECT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN=1; shift ;;
    --max-concurrent) MAX_CONCURRENT="${2:?--max-concurrent needs a value}"; shift 2 ;;
    --per-size)       PER_SIZE=1; shift ;;
    --single-node)    SINGLE_NODE=1; shift ;;
    --collect)        SHOW_COLLECT=1; shift ;;
    -h|--help)        sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "ERROR: unknown option '$1' (try --help)" >&2; exit 1 ;;
  esac
done

if [[ -n "$MAX_CONCURRENT" ]]; then
  if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-concurrent must be a positive integer, got '$MAX_CONCURRENT'" >&2
    exit 1
  fi
  if [[ "$PER_SIZE" -eq 1 || "$SINGLE_NODE" -eq 1 ]]; then
    echo "ERROR: --max-concurrent throttles a SLURM array with %N syntax and has no" >&2
    echo "       meaning for --per-size or --single-node, which submit ordinary jobs." >&2
    exit 1
  fi
fi

if [[ "$PER_SIZE" -eq 1 && "$SINGLE_NODE" -eq 1 ]]; then
  echo "ERROR: --per-size and --single-node are mutually exclusive." >&2
  exit 1
fi

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

# Read the size mapping straight out of the solve script so the two can't drift.
if [[ ! -f "$SOLVE" ]]; then
  echo "ERROR: solve script not found: $SOLVE" >&2
  exit 1
fi
export GUROBI_OPTIMA_MAP_ONLY=1
source "$SOLVE"
unset GUROBI_OPTIMA_MAP_ONLY

# ---------------------------------------------------------------------------
# Resolved mapping table
# ---------------------------------------------------------------------------
echo "============================================================================================"
echo "GUROBI OPTIMA LADDER - RESOLVED RESOURCE MAPPING"
echo "============================================================================================"
echo "  project root:   $PROJECT"
if [[ "$SINGLE_NODE" -eq 1 ]]; then
  echo "  submit mode:    --single-node (ONE job, four concurrent solves inside it)"
  echo "  node alloc:     -N 1 -c ${GUROBI_OPTIMA_NODE_CORES} --mem ${GUROBI_OPTIMA_NODE_MEM} -t ${GUROBI_OPTIMA_NODE_WALLTIME}"
  echo "                  (--mem 0 means all memory on the node)"
elif [[ "$PER_SIZE" -eq 1 ]]; then
  echo "  submit mode:    --per-size (four independently sized jobs, all concurrent)"
else
  echo "  submit mode:    SLURM array 0-3 (four tasks, one allocation each)"
  echo "  array -c/--mem: ${GUROBI_OPTIMA_ARRAY_CORES} cores / ${GUROBI_OPTIMA_ARRAY_MEM} / ${GUROBI_OPTIMA_ARRAY_WALLTIME} per task"
fi
echo
printf "  %-4s %-7s %-20s %-24s %-6s %-6s %-12s %-10s %-9s\n" \
  "TASK" "HUBS" "DATASET" "RUN NAME" "CORES" "MEM" "WALLTIME" "GRB LIMIT" "MARGIN"
printf "  %-4s %-7s %-20s %-24s %-6s %-6s %-12s %-10s %-9s\n" \
  "----" "-------" "--------------------" "------------------------" "------" "------" "------------" "----------" "---------"

# D-HH:MM:SS -> seconds, so the walltime/time-limit invariant can be checked.
walltime_seconds() {
  local w="$1" days=0 hms
  if [[ "$w" == *-* ]]; then days="${w%%-*}"; hms="${w#*-}"; else hms="$w"; fi
  local h m sec
  IFS=: read -r h m sec <<< "$hms"
  echo $(( 10#$days * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${sec:-0} ))
}

TOTAL_CORES=0
INVARIANT_FAILED=0
for idx in 0 1 2 3; do
  gurobi_optima_map "$idx"
  if [[ "$SINGLE_NODE" -eq 1 ]]; then
    EFF_CORES="$SINGLE_NODE_THREADS"; EFF_MEM="shared"; EFF_WALL="$GUROBI_OPTIMA_NODE_WALLTIME"
  elif [[ "$PER_SIZE" -eq 1 ]]; then
    EFF_CORES="$CORES"; EFF_MEM="$MEM"; EFF_WALL="$WALLTIME"
  else
    EFF_CORES="$GUROBI_OPTIMA_ARRAY_CORES"
    EFF_MEM="$GUROBI_OPTIMA_ARRAY_MEM"
    EFF_WALL="$GUROBI_OPTIMA_ARRAY_WALLTIME"
  fi
  TOTAL_CORES=$(( TOTAL_CORES + EFF_CORES ))
  WALL_S=$(walltime_seconds "$EFF_WALL")
  MARGIN=$(( WALL_S - GUROBI_TIME_LIMIT ))
  printf "  %-4s %-7s %-20s %-24s %-6s %-6s %-12s %-10s %-9s\n" \
    "$idx" "${SIZE}" "$DATASET" "$RUN_NAME" "$EFF_CORES" "$EFF_MEM" "$EFF_WALL" \
    "${GUROBI_TIME_LIMIT}s" "${MARGIN}s"
  # The whole point of the Gurobi time limit is that the process exits on its own
  # and writes summary.json with ObjBound. If SLURM kills it first, the bound is
  # lost and the run produces nothing. Refuse to submit in that case.
  if [[ "$MARGIN" -le 0 ]]; then
    echo "  !! FATAL: task $idx gurobi limit ${GUROBI_TIME_LIMIT}s >= walltime ${EFF_WALL} (${WALL_S}s)." >&2
    echo "            SLURM would kill it before it writes ObjBound." >&2
    INVARIANT_FAILED=1
  elif [[ "$MARGIN" -lt 600 ]]; then
    echo "  !! WARNING: task $idx has only ${MARGIN}s of margin for load/build/write." >&2
  fi
  if [[ ! -d "$PROJECT/$DATASET" ]]; then
    echo "  !! WARNING: dataset dir '$DATASET' does not exist under $PROJECT" >&2
  fi
done
if [[ "$INVARIANT_FAILED" -eq 1 ]]; then
  echo >&2
  echo "Refusing to submit. Lower GUROBI_TIME_LIMIT or raise the walltime in" >&2
  echo "sbatch_scripts/gurobi_optima_solve.sh." >&2
  exit 1
fi
echo
echo "  every task passes: --solver gurobi --mip-gap 0 --threads <its core count>"
echo "                     --gurobi-time-limit <below its walltime> --console-log"
if [[ "$SINGLE_NODE" -eq 1 ]]; then
  echo "  total core budget: ${TOTAL_CORES} Gurobi threads inside ONE ${GUROBI_OPTIMA_NODE_CORES}-core allocation"
  if [[ "$TOTAL_CORES" -gt "$GUROBI_OPTIMA_NODE_CORES" ]]; then
    echo "  !! FATAL: thread sum ${TOTAL_CORES} exceeds the ${GUROBI_OPTIMA_NODE_CORES}-core node." >&2
    exit 1
  fi
  echo "                     ${GUROBI_OPTIMA_NODE_CORES} granted, $(( GUROBI_OPTIMA_NODE_CORES - TOTAL_CORES )) left for OS/IO"
else
  echo "  total core budget: ${TOTAL_CORES} cores requested across the four concurrent tasks"
fi
if [[ -n "$MAX_CONCURRENT" ]]; then
  echo "  throttle:          at most ${MAX_CONCURRENT} task(s) running at once (--array=0-3%${MAX_CONCURRENT})"
  echo "                     peak concurrent cores: $(( GUROBI_OPTIMA_ARRAY_CORES * MAX_CONCURRENT ))"
fi
echo "============================================================================================"
echo

# ---------------------------------------------------------------------------
# License check reminder (Goal 4)
# ---------------------------------------------------------------------------
echo "LICENSE CHECK -- run this on the login node BEFORE submitting if you have not:"
echo
echo "    gurobi_cl --license"
echo
echo "  Read the 'Type' line. A 'size-limited' / trial license will not solve 100 hubs at all."
echo "  A floating/token license (Type: token or a TOKENSERVER line in gurobi.lic) has a finite"
echo "  token count -- four concurrent gurobipy processes take four tokens. If the limit is below"
echo "  4, re-run this launcher with --max-concurrent <tokens>."
echo "  A named-user or academic node-locked license has no token ceiling; ignore the throttle."
echo

# ---------------------------------------------------------------------------
# Build and show the submission command
# ---------------------------------------------------------------------------
if [[ "$SINGLE_NODE" -eq 1 ]]; then
  ARGS=( --parsable
         -N 1 -c "$GUROBI_OPTIMA_NODE_CORES" --mem "$GUROBI_OPTIMA_NODE_MEM"
         -t "$GUROBI_OPTIMA_NODE_WALLTIME"
         -J gurobi_optima_singlenode
         -o "logs/%x_%j.out" -e "logs/%x_%j.err"
         --export="ALL,GUROBI_OPTIMA_SINGLE_NODE=1"
         "$SOLVE" )
  echo "Submission command:"
  printf '    sbatch'; printf ' %q' "${ARGS[@]}"; printf '\n'
  echo
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ">>> DRY RUN: nothing submitted."
  else
    JOB_ID=$(sbatch "${ARGS[@]}")
    echo ">>> single-node job submitted: $JOB_ID"
    echo ">>> driver log:  logs/gurobi_optima_singlenode_${JOB_ID}.out"
    echo ">>> per-size logs: logs/gurobi_optima_singlenode_${JOB_ID}_<size>hubs.out"
  fi
elif [[ "$PER_SIZE" -eq 1 ]]; then
  echo "Submission commands (four separate jobs, submitted back to back, all run concurrently):"
  echo
  for idx in 0 1 2 3; do
    gurobi_optima_map "$idx"
    # %A/%a are array-only substitutions; a non-array job uses %j.
    ARGS=( --parsable
           -c "$CORES" --mem "$MEM" -t "$WALLTIME"
           -J "gurobi_optima_${SIZE}hubs"
           -o "logs/%x_%j.out" -e "logs/%x_%j.err"
           --export="ALL,GUROBI_OPTIMA_SIZE_IDX=${idx}"
           "$SOLVE" )
    printf '    sbatch'; printf ' %q' "${ARGS[@]}"; printf '\n'
    if [[ "$DRY_RUN" -eq 0 ]]; then
      JOB_ID=$(sbatch "${ARGS[@]}")
      echo "      -> submitted job $JOB_ID  (logs/gurobi_optima_${SIZE}hubs_${JOB_ID}.out)"
    fi
  done
  echo
  [[ "$DRY_RUN" -eq 1 ]] && echo ">>> DRY RUN: nothing submitted."
else
  ARRAY_SPEC="0-3"
  if [[ -n "$MAX_CONCURRENT" ]]; then
    ARRAY_SPEC="0-3%${MAX_CONCURRENT}"
  fi
  ARGS=( --parsable --array="$ARRAY_SPEC" "$SOLVE" )
  echo "Submission command:"
  printf '    sbatch'; printf ' %q' "${ARGS[@]}"; printf '\n'
  echo
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ">>> DRY RUN: nothing submitted."
  else
    JOB_ID=$(sbatch "${ARGS[@]}")
    echo ">>> solve array submitted: job $JOB_ID (tasks ${ARRAY_SPEC})"
    echo ">>> logs: logs/gurobi_optima_${JOB_ID}_<taskid>.out / .err"
  fi
fi

echo
echo "Queue:    squeue -u \$USER"
echo "Collect:  python collect_gurobi_optima.py"
if [[ "$SHOW_COLLECT" -eq 1 ]]; then
  echo
  echo "Once every task is COMPLETED or TIME_LIMIT, run:"
  echo "    cd $PROJECT && python collect_gurobi_optima.py --output results/gurobi_optima.csv"
fi
