#!/bin/bash
# Run the VA solver over the small instance ladder, by hand, on the VE node.
#
# This is the NO-SLURM path. Run it from an interactive shell you already have
# on the card:
#
#     srun -w sfpga01n -p fpga --pty /bin/bash
#     module load mamba/latest && source activate qubo
#     ./va_run_ladder.sh
#
# va_solve.sh is the sbatch version of the same thing. Use this one when you
# want to watch it happen.
#
#     ./va_run_ladder.sh                                  # all four, small first
#     ./va_run_ladder.sh instances_20hubs                 # just one
#     ./va_run_ladder.sh instances_50hubs instances_100hubs
#     VA_SEED= ./va_run_ladder.sh                         # unseeded
#     VA_RUN_ROOT=results/try2 ./va_run_ladder.sh         # somewhere else
#
# Results land in $VA_RUN_ROOT/<instance>/va/, which is the layout
# va_results.py expects:
#
#     python3 va_results.py --run-root results/va_manual
#
# One instance failing does NOT stop the rest -- each is independent, and a
# failure on the 100-hub run should not cost you the three that already worked.

set -uo pipefail   # deliberately NOT -e; see above

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

VA_RUN_ROOT="${VA_RUN_ROOT:-results/va_manual}"
# Seed makes runs reproducible. Set VA_SEED= (empty) for an unseeded run.
VA_SEED="${VA_SEED-42}"

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(instances_10hubs instances_20hubs instances_50hubs instances_100hubs)
fi

mkdir -p logs "$VA_RUN_ROOT"

# --- VA on PYTHONPATH -----------------------------------------------------
# Discovered, never hardcoded: this cluster has V3.0.0, while the 2022 PoC
# manual documents VApoc_0201. Skipped if it is already there.
if [[ "${PYTHONPATH:-}" != *VectorAnnealing* ]]; then
  VA_PY=$(ls -1d /opt/va/*/libexec/VectorAnnealing/python 2>/dev/null | sort -V | tail -n1 || true)
  if [[ -n "$VA_PY" ]]; then
    export PYTHONPATH="${VA_PY}:${PYTHONPATH:-}"
    echo ">>> PYTHONPATH += $VA_PY"
  else
    echo ">>> WARNING: no VA install under /opt/va/*. Are you on sfpga01n?" >&2
  fi
fi

# --- refuse to run anywhere but the card ----------------------------------
# This script carries no #SBATCH directives -- it runs on whatever node the
# calling shell is already on. Get that shell without `-p fpga -w sfpga01n`
# and you land on SOL's default partition (htc), which has no Vector Engine.
# The solver would then build every QUBO, spend real minutes doing it, and only
# fail at the first sample() call. Check before any of that work happens.
VE_COUNT=$(ls -1 /dev/veslot* /dev/ve[0-9]* 2>/dev/null | wc -l | tr -d ' ')
if [[ "${VE_COUNT:-0}" -eq 0 && -z "${VA_ALLOW_NO_CARD:-}" ]]; then
  echo ""
  echo ">>> ABORT: no VE device visible on $(hostname) -- this is not the VA node."
  echo ">>>        You are probably on SOL's default partition (htc). Get a shell"
  echo ">>>        on the card first:"
  echo ">>>"
  echo ">>>            srun -w sfpga01n -p fpga --pty /bin/bash"
  echo ">>>            module load mamba/latest && source activate qubo"
  echo ">>>            ./va_run_ladder.sh"
  echo ">>>"
  echo ">>>        Set VA_ALLOW_NO_CARD=1 to override (only useful with --dry-run;"
  echo ">>>        an actual solve cannot work without the card)."
  exit 1
fi

echo ">>> host      $(hostname)"
# Pin hash-order determinism before the interpreter starts. Python randomises
# str hashing per process, so set/dict iteration order -- and therefore any
# float sum taken off a set -- varies between runs. See sbatch_scripts/va_env.sh
# for the full rationale. Override with PYTHONHASHSEED=random to run a control
# arm that deliberately measures hash sensitivity.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

echo ">>> python    $(command -v python3) -- $(python3 -V 2>&1)"
echo ">>> VE nodes  $(ls -1 /dev/veslot* /dev/ve[0-9]* 2>/dev/null | tr '\n' ' ' || echo none)"
echo ">>> run root  $VA_RUN_ROOT"
echo ">>> seed      ${VA_SEED:-<unseeded>}"
echo ">>> ladder    ${DATASETS[*]}"

# --- fail fast on the interpreter ----------------------------------------
# Cheaper to find out now than three instances in. The system python on
# sfpga01n is 3.6 and cannot even parse the solver.
if ! python3 -c "
import sys
assert sys.version_info >= (3, 7), 'need python >= 3.7, got %d.%d' % sys.version_info[:2]
compile(open('run_va_fsl_solver.py').read(), 'run_va_fsl_solver.py', 'exec')
import pyqubo, pandas, numpy
" 2>&1; then
  echo ""
  echo ">>> ABORT: this python cannot run the solver."
  echo ">>>        Fix with: module load mamba/latest && source activate qubo"
  exit 1
fi
echo ">>> interpreter OK"

# --- the ladder -----------------------------------------------------------
COMPLETED=""
FAILED=""

for DATASET in "${DATASETS[@]}"; do
  NAME="$(basename "$DATASET")"
  LOG="logs/va_${NAME}.log"

  echo ""
  echo "============================================================"
  echo ">>> $NAME   ($(date '+%H:%M:%S'))"
  echo "============================================================"

  if [[ ! -d "$DATASET" ]]; then
    echo ">>> SKIP: $DATASET does not exist"
    FAILED="$FAILED $NAME(missing)"
    continue
  fi

  CMD=(python3 run_va_fsl_solver.py
       --dataset-dir "$DATASET"
       --run-root "$VA_RUN_ROOT/$NAME")
  if [[ -n "$VA_SEED" ]]; then
    CMD+=(--va-seed "$VA_SEED")
  fi

  START=$(date +%s)
  "${CMD[@]}" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  ELAPSED=$(( $(date +%s) - START ))

  if [[ $RC -eq 0 ]]; then
    COMPLETED="$COMPLETED $NAME"
    echo ">>> $NAME OK in ${ELAPSED}s -> $VA_RUN_ROOT/$NAME/va"
  else
    FAILED="$FAILED $NAME(rc=$RC)"
    echo ">>> $NAME FAILED (rc=$RC) after ${ELAPSED}s -- see $LOG; continuing"
  fi
done

echo ""
echo "============================================================"
echo ">>> completed:${COMPLETED:- <none>}"
echo ">>> failed:  ${FAILED:- <none>}"
echo "============================================================"
echo ">>> Read the results with:"
echo ">>>     python3 va_results.py --run-root $VA_RUN_ROOT"

[[ -z "$FAILED" ]]
