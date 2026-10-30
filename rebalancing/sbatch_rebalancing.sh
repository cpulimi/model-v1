#!/bin/bash
# Base rebalancing model (Gurobi) on Sol -- one synthetic instance per job.
#
# All arguments are passed straight to rebalancing/solve_gurobi.py. Submit from
# the repo root, e.g.
#
#   sbatch rebalancing/sbatch_rebalancing.sh --sites 100 --parts 100  --suppliers 5 --seed 0
#   sbatch rebalancing/sbatch_rebalancing.sh --sites 600 --parts 1000 --suppliers 5 --seed 0 --write-mps
#
# A size ladder, 3 seeds each:
#
#   for n in 50 100 200 600; do for s in 0 1 2; do
#     sbatch -J rebal_s${n} rebalancing/sbatch_rebalancing.sh --sites $n --parts 100 --suppliers 5 --seed $s
#   done; done
#
# Resize per job on the sbatch command line (-c, --mem, -t); the header below
# is only the default. Rough size guide for the synthetic generator with P parts:
#   X vars ~ P * (n/2)^2,  T vars ~ P * (n/2) * suppliers,  rows ~ 2 P n
# so 600 sites x 1000 parts is ~90M X vars -- that needs a big-memory node and
# is past what the synthetic generator should be asked to do. Scale parts first.
#
# --time-limit defaults to (walltime - 15 min) when not given, so Gurobi stops on
# its own, still reports ObjBound, and writes summary.json before SLURM kills it.
#SBATCH -p public
#SBATCH -q private
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-04:00:00
#SBATCH -J rebalancing
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

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

# Threads must match the cores SLURM granted; Threads=0 would size from the node.
THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXTRA=()
if [[ " $* " != *" --time-limit "* ]]; then
  # Walltime of this job in seconds, minus a 15 min margin for build + output.
  WALL_S=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null \
    | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4;
                      else if (NF==3) print $1*3600+$2*60+$3;
                      else if (NF==2) print $1*60+$2; }')
  WALL_S="${WALL_S:-14400}"
  EXTRA+=(--time-limit $(( WALL_S > 1800 ? WALL_S - 900 : WALL_S / 2 )))
fi
if [[ " $* " != *" --threads "* ]]; then
  EXTRA+=(--threads "$THREADS")
fi
if [[ " $* " != *" --out-dir "* ]]; then
  EXTRA+=(--out-dir results/rebalancing/sol)
fi

echo "============================================================================"
echo "REBALANCING BASE MODEL (GUROBI)"
echo "  host:      $(hostname)"
echo "  job id:    ${SLURM_JOB_ID:-n/a}"
echo "  cores:     ${SLURM_CPUS_PER_TASK:-n/a}   threads: ${THREADS}"
echo "  mem/node:  ${SLURM_MEM_PER_NODE:-n/a} MB"
echo "  python:    $(which python)"
echo "  commit:    $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "  args:      $* ${EXTRA[*]}"
echo "============================================================================"

SOLVE_OUT=$(mktemp)
set +e
python -m rebalancing.solve_gurobi "$@" ${EXTRA[@]+"${EXTRA[@]}"} | tee "$SOLVE_OUT"
RC=${PIPESTATUS[0]}
set -e
echo ">>> solve exit code: $RC"

# Independent re-check of the run just written (no Gurobi). The solver's last
# line is "[<run_name>] STATUS ObjVal=... -> <run_dir>".
RUN_DIR=$(grep -o -- '-> .*$' "$SOLVE_OUT" | tail -n1 | cut -c4-)
rm -f "$SOLVE_OUT"
if [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/summary.json" ]]; then
  python -m rebalancing.score_solution --run-dir "$RUN_DIR" || RC=$?
  if command -v seff >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    seff "$SLURM_JOB_ID" > "$RUN_DIR/seff_${SLURM_JOB_ID}.txt" 2>&1 || true
  fi
fi

exit $RC
