#!/bin/bash
# SOLVE job for the NEC Vector Annealing (VA) engine on instances_low.
#
# Runs run_va_fsl_solver.py, which solves the SAME QUBO the OpenJij path solves
# (run_aligned_fsl_comparison.py --solver qubo); only the sampler changes.
# Results land in $RUN_ROOT/va, a sibling of the existing qubo/ and gurobi/ dirs.
#
# EXECUTION TARGET: the physical NEC Vector Engine card on ASU SOL's sfpga01n.
# This runs the QUBO on local hardware through the on-prem VectorAnnealing
# module. There is no cloud path: no SACServiceClient, no REST endpoint, no
# credentials anywhere in this pipeline. pyqubo is the modeling layer only.
#
# Node sfpga01n is confirmed. The PARTITION below is still inferred from the
# node name -- confirm it once with:
#     sinfo -N -n sfpga01n -o '%N %P %t %f'
# and correct -p if that reports something other than 'fpga'. A wrong -p (or -w)
# leaves the job PENDING forever with ReqNodeNotAvail.
#SBATCH -w sfpga01n
#SBATCH -p fpga
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 0-08:00:00
#SBATCH -J va_solve
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs

module purge
set +u
if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc"; fi
source activate qubo 2>/dev/null || conda activate qubo 2>/dev/null || mamba activate qubo 2>/dev/null || true
set -u

# --- NEC Vector Annealing environment -------------------------------------
# Per the NEC Vector Annealing PoC Manual, 2nd Edition (Nov 2022). That manual
# documents the 2022 PoC release (VApoc_0201); the version installed here may
# differ, so adjust these paths to match /opt/va and /opt/nec/ve on this node.
NLC_VARS="${NLC_VARS:-/opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh}"
VA_PYTHON_DIR="${VA_PYTHON_DIR:-/opt/va/VApoc_0201/libexec/VectorAnnealing/python}"

set +u
if [[ -f "$NLC_VARS" ]]; then
  # shellcheck disable=SC1090
  source "$NLC_VARS"
else
  echo ">>> WARNING: $NLC_VARS not found; VA may fail to import." >&2
fi
set -u
export PATH="${PATH}:/opt/nec/ve/bin"
export PYTHONPATH="${VA_PYTHON_DIR}:${PYTHONPATH:-}"

echo ">>> host=$(hostname) project=$PROJECT python=$(which python)"
echo ">>> python version: $(python --version 2>&1)"
echo ">>> PYTHONPATH=$PYTHONPATH"
# VE card inventory. Non-fatal if the tool is absent.
if command -v /opt/nec/ve/bin/vecmd >/dev/null 2>&1; then
  /opt/nec/ve/bin/vecmd info || true
fi

RUN_ROOT="results/va_low/va_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
echo ">>> run root: $RUN_ROOT"

# --- Step 1: preflight ----------------------------------------------------
# Checks every batch against the 100,000-bit / dense-matrix ceiling BEFORE
# touching the VE card. Aborts with a suggested lower --max-z-vars-per-batch
# if any batch is too large. Does not import VectorAnnealing.
python run_va_fsl_solver.py \
  --dataset-dir instances_low \
  --run-root "$RUN_ROOT" \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 50000 \
  --va-max-vars-per-batch 60000 \
  --dry-run

# --- Step 2: solve --------------------------------------------------------
# QUBO construction is byte-identical to the OpenJij path; only the sampler
# changes. PENALTIES: objective scale is OFF by default in run_va_fsl_solver.py,
# so C1-C4 sit flat at --min-penalty 50000 rather than ~2.51M. Compare these
# results against the NO-SCALE OpenJij arm (run_aligned_fsl_comparison_noscale.py,
# scripts/adaptive_scale_sweep_noscale.sh) -- NOT against scale-ON baselines.
# Pass --enable-objective-scale to switch back to the ~2.51M penalties.
# --constraint-multiplier is inert while scale is OFF; kept for config parity.
# ADAPTIVE PENALTY is ON (matches the OpenJij no-scale baseline). 8 iterations,
# not ref's 5: with scale OFF, C3 starts at 50000 against S_lim 500000, and
# escalating past it at growth 1.5 needs 6 iterations.
# RUNTIME: the adaptive loop samples once per iteration, so a batch costs up to
# 8 VA calls, and --va-repeats multiplies on top of that. Repeats is held at 1
# here; raise it only once the calibration run shows there is time budget for it.
# NOTE: VA has no seed parameter, so there is no --seed here; --va-repeats
# characterizes run-to-run spread instead.
# --va-onehot is left OFF for a clean like-for-like first comparison.
python run_va_fsl_solver.py \
  --dataset-dir instances_low \
  --run-root "$RUN_ROOT" \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 50000 \
  --va-max-vars-per-batch 60000 \
  --num-reads 100 \
  --num-sweeps 3000 \
  --penalty-mode adaptive \
  --min-penalty 50000.0 \
  --constraint-multiplier 5.0 \
  --c4-mode auto \
  --hub-prune-max-iterations 500 \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 8 \
  --adaptive-penalty-growth 1.5 \
  --va-repeats 1 \
  --va-max-retries 3 \
  --va-vector-mode ACCURACY \
  --qubo-time-limit 21600

echo ">>> VA run complete. Key artifacts in $RUN_ROOT/va:"
echo "      summary.txt / summary.json"
echo "      batch_summary.csv          (schema matches the qubo/ path)"
echo "      va_batch_summary.csv       (batch_summary columns + VA columns)"
echo "      va_repeat_summary.csv      (per-repeat energy/cost distribution)"
echo "      va_precision_audit.csv     (fp32 vs float64 energy per read)"
echo "      va_batch_plan.csv          (variable counts and dense matrix sizes)"

# --- additive SLURM memory accounting (does NOT affect the solve above) ---
sleep 10  # let SLURM register the running step
if command -v seff >/dev/null 2>&1; then
  seff "${SLURM_JOB_ID}" > "$RUN_ROOT/seff_va_${SLURM_JOB_ID}.txt" 2>&1 || true
fi
SSTAT_LINE=$(sstat -j "${SLURM_JOB_ID}.batch" --format=MaxRSS,MaxVMSize \
  --units=M --noheader --parsable2 2>/dev/null | head -n1 || true)
MAXRSS=$(printf '%s' "$SSTAT_LINE" | awk -F'|' '{print $1}')
MAXVM=$(printf '%s' "$SSTAT_LINE" | awk -F'|' '{print $2}')
SACCT_LINE=$(sacct -j "${SLURM_JOB_ID}" \
  --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
  --units=M --noheader --parsable2 2>/dev/null \
  | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1 || true)
if [[ -z "${MAXRSS:-}" ]]; then MAXRSS=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $3}'); fi
if [[ -z "${MAXVM:-}" ]]; then MAXVM=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $4}'); fi
REQMEM=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $5}')
ELAPSED=$(printf '%s' "$SACCT_LINE" | awk -F'|' '{print $6}')
printf '%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_JOB_ID}" "$MAXRSS" "$MAXVM" "$REQMEM" "$ELAPSED" \
  >> "$RUN_ROOT/slurm_mem_va.tsv"
echo ">>> slurm mem appended to $RUN_ROOT/slurm_mem_va.tsv: jobid=${SLURM_JOB_ID} MaxRSS=$MAXRSS MaxVMSize=$MAXVM ReqMem=$REQMEM Elapsed=$ELAPSED"
