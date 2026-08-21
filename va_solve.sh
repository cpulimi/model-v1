#!/bin/bash
# SOLVE job for the NEC Vector Annealing (VA) engine.
#
# Runs run_va_fsl_solver.py over a LADDER of instances, smallest first, so a
# failure surfaces on the 5-second instance instead of 40 minutes in. Each
# instance gets its own $RUN_ROOT/<instance> folder.
#
# Default ladder is the four small instances -- the goal here is to get the
# code running on the card and produce results. Scaling analytics come later.
#
#     sbatch va_solve.sh                            # the default ladder below
#     sbatch va_solve.sh instances_10hubs           # just one
#     sbatch va_solve.sh instances_low              # the 200-hub instance
#
# SIZES (measured by the solver's own preflight; see outputs/va_benchmark_preflight.csv):
#
#   instance            demand rows      Z       Y      X  total vars  batches  dense
#   instances_10hubs          3,207   3,207   1,180     10      4,397        1   74 MB
#   instances_20hubs          5,996   6,261   2,378     20      8,659        1  286 MB
#   instances_50hubs         15,126  17,449   6,153     50     23,652        1  2.1 GB
#   instances_100hubs        28,192  35,050  12,165    100     47,315        1  8.3 GB
#
# All four fit in ONE batch under the 60,000-variable ceiling, so the batching
# flags below are not load-bearing for them. They matter at 200+ hubs: at
# --max-z-vars-per-batch 50000 the largest instances_low batch compiles to
# 61,635 vars and check_ceiling aborts. 40000 gives 13 batches at <=48,829 vars
# (8.9 GiB dense) and passes. Verified with --dry-run, which needs no card.
#
# CAVEAT ON THE SMALL INSTANCES: instances_10hubs has exactly one eligible hub
# per ZIP, so every routing decision is forced -- C1 pins Z, C2 pins Y, C3 pins X.
# It is a perfect first-light test (known answer, seconds to run) and a useless
# benchmark point. 20hubs has a choice on only 4.4% of demand rows, 50hubs on
# 13.8%, 100hubs on 19.8% -- against 76.5% for instances_low. Treat cost numbers
# from this ladder as "the pipeline works", NOT as evidence about VA quality.
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
#SBATCH -t 0-04:00:00
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
# The 2022 PoC manual documents release VApoc_0201, but SOL carries V3.0.0 (see
# va_probe.py), so the install path is NOT hardcoded: pick the newest install
# actually present on this node. Override by exporting VA_PYTHON_DIR/NLC_VARS.
if [[ -z "${NLC_VARS:-}" ]]; then
  NLC_VARS=$(ls -1d /opt/nec/ve/nlc/*/bin/nlcvars.sh 2>/dev/null | sort -V | tail -n1 || true)
fi
if [[ -z "${VA_PYTHON_DIR:-}" ]]; then
  VA_PYTHON_DIR=$(ls -1d /opt/va/*/libexec/VectorAnnealing/python 2>/dev/null | sort -V | tail -n1 || true)
fi

set +u
if [[ -n "${NLC_VARS:-}" && -f "$NLC_VARS" ]]; then
  # shellcheck disable=SC1090
  source "$NLC_VARS"
else
  echo ">>> WARNING: no nlcvars.sh found under /opt/nec/ve/nlc/*; VA may fail to import." >&2
fi
set -u
export PATH="${PATH}:/opt/nec/ve/bin"
if [[ -n "${VA_PYTHON_DIR:-}" ]]; then
  export PYTHONPATH="${VA_PYTHON_DIR}:${PYTHONPATH:-}"
else
  echo ">>> WARNING: no VA install found under /opt/va/*/libexec/VectorAnnealing/python." >&2
  echo ">>>          Are you really on sfpga01n? Run: srun -w sfpga01n --pty bash -c 'ls -d /opt/va/*'" >&2
fi

echo ">>> host=$(hostname) project=$PROJECT python=$(which python)"
echo ">>> python version: $(python --version 2>&1)"
echo ">>> nlcvars=${NLC_VARS:-<none>}"
echo ">>> VA_PYTHON_DIR=${VA_PYTHON_DIR:-<none>}"
# ${PYTHONPATH:-} not $PYTHONPATH: when no VA install is found the export on
# the branch above never runs, and under `set -u` a bare $PYTHONPATH kills the
# script with "unbound variable" one line after the warning that explains the
# real problem.
echo ">>> PYTHONPATH=${PYTHONPATH:-<unset>}"
# Local VE card inventory -- this is the hardware the QUBO executes on.
echo ">>> VE device nodes visible here:"
ls -1 /dev/veslot* /dev/ve[0-9]* 2>/dev/null || echo "    <none visible from this host>"
if command -v /opt/nec/ve/bin/vecmd >/dev/null 2>&1; then
  /opt/nec/ve/bin/vecmd info || true
fi

# --- the instance ladder --------------------------------------------------
# Smallest first. Override by passing dataset dirs as arguments.
if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(
    instances_10hubs
    instances_20hubs
    instances_50hubs
    instances_100hubs
  )
fi

RUN_ROOT="results/va_ladder/va_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
echo ">>> run root: $RUN_ROOT"
echo ">>> ladder:   ${DATASETS[*]}"

# Plain strings, not arrays: under `set -u`, expanding an EMPTY array as
# "${arr[*]:-x}" is an unbound-variable error on bash <= 4.3, which is exactly
# the "nothing failed" case we most want to print.
FAILED=""
COMPLETED=""

for DATASET in "${DATASETS[@]}"; do
  NAME="$(basename "$DATASET")"
  OUT="$RUN_ROOT/$NAME"
  echo ""
  echo "############################################################"
  echo ">>> INSTANCE: $NAME  ($DATASET)"
  echo "############################################################"

  if [[ ! -d "$DATASET" ]]; then
    echo ">>> SKIP: $DATASET does not exist"
    FAILED="$FAILED $NAME(missing)"
    continue
  fi
  mkdir -p "$OUT"

  # --- preflight --------------------------------------------------------
  # Compiles every batch and checks the 60,000-var ceiling BEFORE touching the
  # card. Does not import VectorAnnealing, so it cannot fail for card reasons.
  if ! python run_va_fsl_solver.py \
      --dataset-dir "$DATASET" \
      --run-root "$OUT" \
      --part-batch-size 1000 \
      --max-z-vars-per-batch 40000 \
      --va-max-vars-per-batch 60000 \
      --dry-run; then
    echo ">>> PREFLIGHT FAILED for $NAME -- skipping its solve, continuing the ladder"
    FAILED="$FAILED $NAME(preflight)"
    continue
  fi

  # --- solve ------------------------------------------------------------
  # PENALTIES: objective scale is OFF by default in run_va_fsl_solver.py, so
  # C1-C4 sit flat at --min-penalty 50000 rather than ~2.51M. These results are
  # comparable to the NO-SCALE OpenJij arm (run_aligned_fsl_comparison_noscale.py)
  # and NOT to any scale-ON baseline. Pass --enable-objective-scale to switch.
  # --constraint-multiplier is inert while scale is OFF; kept for config parity.
  # ADAPTIVE PENALTY is ON. 8 iterations, not the OpenJij path's 5: with scale
  # OFF, C3 starts at 50000 against S_lim 500000, and escalating past it at
  # growth 1.5 needs 6 iterations.
  # RUNTIME: the adaptive loop samples once per iteration, so a batch costs up
  # to 8 VA calls, and --va-repeats multiplies on top. Repeats is 1 until a run
  # shows there is time budget for more.
  # NOTE: the VectorAnnealing module exposes no seed parameter, so there is no
  # --seed here; --va-repeats characterizes run-to-run spread instead.
  # --va-onehot is left OFF for a clean like-for-like first comparison.
  if ! python run_va_fsl_solver.py \
      --dataset-dir "$DATASET" \
      --run-root "$OUT" \
      --part-batch-size 1000 \
      --max-z-vars-per-batch 40000 \
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
      --qubo-time-limit 3600; then
    echo ">>> SOLVE FAILED for $NAME -- continuing the ladder"
    FAILED="$FAILED $NAME(solve)"
    continue
  fi

  COMPLETED="$COMPLETED $NAME"
  echo ">>> $NAME complete -> $OUT/va"
done

echo ""
echo "############################################################"
echo ">>> LADDER DONE. completed:${COMPLETED:- <none>}"
echo ">>>            failed:   ${FAILED:- <none>}"
echo "############################################################"
echo ">>> Key artifacts per instance in $RUN_ROOT/<instance>/va:"
echo "      summary.txt / summary.json"
echo "      batch_summary.csv          (schema matches the qubo/ path)"
echo "      va_batch_summary.csv       (batch_summary columns + VA columns)"
echo "      va_repeat_summary.csv      (per-repeat energy/cost distribution)"
echo "      va_precision_audit.csv     (fp32 vs float64 energy per read)"
echo "      va_batch_plan.csv          (variable counts and dense matrix sizes)"

# --- additive SLURM memory accounting (does NOT affect the solve above) ---
# Guarded on SLURM_JOB_ID: this block is bookkeeping only, and it runs AFTER
# the science. Unguarded under `set -u` it aborts the script when run outside
# sbatch (e.g. srun --pty bash va_solve.sh), which would report a fully
# successful ladder as a FAILED job.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo ">>> not running under SLURM (no SLURM_JOB_ID); skipping memory accounting"
  exit 0
fi
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
