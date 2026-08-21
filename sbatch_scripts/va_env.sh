#!/bin/bash
# Shared environment setup for the VA sbatch scripts. Source it, do not run it.
#
#     source "$PROJECT/sbatch_scripts/va_env.sh"
#
# Source it BY ABSOLUTE PATH off the submit dir. Inside an sbatch job
# ${BASH_SOURCE[0]} / $0 point at /var/spool/slurmd/job<id>/slurm_script, the
# copy SLURM made, so $(dirname ...) is the spool dir and this file is not
# beside it. That source fails silently (no `set -e`) and the job then runs
# without VA on PYTHONPATH and without va_slurm_mem.
#
# Sets up: mamba module + conda env, the VA python dir on PYTHONPATH, an
# interpreter check, and a va_slurm_mem() helper. Deliberately does NOT set
# `set -e` -- the callers manage their own exit codes.
#
# VA_REQUIRE_CARD=0 skips the VE device check (merge does not need the card).
# VA_CONDA_ENV=<name> overrides the conda env, default "qubo".

# --- conda ----------------------------------------------------------------
module purge
# REQUIRED, and easy to forget: without it `source activate` does not exist and
# the job silently runs on the system python, which is 3.6 on sfpga01n and
# cannot even parse the solver. gurobi is not loaded: this path has no Gurobi.
module load mamba/latest

set +u
if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc" || true; fi

# Confirm the env EXISTS before activating. conda's bin/activate calls `exit`
# on an unknown name, and `source` is a POSIX special builtin whose failure
# kills a non-interactive shell outright -- neither is catchable with `|| true`,
# and both would bypass the diagnostics below.
VA_CONDA_ENV="${VA_CONDA_ENV:-qubo}"
if conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$VA_CONDA_ENV"; then
  if command -v activate >/dev/null 2>&1; then
    source activate "$VA_CONDA_ENV" || true
  else
    conda activate "$VA_CONDA_ENV" || true
  fi
  echo ">>> conda env: $VA_CONDA_ENV"
else
  echo ">>> WARNING: conda env '$VA_CONDA_ENV' not found; staying on current python." >&2
fi
set -u

# --- VA module ------------------------------------------------------------
# Discovered, never hardcoded: this cluster carries V3.0.0 while the 2022 PoC
# manual documents VApoc_0201. Override with VA_PYTHON_DIR / NLC_VARS.
if [[ -z "${NLC_VARS:-}" ]]; then
  NLC_VARS=$(ls -1d /opt/nec/ve/nlc/*/bin/nlcvars.sh 2>/dev/null | sort -V | tail -n1 || true)
fi
if [[ -z "${VA_PYTHON_DIR:-}" ]]; then
  VA_PYTHON_DIR=$(ls -1d /opt/va/*/libexec/VectorAnnealing/python 2>/dev/null | sort -V | tail -n1 || true)
fi

set +u
if [[ -n "${NLC_VARS:-}" && -f "${NLC_VARS:-}" ]]; then
  # shellcheck disable=SC1090
  source "$NLC_VARS"
fi
set -u
export PATH="${PATH}:/opt/nec/ve/bin"
if [[ -n "${VA_PYTHON_DIR:-}" ]]; then
  export PYTHONPATH="${VA_PYTHON_DIR}:${PYTHONPATH:-}"
fi

echo ">>> host        $(hostname)"
echo ">>> python      $(command -v python3) -- $(python3 -V 2>&1)"
echo ">>> VA dir      ${VA_PYTHON_DIR:-<none found under /opt/va/*>}"
echo ">>> VE devices  $(ls -1 /dev/veslot* /dev/ve[0-9]* 2>/dev/null | tr '\n' ' ' || echo none)"

# --- must we be on the card? ---------------------------------------------
if [[ "${VA_REQUIRE_CARD:-1}" == "1" ]]; then
  VE_COUNT=$(ls -1 /dev/veslot* /dev/ve[0-9]* 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${VE_COUNT:-0}" -eq 0 ]]; then
    echo ">>> ABORT: no VE device on $(hostname). Wrong node/partition." >&2
    echo ">>>        This job needs '-p fpga -w sfpga01n'." >&2
    exit 1
  fi
fi

# --- interpreter ----------------------------------------------------------
# Fail here, not after the queue wait and a full QUBO compile.
if ! python3 -c "
import sys
assert sys.version_info >= (3, 7), 'need python >= 3.7, got %d.%d' % sys.version_info[:2]
compile(open('run_va_fsl_solver.py').read(), 'run_va_fsl_solver.py', 'exec')
import pyqubo, pandas, numpy
"; then
  echo ">>> ABORT: this python cannot run the solver." >&2
  echo ">>>        Expected: module load mamba/latest && source activate ${VA_CONDA_ENV}" >&2
  exit 1
fi
echo ">>> interpreter OK"

# --- SLURM memory accounting ---------------------------------------------
# Additive instrumentation: SLURM's authoritative MaxRSS as an outside
# cross-check on the rss_peak_mb the solver records. Never fails the caller.
# Usage: va_slurm_mem <run_dir> <jobid> <label>
va_slurm_mem() {
  local run_dir="$1" jobid="$2" label="$3"
  [[ -z "${SLURM_JOB_ID:-}" ]] && return 0
  mkdir -p "$run_dir" || return 0
  sleep 10  # let SLURM register the step
  if command -v seff >/dev/null 2>&1; then
    seff "$jobid" > "$run_dir/seff_${label}_${jobid}.txt" 2>&1 || true
  fi
  # sstat reads the LIVE step so MaxRSS is populated mid-job; sacct only
  # flushes it after the step ends. Try sstat first, fall back to sacct.
  local sstat_line maxrss maxvm sacct_line reqmem elapsed
  sstat_line=$(sstat -j "${SLURM_JOB_ID}.batch" --format=MaxRSS,MaxVMSize \
    --units=M --noheader --parsable2 2>/dev/null | head -n1 || true)
  maxrss=$(printf '%s' "$sstat_line" | awk -F'|' '{print $1}')
  maxvm=$(printf '%s' "$sstat_line" | awk -F'|' '{print $2}')
  sacct_line=$(sacct -j "$jobid" \
    --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
    --units=M --noheader --parsable2 2>/dev/null \
    | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1 || true)
  [[ -z "$maxrss" ]] && maxrss=$(printf '%s' "$sacct_line" | awk -F'|' '{print $3}')
  [[ -z "$maxvm" ]] && maxvm=$(printf '%s' "$sacct_line" | awk -F'|' '{print $4}')
  reqmem=$(printf '%s' "$sacct_line" | awk -F'|' '{print $5}')
  elapsed=$(printf '%s' "$sacct_line" | awk -F'|' '{print $6}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$label" "$maxrss" "$maxvm" "$reqmem" "$elapsed" \
    >> "$run_dir/slurm_mem_va.tsv"
  echo ">>> slurm mem: $label MaxRSS=$maxrss MaxVMSize=$maxvm ReqMem=$reqmem Elapsed=$elapsed"
}
