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
# --- reproducibility ------------------------------------------------------
# PIN THE HASH SEED. Python randomises str/bytes hashing per process, so set and
# dict iteration order differs between runs. Float addition is not associative,
# so any sum taken off a set drifts in the last bits from one process to the
# next. That was observed on the merged cost total (~2e-15 relative).
#
# compute_solution_cost() now sorts and uses math.fsum, which removes the known
# instance, but this pins the whole surface: any future aggregation, and any
# ordering-sensitive tie-break, stays identical run to run. In a repeat study
# that separates ANNEALER variation from harness noise, an unpinned seed makes
# the two indistinguishable.
#
# Must be set BEFORE the interpreter starts -- setting it inside Python is too
# late. Overridable: run with PYTHONHASHSEED=random to deliberately MEASURE hash
# sensitivity as a control arm.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

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
# Additive instrumentation: an outside cross-check on the rss_peak_mb the solver
# records. Never fails the caller.
#
# WHY THIS IS TWO MECHANISMS. seff and `sacct MaxRSS` read the slurmdbd
# accounting database, and that database is not written until the step ENDS.
# Called from inside the job -- the only place these scripts can call it -- both
# therefore report a job that is still RUNNING with "Memory Utilized: 0.00 MB".
# That is exactly what happened to the 10- and 20-hub runs: every seff_*.txt
# says State: RUNNING and every slurm_mem_va.tsv memory column is blank. sstat
# reads the live step and is supposed to cover this, but it returns nothing here
# (it needs a jobacct_gather plugin that actually samples, and the step name it
# wants varies), so it cannot be the only fallback.
#
#   1. va_cgroup_peak (below, in-job)  -- the kernel's own high-water mark for
#      the job cgroup. Always available, needs no accounting database, and is
#      the same number cgroup-based SLURM accounting would eventually report.
#      This is the one that actually saves the data.
#   2. va_par_acct.sh (post-hoc job)   -- runs afterany on solve+merge, once the
#      steps have ended and slurmdbd has flushed, and rewrites the authoritative
#      rows. This is the one that gets a real seff.
#
# Rows carry a `source` column so the two are never silently mixed:
# `in_job` (provisional, cgroup-backed) vs `post_hoc` (final, sacct-backed).

# Kernel high-water mark for this job's cgroup, in MB. Empty if unavailable.
#
# Read AFTER the python child exits but BEFORE the step tears down -- the value
# is a high-water mark the kernel keeps for the whole cgroup, so a dead child's
# peak still counts. Walks up from this process's own cgroup because the batch
# step sits in a nested leaf (.../job_<id>/step_batch/user/task_0) while the
# interesting total is on an ancestor; take the max over the chain.
va_cgroup_peak() {
  local rel base line dir best=0 val f
  # cgroup v2 is "0::<path>"; v1 has a "<n>:memory:<path>" line.
  rel=$(awk -F: '$2 == "" {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)
  if [[ -n "$rel" ]]; then
    base="/sys/fs/cgroup$rel"                       # v2: unified hierarchy
  else
    rel=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)
    [[ -z "$rel" ]] && return 0
    base="/sys/fs/cgroup/memory$rel"                # v1: per-controller mount
  fi
  dir="$base"
  while [[ -n "$dir" && "$dir" != "/sys/fs/cgroup" && "$dir" != "/" ]]; do
    # memory.peak is cgroup v2 (kernel >= 6.8); max_usage_in_bytes is v1.
    for f in "$dir/memory.peak" "$dir/memory.max_usage_in_bytes"; do
      if [[ -r "$f" ]]; then
        val=$(cat "$f" 2>/dev/null || true)
        [[ "$val" =~ ^[0-9]+$ ]] && (( val > best )) && best=$val
      fi
    done
    dir=$(dirname "$dir")
  done
  (( best > 0 )) && awk -v b="$best" 'BEGIN{printf "%.1f", b/1048576}'
}

# Usage: va_slurm_mem <run_dir> <jobid> <label>
va_slurm_mem() {
  local run_dir="$1" jobid="$2" label="$3"
  [[ -z "${SLURM_JOB_ID:-}" ]] && return 0
  mkdir -p "$run_dir" || return 0

  local tsv="$run_dir/slurm_mem_va.tsv"
  # Header, once. Without it the file is six unlabelled columns and the blank
  # memory fields read as "the job used no memory" rather than "not collected".
  if [[ ! -s "$tsv" ]]; then
    printf 'JobID\tLabel\tSource\tCgroupPeakMB\tMaxRSS\tMaxVMSize\tReqMem\tElapsed\tState\n' > "$tsv"
  fi

  # Take the cgroup peak FIRST: it is the number that is actually available now,
  # and the sleep below is dead time during which nothing else reads it.
  local cgpeak
  cgpeak=$(va_cgroup_peak)

  sleep 10  # let SLURM register the step
  if command -v seff >/dev/null 2>&1; then
    seff "$jobid" > "$run_dir/seff_${label}_${jobid}.in_job.txt" 2>&1 || true
  fi

  # sstat reads the LIVE step. Try it, but do not depend on it -- see the note
  # above. --allsteps rather than a guessed ".batch" suffix, because which step
  # carries the numbers differs between a plain job and an array task.
  local maxrss maxvm sacct_line reqmem elapsed state
  maxrss=$(sstat --allsteps -j "${SLURM_JOB_ID}" --format=MaxRSS \
    --units=M --noheader --parsable2 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9.]+M?$' | sort -h | tail -n1 || true)
  maxvm=$(sstat --allsteps -j "${SLURM_JOB_ID}" --format=MaxVMSize \
    --units=M --noheader --parsable2 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9.]+M?$' | sort -h | tail -n1 || true)

  sacct_line=$(sacct -j "$jobid" \
    --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
    --units=M --noheader --parsable2 2>/dev/null \
    | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1 || true)
  [[ -z "$maxrss" ]] && maxrss=$(printf '%s' "$sacct_line" | awk -F'|' '{print $3}')
  [[ -z "$maxvm" ]] && maxvm=$(printf '%s' "$sacct_line" | awk -F'|' '{print $4}')
  reqmem=$(printf '%s' "$sacct_line" | awk -F'|' '{print $5}')
  elapsed=$(printf '%s' "$sacct_line" | awk -F'|' '{print $6}')
  state=$(printf '%s' "$sacct_line" | awk -F'|' '{print $7}')
  # A blank sacct line still means the job is alive; say so rather than leaving
  # a column that reads as missing data.
  [[ -z "$state" ]] && state="RUNNING"

  printf '%s\t%s\tin_job\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jobid" "$label" "${cgpeak:-}" "$maxrss" "$maxvm" "$reqmem" "$elapsed" "$state" \
    >> "$tsv"
  echo ">>> slurm mem: $label cgroup_peak=${cgpeak:-<n/a>}MB MaxRSS=${maxrss:-<pending>}" \
       "MaxVMSize=${maxvm:-<pending>} ReqMem=${reqmem:-<pending>} Elapsed=${elapsed:-<pending>}"
  if [[ -z "$maxrss" ]]; then
    echo ">>>   MaxRSS is blank because this job has not ended yet. va_par_acct.sh" \
         "will append the final post_hoc row."
  fi
}
