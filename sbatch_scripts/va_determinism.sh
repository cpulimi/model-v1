#!/bin/bash
# Determinism study: 20 sequential VA runs on one card, then one summary table.
#
#     sbatch sbatch_scripts/va_determinism.sh
#
# Fire and forget. Nothing to babysit: the driver appends each run's row to the
# CSV as it finishes, so a walltime kill keeps everything already done, and
# resubmitting the SAME command resumes from where it stopped rather than
# starting over.
#
# ONE JOB, NOT AN ARRAY. There is a single Vector Engine, so the runs serialise
# either way; a plain loop in one job avoids needing a dependent collector job to
# assemble the summary afterwards.
#
# WALLTIME. The 10-hub run took ~620 s. 20 runs is ~3.5 h; 8 h is requested to
# absorb a slow seed without losing the study. If it is killed anyway, just
# resubmit -- completed runs are skipped.
#
# Override with environment variables:
#     VA_DET_DATASET=instances_20hubs sbatch sbatch_scripts/va_determinism.sh
#     VA_DET_RUNS=5 VA_DET_OUT=results/det_quick sbatch sbatch_scripts/va_determinism.sh
#SBATCH -w sfpga01n
#SBATCH -p fpga
# Not --exclusive / --mem=0: the fpga partition has one node and a whole-node
# request can sit PD (Resources) forever behind an interactive session. Ask for
# what one solver process actually uses. Peak host RSS at 10 hubs is ~254 MB.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -J va_determinism
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs

# Source by ABSOLUTE path off the submit dir: inside a job $0 points at SLURM's
# copy in the spool dir, so a relative source silently fails and the job then
# runs without VA on PYTHONPATH.
VA_ENV="$PROJECT/sbatch_scripts/va_env.sh"
if [[ -f "$VA_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VA_ENV"
else
  echo ">>> ABORT: $VA_ENV not found" >&2
  exit 1
fi

# va_env.sh already exports this; repeated here because the whole study is void
# without it -- an unpinned hash seed would put harness noise into arm A and make
# the annealer's contribution unidentifiable.
export PYTHONHASHSEED=0

VA_DET_DATASET="${VA_DET_DATASET:-instances_10hubs}"
VA_DET_OUT="${VA_DET_OUT:-results/determinism}"
VA_DET_RUNS="${VA_DET_RUNS:-10}"
VA_DET_SEED="${VA_DET_SEED:-42}"
# AB = both arms. B = varying-seed only (half the card time, but no control arm,
# so any variation cannot be attributed to the annealer rather than the harness).
VA_DET_ARMS="${VA_DET_ARMS:-AB}"

echo ">>> determinism study"
echo ">>>   dataset       $VA_DET_DATASET"
echo ">>>   arms          $VA_DET_ARMS"
echo ">>>   runs per arm  $VA_DET_RUNS  (arm A seed $VA_DET_SEED, arm B seeds 1..$VA_DET_RUNS)"
echo ">>>   output        $VA_DET_OUT"
echo ">>>   PYTHONHASHSEED=$PYTHONHASHSEED"

python3 va_determinism_study.py \
  --dataset-dir "$VA_DET_DATASET" \
  --out "$VA_DET_OUT" \
  --runs-per-arm "$VA_DET_RUNS" \
  --fixed-seed "$VA_DET_SEED" \
  --arms "$VA_DET_ARMS"
RC=$?

# Memory accounting for the study job itself, same helper the other VA jobs use.
if declare -F va_slurm_mem >/dev/null 2>&1; then
  va_slurm_mem "$VA_DET_OUT" "${SLURM_JOB_ID:-none}" determinism || true
fi

echo ">>> determinism study exited rc=$RC"
echo ">>> re-read the summary any time without re-running:"
echo ">>>   python3 va_determinism_study.py --out $VA_DET_OUT --dataset-dir $VA_DET_DATASET --analyze-only"
exit $RC
