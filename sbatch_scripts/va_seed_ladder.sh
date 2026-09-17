#!/bin/bash
# SEED LADDER: does the VA seed's MAGNITUDE change solution quality?
#
#     sbatch sbatch_scripts/va_seed_ladder.sh
#
# THE QUESTION. Three 20-hub runs -- seeds 42, 1000 and 2026 -- came back in
# descending cost order, which reads as "bigger seed, better answer". Three
# points land in that order 1 time in 6 by chance, so the observation carries no
# evidence either way. This job collects enough seeds to settle it.
#
# THE SAMPLE. Twelve seeds spanning six orders of magnitude, deliberately NOT
# submitted in magnitude order, so a drift in card state over the job cannot
# masquerade as a seed effect:
#
#     42, 7, 1000, 123456, 3, 2026, 99991, 500, 8675309, 17, 31337, 2
#
# 20 HUBS, NOT 50. A 20-hub run is ~27 min against ~2.7 h at 50, so the same
# walltime buys four times the sample -- and this question is about the number
# of seeds, not the size of the instance. 20 hubs also already has three runs
# in hand to reuse.
#
# THE THREE EXISTING RUNS ARE IMPORTED, NOT REPEATED. --import-run folds the
# finished results/va_parallel/va_20hubs* directories in as Arm B samples,
# reading each one's seed out of its own summary.json. That is ~80 minutes of
# card time not spent.
#
# ARM A IS THE GATE. Three runs at one fixed seed, to prove the harness itself
# is reproducible. Without it, variation in Arm B cannot be attributed to the
# annealer rather than to the pipeline around it.
#
# WALLTIME. 3 Arm A + 9 new Arm B runs at ~27 min = ~5.4 h. 10 h is requested so
# one slow run does not cost the study. If it is killed anyway, resubmit the
# same command -- finished runs are skipped and the CSV is appended in place.
#
# Override with environment variables:
#     VA_LADDER_DATASET=instances_50hubs sbatch sbatch_scripts/va_seed_ladder.sh
#     VA_LADDER_SEEDS=5,50,500,5000 sbatch sbatch_scripts/va_seed_ladder.sh
#SBATCH -w sfpga01n
#SBATCH -p fpga
# Not --exclusive / --mem=0: the fpga partition has one node and a whole-node
# request can sit PD (Resources) forever behind an interactive session. Peak host
# RSS at 20 hubs is ~481 MB, so 64G is already generous.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-10:00:00
#SBATCH -J va_seed_ladder
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs

# Absolute path: inside a job $0 points at SLURM's spool copy, so a relative
# source silently fails and the job then runs without VA on PYTHONPATH.
VA_ENV="$PROJECT/sbatch_scripts/va_env.sh"
if [[ -f "$VA_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VA_ENV"
else
  echo ">>> ABORT: $VA_ENV not found" >&2
  exit 1
fi

# Repeated even though va_env.sh exports it: an unpinned hash seed puts harness
# noise into Arm A and makes the annealer's contribution unidentifiable.
export PYTHONHASHSEED=0

VA_LADDER_DATASET="${VA_LADDER_DATASET:-instances_20hubs}"
VA_LADDER_OUT="${VA_LADDER_OUT:-results/determinism_20hubs}"
VA_LADDER_SEEDS="${VA_LADDER_SEEDS:-}"
VA_LADDER_FIXED="${VA_LADDER_FIXED:-42}"
VA_LADDER_ARMA="${VA_LADDER_ARMA:-3}"

# Finished runs to fold in rather than repeat. Only those matching the dataset
# are useful; the driver skips any whose seed is already in the CSV.
IMPORTS=()
if [[ "$VA_LADDER_DATASET" == "instances_20hubs" ]]; then
  for d in results/va_parallel/va_20hubs \
           results/va_parallel/va_20hubs_seed1000 \
           results/va_parallel/va_20hubs_seed2026; do
    [[ -f "$d/va/summary.json" ]] && IMPORTS+=(--import-run "$d")
  done
fi

if [[ -n "$VA_LADDER_SEEDS" ]]; then
  SEED_ARGS=(--seeds "$VA_LADDER_SEEDS")
else
  SEED_ARGS=(--seed-ladder)
fi

echo ">>> VA seed ladder"
echo ">>>   dataset      $VA_LADDER_DATASET"
echo ">>>   seeds        ${VA_LADDER_SEEDS:-<built-in ladder>}"
echo ">>>   arm A        $VA_LADDER_ARMA runs at fixed seed $VA_LADDER_FIXED (the gate)"
echo ">>>   imports      ${#IMPORTS[@]} argument(s)"
echo ">>>   output       $VA_LADDER_OUT"
echo ">>>   PYTHONHASHSEED=$PYTHONHASHSEED"

python3 va_determinism_study.py \
  --dataset-dir "$VA_LADDER_DATASET" \
  --out "$VA_LADDER_OUT" \
  --arms AB \
  --runs-per-arm "$VA_LADDER_ARMA" \
  --fixed-seed "$VA_LADDER_FIXED" \
  "${SEED_ARGS[@]}" \
  "${IMPORTS[@]}"
RC=$?

# Same memory-accounting helper the other VA jobs use.
if declare -F va_slurm_mem >/dev/null 2>&1; then
  va_slurm_mem "$VA_LADDER_OUT" "${SLURM_JOB_ID:-none}" seed_ladder || true
fi

echo ">>> seed ladder exited rc=$RC"
echo ">>> re-read the summary any time without re-running:"
echo ">>>   python3 va_determinism_study.py --out $VA_LADDER_OUT --dataset-dir $VA_LADDER_DATASET --analyze-only"
exit $RC
