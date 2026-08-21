#!/bin/bash
# VA parallel-batch SOLVE. One array task per batch; --array is set by
# va_par_launch.sh. This is the only stage that needs the Vector Engine.
#
# THROTTLED TO ONE TASK AT A TIME (%1 in the launcher). sfpga01n exposes
# /dev/veslot0 and /dev/ve0 -- two device nodes for ONE physical card -- so
# concurrent tasks would contend for it. The gain here is not wall clock, it is
# checkpointing: each finished batch is pickled, so a walltime kill costs only
# the batch in flight and you resubmit just the missing ids.
#
# Every flag below MUST match va_par_split/merge (same run-name, output-dir,
# dataset, batching flags) or batch ids will not line up. run_va_parallel_batches.py
# records a batching signature and refuses to merge mismatched pieces.
#SBATCH -w sfpga01n
#SBATCH -p fpga
# NOT --exclusive / --mem=0. The fpga partition has one node, and an
# interactive session on it (an OnDemand VS Code job, or your own srun shell)
# is enough to make a whole-node request unschedulable forever -- squeue shows
# PD (Resources) and nothing ever starts. Nothing here needs the whole node:
# the VS Code server does not touch the Vector Engine, and the solve array is
# throttled to %1 so only one job uses the card at a time. Request what the
# process actually uses instead.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH -J va_par_solve
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -uo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs

source "$(dirname "${BASH_SOURCE[0]}")/va_env.sh"

DATASET="${VA_DATASET:-instances_low}"
RUN_NAME="${VA_RUN_NAME:-va_par}"
OUTDIR="${VA_OUTDIR:-results/va_parallel}"

echo ">>> batch ${SLURM_ARRAY_TASK_ID} of run ${RUN_NAME} (${DATASET})"

python3 run_va_parallel_batches.py \
  --mode solve \
  --batch-id "${SLURM_ARRAY_TASK_ID}" \
  --dataset-dir "$DATASET" \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTDIR" \
  --part-batch-size "${VA_PART_BATCH_SIZE:-1000}" \
  --max-z-vars-per-batch "${VA_MAX_Z:-40000}" \
  --va-max-vars-per-batch "${VA_MAX_VARS:-60000}" \
  --num-reads "${VA_NUM_READS:-100}" \
  --num-sweeps "${VA_NUM_SWEEPS:-3000}" \
  --penalty-mode adaptive \
  --min-penalty "${VA_MIN_PENALTY:-50000.0}" \
  --constraint-multiplier 5.0 \
  --c4-mode auto \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations "${VA_ADAPTIVE_ITERS:-8}" \
  --adaptive-penalty-growth 1.5 \
  --va-repeats "${VA_REPEATS:-1}" \
  --va-max-retries 3 \
  --va-vector-mode "${VA_VECTOR_MODE:-ACCURACY}" \
  ${VA_SEED:+--va-seed "$VA_SEED"} \
  ${VA_PRECISION:+--va-precision "$VA_PRECISION"} \
  --qubo-time-limit "${VA_TIME_LIMIT:-13000}"
RC=$?

va_slurm_mem "$OUTDIR/$RUN_NAME" "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
             "batch${SLURM_ARRAY_TASK_ID}"

exit $RC
