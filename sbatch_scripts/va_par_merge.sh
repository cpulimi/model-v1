#!/bin/bash
# VA parallel-batch MERGE. Runs ONCE after every solve array task succeeds
# (afterok dependency from va_par_launch.sh). Combines all solved batches, runs
# the same repair / trim / hub-prune post-pass the sequential runner does, and
# writes the final va/ outputs + summary.json.
#
# NO VECTOR ENGINE NEEDED. Merge is pure CPU -- aggregation, repair, hub-prune,
# cost accounting -- so it runs on a normal partition and leaves the single VE
# node free. That is why VA_REQUIRE_CARD=0 below.
#
# Every flag MUST match va_par_solve.sh. run_va_parallel_batches.py records a
# batching signature in each checkpoint and refuses to merge mismatched pieces,
# because batch ids are positional and a silent mismatch would produce a
# plausible-looking but wrong answer.
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-02:00:00
#SBATCH -J va_par_merge
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs

VA_REQUIRE_CARD=0
source "$(dirname "${BASH_SOURCE[0]}")/va_env.sh"

DATASET="${VA_DATASET:-instances_low}"
RUN_NAME="${VA_RUN_NAME:-va_par}"
OUTDIR="${VA_OUTDIR:-results/va_parallel}"

python3 run_va_parallel_batches.py \
  --mode merge \
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
  --va-vector-mode "${VA_VECTOR_MODE:-ACCURACY}" \
  ${VA_SEED:+--va-seed "$VA_SEED"} \
  ${VA_PRECISION:+--va-precision "$VA_PRECISION"} \
  --hub-prune-max-iterations "${VA_HUB_PRUNE_ITERS:-500}"
RC=$?

va_slurm_mem "$OUTDIR/$RUN_NAME" "${SLURM_JOB_ID}" "merge"

echo ">>> final outputs in $OUTDIR/$RUN_NAME/va/"
echo ">>> read them with: python3 va_results.py --run-root $OUTDIR"

exit $RC
