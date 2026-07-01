#!/bin/bash
# Parallel-batch SOLVE job (Option 3). One array task per batch.
#
# STEP 1 (login node, once): discover how many batches you have, then set --array below.
#   python run_parallel_batches.py --mode split \
#     --dataset-dir instances_medium --run-name med_par \
#     --part-batch-size 1000 --max-z-vars-per-batch 600000
#
# STEP 2: submit this array, then submit med_par_merge.sh with a dependency:
#   sid=$(sbatch --parsable sbatch_scripts/med_par_solve.sh)
#   sbatch --dependency=afterok:$sid sbatch_scripts/med_par_merge.sh
#
# --array is set by med_par_launch.sh (or pass -t on sbatch command line).
#SBATCH -p lightwork
#SBATCH -q public
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-06:00:00
#SBATCH -J med_par_solve
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

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
echo ">>> host=$(hostname) project=$PROJECT python=$(which python)"

# IMPORTANT: --run-name, --output-dir, and all batching/QUBO flags MUST be
# identical here and in med_par_merge.sh so batch ids and seeds line up.
python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_medium \
  --run-name med_par \
  --output-dir results/medium_adaptive_parallel \
  --seed 42 \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 600000 \
  --num-reads 30 \
  --num-sweeps 500 \
  --max-stages 2 \
  --retry-reads-boost 2.0 \
  --penalty-mode adaptive \
  --min-penalty 50000.0 \
  --constraint-multiplier 5.0 \
  --constraint-multiplier-c2 3.0 \
  --constraint-multiplier-c3 2.0 \
  --c4-mode auto \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 5 \
  --adaptive-penalty-growth 1.5
