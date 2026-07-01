#!/bin/bash
# Parallel-batch SOLVE job for instances_low (Option 3), HEAVY iteration budget.
# One array task per batch. --array is set by low_par_launch.sh.
#
# Purpose: test whether increasing the SA sampling budget (reads/sweeps/stages)
# lowers QUBO cost further on the small instance, toward Gurobi's $74.70M.
# Batching + penalties are kept IDENTICAL to the prior $77.25M low baseline so
# the only variable is the iteration count.
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 128
#SBATCH --mem 200G
#SBATCH -t 0-08:00:00
#SBATCH -J low_par_solve
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
echo ">>> host=$(hostname) project=$PROJECT python=$(which python) batch=${SLURM_ARRAY_TASK_ID}"

# IMPORTANT: --run-name, --output-dir, and all batching/QUBO flags MUST be
# identical here and in low_par_merge.sh so batch ids and seeds line up.
# HEAVY budget: num-reads 500, num-sweeps 12000, max-stages 5 (vs baseline
# 100/3000/3). Batching (part-batch 1000 / max-z 50000) and penalties match
# the prior low baseline so the comparison isolates the iteration effect.
python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_low \
  --run-name low_par_heavy \
  --output-dir results/low_adaptive_parallel \
  --seed 42 \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 50000 \
  --num-reads 500 \
  --num-sweeps 12000 \
  --max-stages 5 \
  --retry-reads-boost 2.0 \
  --penalty-mode adaptive \
  --min-penalty 50000.0 \
  --constraint-multiplier 5.0 \
  --c4-mode auto \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 5 \
  --adaptive-penalty-growth 1.5
