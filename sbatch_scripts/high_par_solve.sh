#!/bin/bash
# Parallel-batch SOLVE job for instances_high (Option 3). One array task per batch.
# --array is set by high_par_launch.sh (or pass --array=1-N on the sbatch line).
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 32
#SBATCH --mem 100G
#SBATCH -t 0-08:00:00
#SBATCH -J high_par_solve
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
# identical here and in high_par_merge.sh so batch ids and seeds line up.
python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_high \
  --run-name high_par \
  --output-dir results/high_adaptive_parallel \
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
