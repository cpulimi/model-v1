#!/bin/bash
# MAX-RESOURCE parallel-batch MERGE for instances_high (Option 3).
# Runs ONCE after all solve tasks finish (afterok). Flags MUST match
# high_par_max_solve.sh.
#SBATCH -p general
#SBATCH -q private
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00
#SBATCH -J high_par_max_merge
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
echo ">>> host=$(hostname) project=$PROJECT python=$(which python) threads=$OMP_NUM_THREADS"

python run_parallel_batches.py \
  --mode merge \
  --dataset-dir instances_high \
  --run-name high_par_max \
  --output-dir results/high_adaptive_parallel_max \
  --seed 42 \
  --part-batch-size 1000 \
  --max-z-vars-per-batch 600000 \
  --num-reads 100 \
  --num-sweeps 3000 \
  --max-stages 3 \
  --retry-reads-boost 2.0 \
  --penalty-mode adaptive \
  --min-penalty 50000.0 \
  --constraint-multiplier 5.0 \
  --constraint-multiplier-c2 3.0 \
  --constraint-multiplier-c3 2.0 \
  --c4-mode auto \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 5 \
  --adaptive-penalty-growth 1.5 \
  --hub-prune-max-iterations 500
