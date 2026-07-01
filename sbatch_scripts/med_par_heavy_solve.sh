#!/bin/bash
# MAX-RESOURCE parallel-batch SOLVE for instances_medium (Option 3).
# Whole node: --exclusive + --mem=0. One array task per batch.
# SA budget: 100 reads / 3000 sweeps / 3 stages (~20x standard med_par).
# Expected wall: ~28-30h (slowest of 2 parallel batches).
#SBATCH -p general
#SBATCH -q private
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -J med_par_heavy_solve
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
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
echo ">>> host=$(hostname) project=$PROJECT python=$(which python) batch=${SLURM_ARRAY_TASK_ID} threads=$OMP_NUM_THREADS cpus=$SLURM_CPUS_ON_NODE mem=${SLURM_MEM_PER_NODE:-exclusive}"

python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_medium \
  --run-name med_par_heavy \
  --output-dir results/medium_adaptive_parallel_heavy \
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
  --adaptive-penalty-growth 1.5
