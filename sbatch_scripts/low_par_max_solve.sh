#!/bin/bash
# MAX-RESOURCE parallel-batch SOLVE for instances_low (Option 3).
# Whole node: --exclusive (all cores) + --mem=0 (all RAM). No -t => uses the
# partition/QOS maximum allowed time. One array task per batch (--array set by
# low_par_max_launch.sh). Heavy iteration budget: 500 reads / 12000 sweeps / 5 stages.
#SBATCH -p general
#SBATCH -q private
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00
#SBATCH -J low_par_max_solve
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
# Use every core on the exclusively-held node for the SA sampling.
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
echo ">>> host=$(hostname) project=$PROJECT python=$(which python) batch=${SLURM_ARRAY_TASK_ID} threads=$OMP_NUM_THREADS"

# Batching/penalties match the prior low baseline; only the iteration budget is
# raised, so the run isolates the effect of more SA iterations.
python run_parallel_batches.py \
  --mode solve \
  --batch-id ${SLURM_ARRAY_TASK_ID} \
  --dataset-dir instances_low \
  --run-name low_par_max \
  --output-dir results/low_adaptive_parallel_max \
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
