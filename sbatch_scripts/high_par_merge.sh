#!/bin/bash
# Parallel-batch MERGE job for instances_high (Option 3). Runs ONCE after all
# solve array tasks finish (afterok dependency). Combines every solved batch,
# runs the SAME hub-prune/post-process as the sequential solver, and writes the
# final qubo/ outputs + summary.json.
#
# Every flag below MUST match high_par_solve.sh (same run-name, output-dir,
# dataset, and batching flags) so it loads the right batch checkpoints.
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 32
#SBATCH --mem 100G
#SBATCH -t 0-10:00:00
#SBATCH -J high_par_merge
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
echo ">>> host=$(hostname) project=$PROJECT python=$(which python)"

python run_parallel_batches.py \
  --mode merge \
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
  --adaptive-penalty-growth 1.5 \
  --hub-prune-max-iterations 500
