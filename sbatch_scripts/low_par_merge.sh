#!/bin/bash
# Parallel-batch MERGE job for instances_low (Option 3), HEAVY iteration budget.
# Runs ONCE after all solve array tasks finish (afterok dependency). Combines
# every solved batch, runs the SAME hub-prune/post-process as the sequential
# solver, and writes the final qubo/ outputs + summary.json.
#
# Every flag below MUST match low_par_solve.sh (same run-name, output-dir,
# dataset, and batching flags) so it loads the right batch checkpoints.
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 128
#SBATCH --mem 200G
#SBATCH -t 0-02:00:00
#SBATCH -J low_par_merge
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
  --adaptive-penalty-growth 1.5 \
  --hub-prune-max-iterations 500
