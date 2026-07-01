#!/bin/bash
# Parallel-batch MERGE job (Option 3). Runs ONCE after all solve array tasks
# finish. Combines every solved batch, runs the SAME hub-prune/post-process as
# the sequential solver, and writes the final qubo/ outputs + summary.json.
#
# Submit with a dependency on the solve array:
#   sid=$(sbatch --parsable sbatch_scripts/med_par_solve.sh)
#   sbatch --dependency=afterok:$sid sbatch_scripts/med_par_merge.sh
#
# Every flag below MUST match med_par_solve.sh (same run-name, output-dir,
# dataset, and batching flags) so it loads the right batch checkpoints.
#SBATCH -p lightwork
#SBATCH -q public
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-02:00:00
#SBATCH -J med_par_merge
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
  --adaptive-penalty-growth 1.5 \
  --hub-prune-max-iterations 500
