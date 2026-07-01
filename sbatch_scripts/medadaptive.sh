#!/bin/bash
#SBATCH -p general
#SBATCH -q public
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 0-10:00:00
#SBATCH -J med_adapt
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

source "$(dirname "$0")/common.sh"
cd "$PROJECT"
mkdir -p logs
module purge
module load mamba/latest gurobi/13.0.1
source activate qubo
export PYTHONHASHSEED=0

python run_aligned_fsl_comparison.py \
  --dataset-dir instances_medium \
  --solver both \
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
  --qubo-time-limit 5400.0 \
  --hub-prune-max-iterations 500 \
  --output-dir results/medium_adaptive


