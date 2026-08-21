#!/bin/bash
# Adaptive-penalty data-scaling sweep (SA sampler, local).
# Holds the sampler budget FIXED (weak: 10 reads / 150 sweeps / 2 stages) and
# grows --max-parts-total to find where adaptive stops converging at iter 1/2.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/opt/anaconda3/bin/python
OUT=outputs/adaptive_scale_test
LOG=$OUT/sweep_console
mkdir -p "$LOG"

SIZES="25 50 100 200 300 450 600"

for N in $SIZES; do
  RUN=$(printf "p%03d" "$N")
  echo "=================================================================="
  echo ">>> RUN max-parts-total=$N  (run=$RUN)  $(date +%H:%M:%S)"
  echo "=================================================================="
  PYTHONHASHSEED=0 "$PY" run_aligned_fsl_comparison.py \
    --dataset-dir instances_low \
    --solver qubo --sampler sa \
    --max-parts-total "$N" \
    --num-reads 10 --num-sweeps 150 --max-stages 2 \
    --penalty-mode adaptive --min-penalty 50000 --constraint-multiplier 5.0 --c4-mode auto \
    --adaptive-penalty-mode within-batch --adaptive-penalty-iterations 5 --adaptive-penalty-growth 1.5 \
    --seed 42 \
    --output-dir "$OUT" --run-name "$RUN" \
    > "$LOG/$RUN.log" 2>&1
  echo ">>> done $RUN rc=$? $(date +%H:%M:%S)"
done
echo ">>> SWEEP COMPLETE $(date +%H:%M:%S)"
