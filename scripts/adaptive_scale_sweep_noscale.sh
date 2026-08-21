#!/bin/bash
# NO-SCALE adaptive data-scaling sweep (SA sampler, local).
# Same protocol as adaptive_scale_sweep.sh, but uses run_aligned_fsl_comparison_noscale.py
# (objective_scale disabled => penalties equal the base --min-penalty 50000, not millions).
# Holds the sampler budget FIXED and grows --max-parts-total to the full low instance (600).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/opt/anaconda3/bin/python
OUT=outputs/adaptive_noscale_test
LOG=$OUT/sweep_console
mkdir -p "$LOG"

SIZES="25 50 100 200 300 450 600"

for N in $SIZES; do
  RUN=$(printf "p%03d" "$N")
  echo "=================================================================="
  echo ">>> RUN max-parts-total=$N  (run=$RUN)  $(date +%H:%M:%S)"
  echo "=================================================================="
  PYTHONHASHSEED=0 "$PY" run_aligned_fsl_comparison_noscale.py \
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
