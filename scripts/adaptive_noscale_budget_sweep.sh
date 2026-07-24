#!/bin/bash
# NO-SCALE + STRONG BUDGET test (SA sampler, local).
# Purpose: separate the two oscillation hypotheses. Same no-scale penalties
# (base 50000, objective_scale disabled), but raise the sampling budget from the
# weak 10r/150s to 100 reads / 3000 sweeps. If batches still EXHAUST/oscillate ->
# structural (penalty magnitude). If they now converge -> was under-sampling.
# Sizes 25 -> 200 only (per request).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/opt/anaconda3/bin/python
OUT=outputs/adaptive_noscale_budget_test
LOG=$OUT/sweep_console
mkdir -p "$LOG"

SIZES="25 50 100 200"

for N in $SIZES; do
  RUN=$(printf "p%03d" "$N")
  echo "=================================================================="
  echo ">>> RUN max-parts-total=$N  (run=$RUN)  $(date +%H:%M:%S)"
  echo "=================================================================="
  PYTHONHASHSEED=0 "$PY" run_aligned_fsl_comparison_noscale.py \
    --dataset-dir instances_low \
    --solver qubo --sampler sa \
    --max-parts-total "$N" \
    --num-reads 100 --num-sweeps 3000 --max-stages 2 \
    --penalty-mode adaptive --min-penalty 50000 --constraint-multiplier 5.0 --c4-mode auto \
    --adaptive-penalty-mode within-batch --adaptive-penalty-iterations 5 --adaptive-penalty-growth 1.5 \
    --seed 42 \
    --output-dir "$OUT" --run-name "$RUN" \
    > "$LOG/$RUN.log" 2>&1
  echo ">>> done $RUN rc=$? $(date +%H:%M:%S)"
done
echo ">>> SWEEP COMPLETE $(date +%H:%M:%S)"
