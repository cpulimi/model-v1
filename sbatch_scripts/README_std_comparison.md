# Matched-budget SA vs SQA comparison (low instance)

Fair, apples-to-apples comparison of the SA and SQA samplers at the **standard
budget**: `--num-reads 100 --num-sweeps 3000 --max-stages 3`. Both halves share
identical batching, penalties, and seeds; the **only** deliberate differences
are the sampler backend and the SQA resource bump (trotter memory/time).

Results are written to a **new** folder, `results/low_std_comparison/`, so the
earlier `results/low_adaptive_parallel/` runs (low_par_heavy, low_par_sqa) are
left untouched.

## Run on SOL (login node)

```bash
bash sbatch_scripts/low_par_std_launch.sh        # SA,  100/3000/3
bash sbatch_scripts/low_par_sqa_std_launch.sh    # SQA, 100/3000/3 (+ gamma 1.0)
```

Each launcher submits split -> solve array -> dependent merge (afterok).

| Half | run-name | output | notes |
|------|----------|--------|-------|
| SA  | `low_par_std_sa`  | `results/low_std_comparison/low_par_std_sa/`  | `--mem 200G`, 8h |
| SQA | `low_par_std_sqa` | `results/low_std_comparison/low_par_std_sqa/` | `--sampler sqa --sqa-gamma 1.0`, `--mem 0`, 1 day (trotter=8 overhead) |

## Confirm SQA actually engaged

```bash
grep -E "sampler|sqa (beta|trotter|gamma)" logs/low_par_sqa_std_solve_*.out
```

Diagnostic CSVs land in each `qubo/` dir (`timing_memory_summary.csv`,
`adaptive_iteration_log.csv`), plus `slurm_mem_low.tsv` in the run root.
