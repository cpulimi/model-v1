# Bit-identical regression record — profiling + adaptive-penalty diagnostics

This is a **record of an executed test**, not a checklist. It backs the commit
claim that the additive instrumentation produces bit-identical solutions when the
new flags are at their defaults.

## What was compared

- **Baseline:** commit `e4e17bb` (pre-instrumentation), run from a clean
  `git worktree`.
- **Current:** working tree (instrumentation commit `a9aee1d` + the SQA-gamma /
  sstat / comment fixes on top).

Both were run through the full parallel pipeline (`split` -> `solve` x11 -> `merge`)
on `instances_low`, with **identical flags** and `PYTHONHASHSEED=0` (the production
setting in the sbatch scripts). Sampling budget was reduced only for speed and was
identical on both sides:

```
--seed 42 --part-batch-size 1000 --max-z-vars-per-batch 50000 \
--num-reads 20 --num-sweeps 150 --max-stages 2 --retry-reads-boost 2.0 \
--penalty-mode adaptive --min-penalty 50000.0 --constraint-multiplier 5.0 --c4-mode auto \
--adaptive-penalty-mode within-batch --adaptive-penalty-iterations 3 --adaptive-penalty-growth 1.5
```

## Result: IDENTICAL

- `summary.json -> final_solution`: **byte-identical** (JSON-normalized).
  `total_cost = 79,927,507.70299968`, `open_hubs = 114`, `stocked_pairs = 44,671`,
  `assignments = 115,710`, all C1-C4 violation audits `= 0`.
- `summary.json -> extra`: identical.
- All 9 solution CSVs byte-identical (md5): `open_hubs.csv`, `closed_hubs.csv`,
  `hubs_open_closed.csv`, `stocked_pairs.csv`, `stocked_hub_part_pairs.csv`,
  `hub_zip_part_pairings.csv`, `raw_qubo_open_hubs.csv`, `raw_qubo_stocked_pairs.csv`,
  `raw_qubo_hub_zip_part_pairings.csv`.
- `summary.json -> runtime`: differs **only** in timing/memory fields — the new
  additive keys (`per_batch_build/sample/eval_seconds`, `postprocess_seconds`,
  `hub_prune_seconds`, `cost_seconds`, `total_pipeline_seconds`,
  `merge_rss_peak_mb`, `per_batch_rss_peak_mb`, `max_single_batch_rss_mb`,
  `sum_of_batch_peaks_rss_mb`) plus wall-clock/tracemalloc values that vary
  run-to-run by nature. No solution field is affected.
- `summary.json -> dataset`: only `dataset_dir` differs (absolute path of the
  worktree vs the repo) — not a solution difference.

## New diagnostics confirmed populated (current run)

- `adaptive_iteration_log.csv`: 24 iteration rows across 11 batches, with
  `objective_scale`, per-constraint `min_pen/scaled/chosen/binding_branch/mult/viol`
  for c1-c4, `num_vars`, `num_interactions`, `seed_used`, `num_reads`, `exit_reason`.
  - `binding_branch_c1..c4` = `scaled` for all rows (scaled value ~2.5M >> floor
    50k, so the floor is never binding at this scale).
  - `objective_scale`: min 5.018e5, max 5.034e5, mean 5.023e5, std ~442.
  - `exit_reason`: `feasible` on the converging iteration (empty on prior iters).
- `timing_memory_summary.csv`: per-batch rows show `rss_peak_mb` ~1,199 MB vs
  `python_peak_tracemalloc_mb` ~492 MB — i.e. the true OS peak is ~2.4x the
  tracemalloc figure, which is exactly why the RSS instrumentation was added.
  `slurm_max_rss` is blank locally (no SLURM); it is joined from
  `slurm_mem_<scale>.tsv` on SOL.

## How to reproduce

Run each side with the flags above (baseline from a `git worktree add <tmp> e4e17bb`),
then compare `final_solution` in `summary.json` and md5 the solution CSVs.
