# Parallel + Adaptive QUBO: Results vs Gurobi

All runs use seed 42, `PYTHONHASHSEED=0`, identical penalty settings
(min-penalty 50000, constraint-multiplier 5.0, c4 auto), and the same classical
post-process (assignment repair + hub-prune, max 500 iterations).

## 1. Pipeline is explicitly two-stage

The QUBO solver is **not** a standalone optimizer; it is a two-stage hybrid:

1. **Stage 1 - sampling (per batch):** simulated annealing (OpenJij) on a
   penalty-encoded QUBO, batched to fit memory (~600k Z vars per batch on
   medium, ~50k on low).
2. **Stage 2 - classical post-process (global):** repair of
   missing/duplicate assignments, then an iterative hub-prune pass that closes
   hubs and relocates their assignments to cheaper open hubs.

Stage 2 does a large share of the final work. Measured on the final runs:

| Instance | Raw (stage 1) cost | Final cost | Hub-prune closures | Relocated assignments |
|---|---|---|---|---|
| low (parallel) | $132.37M (200 hubs) | $77.25M (110 hubs) | 90 | 75,874 / 115,710 = **65.6%** |
| medium (parallel, SOL) | $310.45M (498 hubs) | $129.49M (192 hubs) | 299 | 71,379 / 147,293 = **48.5%** |

In other words, roughly half to two-thirds of final assignments are placed by
the classical relocation pass, not by the annealer directly. The annealer's
contribution is the feasible stocking/assignment structure that the prune pass
then consolidates. Any claim about "QUBO solution quality" below is a claim
about this hybrid, and we state that explicitly.

## 2. Head-to-head vs Gurobi (same datasets, same cost model)

Gurobi solved both instances to proven (near-)optimality
(low: MIP gap 0.0, medium: MIP gap 0.00098), so the gaps below are true
optimality gaps, not gaps to another heuristic.

### instances_low (200 hubs / 600 parts / 2,000 zips; 115,710 demand pairs)

| | Gurobi (local) | QUBO parallel+adaptive (local, 11 batches) |
|---|---|---|
| Total cost | $74,703,680 (OPTIMAL) | $77,247,692 |
| **Cost gap** | - | **+3.41%** |
| Open hubs | 109 | 110 |
| Structural violations | 0 | 0 |
| Wall time | 380 s | **390 s critical path** (3,253 s sequential-equivalent) |
| Peak memory | 0.7 GB | 1.8 GB per batch job |

QUBO settings: 100 reads / 3,000 sweeps / 3 stages, adaptive within-batch.

### instances_medium (500 hubs / 800 parts / 2,000 zips; 147,293 demand pairs)

| | Gurobi (SOL) | QUBO parallel+adaptive (SOL, 2 batches) |
|---|---|---|
| Total cost | $118,998,537 (OPTIMAL, gap 0.001) | $129,494,237 |
| **Cost gap** | - | **+8.82%** |
| Open hubs | 178 | 192 |
| Structural violations | 0 | 0 |
| Wall time | 967 s (solver 243 s) | **6,881 s critical path** (12,353 s sequential-equivalent) |
| Peak memory | 6.0 GB | 27.5 GB (batch 1) / 19.8 GB (batch 2) |

QUBO settings: 30 reads / 500 sweeps / 2 stages, adaptive within-batch.

Cross-check: the sequential off-mode full run on SOL (2/2 batches, 11,979 s)
landed at $129,493,741 - within 0.0004% of the parallel+adaptive result. The
parallel decomposition reproduces the sequential result while cutting wall
clock 1.74x on medium (and 8.3x on low, where there are 11 batches).

### Honest read

- QUBO+post-process is **feasible everywhere and within 3.4-8.8% of proven
  optimal**, with the gap growing on the larger/denser instance.
- Gurobi is also **faster** at these scales (and dramatically so on medium).
  The QUBO pipeline's wall-clock benefit comes from batch parallelism, not
  from beating branch-and-bound per node-hour.
- The medium gap decomposes as: +14 open hubs (+$7.0M fixed), +$1.6M
  inventory, +$1.6M transport vs Gurobi.

## 3. Adaptive penalty: what it does and does not do

Design: per batch, sample -> evaluate violations -> multiply violated
constraints' penalties by 1.5 -> resample, up to 5 iterations
(Lemonge & Barbosa-style schedule).

### At production SA budgets, adaptive never triggers

Every batch of both final runs (11/11 low, 2/2 medium) was feasible at
iteration 1 with multipliers (1.0, 1.0, 1.0, 1.0). The base penalty scaling
is already sufficient when SA gets enough reads/sweeps. Off-mode and
adaptive-mode are equivalent in this regime (verified: bit-identical off-mode
regression vs baseline; medium parallel-adaptive vs sequential-off costs agree
to 0.0004%).

### Under a deliberately weak SA budget, adaptive demonstrably restores feasibility

Controlled experiment, full instances_low, 10 reads / 100 sweeps / 1 stage
(no retry fallback), seed 42, identical post-process:

| | adaptive OFF | adaptive WITHIN-BATCH |
|---|---|---|
| Raw structural violations (stage 1, all batches) | **50** (9 of 11 batches infeasible, worst batch 25) | **0** (all 11 batches feasible) |
| Adaptive iterations used | - | 1-3 per batch (mean 2.3) |
| Final multipliers reached | - | up to c1=2.25, c2=1.5 |
| Final cost after post-process | $78.76M | $79.95M |
| Final structural violations | 0 (repaired classically) | 0 (native) |
| QUBO wall time | 544 s | 1,141 s |

Findings:

1. **Adaptive does X, where X = solver-native feasibility.** Under a weak
   budget it converts 50 raw violations to 0 within at most 3 penalty
   escalations per batch, exactly as designed.
2. **It is not a cost mechanism.** The classical repair also recovers
   feasibility from the 50 violations, and the post-processed costs end up
   within ~1.5% of each other (repair was slightly cheaper here). Adaptive's
   value is removing the *dependence* on classical repair for feasibility -
   relevant if stage 2 were ever restricted (e.g., pure-QUBO deployments or
   hardware annealers where classical repair is out of scope).
3. **Recommended framing:** adaptive penalties are a *verified feasibility
   safety net*. Production settings do not need it (it costs nothing there -
   iteration 1 is the same sample as off-mode); weak budgets provably benefit.

## 4. Parallel decomposition: quality-preserving speedup

Per-batch OpenJij seeds depend only on (seed, batch_id, stage) and batching is
deterministic, so a batch solved in its own job is identical to the same batch
solved sequentially. The merge step runs the same global post-process once.

| Instance | Batches | Sequential-equivalent | Parallel critical path | Speedup |
|---|---|---|---|---|
| low | 11 | 3,253 s | 390 s | **8.3x** |
| medium | 2 | 12,353 s | 6,881 s | **1.8x** |

Speedup tracks batch count; the merge is cheap (8 s low / 143 s medium + a
60-70 s hub-prune at 500 iterations). Verified bit-identical final CSVs vs a
sequential run on a reduced instance, and cost agreement to 0.0004% on
medium full.

## 5. Methodological caveats (for the record)

- An earlier low-instance parallel result of $122.8M / 190 hubs was an
  artifact of capping hub-prune at 10 iterations in the merge; re-merging the
  same batch checkpoints with the baseline cap (500) gives the $77.25M / 110
  hubs reported here. Conclusions drawn from the capped run (e.g., "low and
  medium costs are nearly equal") do not survive the correction: hub count is
  cost-driven, not coverage-saturated, once pruning runs to convergence.
- `results/medium_adaptive/aligned_fsl_20260611_142021/combined_summary.csv`
  contains a Gurobi-vs-QUBO pair, but its QUBO row is a **time-limited run that
  completed only 1 of 2 batches** (stopped_due_to_time_limit=true). The valid
  medium head-to-head uses that run's Gurobi row against the parallel
  `med_par` QUBO run (full coverage), as reported above.
- SLA distance violations (soft, priced into transport cost) remain nonzero
  for both solvers (Gurobi 33.0k / QUBO 37.5k on medium); they are not
  constraint breaches.

## Artifact map

| Result | Path |
|---|---|
| Low Gurobi (optimal) | `outputs/aligned_fsl/low_full_20260521_191929/gurobi/` |
| Low QUBO parallel+adaptive | `outputs/low_par_full/low_par_full/qubo/` (prune-10 variant preserved at `qubo_prune10/`) |
| Medium Gurobi (SOL) | `results/medium_adaptive/aligned_fsl_20260611_142021/gurobi/` |
| Medium QUBO parallel+adaptive (SOL) | `results/medium_adaptive_parallel/med_par/qubo/` |
| Medium QUBO sequential off-mode (SOL) | `results/medium_adaptive/aligned_fsl_20260611_171813/qubo/` |
| Adaptive demo (weak SA, off/on) | `outputs/adaptive_demo/weak_off/`, `outputs/adaptive_demo/weak_on/` |
