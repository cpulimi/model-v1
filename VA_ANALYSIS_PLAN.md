# VA Scaling Study — Analysis Plan

Status: 10-hub and 20-hub results in hand. 50/100 pending.
Scope: VA-only. No Gurobi baseline, no OpenJij comparison.

---

## 0. Instance structure — read this before any chart

**At 10 hubs the problem has exactly one feasible solution.** Every one of the 200 ZIPs has
exactly one hub within `max_service_miles = 180`, so `z` is forced by C1, `y` by C2, `x` by C3.
The predicted forced solution is z=3207, y=1180, x=10, total=4397, SLA rows=638 — bit-for-bit
what VA returned, with all 100 reads at identical energy and the adaptive penalty converging at
iteration 1. VA spent 617.6 s of 621.1 s wall (99.4%) annealing a problem with zero degrees of
freedom. Good plumbing validation; **not** an optimization result.

Degeneracy decays slowly, so this taints the whole ladder:

| Hubs | ZIPs | Eligible (zip,hub) ≤180 mi | Mean hubs/ZIP | ZIPs with one option | Real choice |
|-----:|-----:|-----:|-----:|-----:|-----:|
| 10  | 200  | 200   | 1.00 | 200/200   | **0%** |
| 20  | 400  | 416   | 1.04 | 384/400   | 4% |
| 50  | 1000 | 1,149 | 1.15 | 867/1000  | 13% |
| 100 | 2000 | 2,494 | 1.25 | 1600/2000 | **20%** |

Even at 100 hubs, 80% of assignments are pinned by the service radius. Scope every quality
claim to the multi-option ZIPs, or it measures the data generator rather than the annealer.

---

## 1. The objective, and why it is hard *for an annealer*

```
total = Σ_stocked P_k                                  inventory
      + S_lim · |open hubs|            $500,000/hub    fixed
      + S_var · overflow_units         $12/unit >L     capacity overflow
      + C · new-hub transfers          $50/pair, T_j=0 transfer
      + Σ_assign [ λ1·h_s·b  +  λ2·h_d·b·(d−20)⁺  +  λ3·b·(d−130)⁺ ]
                   linehaul              distance         SLA penalty
```

| Term | 10 hubs | 20 hubs | Share (20h) |
|---|---:|---:|---:|
| Fixed open-hub | $5,000,000.00 | $10,000,000.00 | **91.0%** |
| Assignment transport | $286,685.06 | $529,038.22 | 4.8% |
| Inventory | $194,158.12 | $419,491.85 | 3.8% |
| New-hub transfer | $18,100.00 | $35,800.00 | 0.3% |
| Overflow storage | $0.00 | $0.00 | 0.0% |
| **Total** | **$5,498,943.18** | **$10,984,330.07** | |

Three consequences, each a slide:

**(a) The money is in 0.2% of the variables.** 20 `x` bits of 8,659 carry 91% of the cost.
Flipping one `x` correctly is worth $500,000; flipping one `z` is worth ~$88.

**(b) The penalty scale swamps the objective gradient.** Penalty is 50,000 per violation
against a ~$88 average assignment cost. The search is driven by feasibility, not cost. This is
the central formulation nuance and it is why `min_penalty` pins every multiplier (§3, Q2).

**(c) C4 is dead.** `L = 50,000` pairs/hub against 2,341 pairs stocked across 20 hubs.
`overflow_units = 0` at both sizes. Either shrink `L` so it binds, or say plainly it is
inactive and stop reporting it as a satisfied constraint.

---

## 2. Metrics to extract from every run

All of it is in `results/va_parallel/va_<N>hubs/va/summary.json` + `va_batch_summary.csv`.
Build one harvester over the four runs → a tidy `va_ladder.csv`, one row per (run, batch).

### 2.1 Problem size
`hubs`, `zips`, `parts`, `active_demand_rows`, `num_x/y/z`, `binary_variables`,
`qubo_interactions`, `nonzero_cells`, `matrix_density`, `avg_couplings_per_var`, `total_batches`

| Hubs | z | y | x | total vars | interactions | density | dense fp32 | waste |
|-----:|--:|--:|--:|--:|--:|--:|--:|--:|
| 10  | 3,207  | 1,180  | 10  | 4,397  | 8,784  | 0.0681% | 73.8 MiB | 734× |
| 20  | 6,261  | 2,378  | 20  | 8,659  | 17,563 | 0.0353% | 286 MiB  | ~1,460× |
| 50  | 17,449 | 6,153  | 50  | 23,652 | ~48,000 | ~0.013% | 2.08 GiB | ~4,000× |
| 100 | 35,050 | 12,165 | 100 | 47,315 | ~97,000 | ~0.006% | 8.34 GiB | ~8,000× |

(10/20 measured; 50/100 vars exact from the instance files, interactions extrapolated.)
All four fit one batch — the ladder is genuinely single-shot VA, not a merge artifact.

### 2.2 Runtime
`wall_seconds`, `qubo_build_seconds`, `qubo_sample_seconds` (= `annealing_seconds`),
`sample_eval_seconds`, `merge_seconds`, `postprocess_seconds`, `annealing_share_of_wall`

| Hubs | build | anneal | eval | merge | wall | anneal share |
|-----:|---:|---:|---:|---:|---:|---:|
| 10 | 0.25 s | 617.64 s | 2.58 s | 0.24 s | 621.12 s | 99.44% |
| 20 | 0.50 s | 1,597.39 s | 5.79 s | 0.47 s | 1,604.98 s | 99.53% |

### 2.3 Memory — **label every number; never add host to device**

`memory_accounting_version: 2` marks the post-fix semantics. A `summary.json` **without**
that key predates the fix and its `peak_memory_mb` is a tracemalloc figure, not RSS.

| Field | What it is | 10 hubs | 20 hubs |
|---|---|---:|---:|
| `peak_memory_mb` | **host RSS high-water mark**, max over batch and merge processes | 253.6 MB | 480.7 MB |
| `current_memory_mb` | host RSS at report time (merge process) | — | 147.0 MB |
| `max_batch_python_peak_tracemalloc_mb` | Python objects only — blind to pandas/pyqubo/VA | 53.0 MB | 114.5 MB |
| `max_batch_cgroup_peak_mb` | whole SLURM step; **size `--mem` against this one** | collect | collect |
| `max_batch_rss_children_peak_mb` | largest finished child — 0.0 expected, nothing forks | 0.0 | 0.0 |
| `max_dense_matrix_bytes` | VA's dense QUBO allocation — **device**, predicted | 77.3 MB | 286 MB |
| `ve_device.observed_peak_device_bytes` | **device, measured** — null unless a source exists | null | null |

Two rows that used to be one: `peak_memory_mb` was the tracemalloc figure through
`va_10hubs` and `va_20hubs`. It read 53.0 MB against a true 253.6 MB, and 114.5 MB against
a true 480.7 MB. `rss_peak_mb` was correct throughout — no measurement was ever lost, only
mislabelled. Do not turn those pairs into a multiplier: two instances is not a trend, and
tracemalloc's share moves with instance shape.

**`peak_memory_mb` is a max, not a sum.** It is the largest RSS any single process reached.
The batches ran as separate SLURM array tasks serialised on one card, so they were never
resident together. For a `--mem` request use `cgroup_peak_mb`.

**Host and device are separate resources and are never added.**

### 2.3a Canonical scaling table — regenerate, do not retype

`python3 va_scaling_table.py --csv results/va_scaling.csv` reads the run artefacts, so it
cannot drift from what was actually measured. **Use it as the source for any slide.**

| instance | hubs | total_vars | host RSS MB | cgroup MB | device MB | batches | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| | | | *measured* | *measured* | *predicted* | | |
| instances_10hubs | 10 | 4,397 | 253.6 | — | 73.8 | 1 | 621.1 |
| instances_20hubs | 20 | 8,659 | 480.7 | — | 286.0 | 1 | 1,605.0 |
| instances_50hubs | 50 | 23,652 | **—** | — | 2,134.0 | 1 | — |

Two gaps, both real, neither back-filled:
- **50-hub host RSS was never measured.** The solve was killed in its first adaptive
  iteration. `total_vars` and the dense prediction survive because they come from the
  compile, not the anneal.
- **cgroup peak was never captured for any run.** The pre-fix TSVs are headerless with
  blank memory columns; `seff` ran in-job and reported `State: RUNNING / 0.00 MB`. The
  capture is fixed now (§5.1) but postdates these runs.

#### The "240 MB → 2.1 GB, roughly 9×" claim is wrong — do not present it

It compares **253.6 MB host RSS, measured, on the 10-hub instance** against **2,134 MB
device dense, predicted, on the 50-hub instance**. Different resource, different instance,
different scaling law, and one is a measurement while the other is arithmetic. The ratio
describes nothing.

What can honestly be said, each **within one resource**:

- host RSS **253.6 → 480.7 MB** across 10→20 hubs (**1.90×**) — both measured
- device dense **73.8 → 2,134.0 MB** across 10→50 hubs (**28.9×**) — both predicted

Never add them and never divide one into the other.

### 2.3b Which resource binds — **not currently determinable**

Device cost is `4N²` (quadratic, exact). Host cost is *modelled* as `a + b·N` — linear in
interaction count, because the host never densifies (§3). The crossover is where they meet.

**`runtime.host_device_crossover.crossover_vars` is `null` for every run in this repo, and
that is the correct answer.** It is a least-squares fit over per-batch
`(total_vars, rss_peak_mb)` pairs and refuses below 3 batches. Every VA run recorded so far
is a **single batch**, so there is one point — no slope, no intercept, no fit.

Why the refusal matters. Three defensible methods on the same two runs disagree:

| method | crossover |
|---|---:|
| two-point fit, free intercept (a=19.2, b=0.0533) | 14,321 |
| proportional through the origin (10- and 20-hub mean) | 14,553 |
| 10-hub point alone, through the origin | 15,120 |

The spread is the intercept: host RSS has a real fixed floor (~19 MB of interpreter, pandas
and instance data) that an origin-forced model cannot express. A single-point estimate
picks one of these arbitrarily and reports it to five digits.

**Is host RSS actually linear? Unknown — and untestable with today's data.** Two points
define a line exactly; there is no residual, so linearity is an *assumption*, not a finding.
A third point would be the first real test. The 50-hub run cannot supply it: it died during
its first adaptive iteration (`logs/va_par_solve_61924260_1.out` ends at
`adaptive iter 1/8`, no checkpoint, no `rss_peak` line), so `total_vars = 23,652` was
measured but **host RSS never was**.

To make the crossover computable, either:
- run a ladder rung with `--part-batch-size` low enough to produce ≥3 batches, or
- complete 50 and 100 hubs and fit across runs rather than within one.

Until then, state the regime qualitatively: both completed runs are host-bound at the sizes
measured (253.6 MB host vs 73.8 MB dense at 10 hubs; 480.7 vs 286.0 at 20 hubs), device
cost grows quadratically and host cost does not, so device must dominate eventually — but
*where* is not yet measured.

### 2.3b-ii The redundant preflight compile is a MEMORY cost, not just a time cost

`build_batch_plan()` compiles every batch during preflight; `solve_va_batch()` compiles each
one again. The wall-time cost is small (~0.3 s/batch at 20 hubs, ~1.4 s at 50, against a
600–1,600 s anneal) and on that basis alone it is not worth fixing.

Memory is a different argument. Same process, same instance, card-free phases only,
solve run with and without the preceding preflight:

| instance | solve only | preflight + solve | preflight's cost |
|---|---:|---:|---:|
| 10 hubs | 167.5 MB | 201.0 MB | **+33.5 MB (+20%)** |
| 20 hubs | 209.2 MB | 278.0 MB | **+68.8 MB (+33%)** |

RSS at the moment the solve starts is 207 MB after preflight versus 127 MB fresh (20 hubs),
after an explicit `gc.collect()` — preflight's models are dropped but the memory is not
returned before the solve begins.

Two caveats, both material:
- **Measured on macOS**, whose allocator retains freed blocks. glibc returns large mmap'd
  allocations on free, so the effect may be smaller or absent on the VE node. The VmHWM
  self-delta columns settle it there; run `va_memory_selftest.py` first.
- These are **card-free phases only**. The real 20-hub solve peaked at 480.7 MB with the
  card, well above the 210 MB preflight, so on a real sequential run the solve still sets
  the peak — preflight raises the *floor* it starts from rather than setting the maximum.

So: preflight inflates sequential host memory by roughly the un-returned floor, and the
timing verdict does not cover that. It does not change which phase sets the peak.

### 2.3b-iii Hash-order determinism — required for the repeat study

`PYTHONHASHSEED` is now exported by every VA launch script (`va_env.sh`,
`va_par_solve.sh`, `va_par_merge.sh`, `va_par_launch.sh`, `va_solve.sh`,
`va_run_ladder.sh`), defaulting to `0` and overridable. Every OpenJij-era sbatch script
already did this; the VA path had lost it.

Each `summary.json` now records what actually happened:

| field | meaning |
|---|---|
| `python_hash_seed` | what the environment asked for (`"0"`, `"random"`, `"<unset>"`) |
| `hash_randomization_active` | what the interpreter did — the ground truth |
| `hash_order_deterministic` | convenience inverse of the above |
| `batch_python_hash_seeds` | seeds the SOLVE processes ran under, carried through the merge |

The last one matters because solve and merge are separate jobs: a pinned merge over
unpinned solves is not a pinned run. Batches from before this change report
`"<unrecorded>"` — which is what `va_10hubs` and `va_20hubs` show, correctly, since their
solves ran without the export.

`compute_solution_cost()` also sorts and uses `math.fsum`, so the known cost drift is gone
independently of the seed. Pinning covers the rest of the surface — any other aggregation
or ordering-sensitive tie-break.

**For a repeat study:** run the ladder with the default `PYTHONHASHSEED=0`, and if you want
to bound harness noise, run one rung with `PYTHONHASHSEED=random` as a control. Any spread
in that arm that is absent from the pinned arm is harness, not annealer.

### 2.3c Phase attribution

`va_memory_trace*.csv` plus `runtime.phase_peaks` carry per-phase host memory under nested
labels (`preflight.pyqubo_compile` vs `solve.pyqubo_compile` — the preflight compiles every
batch and the solve compiles it again). Two columns, deliberately distinct:

- `rss_peak_mb` — sampled every 0.5 s. Draws the shape; can miss a spike between polls.
- `vm_hwm_delta_self_mb` — exact rise in `/proc/self/status` VmHWM across the phase,
  exclusive of nested phases. **This is the attribution**: these sum to the run's total
  high-water rise. Linux only; reads 0.0 on macOS.

### 2.3d Instance degeneracy — the hub-siting result is geometry, not optimisation

`va_instance_geometry.py` (CPU, no card) measures how much siting freedom the instances
actually offer. Both are near-total degenerate:

| | 10 hubs | 20 hubs |
|---|---:|---:|
| demand rows | 3,207 | 5,996 |
| rows with exactly 1 reachable hub (service radius) | **3,207 (100.0%)** | 5,731 (95.6%) |
| rows with 2 or fewer | 100.0% | 100.0% |
| max reachable hubs for any row | **1** | 2 |
| best single hub's coverage | 13.4% | 8.8% |
| hubs forced open (sole option for some row) | **10 of 10** | **20 of 20** |
| rows unservable within the SLA radius | 638 (19.9%) | 874 (14.6%) |

**No hub can be closed in either instance.** Every hub is the only reachable option for at
least one demand row, so opening all of them is the only feasible answer. On the 10-hub
instance *every row* has exactly one candidate — C1 has a single feasible option per row,
so the assignment is fully determined too.

This is validated against the runs, not asserted: the geometric floor on SLA violations
matches `va_10hubs` exactly (638 = 638, 0 solver-attributable) and accounts for 92.7% of
`va_20hubs` (874 of 943). Both runs opened exactly the forced number of hubs.

**Consequence for the writeup.** `open_hubs = 10/10` and `20/20` carry no information about
solution quality, and neither does the 91% fixed-cost share — it is fixed cost because
every hub is forced open. Any hub-siting claim must be dropped or re-run on an instance
where siting is a real decision (wider service radius, or hubs sited with overlapping
coverage). What the runs *can* still speak to is assignment and stocking on the 20-hub
instance, and the engineering results (scaling, memory, precision), which do not depend on
siting freedom.

### 2.4 Demand pairs and solution shape
`active_demand_pairs`, `assignments_count`, `stocked_pairs_count`, `open_hubs_count`,
`closed_hubs_count`, plus the six cost terms.

### 2.5 Solution quality
- `audit.total_structural_violations` — 0 at both sizes; pass/fail, not a chart
- SLA violation **rate** = `sla_distance_violations` / `assignments_count`
- `structurally_feasible_reads / total_reads` — 100/100 and 112/112
- `energy_min / median / max`, `cost_min / median / max` → read dispersion
- `extra.postprocess` — separates what VA produced from what repair fixed
- **vs. the nearest-hub baseline (§4)** — the actual quality yardstick

### 2.6 Penalty behaviour
`adaptive_iteration_log.csv`: iterations used, exit reason, final `mult_c1..c4`, and per
constraint `min_pen` vs `scaled` vs `chosen` vs `binding_branch`.

### 2.7 Precision
`va_precision_audit.csv`: 100/100 then 112/112 reads, **max |rel diff| = 0.0** at both sizes.
Exact agreement between VA-reported and host-recomputed energy. One line, but it is the
credibility anchor for every other number.

---

## 3. Open questions — three now answered

### Q1 — Is VA time overhead-dominated? **No. Answered, and my earlier guess was wrong.**
I predicted 20 hubs would land near 600 s again. It came back at 1,597 s.

Per-read annealing: 6.176 s → 14.262 s = **2.31×**, for 1.97× the variables and 2.00× the
interactions → per-read cost scales as **vars^1.24**. On top of that `suggested_num_reads`
grows the read count as `√(num_z/5000)`, so *total* annealing time scales as roughly
**vars^1.7**. The two compound.

Projection from that fit:

| Hubs | vars | auto reads | per read | anneal | vs limits |
|-----:|-----:|-----:|---:|---:|---|
| 50  | 23,652 | 187 | ~49 s | **~9,225 s (2.56 h)** | fits, thin margin |
| 100 | 47,315 | 265 | ~116 s | **~30,778 s (8.55 h)** | **blows both limits** |

`VA_TIME_LIMIT = 13000 s` (3.61 h) and `#SBATCH -t 0-04:00:00` (4.00 h). **The 100-hub run as
currently configured will hit `--qubo-time-limit`, return
`stopped_due_to_time_limit: true` / `full_batch_coverage: false`, and burn four hours of the
single VE card producing nothing.** See §5.2 — this needs a decision before it is launched.

### Q2 — When does the adaptive penalty engage? **Not yet. Still inert at 20 hubs.**
`binding_branch_c1..c4 = floor` at both sizes (`min_pen = 50,000` vs `scaled = 5.0`),
multipliers stayed at 1.0, feasible at iteration 1. `min_penalty` is doing all the work and the
scaled branch is vestigial at these sizes. Worth asking whether a 50,000 floor is well-set
against the ~$88 assignment cost it competes with.

### Q3 — Does VA ever close a hub? **Not yet.** Zero closures at 10 and 20;
`hub_prune_closures = 0`, `hub_prune_relocations = 0`. Still the highest-value number in the
study at $500,000/hub. Report VA-native and post-processed closures separately, never summed.

### Q4 — Does read diversity appear? **Yes, exactly when choice does.** At 10 hubs
min = median = max energy (one feasible point). At 20 hubs the spread opens:
cost 10,984,542 / 10,985,900 / 10,986,739 — a $2,197 band, 0.02% of total. Small but real, and
it confirms dispersion tracks the free ZIPs rather than the annealer's temperature schedule.

Also new at 20 hubs: post-processing changed the answer for the first time. Raw VA cost
$10,984,542.22 → final $10,984,330.07, a **$212.15 improvement** from assignment repair and
stock trimming (at 10 hubs raw and final were identical). Track this gap — it is "how much of
the final answer VA did not produce."

### Q5 — Where is the memory wall? Dense fp32 alloc is exactly `vars² × 4`, verified against
both manifests, so it extrapolates in closed form:

| Hubs | est. vars | dense fp32 |
|---:|---:|---:|
| 100 | 47,315 | 8.34 GiB |
| 150 | ~71,000 | ~18.8 GiB |
| 200 | ~95,000 | ~33.6 GiB |
| 250 | ~118,000 | ~51.9 GiB |
| 300 | ~142,000 | **~75.1 GiB — over the 64 GB request** |

Against a matrix 99.99% zeros. But note Q1: **time runs out before memory does.** The dense
allocation would end the ladder near 250 hubs; the vars^1.7 time curve ends it at 100.

---

## 4. The quality yardstick — a baseline that needs no Gurobi

The instance structure supplies its own bracket. Each ZIP picks one hub from its eligible set,
so two trivial heuristics bound the assignment decision: **nearest eligible hub** and
**farthest eligible hub**. Computed on the 20-hub instance:

| | Total cost | Inventory | Transport | Stocked pairs | SLA violations |
|---|---:|---:|---:|---:|---:|
| Nearest-hub greedy | **$10,979,301.71** | $423,134.27 | $520,367.44 | 2,352 | 874 (14.6%) |
| **VA** | $10,984,330.07 | $419,491.85 | $529,038.22 | 2,341 | 943 (15.7%) |
| Farthest-hub | $11,006,450.19 | $425,195.95 | $545,504.24 | 2,354 | 1,049 (17.5%) |

**VA is beaten by a one-line nearest-hub greedy, by $5,028 (0.046% of total).** It sits 18.5%
of the way across the nearest→farthest span, so it captured 81.5% of the available spread —
while the trivial heuristic captured 100%.

Two honest caveats to state alongside it:
- VA does find *fewer stocked pairs* (2,341 vs 2,352) and the lowest inventory cost of the
  three. It is trading inventory against transport, which the greedy does not do at all. The
  trade just does not pay off here.
- Nearest-hub is trivially near-optimal **only because** this instance is near-degenerate and
  C4 never binds. Once capacity binds or hub closure becomes possible, the greedy stops being a
  serious competitor. That is an argument for fixing the instances, not for dropping the
  baseline.

This is the single most useful comparison available under the no-Gurobi/no-OpenJij constraint,
because it is derived entirely from the instance's own structure. Recompute it for 50 and 100.

---

## 5. Actions

### 5.1 SLURM memory capture — **fixed**
Root cause: `seff` and `sacct MaxRSS` read slurmdbd, which is not written until a step *ends*.
`va_slurm_mem()` runs inside the solve/merge jobs, so it could only ever see `State: RUNNING`
/ `Memory Utilized: 0.00 MB`. `sstat` should have covered the live case and returned nothing.

Two mechanisms now, in `sbatch_scripts/va_env.sh` and `sbatch_scripts/va_par_acct.sh`:
1. **`va_cgroup_peak()` (in-job)** — reads the kernel's own high-water mark from the job cgroup
   (`memory.peak` on v2, `memory.max_usage_in_bytes` on v1), walking up the hierarchy and
   taking the max. Needs no accounting database, so it always produces a number.
2. **`va_par_acct.sh` (post-hoc)** — a tiny job submitted `afterany` on both solve and merge,
   which waits for the slurmdbd flush (polling, up to 3 min) and writes the real `MaxRSS` and a
   real `seff`. `afterany`, not `afterok`, because a job killed by OOM is the one whose memory
   matters most.

`cgroup_peak_mb` is now ALSO read in-process by the solver (a Python port of
`va_cgroup_peak()`) and lands in `summary.json` alongside the rest of the accounting, so it
no longer lives only in a side TSV. The bash version stays as an independent cross-check.

`slurm_mem_va.tsv` now has a header and a `Source` column (`in_job` vs `post_hoc`) so the
provisional and final rows are never mixed. `va_par_launch.sh` submits the accounting job as
stage 4 of 4.

**Recover the 10- and 20-hub numbers now** — they are still in slurmdbd but will age out:

```bash
VA_ACCT_JOBS='61920528 61920529' VA_RUN_NAME=va_10hubs sbatch sbatch_scripts/va_par_acct.sh
VA_ACCT_JOBS='61921976 61921977' VA_RUN_NAME=va_20hubs sbatch sbatch_scripts/va_par_acct.sh
```

### 5.2 Decide the 100-hub time budget — **before launching it**
Per Q1 the run needs ~8.55 h of annealing against a 3.61 h internal limit and a 4 h walltime.
Two levers, and the first is better methodology anyway:

- **Pin the reads.** `VA_NUM_READS` currently auto-scales as `√(num_z/5000)` — 100 → 112 → 187
  → 265 across the ladder. That makes reads a *confound*: the 10→20 time growth mixes problem
  growth with a 12% read increase, so "time vs N" is not a clean scaling curve. Holding reads
  at 100 for every size both fixes the science and brings 100 hubs to ~11,600 s (3.22 h), which
  fits. Set `VA_NUM_READS=100` and add a `--num-reads` override that defeats the scaling.
- **Or raise the ceilings**: `VA_TIME_LIMIT` past 31,000 s and `#SBATCH -t` past 9 h in
  `va_par_solve.sh`. Costs the single VE card for most of a day.

Recommendation: pin the reads, and if you want a diversity measurement, do it as a separate
reads sweep at one fixed size rather than confounding the scaling ladder.

### 5.3 Still open
- **Run 10 hubs with `--va-onehot`.** `va_onehot_groups = 0`; C1 is exactly one-hot (each
  (zip,part) → exactly one hub) and VA V3.0.0 takes native one-hot groups. Feeding them
  natively confines the search to the feasible manifold instead of penalising departures,
  removing the 50,000 penalty and the ~$300 M offset from the energy scale. Highest-leverage
  experiment available, costs one flag. Even on the degenerate 10-hub instance it isolates the
  timing effect cleanly — and per Q1, timing is now the binding constraint on the whole ladder.
- **Decide what to do about `L = 50,000`.** Make it bind or declare it inactive.
- **Seed sweep at one size.** Everything is `seed = 42`, `repeats = 1`. One size × 5 seeds
  gives error bars; without them the quality slide is a single sample.
- **Build the harvester.** Four runs in, tidy `va_ladder.csv` out, columns per §2.
  `va_results.py` already reads ladder summaries and is the natural place to extend.

---

## 6. Proposed slides

1. **What we solved.** Objective block + the 91% fixed-cost donut.
2. **The instances are near-degenerate.** The §0 table. Frames everything, pre-empts the
   obvious reviewer question.
3. **Problem size vs. hubs.** Stacked z/y/x bars, interactions and density on a second axis.
4. **Where the time goes.** Stacked wall-time bar per N + the vars^1.7 fit and the walltime
   ceiling. This is now the study's headline scaling result.
5. **Memory, four ways.** Grouped bars (tracemalloc / RSS / dense alloc / cgroup peak) with the
   §2.3 legend, overlaying `vars² × 4` and the 64 GB line.
6. **Feasibility and precision.** Structural violations 0, feasible-read fraction, rel-diff 0.
7. **Quality vs. the nearest/farthest bracket.** The §4 table as a bar with VA between the
   bounds. The honest centrepiece.
8. **Penalty behaviour.** Floor-vs-scaled binding branch and multiplier trajectory.
9. **Where this ends.** Time wall at ~100 hubs, memory wall at ~250, sparse/dense waste factor.

Per house style: charts carry **no embedded titles** — the slide supplies the title. Keep the
caveat subtitle on any chart whose N=10 point is a forced solution.

---

## 7. What this study can and cannot claim

**Can claim:** VA runs the full FSL formulation end-to-end on the VE card and returns
structurally feasible solutions with exact energy agreement; QUBO size, interaction count,
density, wall-time composition and memory scale in a measured and extrapolable way; the time
wall and the memory wall are both located and quantified; the formulation's cost structure and
constraint activity are characterised.

**Cannot claim:** that VA finds a *good* solution. At 10 hubs there was only one; at 20 hubs a
nearest-hub greedy beats it; at 100 hubs four-fifths of decisions are still forced. Making
quality claims askable needs an instance generator with real hub redundancy and a binding
capacity constraint — as it stands, the ladder answers "can VA handle the scale," not "does VA
find better networks."
