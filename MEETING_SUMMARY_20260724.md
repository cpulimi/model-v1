# Adaptive Penalty — One-Page Summary (24 July 2026)

**Setup common to everything below:** `instances_low`, SA sampler (OpenJij), seed 42,
`PYTHONHASHSEED=0`, adaptive within-batch (growth 1.5×, cap 5 iterations),
`part-batch-size 1000`, `max-z-vars-per-batch 50000`, identical classical post-process.
The swept axis is **parts (SKUs)**; **hubs are fixed at 200** and zips at 2,000 throughout.

---

## The three experiments

| # | What changed | Result |
|---|---|---|
| **1** | Data 25 → 600 parts, penalty scaled (~2.5M), weak budget (10 reads / 150 sweeps) | **0 of 33 batches** exhausted the cap; all feasible in ≤ 3 iterations |
| **2** | Same sweep, objective-scale **removed** (penalty = base 50,000) | **20 of 33 batches** exhausted the cap — adaptive stalls |
| **3** | Keep 50k penalty, raise budget to 100 reads / 3000 sweeps | The exhausting batch is **feasible at iteration 1** |

### Experiment 1 — growing the data does not stress adaptive

| parts | 25 | 50 | 100 | 200 | 300 | 450 | 600 |
|---|---|---|---|---|---|---|---|
| batches | 1 | 1 | 2 | 4 | 6 | 8 | 11 |
| max iterations to feasibility | 2 | 2 | 2 | 3 | 3 | 3 | 3 |
| exhausted | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Batching caps every batch at ~50k Z-variables, so more parts means **more batches of the
same difficulty**, not harder batches. To actually load adaptive we must raise
`--max-z-vars-per-batch`, not `--max-parts-total`.

Note on the earlier "adaptive never moves off iteration 1" observation: that is a
**sampling-budget** effect. At production budget (100 reads / 3000 sweeps) SA is feasible
at iteration 1, so adaptive is a no-op. At the weak budget used here it does move (2–3
iterations) and still converges everywhere.

### Experiment 2 — the objective-scale is load-bearing at a weak budget

Exhausted batches by size: 25:0/1 · 50:0/1 · 100:**1**/2 · 200:**3**/4 · 300:**4**/6 ·
450:**5**/8 · 600:**7**/11 → **20 of 33**.

But the **delivered answer is unchanged**: final cost differs by **≤ 0.04%** at every size
and structural violations are **0** either way — the classical retry + repair recovers
feasibility regardless. The cost is paid in **time**: 1.5–2.8× slower (full low instance
544 s → 1,403 s), because each exhausted batch triggers the heavy retry-reads fallback.

**Definitions, precisely:**
- *"Exhausted"* = adaptive never produced a 0-violation sample within its 5 iterations.
  It is a statement about the **QUBO-native mechanism**, not about the delivered solution.
- *"Cost Δ"* = no-scale vs scaled **full-pipeline final cost** at the same part count
  (not versus Gurobi, and not the per-iteration cost printed inside the adaptive loop).

### The failure mode: a limit cycle, not slow convergence

Same batch every time (71 parts, Z ≈ 49,879), penalty 50k, weak budget — violations C1/C2/C3:

```
iter 1: 44 / 6 / 0
iter 2:  0 / 0 / 1     fix C1,C2 -> C3 breaks
iter 3: 31 / 6 / 0     fix C3    -> C1,C2 return
iter 4:  0 / 0 / 1
iter 5: 13 / 2 / 0  -> EXHAUSTED
```

Mechanism: each iteration multiplies by 1.5× **only** the constraints violated right now,
multipliers **only ratchet up**, and C1/C2 and C3 **share the same Z and Y variables**.

### Experiment 3 — the separation test (this overturned our hypothesis)

| batch | weak 10r/150s | strong 100r/3000s |
|---|---|---|
| Z ≈ 49,879 (71 parts) | **EXHAUSTED at 5** | **feasible at 1** |
| Z ≈ 35,905 (50 parts) | feasible at 3 | feasible at 1 |
| Z ≈ 24,370 (29 parts) | feasible at 5 | feasible at 3 (monotone) |
| Z ≈ 19,358 (25 parts) | feasible at 4 | feasible at 3 (monotone) |

**Verdict: under-sampling, not structural.** The penalty was sufficient all along. Where
iterations are still needed, violations now fall **monotonically** (10/17/40 → 3/4/10 → 0)
instead of ping-ponging. Our going-in hypothesis — "the penalty is too small, it's
structural" — was **wrong**; the double-digit C1 reversions were the weak sampler landing
in a different marginal local minimum each iteration (fresh seed, only 10 reads).

---

## The model that ties it together

Feasibility has **two independent levers**, and only starving **both at once** breaks it:

| | weak sampling | strong sampling |
|---|---|---|
| **penalty ~2.5M** | converges ≤ 3 iters (adaptive earns its keep) | **PRODUCTION** — feasible at iter 1, adaptive is a no-op safety net |
| **penalty 50k** | **PATHOLOGICAL** — limit cycle, 20/33 exhaust | feasible, usually at iter 1 |

The objective-scale specifically earns its keep when the **sampling budget is constrained**
— hardware annealers, or tight time budgets — because it delivers feasibility even from a
weak sampler.

---

## Next steps

1. **Stress the right axis** — hold parts at 600, raise `--max-z-vars-per-batch`
   (100k → 400k). Batch size, not part count, is what loads adaptive.
2. **Improve the schedule** — escalate all *recently*-violated constraints together, or
   grow proportionally to violation count, instead of a flat 1.5× on the current violator.
3. **Hubs axis** — today varied parts with hubs fixed at 200. A hubs-first framing needs a
   subset of `hubs.csv`; there is no `--max-hubs` flag (only `--top-hubs-per-zip`, which
   changes coupling density, not hub count).

## Caveats stated plainly

- Every final solution in all three experiments was structurally feasible (0 violations),
  and final costs agree within 0.04%. **None of this changes the delivered answer today.**
- Experiment 3 was run to 100 parts; the 200-part run was stopped for time once the
  decisive batch (Z ≈ 49,879) had answered the question.

## Artifacts

| Item | Path |
|---|---|
| Slide deck | `Adaptive_Penalty_Findings_20260724.pptx` |
| Exp 1 (scaled sweep) | `outputs/adaptive_scale_test/` + `RESULTS_adaptive_scaling.md` |
| Exp 2 (no-scale sweep) | `outputs/adaptive_noscale_test/` + `RESULTS_noscale_vs_scaled.md` |
| Exp 3 (budget test) | `outputs/adaptive_noscale_budget_test/` + `RESULTS_budget_test.md` |
| No-scale solver variant | `run_aligned_fsl_comparison_noscale.py` |
| Drivers / parser | `scripts/adaptive_scale_sweep.sh`, `scripts/adaptive_scale_sweep_noscale.sh`, `scripts/adaptive_noscale_budget_sweep.sh`, `scripts/parse_adaptive_sweep.py` |
