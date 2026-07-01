# Adaptive Penalty for QUBO Solver — Cursor Build Doc

## What we're building

A **within-batch adaptive penalty mechanism** for `standalone_qubo_solver_aligned.py`. After each SA sampling round, inspect which constraints (C1, C2, C3) were violated, scale those specific penalties by a growth factor α, rebuild the QUBO, and resample. Repeat until feasible or max iterations hit. If still infeasible, fall back to existing retry-with-more-reads logic as a safety net.

This is a research contribution — the standard QUBO literature (Smith & Coit 1997) treats penalty weights as static hyperparameters. We're implementing the per-constraint adaptive scheme of Lemonge & Barbosa (2004).

---

## Why we're doing this

The baseline results show QUBO opens too many hubs at scale (200 → 800 hubs: +5 → +14 → +34 hubs vs Gurobi; cost gap 7% → 9% → 17%). The hypothesis: penalty mass dilution. Fixed hub closure reward ($500K) stays constant while penalty mass scales with batch objective. As batches grow, closing a hub becomes a smaller relative reward, so SA plays safe and keeps hubs open.

Adaptive penalty fixes this by **letting the solver discover the right penalty levels per constraint** rather than guessing once at QUBO construction.

---

## File to modify

**`standalone_qubo_solver_aligned.py`** — modify in place. No new files.

---

## Code changes by location

### 1. New CLI args (in the `argparse` block, near line 1575)

```python
p.add_argument("--adaptive-penalty-mode", type=str, default="off",
               choices=["off", "within-batch"],
               help="Adaptive penalty strategy. 'off'=current static behavior. "
                    "'within-batch'=iteratively scale violated constraint penalties.")
p.add_argument("--adaptive-penalty-iterations", type=int, default=5,
               help="Max adaptive penalty iterations per batch (only used when mode != off).")
p.add_argument("--adaptive-penalty-growth", type=float, default=1.5,
               help="Multiplicative growth factor for violated constraint penalties. "
                    "Typical range 1.5-2.0. Lemonge & Barbosa (2004) default ~1.5.")
p.add_argument("--adaptive-penalty-initial-source", type=str, default="batch-scale",
               choices=["batch-scale", "external"],
               help="Where initial penalties come from. 'batch-scale'=current "
                    "penalty_weights() behavior. 'external'=hook for cross-batch "
                    "(not used in v1).")
```

### 2. Modify `penalty_weights()` (around line 462)

Currently returns `{"c1": base, "c2": base, "c3": base, "c4": base}`. Add an optional `multipliers` parameter that scales the returned penalties per constraint:

```python
def penalty_weights(batch_df, data, args, multipliers=None):
    # ... existing scale + base computation unchanged ...
    out = {"c1": base, "c2": base, "c3": base, "c4": base}
    # ... existing per-constraint overrides unchanged ...

    # NEW: apply adaptive multipliers if provided
    if multipliers is not None:
        for c in ("c1", "c2", "c3", "c4"):
            if c in multipliers:
                out[c] = out[c] * float(multipliers[c])
    return out
```

**Important:** `multipliers` defaults to `None`, so all existing call sites continue working unchanged.

### 3. New helper function — adaptive penalty loop

Add this function above `solve_batch`:

```python
def run_adaptive_penalty_loop(batch, batch_df, data, args, sampler, base_reads):
    """
    Within-batch adaptive penalty: iteratively grow penalties for violated constraints,
    rebuild QUBO, resample. Returns (best_eval, qubo_meta_final, multipliers_final,
    adaptive_iterations_used, was_feasible).
    """
    multipliers = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    growth = float(args.adaptive_penalty_growth)
    max_iter = int(args.adaptive_penalty_iterations)
    best_eval = None
    qubo_meta_final = None
    adaptive_iter_used = 0
    was_feasible = False

    for iteration in range(1, max_iter + 1):
        adaptive_iter_used = iteration

        # Rebuild QUBO with current multipliers
        penalties = penalty_weights(batch_df, data, args, multipliers=multipliers)
        Q, qubo_meta = build_qubo(batch_df, data, args, penalties)
        qubo_meta_final = qubo_meta

        # Single sampling pass at base reads (NOT the retry-reads escalation yet)
        sample_kwargs = {"num_reads": base_reads}
        if args.seed is not None and int(args.seed) >= 0:
            sample_kwargs["seed"] = int(args.seed) + batch.batch_id * 1000 + iteration
        if int(args.num_sweeps or 0) > 0:
            sample_kwargs["num_sweeps"] = int(args.num_sweeps)

        print(f"    adaptive iter {iteration}/{max_iter} | "
              f"multipliers c1={multipliers['c1']:.2f} c2={multipliers['c2']:.2f} "
              f"c3={multipliers['c3']:.2f}", flush=True)
        response = sampler.sample_qubo(Q, **sample_kwargs)

        # Pick best sample (use existing scoring tuple)
        iter_best = None
        for sample, energy in iter_openjij_samples(response):
            ev = evaluate_sample(sample, energy, qubo_meta, batch_df, data)
            key = (ev["total_violations"], ev["c1"], ev["c2"], ev["c3"], ev["cost"], ev["energy"])
            if iter_best is None:
                iter_best = ev
            else:
                old_key = (iter_best["total_violations"], iter_best["c1"],
                           iter_best["c2"], iter_best["c3"], iter_best["cost"],
                           iter_best["energy"])
                if key < old_key:
                    iter_best = ev

        # Update best_eval if this iteration beat it
        if best_eval is None:
            best_eval = iter_best
        else:
            old_key = (best_eval["total_violations"], best_eval["c1"], best_eval["c2"],
                       best_eval["c3"], best_eval["cost"], best_eval["energy"])
            new_key = (iter_best["total_violations"], iter_best["c1"], iter_best["c2"],
                       iter_best["c3"], iter_best["cost"], iter_best["energy"])
            if new_key < old_key:
                best_eval = iter_best

        print(f"    adaptive iter {iteration} | violations C1={iter_best['c1']} "
              f"C2={iter_best['c2']} C3={iter_best['c3']} | "
              f"cost={iter_best['cost']:.2f}", flush=True)

        # Check feasibility
        if int(iter_best["total_violations"]) == 0:
            was_feasible = True
            print(f"    adaptive feasible at iter {iteration}", flush=True)
            break

        # Grow multipliers for violated constraints
        if int(iter_best["c1"]) > 0:
            multipliers["c1"] *= growth
        if int(iter_best["c2"]) > 0:
            multipliers["c2"] *= growth
        if int(iter_best["c3"]) > 0:
            multipliers["c3"] *= growth
        if int(iter_best.get("c4", 0)) > 0:
            multipliers["c4"] *= growth

    return best_eval, qubo_meta_final, multipliers, adaptive_iter_used, was_feasible
```

### 4. Modify `solve_batch()` (around line 786)

Branch on `args.adaptive_penalty_mode`:

```python
def solve_batch(batch, active, data, args):
    # ... existing batch prep code unchanged through QUBO build ...

    sampler = openjij.SASampler()
    base_reads = suggested_num_reads(num_z, args)
    best_eval = None
    sample_seconds = 0.0
    eval_seconds = 0.0
    stage_used = 0
    adaptive_iters_used = 0
    adaptive_was_feasible = False
    final_multipliers = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}

    # NEW: adaptive penalty phase (if enabled)
    if args.adaptive_penalty_mode == "within-batch":
        print("  [2a/3] Adaptive penalty phase...", flush=True)
        t_adapt = time.time()
        (best_eval, qubo_meta, final_multipliers,
         adaptive_iters_used, adaptive_was_feasible) = run_adaptive_penalty_loop(
            batch, batch_df, data, args, sampler, base_reads
        )
        sample_seconds += time.time() - t_adapt

        if adaptive_was_feasible:
            # Skip retry-reads fallback; we're feasible
            print("  [2b/3] Adaptive succeeded - skipping retry-reads fallback", flush=True)
        else:
            print(f"  [2b/3] Adaptive exhausted ({adaptive_iters_used} iters) - "
                  f"running retry-reads fallback with final multipliers", flush=True)

    # EXISTING retry-reads logic - runs always when adaptive is off,
    # or as fallback when adaptive didn't reach feasibility
    if args.adaptive_penalty_mode == "off" or not adaptive_was_feasible:
        # Rebuild Q with the final multipliers (in case adaptive ran)
        penalties = penalty_weights(batch_df, data, args,
                                    multipliers=final_multipliers if args.adaptive_penalty_mode == "within-batch" else None)
        Q, qubo_meta = build_qubo(batch_df, data, args, penalties)

        print("  [2/3] Sampling QUBO with OpenJij (retry-reads loop)...", flush=True)
        for stage in range(1, int(args.max_stages) + 1):
            # ... EXISTING stage loop body unchanged ...
            pass  # keep all existing code from line 833-907
    else:
        # Adaptive succeeded; record that retry stages did not run
        stage_used = 0

    # ... existing post-processing unchanged ...
```

### 5. Per-batch result schema — add adaptive metadata

`BatchResult` dataclass (line 100): add fields

```python
adaptive_iterations_used: int = 0
adaptive_was_feasible: bool = False
final_penalty_multipliers: dict[str, float] = field(default_factory=lambda: {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0})
```

Populate these when constructing the `BatchResult` near line 921.

### 6. CLI summary print (around line 1616)

Add to the parameter dump at solver start:

```python
print(f"  adaptive penalty:      {args.adaptive_penalty_mode}")
if args.adaptive_penalty_mode != "off":
    print(f"    max iterations:      {args.adaptive_penalty_iterations}")
    print(f"    growth factor:       {args.adaptive_penalty_growth}")
```

### 7. Top-level summary metadata (around line 1455)

In the JSON summary dump, add:

```python
"adaptive_penalty_mode": args.adaptive_penalty_mode,
"adaptive_penalty_iterations_max": int(args.adaptive_penalty_iterations),
"adaptive_penalty_growth": float(args.adaptive_penalty_growth),
```

---

## Backward compatibility requirements

**This is critical.** When `--adaptive-penalty-mode off` (the default), the solver must behave **bit-identically** to the current baseline. Same seeds, same QUBO, same results.

- Default mode is `"off"` — no surprises
- `penalty_weights()` with `multipliers=None` must return exactly what it did before
- The retry-reads stage loop runs unchanged when adaptive mode is off
- New CLI args have defaults so existing scripts don't break

---

## Out of scope for v1 (do not implement)

- Cross-batch penalty learning (hook is present via `--adaptive-penalty-initial-source external` but not wired)
- Decreasing penalties on feasibility (Lemonge & Barbosa's "reward" mechanism)
- Parallel batch processing (next phase)
- Multi-seed best-of-N (next phase)
- Any changes to `solve_fsl_risk_optimization_aligned.py` (Gurobi solver) — leave it alone
- Any changes to `run_aligned_fsl_comparison.py` — leave it alone
