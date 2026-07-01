# "How I'll Know Cursor Did This Right" — Verification Checklist

You need to be able to look at what Cursor produces and tell whether it's correct.
This is the part where the old "AI-delegated work" trap lives — Cursor will produce
code that *runs* but may not produce code that *does what we wanted*. Go through
this list before you trust the output.

---

## Layer 1: Static inspection (do this first, takes 10 minutes)

Read the `git diff` and answer each question. If any answer is "I'm not sure,"
stop and ask Claude (me) before running anything.

### 1.1 Backward compatibility
- [ ] **Find the new `multipliers` parameter in `penalty_weights()`. What is its default value?**
      Correct answer: `None`. If it's `{}`, `1.0`, or anything else, that's wrong — the
      function must behave identically to before when called without this kwarg.
- [ ] **Search for `args.adaptive_penalty_mode == "off"`. In all branches where this
      condition is true, is any existing line of code changed?**
      Correct answer: no. The `off` path should be the existing code, untouched.
- [ ] **Did any existing CLI argument get its `default=` value changed?**
      Correct answer: no. Only new flags should be added.

### 1.2 The adaptive loop itself
- [ ] **In `run_adaptive_penalty_loop()`, find where multipliers get updated. Confirm:
      a multiplier only grows for a constraint that has `violations > 0` in that iteration.**
      Wrong implementation: growing all multipliers every iteration. That's not adaptive,
      that's just exponential blowup.
- [ ] **Find where `build_qubo()` is called inside the adaptive loop. Confirm it's called
      *every iteration*, not just once.**
      If Cursor builds Q once and tries to scale entries in place, the math will be wrong
      and the result will diverge from a true rebuild.
- [ ] **Confirm the adaptive loop uses `base_reads` (not `base_reads * retry_reads_boost^stage`)
      for `num_reads`.** The adaptive loop is a separate mechanism — it shouldn't use the
      retry-reads escalation.
- [ ] **Find the early-exit condition. It should check `total_violations == 0`. Anything
      else (e.g., checking only C2, or using a tolerance) is wrong.**

### 1.3 Fallback to retry-reads
- [ ] **When `adaptive_was_feasible == True`, does the existing stage loop run?**
      Correct answer: no. We skip it because we're already feasible.
- [ ] **When `adaptive_was_feasible == False`, does the existing stage loop run with
      the `final_multipliers` from the adaptive phase (not the un-multiplied baseline)?**
      This is subtle. If the stage loop rebuilds Q with multipliers=None, you've thrown
      away everything the adaptive loop learned.

### 1.4 Metadata
- [ ] **Are `adaptive_iterations_used`, `adaptive_was_feasible`, and
      `final_penalty_multipliers` all present in `BatchResult` and populated when the
      result is constructed?**
- [ ] **Are the three new fields included in the JSON summary that gets written out?**
      If they're computed but never written to disk, you can't analyze them later.

---

## Layer 2: Runtime smoke tests (10 minutes)

After Cursor finishes, run these commands and confirm the expected outputs.

### 2.1 Help text shows the new flags
```bash
python standalone_qubo_solver_aligned.py --help | grep -i adaptive
```
**Expected:** four lines, one per new flag.
**If you see fewer:** Cursor missed at least one flag. Re-check `argparse` block.

### 2.2 Default behavior unchanged — the critical regression test
Run the solver on the smallest test instance with the **exact same command** you used for
your baseline `low` instance run, *without* any of the new flags.

```bash
python standalone_qubo_solver_aligned.py \
  --seed 1 \
  [...all other flags exactly as in your baseline low run...]
```

**Expected:** the `total_cost`, `open_hubs`, and `sla_distance_violations` from the new
run should match your baseline `low` run **to within rounding error** (cost should match
to 2 decimal places; counts should match exactly). Your baseline low result was
$80,189,859.87 with 114 open hubs and 30,770 SLA violations.

**If results differ:** something in the default code path got accidentally changed.
**Do not proceed.** Show Claude the diff.

### 2.3 Adaptive mode actually runs differently
Run the same low instance with adaptive turned on:

```bash
python standalone_qubo_solver_aligned.py \
  --seed 1 \
  --adaptive-penalty-mode within-batch \
  --adaptive-penalty-iterations 5 \
  --adaptive-penalty-growth 1.5 \
  [...all other flags exactly as in your baseline low run...]
```

**Expected stdout patterns:**
- For each batch, you should see lines like:
  `  [2a/3] Adaptive penalty phase...`
  `    adaptive iter 1/5 | multipliers c1=1.00 c2=1.00 c3=1.00`
  `    adaptive iter 1 | violations C1=X C2=Y C3=Z | cost=...`
- For at least some batches you should see multipliers grow on iteration 2, 3, etc.
  (e.g., `multipliers c1=1.50 c2=1.00 c3=1.50` after C1 and C3 were violated in iter 1).
- For batches where adaptive succeeds, you should see `adaptive feasible at iter N`
  and *no* subsequent `stage N/3: sampling...` lines.

**If you see multipliers stuck at 1.00 forever:** the growth update is broken.
**If you see no `adaptive` lines at all:** the branch on `adaptive_penalty_mode` is wrong.
**If you see both adaptive *and* the old stage loop running every batch:** the
fallback condition is inverted.

### 2.4 The JSON summary contains the new fields

After the adaptive run, open the output summary JSON:

```bash
cat outputs/[...].../qubo/summary.json | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('adaptive_penalty_mode'), d.get('adaptive_penalty_iterations_max'), d.get('adaptive_penalty_growth'))"
```

**Expected:** `within-batch 5 1.5`
**If you see `None None None`:** the metadata wasn't wired through.

---

## Layer 3: Sanity checks on results (this tells you if the *method* is working)

After 2.3 succeeds, compare adaptive-on vs adaptive-off on the same seed.

### 3.1 Adaptive should not make things wildly worse
The cost from the adaptive run (seed=1) should be **within ±15% of the baseline cost** for
the same seed. If it's 50% worse, something is fundamentally wrong with the penalty math.
If it's 30% better, also be suspicious — that would be too good for a first attempt.

### 3.2 SLA violations should generally drop (or stay similar)
If adaptive penalty is working, the number of `sla_distance_violations` should go down or
stay about the same compared to the baseline. If it goes up significantly, the penalty
growth might be crushing the objective (a known failure mode I warned about — α too high).

### 3.3 Number of open hubs should generally decrease (or stay similar)
This is the actual hypothesis we're testing — that adaptive penalty fixes the
"opens too many hubs" problem. On low instance you may not see much change (the gap was
only +5 hubs). On high instance the effect should be more visible.

### 3.4 Look at `final_penalty_multipliers` distribution across batches
Open the per-batch results and look at the final multipliers. If most batches converged
at multipliers of 1.0 (no adaptation needed) or 1.5-2.25 (1-2 iterations), that's healthy.
If you see multipliers like 11.4 (six iterations of 1.5×), it means the constraint was
never satisfied — investigate that constraint specifically.

---

## What to do if something is wrong

1. **Don't ask Cursor to fix it blindly.** Cursor will happily produce a "fix" that
   makes the symptom go away while breaking something else. Diagnose first.
2. **Save the broken diff.** `git stash` or `git diff > broken_attempt.diff`. Bring it
   to Claude (me). I'll help diagnose specifically what's wrong, then you go back to
   Cursor with a targeted instruction.
3. **If only Layer 2.2 (backward compat) fails:** the bug is in the "off" path.
   Easy fix — find what changed and revert it.
4. **If only Layer 2.3 fails (adaptive doesn't run):** the bug is in the branch logic
   in `solve_batch`. Check the `if args.adaptive_penalty_mode == "within-batch":` block.
5. **If Layer 3.1 fails (adaptive makes things much worse):** the bug is likely in
   the multiplier update logic — possibly growing all multipliers instead of only
   violated ones, or applying the multiplier in the wrong direction.

---

## What "done" looks like

You should be able to say all of the following truthfully:

- [ ] I read the entire `git diff` and understood every line
- [ ] Backward compat smoke test (2.2) produces matching results to baseline
- [ ] Adaptive mode smoke test (2.3) shows multipliers growing across iterations
- [ ] JSON summary contains the new adaptive metadata fields
- [ ] I could explain to a teammate **why** rebuilding the QUBO every iteration is
      necessary (not just *that* we do it)
- [ ] I could explain **why** the growth factor is applied only to violated constraints
      (and what would happen if we grew all of them every time)

If you check all six, you're ready to do real experiments with the new solver.
