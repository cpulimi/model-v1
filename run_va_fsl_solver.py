#!/usr/bin/env python3
"""
Standalone FSL QUBO solver running on the NEC Vector Annealing (VA) engine.

This is a sibling of run_aligned_fsl_comparison.py's QUBO path. The QUBO
construction is byte-identical to the OpenJij path -- same encoding, same
batching, same variables -- so the sampler is the only thing that changes.

PENALTY SCALE: objective-scale normalization is OFF by default here, so C1-C4
penalties sit flat at --min-penalty (50,000) instead of being lifted to
~scale*multiplier (~2.51M on instances_low). That makes this run comparable to
the NO-SCALE OpenJij arm (run_aligned_fsl_comparison_noscale.py, e.g.
scripts/adaptive_scale_sweep_noscale.sh), and NOT to scale-ON baselines.
Pass --enable-objective-scale to restore the scale-ON penalties.

    python run_va_fsl_solver.py --dataset-dir instances_low --dry-run
    python run_va_fsl_solver.py --dataset-dir instances_low --run-root results/va_low

Everything about the model -- batching, variable naming, penalty weighting,
constraint encoding, sample evaluation, post-processing, output schemas -- is
imported from run_aligned_fsl_comparison (aliased `ref`) and is NOT
reimplemented here. This file only:

  1. checks every batch against VA's 100,000-bit / dense-matrix memory ceiling,
  2. calls VectorAnnealing instead of openjij,
  3. audits VA's single-precision energies against a float64 recompute.

openjij is never imported here, at any scope.

Outputs land in <run-root>/va, a sibling of the existing "qubo" and "gurobi"
directories, using the same filenames and column schemas run_qubo_solver
writes, so ref.build_combined_outputs and the existing comparison tooling can
read it unchanged.

DELIBERATE SCOPE NOTES
----------------------
* No seeding. The VA PoC API exposes no seed parameter, so VA runs are not
  reproducible read-for-read. Use --va-repeats N to characterize the spread
  instead; per-repeat statistics land in va_batch_summary.csv.
* Adaptive penalty IS implemented, and is ON by default. It mirrors
  ref.run_adaptive_penalty_loop -- rebuild the QUBO with grown multipliers for
  violated constraints, resample, repeat until feasible -- with the seed line
  dropped, since VA has no seed and the escalation never depended on one.
  It is on by default because with objective scale OFF, C3's penalty starts at
  50,000 against an S_lim of 500,000, so stocking at a closed hub is initially
  10x cheaper than opening one; only the escalation corrects that.
  --adaptive-penalty-iterations defaults to 8 rather than ref's 5, because
  ceil(log(500000/50000)/log(1.5)) = 6 iterations are needed to clear S_lim.
  Pass --adaptive-penalty-mode off for a single static pass.
* No retry-reads escalation (--max-stages / --retry-reads-boost). Each batch
  gets one VA sampling call per repeat, plus constraint-failure retries.

Environment (per NEC Vector Annealing PoC Manual, 2nd Edition, Nov 2022):
    Python 3.8, numpy >= 1.22.3, pyqubo >= 1.0.13, and
    export PYTHONPATH=/opt/va/VApoc_0201/libexec/VectorAnnealing/python:${PYTHONPATH}
That manual documents the 2022 PoC release; the version installed on this
cluster may differ. The VA module version is printed at startup when it
exposes one.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

# Import the reference implementation as a library. It imports openjij only
# inside build_sampler() and solve_qubo_batch(), neither of which this file
# calls, so importing it pulls in no sampler backend.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aligned_fsl_comparison as ref  # noqa: E402


# VA hard specification, from the PoC manual.
VA_HARD_MAX_VARS = 100_000          # "Binary data size: up to 100 thousand bit"
VA_DENSE_BYTES_PER_ENTRY = 4        # 32-bit resolution, dense full-connection matrix
VA_MANUAL_REF = "NEC Vector Annealing PoC Manual, 2nd Edition (Nov 2022)"


# ---------------------------------------------------------------------------
# VA environment
# ---------------------------------------------------------------------------


def import_vector_annealing() -> Any:
    """Import VectorAnnealing or exit with actionable guidance. No fallback."""
    try:
        import VectorAnnealing  # type: ignore
    except Exception as exc:  # ImportError, but VA can also fail on VE probing
        print("=" * 76, file=sys.stderr)
        print("ERROR: could not import VectorAnnealing.", file=sys.stderr)
        print(f"  underlying error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Source the VA environment and put the VA python dir on PYTHONPATH:", file=sys.stderr)
        print("    source /opt/nec/ve/nlc/<ver>/bin/nlcvars.sh", file=sys.stderr)
        print("    export PATH=${PATH}:/opt/nec/ve/bin", file=sys.stderr)
        print("    export PYTHONPATH=/opt/va/VApoc_0201/libexec/VectorAnnealing/python:${PYTHONPATH}", file=sys.stderr)
        print(f"  (paths per {VA_MANUAL_REF}; adjust to the installed version)", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"  active python:   {sys.executable}", file=sys.stderr)
        print(f"  python version:  {sys.version.splitlines()[0]}", file=sys.stderr)
        print(f"  PYTHONPATH:      {__import__('os').environ.get('PYTHONPATH', '<unset>')}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  This solver does not fall back to any other sampler.", file=sys.stderr)
        print("  Run with --dry-run to validate the batch plan without a VE card.", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        raise SystemExit(2)

    version = None
    for attr in ("__version__", "VERSION", "version"):
        value = getattr(VectorAnnealing, attr, None)
        if value is not None and not callable(value):
            version = str(value)
            break

    print("VectorAnnealing imported.", flush=True)
    print(f"  module file:     {getattr(VectorAnnealing, '__file__', '<unknown>')}", flush=True)
    print(f"  module version:  {version if version else '<module exposes no version attribute>'}", flush=True)
    print(f"  python:          {sys.version.splitlines()[0]}", flush=True)
    return VectorAnnealing


# ---------------------------------------------------------------------------
# Ceiling arithmetic and the batch plan
# ---------------------------------------------------------------------------


def dense_matrix_bytes(num_vars: int) -> int:
    """VA stores the problem as a dense matrix; QUBO sparsity does not help.

    100000^2 * 4 B = 40.0 GB, consistent with the manual's 48 GB requirement for
    full coupling at 100k bits; 70000^2 * 4 B = 19.6 GB matches its 24 GB tier.
    """
    return int(num_vars) * int(num_vars) * VA_DENSE_BYTES_PER_ENTRY


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:,.1f} {unit}"
        x /= 1024.0
    return f"{x:,.1f} TiB"


def build_batch_plan(
    data: dict[str, Any],
    batches: list[ref.BatchSpec],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build every batch's QUBO once to learn its TRUE variable count.

    ref.build_batches caps Z-vars only (--max-z-vars-per-batch); Y and X are
    whatever the batch's hub/part footprint implies. The VA ceiling applies to
    len(z_name) + len(y_name) + len(x_name), so it can only be checked after
    the QUBO is built. Each QUBO is discarded immediately; only counts are kept.
    """
    plan: list[dict[str, Any]] = []
    active = data["active"]

    for batch in batches:
        batch_df = active.iloc[batch.row_indices].reset_index(drop=True)
        t0 = time.perf_counter()
        qubo_meta = ref.build_qubo_for_batch(batch_df, data, args)
        build_seconds = time.perf_counter() - t0

        num_z = len(qubo_meta["z_name"])
        num_y = len(qubo_meta["y_name"])
        num_x = len(qubo_meta["x_name"])
        total_vars = num_z + num_y + num_x
        qubo_vars = len({name for key in qubo_meta["Q"] for name in key})

        plan.append(
            {
                "batch_id": int(batch.batch_id),
                "num_parts": int(batch_df["part_id"].nunique()),
                "num_rows": int(len(batch_df)),
                "num_z": int(num_z),
                "num_y": int(num_y),
                "num_x": int(num_x),
                "total_vars": int(total_vars),
                "qubo_vars_in_Q": int(qubo_vars),
                "interactions": int(len(qubo_meta["Q"])),
                "dense_matrix_bytes": int(dense_matrix_bytes(total_vars)),
                "penalty_c1": float(qubo_meta["penalties"]["c1"]),
                "penalty_c2": float(qubo_meta["penalties"]["c2"]),
                "penalty_c3": float(qubo_meta["penalties"]["c3"]),
                "c4_note": str(qubo_meta["c4_note"]),
                "build_seconds": float(build_seconds),
                "note": str(batch.note),
            }
        )

        del qubo_meta
        gc.collect()

    return plan


def print_batch_plan(plan: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ceiling = int(args.va_max_vars_per_batch)
    print("\n" + "=" * 108, flush=True)
    print("VA BATCH PLAN", flush=True)
    print("=" * 108, flush=True)
    print(f"  configured ceiling (--va-max-vars-per-batch): {ceiling:,} vars/batch", flush=True)
    print(f"  VA hard maximum (per manual):                 {VA_HARD_MAX_VARS:,} vars", flush=True)
    print(f"  dense matrix at the hard max:                 {human_bytes(dense_matrix_bytes(VA_HARD_MAX_VARS))}", flush=True)
    print(f"  dense matrix at the ceiling:                  {human_bytes(dense_matrix_bytes(ceiling))}", flush=True)
    print("  NOTE: VA stores the problem densely; QUBO sparsity does not reduce this.", flush=True)
    print("-" * 108, flush=True)
    header = (
        f"{'batch':>6}  {'parts':>6}  {'rows':>8}  {'Z':>9}  {'Y':>8}  {'X':>6}  "
        f"{'total_vars':>11}  {'dense_bytes':>16}  {'dense':>11}  {'status':>9}"
    )
    print(header, flush=True)
    print("-" * 108, flush=True)

    for row in plan:
        over = row["total_vars"] > ceiling
        status = "OVER" if over else ("OK" if row["total_vars"] <= VA_HARD_MAX_VARS else "OVER-HARD")
        print(
            f"{row['batch_id']:>6}  {row['num_parts']:>6,}  {row['num_rows']:>8,}  "
            f"{row['num_z']:>9,}  {row['num_y']:>8,}  {row['num_x']:>6,}  "
            f"{row['total_vars']:>11,}  {row['dense_matrix_bytes']:>16,}  "
            f"{human_bytes(row['dense_matrix_bytes']):>11}  {status:>9}",
            flush=True,
        )

    print("-" * 108, flush=True)
    if plan:
        worst = max(plan, key=lambda r: r["total_vars"])
        totals = sum(r["total_vars"] for r in plan)
        print(
            f"  batches: {len(plan):,} | largest batch: #{worst['batch_id']} at "
            f"{worst['total_vars']:,} vars ({human_bytes(worst['dense_matrix_bytes'])} dense) | "
            f"summed vars across batches: {totals:,}",
            flush=True,
        )
        mismatched = [r for r in plan if r["qubo_vars_in_Q"] != r["total_vars"]]
        if mismatched:
            print(
                f"  note: {len(mismatched)} batch(es) have variables that cancelled out of Q entirely; "
                "the ceiling check uses the larger len(z)+len(y)+len(x) count.",
                flush=True,
            )
    print("=" * 108, flush=True)


def check_ceiling(plan: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Abort loudly if any batch exceeds the ceiling. Never truncate or split."""
    ceiling = int(args.va_max_vars_per_batch)
    if ceiling > VA_HARD_MAX_VARS:
        raise SystemExit(
            f"--va-max-vars-per-batch={ceiling:,} exceeds VA's hard limit of "
            f"{VA_HARD_MAX_VARS:,} binary variables ({VA_MANUAL_REF})."
        )

    offenders = [r for r in plan if r["total_vars"] > ceiling]
    if not offenders:
        return

    current_max_z = int(args.max_z_vars_per_batch)
    suggestions = []
    lines = [
        "",
        "=" * 108,
        "ABORT: batch exceeds the VA variable ceiling",
        "=" * 108,
        f"  ceiling (--va-max-vars-per-batch): {ceiling:,} total variables per batch",
        f"  VA hard maximum:                   {VA_HARD_MAX_VARS:,} ({VA_MANUAL_REF})",
        "",
        "  Offending batches (total = len(z_name) + len(y_name) + len(x_name)):",
    ]
    for r in offenders:
        ratio = float(ceiling) / float(r["total_vars"])
        suggested = max(1, int(current_max_z * ratio))
        suggestions.append(suggested)
        lines.append(
            f"    batch {r['batch_id']:>4}: {r['total_vars']:,} vars "
            f"(Z={r['num_z']:,} Y={r['num_y']:,} X={r['num_x']:,}) "
            f"-> dense matrix {human_bytes(r['dense_matrix_bytes'])}, "
            f"over by {r['total_vars'] - ceiling:,}"
        )

    suggested_max_z = min(suggestions)
    lines.extend(
        [
            "",
            f"  Current --max-z-vars-per-batch is {current_max_z:,}, which caps Z only.",
            f"  Suggested retry:  --max-z-vars-per-batch {suggested_max_z:,}",
            "  (Y and X do not shrink proportionally with Z, so re-run --dry-run to confirm.)",
            "",
            "  Not truncating, splitting, or proceeding. This is a hard memory wall.",
            "=" * 108,
        ]
    )
    raise SystemExit("\n".join(lines))


# ---------------------------------------------------------------------------
# float64 energy recompute (precision instrumentation)
# ---------------------------------------------------------------------------


def recompute_energy_float64(Q: dict[tuple[str, str], float], spin: dict[str, Any], offset: float = 0.0) -> float:
    """Recompute a sample's QUBO energy on the host in double precision.

    VA computes in single precision (manual: "VA processes using single
    precision floating point arithmetic"). This is the fp64 reference value.
    math.fsum keeps the summation exact so the reported difference is
    attributable to VA, not to our accumulation order.
    """
    on = {name for name, value in spin.items() if ref.sample_is_one(value)}
    terms = [float(c) for (u, v), c in Q.items() if u in on and v in on]
    return math.fsum(terms) + float(offset)


def relative_difference(va_energy: float, recomputed: float) -> float:
    denom = abs(float(recomputed))
    if denom <= 1e-12:
        return float("nan")
    return abs(float(va_energy) - float(recomputed)) / denom


# ---------------------------------------------------------------------------
# VA sampling
# ---------------------------------------------------------------------------


def parse_beta_range(text: str) -> list[float] | None:
    """Parse "start,end[,nsteps]" into VA's beta_range. Empty -> VA default."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace("[", "").replace("]", "").split(",") if p.strip()]
    if len(parts) not in (2, 3):
        raise SystemExit(
            f"--va-beta-range must be 'start,end' or 'start,end,nsteps'; got {text!r}. "
            f"VA default is [10,100,200] ({VA_MANUAL_REF})."
        )
    values: list[float] = [float(parts[0]), float(parts[1])]
    if len(parts) == 3:
        values.append(int(float(parts[2])))
    return values


def build_one_hot_list(qubo_meta: dict[str, Any]) -> list[list[str]]:
    """C1 as a VA one-hot flip group: one hub per active (zip, part) demand row.

    Per the manual, flip-option constraints "must be included in Hamiltonian's
    formulation" -- so this is declared IN ADDITION to the C1 penalty terms,
    which are left untouched in the QUBO.
    """
    groups: list[list[str]] = []
    for entries in qubo_meta["demand_groups"].values():
        names = [zn for _, zn in entries]
        if names:
            groups.append(names)
    return groups


def va_sample_once(
    VectorAnnealing: Any,
    Q: dict[tuple[str, str], float],
    args: argparse.Namespace,
    num_reads: int,
    one_hot_list: list[list[str]] | None,
    beta_range: list[float] | None,
) -> tuple[list[Any], float]:
    """One VectorAnnealing.sample() call. Returns (results, wall seconds).

    offset is 0.0: ref.build_qubo_for_batch omits the C1 constant term, and the
    OpenJij path likewise samples with no offset. Keeping it at 0 makes VA
    energies directly comparable to the OpenJij baseline's.
    """
    model_kwargs: dict[str, Any] = {}
    if one_hot_list:
        model_kwargs["onehot"] = one_hot_list

    va_model = VectorAnnealing.model(Q, 0.0, **model_kwargs)
    sampler = VectorAnnealing.sampler()

    sample_kwargs: dict[str, Any] = {
        "num_reads": int(num_reads),
        # None/1 returns a single best solution; num_results == num_reads
        # returns every annealing result, which the precision audit needs.
        "num_results": int(num_reads),
        "vector_mode": str(args.va_vector_mode),
    }
    if int(args.num_sweeps or 0) > 0:
        sample_kwargs["num_sweeps"] = int(args.num_sweeps)
    if beta_range is not None:
        sample_kwargs["beta_range"] = beta_range

    t0 = time.perf_counter()
    result = sampler.sample(va_model, **sample_kwargs)
    elapsed = time.perf_counter() - t0

    results = list(result) if not isinstance(result, list) else result
    del va_model, sampler
    return results, elapsed


def va_sample_with_retries(
    VectorAnnealing: Any,
    Q: dict[tuple[str, str], float],
    args: argparse.Namespace,
    base_reads: int,
    one_hot_list: list[list[str]] | None,
    beta_range: list[float] | None,
    label: str,
) -> tuple[list[Any], float, int, int]:
    """Sample until VA returns something usable.

    Retries only when EVERY read of a call came back with a broken constraint --
    the "output result with broken constraint / INVALID" case the manual
    documents, whose advice is to re-run. Returns
    (results, seconds, attempts_used, retries). An empty result list means every
    attempt failed; the caller decides what to do about it.
    """
    seconds = 0.0
    retries = 0
    attempts_used = 0

    for attempt in range(1, int(args.va_max_retries) + 2):
        attempts_used = attempt
        results, elapsed = va_sample_once(VectorAnnealing, Q, args, base_reads, one_hot_list, beta_range)
        seconds += elapsed

        if not results:
            retries += 1
            print(f"    {label} attempt {attempt}: VA returned no results ({elapsed:.2f}s) - retrying", flush=True)
            continue

        flags = [getattr(r, "constraint", None) for r in results]
        all_broken = len(flags) > 0 and all(f is False for f in flags)

        if not all_broken:
            print(
                f"    {label} attempt {attempt}: {len(results)} reads in {elapsed:.2f}s "
                f"| constraint satisfied {sum(1 for f in flags if f is True)}/{len(flags)}"
                + (" (VA reported no constraint flag)" if all(f is None for f in flags) else ""),
                flush=True,
            )
            return results, seconds, attempts_used, retries

        retries += 1
        print(
            f"    {label} attempt {attempt}: VA returned INVALID (constraint broken on all "
            f"{len(flags)} reads, {elapsed:.2f}s) - retry {attempt}/{int(args.va_max_retries)}",
            flush=True,
        )

    return [], seconds, attempts_used, retries


def solve_va_batch(
    VectorAnnealing: Any,
    batch: ref.BatchSpec,
    data: dict[str, Any],
    args: argparse.Namespace,
    beta_range: list[float] | None,
) -> tuple[ref.BatchResult, list[dict[str, Any]], dict[str, Any]]:
    """Solve one batch on VA. Returns (BatchResult, precision rows, va stats).

    Mirrors ref.run_adaptive_penalty_loop when --adaptive-penalty-mode is
    within-batch: rebuild the QUBO with grown multipliers for whichever
    constraints are still violated, resample, repeat until feasible or the
    iteration budget runs out. The seed line is the only thing dropped -- VA has
    no seed parameter, and the escalation logic never depended on one.

    This matters most with objective scale OFF: C3's penalty starts at
    --min-penalty (50,000) while S_lim is 500,000, so stocking at a closed hub
    is initially 10x cheaper than opening one. Only the escalation fixes that.
    """
    batch_start = time.time()
    batch_df = data["active"].iloc[batch.row_indices].reset_index(drop=True)
    num_rows = len(batch_df)
    num_parts = int(batch_df["part_id"].nunique())

    print(
        f"\n=== VA Batch {batch.batch_id} | {num_parts} parts | {num_rows:,} demand rows | "
        f"estimated Z={batch.estimated_z_vars:,} ===",
        flush=True,
    )
    if batch.note:
        print(f"  note: {batch.note}", flush=True)

    adaptive = args.adaptive_penalty_mode == "within-batch"
    multipliers: dict[str, float] = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    max_iterations = int(args.adaptive_penalty_iterations) if adaptive else 1
    growth = float(args.adaptive_penalty_growth)
    stagnation_patience = int(args.adaptive_penalty_stagnation_patience)

    print("  [1/3] Building aligned manual QUBO dictionary (ref.build_qubo_for_batch)...", flush=True)
    t0 = time.time()
    qubo_meta = ref.build_qubo_for_batch(batch_df, data, args)
    build_seconds = time.time() - t0
    Q = qubo_meta["Q"]
    num_z = len(qubo_meta["z_name"])
    num_y = len(qubo_meta["y_name"])
    num_x = len(qubo_meta["x_name"])
    total_vars = num_z + num_y + num_x
    interactions = len(Q)
    print(
        f"    built in {build_seconds:.2f}s | Z={num_z:,} Y={num_y:,} X={num_x:,} "
        f"total_vars={total_vars:,} interactions={interactions:,}",
        flush=True,
    )
    print(
        "    penalties: "
        f"C1={qubo_meta['penalties']['c1']:,.2f} "
        f"C2={qubo_meta['penalties']['c2']:,.2f} "
        f"C3={qubo_meta['penalties']['c3']:,.2f}",
        flush=True,
    )

    # Defence in depth: the preflight already checked this, but a batch must
    # never reach VectorAnnealing.model() over the ceiling.
    if total_vars > int(args.va_max_vars_per_batch):
        raise SystemExit(
            f"Batch {batch.batch_id} has {total_vars:,} variables, over the "
            f"--va-max-vars-per-batch ceiling of {int(args.va_max_vars_per_batch):,}."
        )

    one_hot_list = build_one_hot_list(qubo_meta) if bool(args.va_onehot) else None
    if one_hot_list is not None:
        print(
            f"    flip option: onehot with {len(one_hot_list):,} groups "
            "(C1 penalty terms are RETAINED in the Hamiltonian, per the manual)",
            flush=True,
        )

    base_reads = ref.suggested_num_reads(num_z, args)
    print(
        f"  [2/3] Sampling with VectorAnnealing | reads={base_reads} "
        f"sweeps={args.num_sweeps} vector_mode={args.va_vector_mode} "
        f"beta_range={beta_range if beta_range is not None else 'VA default [10,100,200]'} | "
        f"repeats={args.va_repeats} | adaptive={'within-batch' if adaptive else 'off'}"
        + (f" (max {max_iterations} iters, growth {growth})" if adaptive else ""),
        flush=True,
    )

    best_eval: dict[str, Any] | None = None
    precision_rows: list[dict[str, Any]] = []
    repeat_records: list[dict[str, Any]] = []
    iteration_log: list[dict[str, Any]] = []
    sample_seconds = 0.0
    eval_seconds = 0.0
    read_index = 0
    total_sample_calls = 0
    total_retries = 0
    constraint_ok_count = 0
    constraint_total_count = 0
    iterations_used = 0
    was_feasible = False
    exit_reason = "max_iterations" if adaptive else "static_single_pass"
    best_total_violations: int | None = None
    no_improve_count = 0

    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration

        # Rebuild the QUBO at this iteration's multipliers. Variable counts are
        # invariant under multipliers (only coefficients change), so the preflight
        # ceiling check stays valid across iterations.
        if iteration > 1:
            t_rebuild = time.time()
            qubo_meta = ref.build_qubo_for_batch(batch_df, data, args, multipliers=multipliers)
            build_seconds += time.time() - t_rebuild
            Q = qubo_meta["Q"]

        _, diag = ref.penalty_weights(
            batch_df, data, args, multipliers=multipliers, return_diagnostics=True
        )
        if adaptive:
            print(
                f"    adaptive iter {iteration}/{max_iterations} | "
                f"multipliers c1={multipliers['c1']:.2f} c2={multipliers['c2']:.2f} "
                f"c3={multipliers['c3']:.2f} c4={multipliers['c4']:.2f} | "
                f"penalties C1={diag['c1']['final_value']:,.0f} C2={diag['c2']['final_value']:,.0f} "
                f"C3={diag['c3']['final_value']:,.0f}",
                flush=True,
            )

        iter_evals: list[dict[str, Any]] = []
        for repeat in range(1, int(args.va_repeats) + 1):
            label = f"iter {iteration} repeat {repeat}" if adaptive else f"repeat {repeat}"
            results, seconds, attempts_used, retries = va_sample_with_retries(
                VectorAnnealing, Q, args, base_reads, one_hot_list, beta_range, label
            )
            sample_seconds += seconds
            total_sample_calls += attempts_used
            total_retries += retries

            if not results:
                raise RuntimeError(
                    f"VA batch {batch.batch_id} ({label}): every read had a broken constraint "
                    f"after {attempts_used} attempts (--va-max-retries "
                    f"{int(args.va_max_retries)}). Failing this batch."
                )

            t_eval = time.time()
            repeat_evals: list[dict[str, Any]] = []
            for r in results:
                spin = dict(getattr(r, "spin", {}) or {})
                va_energy = float(getattr(r, "energy", 0.0))
                constraint_flag = getattr(r, "constraint", None)

                recomputed = recompute_energy_float64(Q, spin, offset=0.0)
                precision_rows.append(
                    {
                        "batch_id": int(batch.batch_id),
                        "read_index": int(read_index),
                        "num_vars": int(total_vars),
                        "va_reported_energy": float(va_energy),
                        "recomputed_energy": float(recomputed),
                        "abs_diff": float(abs(va_energy - recomputed)),
                        "rel_diff": float(relative_difference(va_energy, recomputed)),
                        "constraint_ok": constraint_flag if constraint_flag is None else bool(constraint_flag),
                    }
                )
                read_index += 1

                constraint_total_count += 1
                if constraint_flag is True:
                    constraint_ok_count += 1

                # Identical accounting to ref.solve_qubo_batch: VA's own reported
                # energy is fed to evaluate_sample, exactly as OpenJij's is.
                ev = ref.evaluate_sample(spin, va_energy, qubo_meta, data)
                repeat_evals.append(ev)
                iter_evals.append(ev)

                key = (ev["total_violations"], ev["c1"], ev["c2"], ev["c3"], ev["cost"], ev["energy"])
                if best_eval is None:
                    best_eval = ev
                else:
                    old_key = (
                        best_eval["total_violations"],
                        best_eval["c1"],
                        best_eval["c2"],
                        best_eval["c3"],
                        best_eval["cost"],
                        best_eval["energy"],
                    )
                    if key < old_key:
                        best_eval = ev
            eval_seconds += time.time() - t_eval

            energies = [float(e["energy"]) for e in repeat_evals]
            costs = [float(e["cost"]) for e in repeat_evals]
            feasible_reads = sum(1 for e in repeat_evals if int(e["total_violations"]) == 0)
            repeat_best = min(
                repeat_evals,
                key=lambda e: (e["total_violations"], e["c1"], e["c2"], e["c3"], e["cost"], e["energy"]),
            )
            repeat_records.append(
                {
                    "iteration": int(iteration),
                    "repeat": int(repeat),
                    "attempts_used": int(attempts_used),
                    "reads": int(len(repeat_evals)),
                    "energy_min": min(energies),
                    "energy_median": statistics.median(energies),
                    "energy_max": max(energies),
                    "cost_min": min(costs),
                    "cost_median": statistics.median(costs),
                    "cost_max": max(costs),
                    "constraint_satisfied_reads": int(
                        sum(1 for r in results if getattr(r, "constraint", None) is True)
                    ),
                    "structurally_feasible_reads": int(feasible_reads),
                    "best_total_violations": int(repeat_best["total_violations"]),
                    "best_cost": float(repeat_best["cost"]),
                }
            )
            print(
                f"    {label} | energy min/med/max = {min(energies):,.2f} / "
                f"{statistics.median(energies):,.2f} / {max(energies):,.2f} | cost min/med/max = "
                f"{min(costs):,.2f} / {statistics.median(costs):,.2f} / {max(costs):,.2f} | "
                f"structurally feasible reads {feasible_reads}/{len(repeat_evals)}",
                flush=True,
            )

        iter_best = min(
            iter_evals,
            key=lambda e: (e["total_violations"], e["c1"], e["c2"], e["c3"], e["cost"], e["energy"]),
        )

        # ref.adaptive_iteration_log_dataframe's schema. seed_used is None: VA has
        # no seed parameter, but the column is kept so the CSV matches the
        # OpenJij path's and both can be concatenated.
        log_row: dict[str, Any] = {
            "batch_id": int(batch.batch_id),
            "iteration": int(iteration),
            "objective_scale": float(diag["c1"]["objective_scale"]),
            "total_violations": int(iter_best["total_violations"]),
            "cost": float(iter_best["cost"]),
            "energy": float(iter_best["energy"]),
            "num_vars": int(total_vars),
            "num_interactions": int(len(Q)),
            "seed_used": None,
            "num_reads": int(base_reads),
            "exit_reason": "",
        }
        for c in ("c1", "c2", "c3", "c4"):
            log_row[f"min_pen_{c}"] = float(diag[c]["min_pen"])
            log_row[f"scaled_{c}"] = float(diag[c]["scaled_value"])
            log_row[f"chosen_{c}"] = float(diag[c]["chosen_value"])
            log_row[f"binding_branch_{c}"] = str(diag[c]["binding_branch"])
            log_row[f"mult_{c}"] = float(multipliers[c])
            log_row[f"viol_{c}"] = int(iter_best.get(c, 0))
        iteration_log.append(log_row)

        if int(iter_best["total_violations"]) == 0:
            was_feasible = True
            exit_reason = "feasible"
            iteration_log[-1]["exit_reason"] = exit_reason
            if adaptive:
                print(f"    adaptive feasible at iter {iteration}", flush=True)
            break

        if not adaptive:
            break

        # Stagnation exit, after the feasibility break so feasibility always wins.
        if best_total_violations is None or int(iter_best["total_violations"]) < best_total_violations:
            best_total_violations = int(iter_best["total_violations"])
            no_improve_count = 0
        else:
            no_improve_count += 1
        if stagnation_patience > 0 and no_improve_count >= stagnation_patience:
            exit_reason = "stagnated"
            iteration_log[-1]["exit_reason"] = exit_reason
            print(
                f"    adaptive stagnated at iter {iteration} "
                f"(no strict improvement for {no_improve_count} iters)",
                flush=True,
            )
            break

        # Grow multipliers for violated constraints only.
        for c in ("c1", "c2", "c3", "c4"):
            if int(iter_best.get(c, 0)) > 0:
                multipliers[c] *= growth
        print(
            f"    adaptive iter {iteration} violations C1={iter_best['c1']} C2={iter_best['c2']} "
            f"C3={iter_best['c3']} C4={iter_best.get('c4', 0)} | cost={iter_best['cost']:,.2f} "
            f"-> growing penalties",
            flush=True,
        )

    if adaptive and exit_reason == "max_iterations" and iteration_log:
        iteration_log[-1]["exit_reason"] = exit_reason
        if not was_feasible:
            print(
                f"    adaptive exhausted {iterations_used} iterations without feasibility "
                f"(exit_reason=max_iterations). Consider raising "
                f"--adaptive-penalty-iterations or --adaptive-penalty-growth.",
                flush=True,
            )

    if best_eval is None:
        raise RuntimeError(f"VA batch {batch.batch_id}: no sample was selected")

    total_seconds = time.time() - batch_start
    status = "OK" if best_eval["total_violations"] == 0 else f"{best_eval['total_violations']} structural violations"
    print(
        f"  [3/3] Batch selected | {status} | aligned cost={ref.money(best_eval['cost'])} | "
        f"total_time={total_seconds:.2f}s",
        flush=True,
    )

    all_energies = [float(row["va_reported_energy"]) for row in precision_rows]
    all_rel = [row["rel_diff"] for row in precision_rows if math.isfinite(row["rel_diff"])]
    va_stats = {
        "batch_id": int(batch.batch_id),
        "total_vars": int(total_vars),
        "dense_matrix_bytes": int(dense_matrix_bytes(total_vars)),
        "va_repeats": int(args.va_repeats),
        "va_sample_calls": int(total_sample_calls),
        "va_constraint_retries": int(total_retries),
        "va_reads_per_call": int(base_reads),
        "va_total_reads": int(len(precision_rows)),
        "va_constraint_satisfied_reads": int(constraint_ok_count),
        "va_constraint_flagged_reads": int(constraint_total_count),
        "va_onehot_groups": int(len(one_hot_list)) if one_hot_list else 0,
        "va_vector_mode": str(args.va_vector_mode),
        "energy_min": min(all_energies) if all_energies else float("nan"),
        "energy_median": statistics.median(all_energies) if all_energies else float("nan"),
        "energy_max": max(all_energies) if all_energies else float("nan"),
        "cost_min": min(r["cost_min"] for r in repeat_records),
        "cost_median": statistics.median([r["cost_median"] for r in repeat_records]),
        "cost_max": max(r["cost_max"] for r in repeat_records),
        "structurally_feasible_reads": int(sum(r["structurally_feasible_reads"] for r in repeat_records)),
        "max_abs_rel_diff": max(all_rel) if all_rel else float("nan"),
        "median_abs_rel_diff": statistics.median(all_rel) if all_rel else float("nan"),
        "adaptive_iterations_used": int(iterations_used),
        "adaptive_was_feasible": bool(was_feasible),
        "adaptive_exit_reason": str(exit_reason),
        "final_mult_c1": float(multipliers["c1"]),
        "final_mult_c2": float(multipliers["c2"]),
        "final_mult_c3": float(multipliers["c3"]),
        "final_penalty_c1": float(qubo_meta["penalties"]["c1"]),
        "final_penalty_c3": float(qubo_meta["penalties"]["c3"]),
        "repeat_records": repeat_records,
    }

    result = ref.BatchResult(
        batch_id=batch.batch_id,
        num_rows=num_rows,
        num_parts=num_parts,
        num_z=num_z,
        num_y=num_y,
        num_x=num_x,
        interactions=interactions,
        build_seconds=build_seconds,
        sample_seconds=sample_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total_seconds,
        # stage_used is an OpenJij retry-stage concept with no VA analogue.
        # Held at 0; VA call counts live in va_batch_summary.csv.
        stage_used=0,
        suggested_reads=base_reads,
        energy=float(best_eval["energy"]),
        reconstructed_cost=float(best_eval["cost"]),
        c1_violations=int(best_eval["c1"]),
        c2_violations=int(best_eval["c2"]),
        c3_violations=int(best_eval["c3"]),
        c4_violations=int(best_eval["c4"]),
        penalty_c1=float(qubo_meta["penalties"]["c1"]),
        penalty_c2=float(qubo_meta["penalties"]["c2"]),
        penalty_c3=float(qubo_meta["penalties"]["c3"]),
        open_hubs=list(best_eval["open_hubs"]),
        stocked_pairs=list(best_eval["stocked_pairs"]),
        assignments=list(best_eval["assignments"]),
        adaptive_iterations_used=int(iterations_used),
        adaptive_was_feasible=bool(was_feasible),
        final_penalty_multipliers=dict(multipliers),
        adaptive_iteration_log=list(iteration_log),
        adaptive_exit_reason=str(exit_reason),
    )

    del Q, qubo_meta
    gc.collect()
    return result, precision_rows, va_stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def va_batch_summary_dataframe(
    results: list[ref.BatchResult], va_stats: list[dict[str, Any]]
) -> pd.DataFrame:
    """ref.batch_summary_dataframe's columns plus the VA-specific ones."""
    base = ref.batch_summary_dataframe(results)
    extra = pd.DataFrame(
        [{k: v for k, v in s.items() if k != "repeat_records"} for s in va_stats]
    )
    if base.empty or extra.empty:
        return base
    return base.merge(extra, on="batch_id", how="left")


def va_repeat_dataframe(va_stats: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in va_stats:
        for rec in s["repeat_records"]:
            row = {"batch_id": int(s["batch_id"]), "total_vars": int(s["total_vars"])}
            row.update(rec)
            rows.append(row)
    return pd.DataFrame(rows)


def print_precision_report(precision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The experimental headline: can fp32 VA resolve costs at this scale?"""
    rel = [r["rel_diff"] for r in precision_rows if math.isfinite(r["rel_diff"])]
    abs_diffs = [r["abs_diff"] for r in precision_rows]

    stats: dict[str, Any] = {
        "reads_audited": int(len(precision_rows)),
        "reads_with_finite_rel_diff": int(len(rel)),
        "max_abs_rel_diff": float(max(rel)) if rel else None,
        "median_abs_rel_diff": float(statistics.median(rel)) if rel else None,
        "max_abs_diff": float(max(abs_diffs)) if abs_diffs else None,
        "median_abs_diff": float(statistics.median(abs_diffs)) if abs_diffs else None,
    }

    bar = "#" * 92
    print("\n" + bar, flush=True)
    print("VA FLOAT32 PRECISION AUDIT", flush=True)
    print(bar, flush=True)
    print("  VA computes in single precision; recomputed_energy is the float64 host reference.", flush=True)
    print(f"  reads audited:            {stats['reads_audited']:,}", flush=True)
    if rel:
        print(f"  MAX  |rel_diff|:          {stats['max_abs_rel_diff']:.6e}", flush=True)
        print(f"  MEDIAN |rel_diff|:        {stats['median_abs_rel_diff']:.6e}", flush=True)
        print(f"  max  |abs_diff|:          {stats['max_abs_diff']:,.4f}", flush=True)
        print(f"  median |abs_diff|:        {stats['median_abs_diff']:,.4f}", flush=True)
        print("", flush=True)
        print("  Read this against your cost scale: an |abs_diff| larger than the cost", flush=True)
        print("  differences you care about means VA cannot rank those solutions.", flush=True)
    else:
        print("  No finite relative differences (all recomputed energies were ~0).", flush=True)
    print(f"  detail CSV:               va_precision_audit.csv", flush=True)
    print(bar, flush=True)
    return stats


def print_va_header(
    data: dict[str, Any], batches: list[ref.BatchSpec], args: argparse.Namespace, run_dir: Path
) -> None:
    avg_hubs = sum(len(v) for v in data["zip_to_hubs"].values()) / max(1, len(data["zip_to_hubs"]))
    print("\n" + "=" * 76, flush=True)
    print("VA QUBO MODEL - ALIGNED COST BASIS (identical QUBO to the OpenJij path)", flush=True)
    print("=" * 76, flush=True)
    print(f"  dataset:                  {data['dataset_name']}", flush=True)
    print(f"  active demand pairs:      {len(data['active']):,}", flush=True)
    print(f"  parts loaded:             {len(data['K']):,}", flush=True)
    print(f"  hubs loaded:              {len(data['J']):,}", flush=True)
    print(
        f"  candidate hubs/zip:       "
        f"{'all' if data['top_hubs_per_zip'] is None else data['top_hubs_per_zip']} (avg {avg_hubs:.2f})",
        flush=True,
    )
    print(f"  total QUBO batches:       {len(batches):,}", flush=True)
    print(f"  max Z vars/batch:         {args.max_z_vars_per_batch:,}", flush=True)
    print(f"  VA max total vars/batch:  {args.va_max_vars_per_batch:,}", flush=True)
    print(f"  sampler:                  NEC Vector Annealing", flush=True)
    print(f"    vector_mode:            {args.va_vector_mode}", flush=True)
    print(f"    num_reads (base):       {args.num_reads}", flush=True)
    print(f"    num_sweeps:             {args.num_sweeps}", flush=True)
    print(f"    beta_range:             {args.va_beta_range or 'VA default [10,100,200]'}", flush=True)
    print(f"    repeats:                {args.va_repeats}  (no seed parameter exists in VA)", flush=True)
    print(f"    max constraint retries: {args.va_max_retries}", flush=True)
    print(f"    onehot flip option:     {'ON' if args.va_onehot else 'OFF'}", flush=True)
    print(f"  penalty mode:             {args.penalty_mode}", flush=True)
    print(
        f"  objective scale:          "
        f"{'ON  (penalties ~ scale x multiplier)' if args.enable_objective_scale else 'OFF (penalties flat at --min-penalty)'}",
        flush=True,
    )
    print(f"  min penalty (floor):      {float(args.min_penalty):,.2f}", flush=True)
    print(f"  adaptive penalty:         {args.adaptive_penalty_mode}", flush=True)
    if args.adaptive_penalty_mode == "within-batch":
        print(f"    max iterations:         {args.adaptive_penalty_iterations}", flush=True)
        print(f"    growth factor:          {args.adaptive_penalty_growth}", flush=True)
        s_lim = float(data["scalar"]["S_lim"])
        base = float(args.min_penalty)
        if not args.enable_objective_scale and base < s_lim:
            need = math.ceil(math.log(s_lim / base) / math.log(float(args.adaptive_penalty_growth)))
            print(
                f"    NOTE: C3 starts at {base:,.0f} vs S_lim {s_lim:,.0f}; stocking at a closed "
                f"hub is initially cheaper than opening one.",
                flush=True,
            )
            print(
                f"          escalation needs ~{need} iterations to pass S_lim "
                f"(budget is {args.adaptive_penalty_iterations}).",
                flush=True,
            )
    print(f"  constraint multiplier:    {float(args.constraint_multiplier):,.2f}"
          f"{'' if args.enable_objective_scale else '  (inactive: scale is OFF)'}", flush=True)
    print(f"  base miles:               {data['scalar']['base_miles']}", flush=True)
    print(f"  penalty start miles:      {data['scalar']['penalty_start_miles']}", flush=True)
    print(f"  max service miles:        {data['scalar']['max_service_miles']}", flush=True)
    print(f"  output folder:            {run_dir}", flush=True)
    print("=" * 76, flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_va_solver(args: argparse.Namespace) -> dict[str, Any] | None:
    tracemalloc.start()
    start = time.perf_counter()

    data = ref.load_problem_data(
        args.dataset_dir,
        max_service_miles_override=args.max_service_miles,
        penalty_start_miles_override=args.penalty_start_miles,
        top_hubs_per_zip=None if int(args.top_hubs_per_zip) < 0 else int(args.top_hubs_per_zip),
        max_parts_total=None if int(args.max_parts_total) < 0 else int(args.max_parts_total),
    )

    run_dir = Path(args.run_root).expanduser().resolve() / "va"
    batches = ref.build_batches(
        data["active"],
        data["part_order"],
        data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )

    if not bool(args.dry_run):
        run_dir.mkdir(parents=True, exist_ok=True)
    print_va_header(data, batches, args, run_dir)

    # Preflight: build every QUBO, learn the TRUE variable counts, check the
    # ceiling. Nothing is sampled and VectorAnnealing is not imported yet.
    print("\nPreflight: building every batch QUBO to measure true variable counts...", flush=True)
    plan = build_batch_plan(data, batches, args)
    print_batch_plan(plan, args)
    check_ceiling(plan, args)

    if bool(args.dry_run):
        print(
            f"\n--dry-run: all {len(plan)} batch(es) fit under the "
            f"{int(args.va_max_vars_per_batch):,}-variable ceiling. "
            "VectorAnnealing was not imported. Exiting.",
            flush=True,
        )
        return None

    pd.DataFrame(plan).to_csv(run_dir / "va_batch_plan.csv", index=False)

    beta_range = parse_beta_range(args.va_beta_range)
    VectorAnnealing = import_vector_annealing()

    deadline = None
    if args.qubo_time_limit is not None and float(args.qubo_time_limit) > 0:
        deadline = time.perf_counter() + float(args.qubo_time_limit)

    results: list[ref.BatchResult] = []
    all_precision_rows: list[dict[str, Any]] = []
    all_va_stats: list[dict[str, Any]] = []
    stopped_time_limit = False

    for batch in batches:
        if deadline is not None and time.perf_counter() >= deadline:
            print(
                f"\nVA wall-time limit reached before next batch. "
                f"Completed {len(results)}/{len(batches)} batches.",
                flush=True,
            )
            stopped_time_limit = True
            break

        result, precision_rows, va_stats = solve_va_batch(
            VectorAnnealing, batch, data, args, beta_range
        )
        results.append(result)
        all_precision_rows.extend(precision_rows)
        all_va_stats.append(va_stats)

        # Same checkpoint filenames run_qubo_solver writes.
        checkpoint_raw = ref.aggregate_raw_results(results)
        pd.DataFrame(checkpoint_raw["assignments"], columns=["zip_id", "hub_id", "part_id"]).to_csv(
            run_dir / "checkpoint_raw_assignments.csv", index=False
        )
        pd.DataFrame({"hub_id": checkpoint_raw["open_hubs"]}).to_csv(
            run_dir / "checkpoint_raw_open_hubs.csv", index=False
        )
        ref.batch_summary_dataframe(results).to_csv(run_dir / "batch_summary_checkpoint.csv", index=False)
        pd.DataFrame(all_precision_rows).to_csv(run_dir / "va_precision_audit.csv", index=False)

    if not results:
        raise RuntimeError("No VA batches completed; increase --qubo-time-limit or check the instance")

    raw = ref.aggregate_raw_results(results)
    final = ref.postprocess_qubo_solution(
        raw["assignments"],
        raw["stocked_pairs"],
        raw["open_hubs"],
        data,
        repair_assignments=not bool(args.no_repair_assignments),
        trim_unused=not bool(args.no_trim_unused),
        hub_prune=not bool(args.no_hub_prune),
        hub_prune_max_iterations=int(args.hub_prune_max_iterations),
    )

    # Shared schema for the comparison tooling, plus the VA-augmented variant.
    ref.batch_summary_dataframe(results).to_csv(run_dir / "batch_summary.csv", index=False)
    va_batch_summary_dataframe(results, all_va_stats).to_csv(run_dir / "va_batch_summary.csv", index=False)
    va_repeat_dataframe(all_va_stats).to_csv(run_dir / "va_repeat_summary.csv", index=False)

    # Same filenames and schemas run_qubo_solver emits in within-batch mode, so the
    # VA and OpenJij adaptive logs can be concatenated and compared directly.
    if args.adaptive_penalty_mode == "within-batch":
        ref.batch_adaptive_summary_dataframe(results).to_csv(
            run_dir / "batch_adaptive_summary.csv", index=False
        )
        ref.adaptive_iteration_log_dataframe(results).to_csv(
            run_dir / "adaptive_iteration_log.csv", index=False
        )
    pd.DataFrame(all_precision_rows).to_csv(run_dir / "va_precision_audit.csv", index=False)

    ref.assignment_rows_dataframe(raw["assignments"], data).to_csv(
        run_dir / "raw_qubo_hub_zip_part_pairings.csv", index=False
    )
    pd.DataFrame({"hub_id": raw["open_hubs"]}).to_csv(run_dir / "raw_qubo_open_hubs.csv", index=False)
    pd.DataFrame(raw["stocked_pairs"], columns=["hub_id", "part_id"]).to_csv(
        run_dir / "raw_qubo_stocked_pairs.csv", index=False
    )

    wall = time.perf_counter() - start
    peak_mb, current_mb = ref.memory_report_mb()
    runtime = {
        "wall_seconds": float(wall),
        "qubo_build_seconds": float(sum(r.build_seconds for r in results)),
        "qubo_sample_seconds": float(sum(r.sample_seconds for r in results)),
        "sample_eval_seconds": float(sum(r.eval_seconds for r in results)),
        "batch_total_seconds": float(sum(r.total_seconds for r in results)),
        "peak_memory_mb": float(peak_mb),
        "current_memory_mb": float(current_mb),
    }

    precision_summary = print_precision_report(all_precision_rows)

    raw_cost = ref.compute_solution_cost(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    raw_audit = ref.global_audit(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    extra = {
        "completed_batches": int(len(results)),
        "total_batches": int(len(batches)),
        "full_batch_coverage": bool(len(results) == len(batches) and not stopped_time_limit),
        "stopped_due_to_time_limit": bool(stopped_time_limit),
        "raw_solution": {
            "open_hubs_count": int(len(raw["open_hubs"])),
            "stocked_pairs_count": int(len(raw["stocked_pairs"])),
            "assignments_count": int(len(raw["assignments"])),
            "cost": raw_cost,
            "audit": raw_audit,
        },
        "postprocess": {
            "repair_assignments": bool(not args.no_repair_assignments),
            "trim_unused_open_stock": bool(not args.no_trim_unused),
            "missing_unrepaired": int(final.get("missing_unrepaired", 0)),
            "hub_prune_enabled": bool(not args.no_hub_prune),
            "hub_prune_closures": int(final.get("hub_prune_stats", {}).get("closures", 0)),
            "hub_prune_relocations": int(final.get("hub_prune_stats", {}).get("relocations", 0)),
        },
        "va": {
            "engine": "NEC Vector Annealing",
            "manual_reference": VA_MANUAL_REF,
            "module_version": next(
                (
                    str(getattr(VectorAnnealing, a))
                    for a in ("__version__", "VERSION", "version")
                    if getattr(VectorAnnealing, a, None) is not None
                    and not callable(getattr(VectorAnnealing, a))
                ),
                None,
            ),
            "python_version": sys.version.splitlines()[0],
            "objective_scale_enabled": bool(args.enable_objective_scale),
            "min_penalty": float(args.min_penalty),
            "constraint_multiplier": float(args.constraint_multiplier),
            "penalty_note": (
                "objective scale ON: penalties = max(min_penalty, scale*multiplier)"
                if args.enable_objective_scale
                else "objective scale OFF (VA default): penalties flat at min_penalty; "
                     "comparable to the run_aligned_fsl_comparison_noscale.py arm, NOT to "
                     "scale-ON OpenJij baselines"
            ),
            "seeded": False,
            "seed_note": "The VA PoC API exposes no seed parameter; runs are not reproducible read-for-read.",
            "adaptive_penalty": "not implemented on the VA path (static penalties, equivalent to ref --adaptive-penalty-mode off)",
            "num_reads_base": int(args.num_reads),
            "num_sweeps": int(args.num_sweeps),
            "vector_mode": str(args.va_vector_mode),
            "beta_range": beta_range if beta_range is not None else "VA default [10,100,200]",
            "repeats": int(args.va_repeats),
            "max_retries": int(args.va_max_retries),
            "onehot_flip_option": bool(args.va_onehot),
            "max_vars_per_batch_ceiling": int(args.va_max_vars_per_batch),
            "hard_max_vars": int(VA_HARD_MAX_VARS),
            "max_batch_vars_observed": int(max(p["total_vars"] for p in plan)),
            "max_batch_dense_bytes": int(max(p["dense_matrix_bytes"] for p in plan)),
            "total_sample_calls": int(sum(s["va_sample_calls"] for s in all_va_stats)),
            "total_constraint_retries": int(sum(s["va_constraint_retries"] for s in all_va_stats)),
        },
        "precision_audit": precision_summary,
    }

    summary = ref.write_solution_outputs(
        run_dir,
        solver_name="va",
        data=data,
        assignments=final["assignments"],
        stocked_pairs=final["stocked_pairs"],
        open_hubs=final["open_hubs"],
        runtime=runtime,
        extra=extra,
        assignment_sources=final.get("assignment_sources", {}),
    )
    print(ref.final_results_block(summary, "VA FINAL RESULTS"), flush=True)

    # Repeat the precision headline last so it is the final thing on screen.
    if precision_summary.get("max_abs_rel_diff") is not None:
        print(
            f"\n>>> VA fp32 precision: max |rel_diff| = {precision_summary['max_abs_rel_diff']:.6e}, "
            f"median |rel_diff| = {precision_summary['median_abs_rel_diff']:.6e} "
            f"over {precision_summary['reads_audited']:,} reads. See va_precision_audit.csv.",
            flush=True,
        )

    gc.collect()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the FSL QUBO pipeline on the NEC Vector Annealing engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Mirrored from run_aligned_fsl_comparison.parse_args ---------------
    p.add_argument("--dataset-dir", default="instances_low",
                   help="Folder with demand/distances/hubs/parameters/parts/zips CSV files.")
    p.add_argument("--run-root", default="",
                   help="Run folder. Results go to <run-root>/va. Empty picks outputs/va_fsl_<timestamp>.")

    p.add_argument("--max-service-miles", "--max-distance", dest="max_service_miles", type=float, default=None,
                   help="Eligibility cutoff. Default uses parameters.csv max_service_miles.")
    p.add_argument("--penalty-start-miles", type=float, default=None,
                   help="Override parameters.csv penalty_start_miles.")
    p.add_argument("--top-hubs-per-zip", type=int, default=-1,
                   help="Nearest eligible hubs per ZIP. -1 keeps all within max service miles.")
    p.add_argument("--max-parts-total", type=int, default=-1, help="Limit parts for testing. -1 keeps all.")

    p.add_argument("--part-batch-size", type=int, default=1000, help="Soft part count per batch.")
    p.add_argument("--max-z-vars-per-batch", type=int, default=50000,
                   help="Hard Z-variable cap used by ref.build_batches. Caps Z only, not Y or X.")
    p.add_argument("--num-reads", type=int, default=100,
                   help="Base reads; scaled per batch by ref.suggested_num_reads, then passed as VA num_reads.")
    p.add_argument("--num-sweeps", type=int, default=3000, help="VA num_sweeps (VA default is 500).")
    p.add_argument("--penalty-mode", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--min-penalty", type=float, default=50000.0,
                   help="Penalty floor, and -- with objective scale OFF, the default on this path -- "
                        "the flat effective penalty for C1-C4.")
    p.add_argument("--constraint-multiplier", type=float, default=5.0,
                   help="Only takes effect with --enable-objective-scale; otherwise ignored.")
    p.add_argument("--enable-objective-scale", action="store_true",
                   help="Re-enable ref's objective-scale normalization, which lifts penalties to "
                        "~scale*multiplier (~2.51M on instances_low). OFF by default on the VA path, "
                        "so penalties sit flat at --min-penalty (50,000). Matches the arm that "
                        "run_aligned_fsl_comparison_noscale.py produces.")
    for c in ("c1", "c2", "c3", "c4"):
        p.add_argument(f"--min-penalty-{c}", type=float, default=-1.0)
        p.add_argument(f"--constraint-multiplier-{c}", type=float, default=-1.0)
    p.add_argument("--c4-mode", choices=["off", "auto", "on"], default="auto")
    p.add_argument("--x-empty-penalty-factor", type=float, default=0.0,
                   help="Mirrored QUBO-formulation flag. Default 0 (disabled).")
    p.add_argument("--y-overflow-penalty-factor", type=float, default=0.0,
                   help="Mirrored QUBO-formulation flag. Default 0 (disabled).")
    p.add_argument("--qubo-time-limit", type=float, default=5400.0,
                   help="Wall time budget in seconds, checked between batches. 0 disables.")
    p.add_argument("--no-repair-assignments", action="store_true",
                   help="Do not repair missing/multiple assignments in the final output.")
    p.add_argument("--no-trim-unused", action="store_true",
                   help="Do not trim unused open hubs / stocked pairs after final assignments.")
    p.add_argument("--adaptive-penalty-mode", choices=["off", "within-batch"], default="within-batch",
                   help="Adaptive penalty strategy. 'within-batch' iteratively grows the penalties of "
                        "violated constraints and resamples. ON by default here because with objective "
                        "scale OFF, C3 starts at 50,000 against an S_lim of 500,000 and only the "
                        "escalation fixes that. 'off' gives a single static pass.")
    p.add_argument("--adaptive-penalty-iterations", type=int, default=8,
                   help="Max adaptive iterations per batch. Default 8, not ref's 5: escalating C3 past "
                        "S_lim needs ceil(log(500000/50000)/log(1.5)) = 6 iterations at growth 1.5.")
    p.add_argument("--adaptive-penalty-growth", type=float, default=1.5,
                   help="Multiplicative growth for violated constraint penalties.")
    p.add_argument("--adaptive-penalty-stagnation-patience", type=int, default=0,
                   help="Break if total_violations has not strictly improved for N iterations. 0 disables.")
    p.add_argument("--no-hub-prune", action="store_true", help="Disable the global hub-pruning post-pass.")
    p.add_argument("--hub-prune-max-iterations", type=int, default=500,
                   help="Max passes over open hubs in the hub-prune post-pass.")

    # --- VA-specific -------------------------------------------------------
    va = p.add_argument_group("Vector Annealing")
    va.add_argument("--va-max-vars-per-batch", type=int, default=60000,
                    help=f"Ceiling on TOTAL vars per batch (Z+Y+X). VA hard max is {VA_HARD_MAX_VARS:,}.")
    va.add_argument("--va-repeats", type=int, default=1,
                    help="Re-run each batch N times and record the distribution. VA has no seed parameter.")
    va.add_argument("--va-max-retries", type=int, default=3,
                    help="Retries when VA returns a broken constraint on every read of a call.")
    va.add_argument("--va-onehot", action="store_true",
                    help="Declare C1 as VA one-hot flip groups. C1 penalty terms stay in the Hamiltonian.")
    va.add_argument("--va-vector-mode", choices=["SPEED", "ACCURACY"], default="ACCURACY",
                    help="VA vector_mode.")
    va.add_argument("--va-beta-range", default="",
                    help="VA beta_range as 'start,end[,nsteps]'. Empty uses the VA default [10,100,200].")
    va.add_argument("--dry-run", action="store_true",
                    help="Load data, build batches and QUBOs, print the plan, check the ceiling, exit. "
                         "Does not import VectorAnnealing; runs on any machine.")

    args = p.parse_args(argv)

    # ref.penalty_weights reads `disable_objective_scale` via getattr. The VA path
    # defaults to objective scale OFF, so penalties equal --min-penalty flat rather
    # than being lifted into the millions. See the module docstring for why.
    args.disable_objective_scale = not bool(args.enable_objective_scale)

    if int(args.va_repeats) < 1:
        p.error("--va-repeats must be >= 1")
    if int(args.va_max_retries) < 0:
        p.error("--va-max-retries must be >= 0")
    if not args.run_root:
        args.run_root = str(Path("outputs") / f"va_fsl_{ref.now_stamp()}")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_va_solver(args)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        peak_mb, current_mb = ref.memory_report_mb()
        print("ERROR: VA run failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"peak/current memory before exit: {peak_mb:,.1f}/{current_mb:,.1f} MB", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
