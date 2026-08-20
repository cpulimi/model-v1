#!/usr/bin/env python3
"""
Scaling probe for the generated FSL instance suite.

Answers two questions, without needing a VE card or any solver backend:

  1. Does the solver pipeline run clean on each instance? (load -> batch ->
     formulate the pyqubo model -> compile -> emit QUBO -> ceiling check)
  2. How do problem size, build cost and memory trend as hubs increase?

It reuses run_va_fsl_solver's real code path -- the same load_problem_data,
build_batches and build_batch_plan the VA solver runs in preflight -- so a pass
here means the solver would get as far as sampling on that instance.

    python scaling_probe.py                        # probe the fsl_* suite
    python scaling_probe.py --instances outputs/fsl_10_hubs outputs/fsl_50_hubs
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

import run_va_fsl_solver as solver


def probe_instance(dataset_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run the solver's preflight on one instance and time/measure it."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    data = solver.load_problem_data(
        dataset_dir,
        max_service_miles_override=None,
        penalty_start_miles_override=None,
        top_hubs_per_zip=None,
        max_parts_total=None,
    )
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    batches = solver.build_batches(
        data["active"],
        data["part_order"],
        data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )
    t_batch = time.perf_counter() - t1

    t2 = time.perf_counter()
    plan = solver.build_batch_plan(data, batches, args)
    t_plan = time.perf_counter() - t2

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_vars = max(p["total_vars"] for p in plan)
    over = [p for p in plan if p["total_vars"] > int(args.va_max_vars_per_batch)]
    avg_hubs = sum(len(v) for v in data["zip_to_hubs"].values()) / max(1, len(data["zip_to_hubs"]))

    return {
        "instance": dataset_dir.name,
        "hubs": len(data["J"]),
        "zips": len(data["zips"]),
        "parts": len(data["K"]),
        "active_rows": len(data["active"]),
        "avg_eligible_hubs_per_zip": avg_hubs,
        "batches": len(plan),
        "max_batch_vars": total_vars,
        "sum_z": sum(p["num_z"] for p in plan),
        "sum_y": sum(p["num_y"] for p in plan),
        "sum_x": sum(p["num_x"] for p in plan),
        "sum_vars": sum(p["total_vars"] for p in plan),
        "interactions": sum(p["interactions"] for p in plan),
        "max_dense_bytes": max(p["dense_matrix_bytes"] for p in plan),
        "express_s": sum(p["pyqubo_express_seconds"] for p in plan),
        "compile_s": sum(p["pyqubo_compile_seconds"] for p in plan),
        "load_s": t_load,
        "batch_s": t_batch,
        "plan_s": t_plan,
        "total_s": time.perf_counter() - t0,
        "peak_mb": peak / (1024.0 * 1024.0),
        "over_ceiling": len(over),
        "status": "OVER-CEILING" if over else "OK",
    }


def build_probe_args(cli: argparse.Namespace) -> argparse.Namespace:
    """The subset of solver flags build_batch_plan reads, at their defaults."""
    return argparse.Namespace(
        part_batch_size=cli.part_batch_size,
        max_z_vars_per_batch=cli.max_z_vars_per_batch,
        va_max_vars_per_batch=cli.va_max_vars_per_batch,
        va_include_offset=False,
        penalty_mode="adaptive",
        min_penalty=50000.0,
        constraint_multiplier=5.0,
        disable_objective_scale=True,
        c4_mode="auto",
        x_empty_penalty_factor=0.0,
        y_overflow_penalty_factor=0.0,
        **{f"min_penalty_{c}": -1.0 for c in ("c1", "c2", "c3", "c4")},
        **{f"constraint_multiplier_{c}": -1.0 for c in ("c1", "c2", "c3", "c4")},
    )


def print_report(rows: list[dict[str, Any]]) -> bool:
    print("\n" + "=" * 100)
    print("PIPELINE SCALING PROBE  (run_va_fsl_solver preflight: load -> batch -> pyqubo -> QUBO)")
    print("=" * 100)

    h = (f"  {'instance':<14} {'hubs':>5} {'zips':>6} {'rows':>7} {'batch':>6} "
         f"{'Z':>8} {'Y':>7} {'X':>6} {'vars':>8} {'interactions':>13} {'status':>13}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for r in rows:
        print(f"  {r['instance']:<14} {r['hubs']:>5,} {r['zips']:>6,} {r['active_rows']:>7,} "
              f"{r['batches']:>6,} {r['sum_z']:>8,} {r['sum_y']:>7,} {r['sum_x']:>6,} "
              f"{r['sum_vars']:>8,} {r['interactions']:>13,} {r['status']:>13}")

    print("\n  build cost and memory")
    h2 = (f"  {'instance':<14} {'load_s':>8} {'batch_s':>8} {'express_s':>10} {'compile_s':>10} "
          f"{'total_s':>9} {'peak_MB':>9} {'max_dense':>11}")
    print(h2)
    print("  " + "-" * (len(h2) - 2))
    for r in rows:
        print(f"  {r['instance']:<14} {r['load_s']:>8.2f} {r['batch_s']:>8.2f} {r['express_s']:>10.2f} "
              f"{r['compile_s']:>10.2f} {r['total_s']:>9.2f} {r['peak_mb']:>9.1f} "
              f"{solver.human_bytes(r['max_dense_bytes']):>11}")

    # Growth factors relative to the smallest instance: the actual scaling trend.
    print("\n  scaling trend (x relative to the first instance)")
    base = rows[0]
    h3 = f"  {'instance':<14} {'hubs':>7} {'vars':>7} {'interactions':>13} {'build_time':>11} {'peak_mem':>9}"
    print(h3)
    print("  " + "-" * (len(h3) - 2))
    for r in rows:
        def ratio(key: str) -> str:
            b = float(base[key]) or 1.0
            return f"{float(r[key]) / b:.2f}x"
        print(f"  {r['instance']:<14} {ratio('hubs'):>7} {ratio('sum_vars'):>7} "
              f"{ratio('interactions'):>13} {ratio('total_s'):>11} {ratio('peak_mb'):>9}")

    failed = [r for r in rows if r["status"] != "OK"]
    print("\n" + "=" * 100)
    if failed:
        print(f"RESULT: {len(failed)} instance(s) exceeded the VA ceiling: "
              f"{[r['instance'] for r in failed]}")
    else:
        print(f"RESULT: PASS - all {len(rows)} instance(s) load, batch, formulate and compile clean, "
              "and fit the VA ceiling.")
    print("=" * 100)
    return not failed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances", nargs="*", default=None,
                   help="Instance dirs to probe. Default: outputs/fsl_* sorted by hub count.")
    p.add_argument("--part-batch-size", type=int, default=1000)
    p.add_argument("--max-z-vars-per-batch", type=int, default=50000)
    p.add_argument("--va-max-vars-per-batch", type=int, default=60000)
    p.add_argument("--csv", default="", help="Optional path to write the probe table as CSV.")
    cli = p.parse_args(argv)

    if cli.instances:
        dirs = [Path(x) for x in cli.instances]
    else:
        dirs = sorted(Path("outputs").glob("fsl_*_hubs"),
                      key=lambda d: int(d.name.split("_")[1]))
    missing = [d for d in dirs if not d.is_dir()]
    if missing or not dirs:
        raise SystemExit(f"ERROR: no instance directories to probe (missing: {missing})")

    args = build_probe_args(cli)
    rows: list[dict[str, Any]] = []
    for d in dirs:
        print(f"\n>>> probing {d} ...", flush=True)
        try:
            rows.append(probe_instance(d, args))
            print(f"    OK: {rows[-1]['sum_vars']:,} vars in {rows[-1]['batches']} batch(es), "
                  f"{rows[-1]['total_s']:.2f}s", flush=True)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append({
                "instance": d.name, "hubs": 0, "zips": 0, "parts": 0, "active_rows": 0,
                "avg_eligible_hubs_per_zip": 0.0, "batches": 0, "max_batch_vars": 0,
                "sum_z": 0, "sum_y": 0, "sum_x": 0, "sum_vars": 0, "interactions": 0,
                "max_dense_bytes": 0, "express_s": 0.0, "compile_s": 0.0, "load_s": 0.0,
                "batch_s": 0.0, "plan_s": 0.0, "total_s": 0.0, "peak_mb": 0.0,
                "over_ceiling": 0, "status": f"FAILED:{type(exc).__name__}",
            })

    ok = print_report(rows)
    if cli.csv:
        pd.DataFrame(rows).to_csv(cli.csv, index=False)
        print(f"\nprobe table written to {cli.csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
