#!/usr/bin/env python3
"""
Computational performance harness for the NEC Vector Annealing FSL solver.

Produces one consolidated row per instance covering everything needed to
characterise VA on this workload:

  size       hubs, demand rows, batches, Z/Y/X, binary variables,
             QUBO interactions, matrix density, couplings per variable
  memory     VA dense-matrix requirement, sparse equivalent, waste factor,
             true process RSS peak
  time       QUBO construction (pyqubo expression + compile + to_qubo),
             annealing (the VE card's own sample() time), evaluation, total wall
  quality    raw and final cost, raw C1-C4 violations, feasible-read share

Two modes:

  --mode preflight   No VE card needed. Runs the solver's own preflight in
                     process: load -> batch -> formulate -> compile -> to_qubo
                     -> ceiling check. Gives every size, density, construction
                     time and build-memory metric. Annealing time and solution
                     quality are necessarily blank -- nothing is sampled.

  --mode full        Requires the VE card (ASU SOL: sfpga01n). Runs
                     run_va_fsl_solver.py per instance as a subprocess, polling
                     the process tree for true peak RSS, then reads
                     summary.json -> extra.performance for the rest.

    python va_benchmark.py --mode preflight
    python va_benchmark.py --mode full --num-reads 100 --num-sweeps 3000

The sampling budget is held identical across instances so the trend is
attributable to instance size and not to the budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

import run_va_fsl_solver as solver

try:
    import psutil  # type: ignore
except ModuleNotFoundError:
    psutil = None

BLANK = float("nan")


# ---------------------------------------------------------------------------
# preflight mode: no card required
# ---------------------------------------------------------------------------


def preflight_args(cli: argparse.Namespace) -> argparse.Namespace:
    """The solver flags build_batch_plan reads, at the solver's own defaults."""
    return argparse.Namespace(
        part_batch_size=cli.part_batch_size,
        max_z_vars_per_batch=cli.max_z_vars_per_batch,
        va_max_vars_per_batch=cli.va_max_vars_per_batch,
        va_include_offset=False,
        penalty_mode="adaptive",
        min_penalty=cli.min_penalty,
        constraint_multiplier=5.0,
        disable_objective_scale=True,
        c4_mode="auto",
        x_empty_penalty_factor=0.0,
        y_overflow_penalty_factor=0.0,
        **{f"min_penalty_{c}": -1.0 for c in ("c1", "c2", "c3", "c4")},
        **{f"constraint_multiplier_{c}": -1.0 for c in ("c1", "c2", "c3", "c4")},
    )


def run_preflight(dataset_dir: Path, cli: argparse.Namespace) -> dict[str, Any]:
    args = preflight_args(cli)
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
    load_s = time.perf_counter() - t0

    batches = solver.build_batches(
        data["active"], data["part_order"], data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )
    t1 = time.perf_counter()
    plan = solver.build_batch_plan(data, batches, args)
    construction_s = time.perf_counter() - t1

    tm_peak = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
    tracemalloc.stop()

    vars_total = sum(p["total_vars"] for p in plan)
    weighted_density = sum(p["matrix_density"] * p["total_vars"] for p in plan) / max(1, vars_total)
    over = [p for p in plan if p["total_vars"] > int(args.va_max_vars_per_batch)]

    return {
        "instance": dataset_dir.name,
        "mode": "preflight",
        "status": "OVER-CEILING" if over else "OK",
        # size
        "hubs": len(data["J"]),
        "zips": len(data["zips"]),
        "parts": len(data["K"]),
        "demand_rows": len(data["active"]),
        "batches": len(plan),
        "num_z": sum(p["num_z"] for p in plan),
        "num_y": sum(p["num_y"] for p in plan),
        "num_x": sum(p["num_x"] for p in plan),
        "binary_vars": vars_total,
        "max_batch_vars": max(p["total_vars"] for p in plan),
        "interactions": sum(p["interactions"] for p in plan),
        "matrix_density": weighted_density,
        "couplings_per_var": max(p["avg_couplings_per_var"] for p in plan),
        # memory
        "dense_bytes": max(p["dense_matrix_bytes"] for p in plan),
        "sparse_bytes": sum(p["sparse_equivalent_bytes"] for p in plan),
        "dense_waste_factor": max(p["dense_waste_factor"] for p in plan),
        "rss_peak_mb": solver.peak_rss_mb(),
        "tracemalloc_peak_mb": tm_peak,
        # time
        "load_seconds": load_s,
        "construction_seconds": construction_s,
        "pyqubo_express_seconds": sum(p["pyqubo_express_seconds"] for p in plan),
        "pyqubo_compile_seconds": sum(p["pyqubo_compile_seconds"] for p in plan),
        "annealing_seconds": BLANK,
        "evaluation_seconds": BLANK,
        "total_wall_seconds": time.perf_counter() - t0,
        # quality: nothing sampled in preflight
        "raw_cost": BLANK,
        "final_cost": BLANK,
        "raw_structural_violations": BLANK,
        "raw_c1": BLANK, "raw_c2": BLANK, "raw_c3": BLANK,
        "feasible_read_share": BLANK,
        "total_reads": BLANK,
    }


# ---------------------------------------------------------------------------
# full mode: real VA solve on the VE card
# ---------------------------------------------------------------------------


def poll_peak_rss(proc: subprocess.Popen, interval: float = 0.05) -> float:
    """Peak RSS of the process tree in MB, sampled while it runs.

    The solver reports its own ru_maxrss, but polling externally also catches
    any child process and is independent of the solver's instrumentation.
    """
    if psutil is None:
        proc.wait()
        return BLANK
    try:
        p = psutil.Process(proc.pid)
    except Exception:
        proc.wait()
        return BLANK

    peak = 0.0
    while proc.poll() is None:
        try:
            total = p.memory_info().rss
            for child in p.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    pass
            peak = max(peak, total / (1024.0 * 1024.0))
        except Exception:
            break
        time.sleep(interval)
    proc.wait()
    return peak


def run_full(dataset_dir: Path, cli: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(cli.run_root) / dataset_dir.name
    cmd = [
        sys.executable, "run_va_fsl_solver.py",
        "--dataset-dir", str(dataset_dir),
        "--run-root", str(run_root),
        "--part-batch-size", str(cli.part_batch_size),
        "--max-z-vars-per-batch", str(cli.max_z_vars_per_batch),
        "--va-max-vars-per-batch", str(cli.va_max_vars_per_batch),
        "--num-reads", str(cli.num_reads),
        "--num-sweeps", str(cli.num_sweeps),
        "--min-penalty", str(cli.min_penalty),
        "--va-repeats", str(cli.va_repeats),
        "--adaptive-penalty-mode", cli.adaptive_penalty_mode,
        "--adaptive-penalty-iterations", str(cli.adaptive_penalty_iterations),
        "--qubo-time-limit", str(cli.qubo_time_limit),
    ]
    if cli.va_vector_mode:
        cmd += ["--va-vector-mode", cli.va_vector_mode]

    log_path = run_root / "va_benchmark_stdout.log"
    run_root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        rss_peak = poll_peak_rss(proc)
    wall = time.perf_counter() - t0

    summary_path = run_root / "va" / "summary.json"
    if proc.returncode != 0 or not summary_path.is_file():
        tail = ""
        if log_path.is_file():
            tail = "\n      ".join(log_path.read_text(errors="replace").splitlines()[-6:])
        print(f"    FAILED rc={proc.returncode}; last log lines:\n      {tail}", flush=True)
        return {
            "instance": dataset_dir.name, "mode": "full",
            "status": f"FAILED(rc={proc.returncode})",
            "measured_wall_seconds": wall, "rss_peak_polled_mb": rss_peak,
        }

    s = json.loads(summary_path.read_text())
    perf = s.get("extra", {}).get("performance", {})
    audit = s["final_solution"]["audit"]
    reads = float(perf.get("total_reads", 0) or 0)

    return {
        "instance": dataset_dir.name,
        "mode": "full",
        "status": "OK",
        "hubs": perf.get("hubs"),
        "zips": perf.get("zips"),
        "parts": perf.get("parts"),
        "demand_rows": perf.get("active_demand_rows"),
        "batches": perf.get("batches"),
        "num_z": perf.get("num_z"),
        "num_y": perf.get("num_y"),
        "num_x": perf.get("num_x"),
        "binary_vars": perf.get("binary_variables"),
        "max_batch_vars": perf.get("max_batch_binary_variables"),
        "interactions": perf.get("qubo_interactions"),
        "matrix_density": perf.get("matrix_density"),
        "couplings_per_var": perf.get("avg_couplings_per_var"),
        "dense_bytes": perf.get("max_dense_matrix_bytes"),
        # The solver reports dense bytes and the waste factor; their ratio is the
        # sparse-equivalent footprint of the same couplings.
        "sparse_bytes": (
            float(perf["max_dense_matrix_bytes"]) / float(perf["max_dense_waste_factor"])
            if perf.get("max_dense_matrix_bytes") and perf.get("max_dense_waste_factor")
            else BLANK
        ),
        "dense_waste_factor": perf.get("max_dense_waste_factor"),
        "rss_peak_mb": max(float(perf.get("rss_peak_mb") or 0.0), rss_peak or 0.0),
        "tracemalloc_peak_mb": s["runtime"].get("tracemalloc_peak_mb"),
        "load_seconds": BLANK,
        "construction_seconds": perf.get("qubo_construction_seconds"),
        "pyqubo_express_seconds": s["runtime"].get("pyqubo_express_seconds"),
        "pyqubo_compile_seconds": s["runtime"].get("pyqubo_compile_seconds"),
        "annealing_seconds": perf.get("annealing_seconds"),
        "evaluation_seconds": perf.get("evaluation_seconds"),
        "total_wall_seconds": perf.get("total_wall_seconds", wall),
        "measured_wall_seconds": wall,
        "raw_cost": perf.get("raw_cost"),
        "final_cost": s["final_solution"]["cost"]["total_cost"],
        "raw_structural_violations": perf.get("raw_structural_violations"),
        "raw_c1": perf.get("raw_c1_violations"),
        "raw_c2": perf.get("raw_c2_violations"),
        "raw_c3": perf.get("raw_c3_violations"),
        "final_structural_violations": audit["total_structural_violations"],
        "sla_violations": audit["sla_distance_violations"],
        "feasible_read_share": (float(perf.get("structurally_feasible_reads", 0)) / reads) if reads else BLANK,
        "total_reads": perf.get("total_reads"),
        "annealing_share_of_wall": s["runtime"].get("annealing_share_of_wall"),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _n(v: Any, fmt: str = ",.0f", blank: str = "-") -> str:
    try:
        f = float(v)
        if f != f:  # NaN
            return blank
        return format(f, fmt)
    except (TypeError, ValueError):
        return blank


def print_report(rows: list[dict[str, Any]], mode: str) -> None:
    ok = [r for r in rows if r.get("status") == "OK"]

    print("\n" + "=" * 104)
    print(f"VA COMPUTATIONAL PERFORMANCE  (mode={mode})")
    print("=" * 104)

    print("\n1. PROBLEM SIZE")
    h = (f"  {'instance':<14} {'hubs':>5} {'demand':>8} {'batch':>6} {'Z':>8} {'Y':>7} {'X':>5} "
         f"{'binary_vars':>12} {'interactions':>13} {'density':>10} {'cpl/var':>8}")
    print(h); print("  " + "-" * (len(h) - 2))
    for r in rows:
        print(f"  {r['instance']:<14} {_n(r.get('hubs')):>5} {_n(r.get('demand_rows')):>8} "
              f"{_n(r.get('batches')):>6} {_n(r.get('num_z')):>8} {_n(r.get('num_y')):>7} "
              f"{_n(r.get('num_x')):>5} {_n(r.get('binary_vars')):>12} {_n(r.get('interactions')):>13} "
              f"{_n(r.get('matrix_density'), '.5%'):>10} {_n(r.get('couplings_per_var'), '.1f'):>8}")

    print("\n2. MEMORY  -- two different resources, do not add them up")
    print("   VA dense / sparse equiv / waste = DEVICE side. What the VE card must hold:")
    print("   binary_vars^2 x 4B, allocated regardless of how sparse the QUBO is.")
    print("   RSS peak / tracemalloc     = HOST side. Python building the QUBO before transfer.")
    h = (f"  {'instance':<14} {'VA dense':>12} {'sparse equiv':>14} {'waste':>10} "
         f"{'RSS peak MB':>12} {'tracemalloc MB':>15}")
    print(h); print("  " + "-" * (len(h) - 2))
    for r in rows:
        dense = r.get("dense_bytes"); sparse = r.get("sparse_bytes")
        print(f"  {r['instance']:<14} "
              f"{(solver.human_bytes(int(dense)) if dense == dense and dense else '-'):>12} "
              f"{(solver.human_bytes(int(sparse)) if sparse == sparse and sparse else '-'):>14} "
              f"{_n(r.get('dense_waste_factor'), ',.0f') + 'x':>10} "
              f"{_n(r.get('rss_peak_mb'), ',.1f'):>12} {_n(r.get('tracemalloc_peak_mb'), ',.1f'):>15}")

    print("\n3. TIME (seconds)")
    h = (f"  {'instance':<14} {'construct':>10} {'  express':>9} {'compile':>8} {'annealing':>10} "
         f"{'evaluate':>9} {'total wall':>11} {'anneal %':>9}")
    print(h); print("  " + "-" * (len(h) - 2))
    for r in rows:
        share = r.get("annealing_share_of_wall")
        print(f"  {r['instance']:<14} {_n(r.get('construction_seconds'), ',.2f'):>10} "
              f"{_n(r.get('pyqubo_express_seconds'), ',.2f'):>9} {_n(r.get('pyqubo_compile_seconds'), ',.2f'):>8} "
              f"{_n(r.get('annealing_seconds'), ',.2f'):>10} {_n(r.get('evaluation_seconds'), ',.2f'):>9} "
              f"{_n(r.get('total_wall_seconds'), ',.2f'):>11} {_n(share, '.1%'):>9}")

    if mode == "full":
        print("\n4. SOLUTION QUALITY AND CONSTRAINT VIOLATIONS")
        print("   raw_* is what VA returned. final_* is after the repair/prune post-pass --")
        print("   judge the annealer on raw_*, since the post-pass repairs violations to 0.")
        h = (f"  {'instance':<14} {'raw cost':>16} {'final cost':>16} {'raw C1':>8} {'raw C2':>8} "
             f"{'raw C3':>8} {'raw struct':>11} {'final struct':>13} {'feasible reads':>15}")
        print(h); print("  " + "-" * (len(h) - 2))
        for r in rows:
            print(f"  {r['instance']:<14} {_n(r.get('raw_cost'), ',.2f'):>16} "
                  f"{_n(r.get('final_cost'), ',.2f'):>16} {_n(r.get('raw_c1')):>8} "
                  f"{_n(r.get('raw_c2')):>8} {_n(r.get('raw_c3')):>8} "
                  f"{_n(r.get('raw_structural_violations')):>11} "
                  f"{_n(r.get('final_structural_violations')):>13} "
                  f"{_n(r.get('feasible_read_share'), '.1%'):>15}")

    if len(ok) > 1:
        base = ok[0]
        print(f"\n{'5' if mode == 'full' else '4'}. SCALING TREND (x relative to {base['instance']})")
        keys = [("hubs", "hubs"), ("binary_vars", "vars"), ("interactions", "interact"),
                ("dense_bytes", "VA dense"), ("construction_seconds", "construct"),
                ("annealing_seconds", "anneal"), ("rss_peak_mb", "RSS")]
        h = "  " + f"{'instance':<14}" + "".join(f"{lbl:>11}" for _, lbl in keys)
        print(h); print("  " + "-" * (len(h) - 2))
        for r in ok:
            cells = ""
            for key, _ in keys:
                try:
                    b, v = float(base.get(key)), float(r.get(key))
                    if b != b or v != v or not b:  # NaN or zero base -> not comparable
                        raise ValueError
                    cells += f"{v / b:>10.2f}x"
                except (TypeError, ValueError):
                    cells += f"{'-':>11}"
            print(f"  {r['instance']:<14}{cells}")

    failed = [r for r in rows if r.get("status") != "OK"]
    print("\n" + "=" * 104)
    print(f"RESULT: {len(ok)}/{len(rows)} instance(s) completed"
          + (f"; failures: {[r['instance'] for r in failed]}" if failed else "."))
    print("=" * 104)


def fit_growth_exponent(x: list[float], y: list[float]) -> float:
    """Slope of log(y) vs log(x), i.e. k in y ~ x^k.

    k tells you the growth class directly: 1.0 doubles when hubs double, 2.0
    quadruples. Fitted rather than eyeballed so the classification is not a
    judgement call.
    """
    pts = [(math.log(a), math.log(b)) for a, b in zip(x, y)
           if a and b and a > 0 and b > 0 and a == a and b == b]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else float("nan")


def classify_growth(k: float) -> tuple[str, str]:
    """(label, plain-English meaning) for a growth exponent."""
    if k != k:
        return "n/a", "not enough data points"
    if k < -0.3:
        return "SHRINKING", f"falls as size grows (~1/n^{abs(k):.1f})"
    if k < 0.3:
        return "FLAT", "roughly constant regardless of size"
    if k < 0.75:
        return "SUB-LINEAR", "grows slower than the hub count"
    if k < 1.3:
        return "LINEAR", "doubles when hubs double"
    if k < 1.7:
        return "SUPER-LINEAR", "grows faster than hubs, cheaper than squared"
    if k < 2.3:
        return "QUADRATIC", "quadruples when hubs double"
    return "WORSE THAN QUADRATIC", "grows explosively; this will bind first"


def sparkbar(values: list[float], width: int = 22) -> str:
    """A tiny proportional bar so the shape is visible without reading digits."""
    vals = [v for v in values if v == v]
    if not vals or max(vals) <= 0:
        return ""
    top = max(vals)
    return "".join("#" * max(1, int(round(width * (v / top)))) if v == v else "" for v in values[:1])


def print_trend_analysis(rows: list[dict[str, Any]], mode: str) -> None:
    """Turn the recorded numbers into growth laws, per-unit rates and projections."""
    ok = [r for r in rows if r.get("status") == "OK" and r.get("hubs")]
    if len(ok) < 2:
        return
    hubs = [float(r["hubs"]) for r in ok]

    print("\n" + "=" * 104)
    print("TREND ANALYSIS")
    print("=" * 104)
    print("  Growth exponent k fits metric ~ hubs^k across the suite.")
    print("  k=1 linear (doubles when hubs double) | k=2 quadratic (quadruples) | k<0 shrinking.\n")

    metrics = [
        ("demand_rows", "demand rows", ",.0f"),
        ("binary_vars", "binary variables", ",.0f"),
        ("interactions", "QUBO interactions", ",.0f"),
        ("matrix_density", "matrix density", ".5%"),
        ("couplings_per_var", "couplings per variable", ".2f"),
        ("dense_bytes", "VA dense matrix", ",.0f"),
        ("construction_seconds", "QUBO construction time", ",.2f"),
        ("annealing_seconds", "annealing time", ",.2f"),
        ("total_wall_seconds", "total runtime", ",.2f"),
        ("rss_peak_mb", "peak RSS", ",.1f"),
        ("raw_structural_violations", "raw violations", ",.0f"),
        ("raw_cost", "raw cost", ",.0f"),
    ]

    h = f"  {'metric':<24} {'first':>14} {'last':>14} {'growth':>9} {'k':>6}  {'class':<20} meaning"
    print(h)
    print("  " + "-" * (len(h) + 22))
    for key, label, fmt in metrics:
        ys = [float(r.get(key)) if r.get(key) is not None else float("nan") for r in ok]
        ys = [y if y == y else float("nan") for y in ys]
        usable = [y for y in ys if y == y]
        if len(usable) < 2:
            continue
        k = fit_growth_exponent(hubs, ys)
        label_txt, meaning = classify_growth(k)
        first, last = usable[0], usable[-1]
        growth = (last / first) if first else float("nan")
        print(f"  {label:<24} {_n(first, fmt):>14} {_n(last, fmt):>14} "
              f"{_n(growth, ',.1f') + 'x':>9} {_n(k, '+.2f'):>6}  {label_txt:<20} {meaning}")

    # Per-unit rates: these should stay roughly flat if scaling is healthy.
    print("\n  PER-UNIT RATES  (flat down a column = clean scaling)")
    h2 = (f"  {'instance':<14} {'vars/hub':>10} {'interact/var':>13} {'build ms/1k vars':>17} "
          f"{'RSS MB/1k vars':>15} {'VA dense MB/var':>16}")
    print(h2)
    print("  " + "-" * (len(h2) - 2))
    for r in ok:
        v = float(r.get("binary_vars") or 0)
        cs = r.get("construction_seconds")
        print(f"  {r['instance']:<14} {_n(v / float(r['hubs']), ',.1f'):>10} "
              f"{_n(float(r.get('interactions') or 0) / v if v else BLANK, ',.2f'):>13} "
              f"{_n((float(cs) * 1000.0 / (v / 1000.0)) if (cs and v) else BLANK, ',.1f'):>17} "
              f"{_n(float(r.get('rss_peak_mb') or 0) / (v / 1000.0) if v else BLANK, ',.1f'):>15} "
              f"{_n(float(r.get('dense_bytes') or 0) / 1048576.0 / v if v else BLANK, ',.3f'):>16}")

    # Projection to the VA ceilings, using the fitted laws.
    var_k = fit_growth_exponent(hubs, [float(r["binary_vars"]) for r in ok])
    base_hub, base_vars = hubs[0], float(ok[0]["binary_vars"])
    print("\n  PROJECTION  (extrapolating binary variables ~ hubs^%.2f)" % var_k)
    h3 = (f"  {'hubs':>6} {'binary_vars':>13} {'VA dense':>12} {'vs 60k ceiling':>16} "
          f"{'vs 100k hard max':>18}")
    print(h3)
    print("  " + "-" * (len(h3) - 2))
    for target in (100, 150, 200, 250, 300):
        proj = base_vars * (target / base_hub) ** var_k
        dense = solver.dense_matrix_bytes(int(proj))
        c1 = "FITS" if proj <= 60000 else f"OVER by {proj - 60000:,.0f}"
        c2 = "FITS" if proj <= 100000 else f"OVER by {proj - 100000:,.0f}"
        print(f"  {target:>6,} {proj:>13,.0f} {solver.human_bytes(dense):>12} {c1:>16} {c2:>18}")

    # The one-paragraph read.
    dense_k = fit_growth_exponent(hubs, [float(r["dense_bytes"]) for r in ok])
    dens_k = fit_growth_exponent(hubs, [float(r["matrix_density"]) for r in ok])
    print("\n  READ THIS AS:")
    print(f"    - Problem size grows ~linearly (vars ~ hubs^{var_k:.2f}), so the model itself scales cleanly.")
    print(f"    - VA's memory grows ~hubs^{dense_k:.2f} because it stores the matrix densely,")
    print(f"      while density itself falls ~hubs^{dens_k:.2f}. Memory, not variable count, binds first.")
    if mode != "full":
        print("    - Annealing time and solution quality are blank: preflight does not sample.")
        print("      Run --mode full on the VE node (sfpga01n) to fill those columns.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["preflight", "full"], default="preflight",
                   help="preflight needs no VE card; full runs the real VA solve.")
    p.add_argument("--instances", nargs="*", default=None,
                   help="Instance dirs. Default: outputs/fsl_*_hubs by hub count.")
    p.add_argument("--run-root", default="results/va_benchmark",
                   help="full mode: where each instance's solver run is written.")
    p.add_argument("--csv", default="", help="Write the consolidated table here.")
    p.add_argument("--part-batch-size", type=int, default=1000)
    p.add_argument("--max-z-vars-per-batch", type=int, default=50000)
    p.add_argument("--va-max-vars-per-batch", type=int, default=60000)
    p.add_argument("--min-penalty", type=float, default=50000.0)
    p.add_argument("--num-reads", type=int, default=100)
    p.add_argument("--num-sweeps", type=int, default=3000)
    p.add_argument("--va-repeats", type=int, default=1)
    p.add_argument("--va-vector-mode", default="ACCURACY", choices=["SPEED", "ACCURACY"])
    p.add_argument("--adaptive-penalty-mode", default="within-batch", choices=["off", "within-batch"])
    p.add_argument("--adaptive-penalty-iterations", type=int, default=8)
    p.add_argument("--qubo-time-limit", type=float, default=21600.0)
    cli = p.parse_args(argv)

    if cli.instances:
        dirs = [Path(x) for x in cli.instances]
    else:
        dirs = sorted(Path("outputs").glob("fsl_*_hubs"), key=lambda d: int(d.name.split("_")[1]))
    if not dirs or any(not d.is_dir() for d in dirs):
        raise SystemExit(f"ERROR: no instance directories found ({[str(d) for d in dirs]})")

    print("=" * 104)
    print(f"VA BENCHMARK | mode={cli.mode} | {len(dirs)} instance(s)")
    if cli.mode == "full":
        print(f"  sampling budget held constant: num_reads={cli.num_reads} num_sweeps={cli.num_sweeps} "
              f"repeats={cli.va_repeats} vector_mode={cli.va_vector_mode}")
    print("=" * 104)

    rows: list[dict[str, Any]] = []
    for d in dirs:
        print(f"\n>>> {d} ...", flush=True)
        try:
            row = run_preflight(d, cli) if cli.mode == "preflight" else run_full(d, cli)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            row = {"instance": d.name, "mode": cli.mode, "status": f"FAILED({type(exc).__name__})"}
        rows.append(row)
        if row.get("status") == "OK":
            print(f"    OK  vars={_n(row.get('binary_vars'))} "
                  f"density={_n(row.get('matrix_density'), '.5%')} "
                  f"wall={_n(row.get('total_wall_seconds'), ',.2f')}s", flush=True)

    print_report(rows, cli.mode)
    print_trend_analysis(rows, cli.mode)

    if cli.csv:
        Path(cli.csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cli.csv, index=False)
        print(f"\nconsolidated table -> {cli.csv}")
    return 0 if all(r.get("status") == "OK" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
