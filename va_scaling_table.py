#!/usr/bin/env python3
"""Scaling table that keeps HOST and DEVICE memory apart.

Written to kill a specific reporting error: "240 MB to 2.1 GB, roughly 9x".
That compares HOST RSS at 10 hubs against PREDICTED DEVICE dense memory at
50 hubs -- two different resources, two different scaling laws, two different
instances. It is not a growth factor of anything.

  host RSS   -- MEASURED, on the CPU side, roughly linear in interaction count
                because the host never densifies the QUBO (it hands VA a sparse
                dict; see qubo_from_model).
  device      -- PREDICTED, on the card, exactly 4*N^2 bytes. Not a fit: VA
                stores the problem densely regardless of sparsity, so this is
                arithmetic, and the only uncertainty is whether VA allocates
                exactly what the manual implies.

These are never added and never divided into one another.

Everything is read from the run artefacts, so the table updates itself when a
missing run completes. A figure that was not measured is reported as absent --
never back-filled from a fit, which is how the original error happened.

    python3 va_scaling_table.py
    python3 va_scaling_table.py --csv results/va_scaling.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

BAR = "=" * 108
VA_DENSE_BYTES_PER_ENTRY = 4
MB = 1024.0 * 1024.0

# (label, hubs, run dir, solve log). The log is a fallback source of host RSS
# for runs that never wrote a summary.json.
RUNS = [
    ("instances_10hubs", 10, "results/va_parallel/va_10hubs", "logs/va_par_solve_61920528_1.out"),
    ("instances_20hubs", 20, "results/va_parallel/va_20hubs", "logs/va_par_solve_61921976_1.out"),
    ("instances_50hubs", 50, "results/va_parallel/va_50hubs", "logs/va_par_solve_61924260_1.out"),
]


def host_rss_from_log(path: Path) -> float | None:
    """`rss_peak=NNN.N MB` from a solve log's completion line, if it got there.

    A killed run has no such line. Returning None keeps the distinction between
    "measured 0" and "never measured", which is the whole point of this script.
    """
    if not path.is_file():
        return None
    m = None
    for line in path.read_text(errors="replace").splitlines():
        hit = re.search(r"rss_peak=([\d,]+\.?\d*)", line)
        if hit:
            m = hit.group(1).replace(",", "")
    return float(m) if m else None


def cgroup_from_tsv(run_dir: Path) -> tuple[float | None, str]:
    """Cgroup step peak from slurm_mem_va.tsv, if that file has the column.

    Runs from before the SLURM-capture fix wrote a HEADERLESS tsv with blank
    memory columns (seff was called in-job, before slurmdbd flushed, so it
    reported State: RUNNING / 0.00 MB). Those are absent, not zero.
    """
    tsv = run_dir / "slurm_mem_va.tsv"
    if not tsv.is_file():
        return None, "no slurm_mem_va.tsv"
    lines = [l for l in tsv.read_text().splitlines() if l.strip()]
    if not lines:
        return None, "empty"
    if not lines[0].lower().startswith("jobid\t"):
        return None, "pre-fix headerless TSV, memory columns blank"
    header = lines[0].split("\t")
    try:
        idx = header.index("CgroupPeakMB")
    except ValueError:
        return None, "no CgroupPeakMB column"
    vals = []
    for l in lines[1:]:
        parts = l.split("\t")
        if len(parts) > idx and parts[idx].strip():
            try:
                vals.append(float(parts[idx]))
            except ValueError:
                pass
    return (max(vals), "measured") if vals else (None, "column present but blank")


def collect(label: str, hubs: int, run_dir: str, log: str) -> dict:
    root = Path(run_dir)
    row = {
        "instance": label, "hubs": hubs, "total_vars": None,
        "host_rss_peak_mb": None, "host_rss_source": "not measured",
        "cgroup_peak_mb": None, "cgroup_source": "",
        "device_dense_mb": None, "batch_count": None, "wall_seconds": None,
        "status": "no run artefacts",
    }
    summary = root / "va" / "summary.json"
    plan = root / "parallel_work" / "va_batch_plan.csv"

    if summary.is_file():
        s = json.loads(summary.read_text())
        p, rt = s["extra"]["performance"], s["runtime"]
        row.update({
            "total_vars": int(p["max_batch_binary_variables"]),
            "host_rss_peak_mb": float(rt["max_batch_rss_peak_mb"]),
            "host_rss_source": "summary.json (measured)",
            "device_dense_mb": float(p["max_dense_matrix_bytes"]) / MB,
            "batch_count": int(p["batches"]),
            "wall_seconds": float(rt["wall_seconds"]),
            "status": "complete",
        })
    elif plan.is_file():
        # Split ran, solve did not finish. total_vars and the dense prediction
        # are still real -- they come from compiling the QUBO, not from sampling.
        with open(plan, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            row.update({
                "total_vars": max(int(r["total_vars"]) for r in rows),
                "device_dense_mb": max(float(r["dense_matrix_bytes"]) for r in rows) / MB,
                "batch_count": len(rows),
                "status": "INCOMPLETE - solve did not finish",
            })
        rss = host_rss_from_log(Path(log))
        if rss is not None:
            row["host_rss_peak_mb"] = rss
            row["host_rss_source"] = f"{log} (measured)"

    if row["host_rss_peak_mb"] is None and Path(log).is_file():
        rss = host_rss_from_log(Path(log))
        if rss is not None:
            row["host_rss_peak_mb"] = rss
            row["host_rss_source"] = f"{log} (measured)"

    cg, why = cgroup_from_tsv(root)
    row["cgroup_peak_mb"], row["cgroup_source"] = cg, why
    return row


def loglog_fit(pts: list[tuple[float, float]]) -> dict | None:
    """Power-law exponent by least squares on log-log. None below 2 points."""
    if len(pts) < 2:
        return None
    xs = [math.log(x) for x, _ in pts]
    ys = [math.log(y) for _, y in pts]
    n = float(len(pts))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return {"exponent": b, "coef": math.exp(a), "n": len(pts),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None}


def linear_fit(pts: list[tuple[float, float]]) -> dict | None:
    """Host model a + b*N. Returns residuals so linearity can be JUDGED."""
    if len(pts) < 2:
        return None
    n = float(len(pts))
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in pts]
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum(r * r for r in resid)
    return {"intercept": a, "slope": b, "n": len(pts), "residuals": resid,
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None}


def crossover(fit: dict | None) -> float | None:
    """N where 4N^2/2^20 meets the fitted host line a + bN."""
    if not fit:
        return None
    a, b = fit["intercept"], fit["slope"]
    K = MB / VA_DENSE_BYTES_PER_ENTRY
    disc = b * b + 4.0 * a / K
    if disc < 0 or a < 0:
        return None
    root = (b + math.sqrt(disc)) / (2.0 / K)
    return root if root > 0 else None


def fmt(v, spec=",.1f", dash="--"):
    return dash if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="", help="Also write the table to this path.")
    a = ap.parse_args()

    rows = [collect(*r) for r in RUNS]

    print(BAR)
    print("VA SCALING -- HOST AND DEVICE MEMORY ARE SEPARATE RESOURCES, NEVER COMBINED")
    print(BAR)
    hdr = (f"{'instance':<18}{'hubs':>5}{'total_vars':>12}{'host_rss_mb':>13}"
           f"{'cgroup_mb':>11}{'device_mb':>11}{'batches':>9}{'wall_s':>10}  {'status'}")
    print(hdr)
    print(f"{'':18}{'':5}{'':12}{'MEASURED':>13}{'MEASURED':>11}{'PREDICTED':>11}")
    print("-" * 108)
    for r in rows:
        print(f"{r['instance']:<18}{r['hubs']:>5}{fmt(r['total_vars'], ',d'):>12}"
              f"{fmt(r['host_rss_peak_mb']):>13}{fmt(r['cgroup_peak_mb']):>11}"
              f"{fmt(r['device_dense_mb']):>11}{fmt(r['batch_count'], 'd'):>9}"
              f"{fmt(r['wall_seconds'], ',.1f'):>10}  {r['status']}")
    print("-" * 108)
    for r in rows:
        if r["host_rss_peak_mb"] is None:
            print(f"  {r['instance']}: host RSS {r['host_rss_source'].upper()} "
                  f"-- the solve was killed before it recorded one.")
        if r["cgroup_peak_mb"] is None:
            print(f"  {r['instance']}: cgroup peak absent ({r['cgroup_source']}).")

    host_pts = [(float(r["total_vars"]), float(r["host_rss_peak_mb"])) for r in rows
                if r["total_vars"] and r["host_rss_peak_mb"]]
    dev_pts = [(float(r["total_vars"]), float(r["device_dense_mb"])) for r in rows
               if r["total_vars"] and r["device_dense_mb"]]

    print("\n" + BAR)
    print("SEPARATE FITS  (never one curve across both resources)")
    print(BAR)

    print(f"\n  DEVICE -- {len(dev_pts)} point(s). Not a fit: dense_matrix_bytes(N) = 4*N^2 "
          f"is exact arithmetic.")
    d = loglog_fit(dev_pts)
    if d:
        print(f"    log-log exponent {d['exponent']:.4f} (expected exactly 2), "
              f"R^2 {d['r2']:.6f} -- confirms the formula, adds no information.")

    print(f"\n  HOST -- {len(host_pts)} measured point(s).")
    hf = linear_fit(host_pts)
    if hf:
        print(f"    linear fit: {hf['intercept']:,.1f} MB + {hf['slope']:.6f} MB/var")
        hl = loglog_fit(host_pts)
        if hl:
            print(f"    log-log exponent {hl['exponent']:.4f}"
                  + (f", R^2 {hl['r2']:.6f}" if hl['r2'] is not None else ""))

    print("\n" + BAR)
    print("IS HOST RSS LINEAR?")
    print(BAR)
    n = len(host_pts)
    if n < 2:
        print(f"  UNANSWERABLE: only {n} measured host point(s).")
    elif n == 2:
        print("  NOT TESTABLE. There are only TWO measured host RSS points, and two points")
        print("  define a line EXACTLY -- zero residual, R^2 = 1 by construction, no")
        print("  alternative model can be rejected. Linearity is an ASSUMPTION here, not a")
        print("  finding. A quadratic, or any monotone curve, fits these two points equally")
        print("  well.")
        print("\n  The third point does not exist: the 50-hub solve was killed during its")
        print("  first adaptive iteration, so it recorded total_vars (23,652, from the")
        print("  compile) but never a host RSS. Completing that run is what would make this")
        print("  question answerable for the first time.")
    else:
        resid = hf["residuals"]
        worst = max(abs(r) for r in resid)
        rel = worst / max(y for _, y in host_pts)
        print(f"  {n} measured points. Residuals from the linear fit: "
              + ", ".join(f"{r:+,.1f}" for r in resid) + " MB")
        print(f"  largest |residual| {worst:,.1f} MB ({100 * rel:.2f}% of the largest value), "
              f"R^2 {hf['r2']:.6f}")
        if rel < 0.05:
            print("  => CONSISTENT with a linear model.")
        else:
            print("  => NOT consistent with a linear model; residuals are structured.")

    print("\n" + BAR)
    print("HOST / DEVICE CROSSOVER")
    print(BAR)
    x = crossover(hf)
    if x is None:
        print("  Not computable from the available host points.")
    else:
        print(f"  Fitted host line meets 4N^2 at N ~= {x:,.0f} total_vars.")
        if n == 2:
            print("  *** ESTIMATE ONLY: rests on the untestable 2-point host line above.")
            print("      Do not quote this figure without that caveat.")
        for r in rows:
            if not r["total_vars"]:
                continue
            regime = "HOST-bound" if r["total_vars"] < x else "DEVICE-bound"
            print(f"    {r['instance']:<18} {r['total_vars']:>7,} vars -> {regime}")

    print("\n" + BAR)
    print("THE REPORTING ERROR THIS REPLACES")
    print(BAR)
    h10 = next((r for r in rows if r["hubs"] == 10), None)
    d50 = next((r for r in rows if r["hubs"] == 50), None)
    if h10 and d50 and h10["host_rss_peak_mb"] and d50["device_dense_mb"]:
        print(f"  \"240 MB to 2.1 GB, roughly 9x\" compares:")
        print(f"    {h10['host_rss_peak_mb']:,.1f} MB  HOST RSS, MEASURED, 10-hub instance")
        print(f"    {d50['device_dense_mb']:,.1f} MB  DEVICE dense, PREDICTED, 50-hub instance")
        print("  Different resource, different instance, different scaling law, and one is")
        print("  measured while the other is arithmetic. The ratio is meaningless.")
        print("\n  What can honestly be said, each within ONE resource:")
        hs = [r for r in rows if r["host_rss_peak_mb"]]
        if len(hs) >= 2:
            lo, hi = hs[0], hs[-1]
            print(f"    host RSS   {lo['host_rss_peak_mb']:,.1f} -> {hi['host_rss_peak_mb']:,.1f} MB "
                  f"({lo['hubs']}->{hi['hubs']} hubs, {hi['host_rss_peak_mb'] / lo['host_rss_peak_mb']:.2f}x) "
                  f"-- both MEASURED")
        ds = [r for r in rows if r["device_dense_mb"]]
        if len(ds) >= 2:
            lo, hi = ds[0], ds[-1]
            print(f"    device     {lo['device_dense_mb']:,.1f} -> {hi['device_dense_mb']:,.1f} MB "
                  f"({lo['hubs']}->{hi['hubs']} hubs, {hi['device_dense_mb'] / lo['device_dense_mb']:.2f}x) "
                  f"-- both PREDICTED")
    print(BAR)

    if a.csv:
        out = Path(a.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["instance", "hubs", "total_vars", "host_rss_peak_mb", "cgroup_peak_mb",
                "device_dense_mb", "batch_count", "wall_seconds",
                "host_rss_source", "cgroup_source", "status"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n  CSV -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
