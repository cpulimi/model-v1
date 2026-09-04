#!/usr/bin/env python3
"""Collect the Gurobi proven-optimum ladder into one CSV.

Walks the per-size output folders written by sbatch_scripts/gurobi_optima_solve.sh,
reads each Gurobi summary.json, and emits gurobi_optima.csv plus the same table
on stdout.

The point of the table is the optimality bracket. For a MINIMIZE model the true
optimum lies in [obj_bound, total_cost]; proven_optimal says whether that bracket
has actually collapsed. A TIME_LIMIT row is still useful -- it gives a valid
lower bound on the optimum, which is exactly what an annealing gap needs to be
measured against -- so those rows are kept, not dropped.

Usage:
    python collect_gurobi_optima.py
    python collect_gurobi_optima.py --run-root results/gurobi_optima --output results/gurobi_optima.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Must match gurobi_optima_map() in sbatch_scripts/gurobi_optima_solve.sh.
LADDER: list[tuple[int, str]] = [
    (10, "gurobi_optima_10hubs"),
    (20, "gurobi_optima_20hubs"),
    (50, "gurobi_optima_50hubs"),
    (100, "gurobi_optima_100hubs"),
]

# Gurobi's own default MIPGap tolerance is 1e-4; a gap this tight is numerically
# indistinguishable from closed.
OPTIMAL_GAP_TOL = 1e-9

COLUMNS = [
    "instance",
    "n_hubs",
    "status",
    "total_cost",
    "obj_bound",
    "mip_gap_achieved",
    "proven_optimal",
    "gurobi_runtime_s",
    "wall_time_s",
    "node_count",
    "threads",
    "peak_rss_mb",
    "open_hubs",
    "closed_hubs",
]


def _get(d: Any, *path: str) -> Any:
    """Nested dict lookup that returns None instead of raising."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    v = _num(value)
    return None if v is None else int(v)


def read_row(n_hubs: int, run_name: str, run_root: Path) -> dict[str, Any]:
    summary_path = run_root / run_name / "gurobi" / "summary.json"
    row: dict[str, Any] = {c: None for c in COLUMNS}
    row["instance"] = run_name
    row["n_hubs"] = n_hubs

    if not summary_path.is_file():
        row["status"] = "MISSING_SUMMARY"
        row["proven_optimal"] = False
        return row

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # truncated write, partial job, bad json
        row["status"] = f"UNREADABLE_SUMMARY ({type(exc).__name__})"
        row["proven_optimal"] = False
        return row

    extra = summary.get("extra") or {}
    runtime = summary.get("runtime") or {}
    final = summary.get("final_solution") or {}

    dataset_name = _get(summary, "dataset", "dataset_name")
    if dataset_name:
        row["instance"] = str(dataset_name)
    hubs_from_summary = _int(_get(summary, "dataset", "hubs"))
    if hubs_from_summary is not None:
        row["n_hubs"] = hubs_from_summary

    status = extra.get("gurobi_status")
    row["status"] = str(status) if status else "UNKNOWN"

    row["total_cost"] = _num(_get(final, "cost", "total_cost"))
    if row["total_cost"] is None:
        row["total_cost"] = _num(extra.get("gurobi_objective"))

    row["obj_bound"] = _num(extra.get("obj_bound"))

    # gurobi_result_version 2 writes mip_gap_achieved; v1 summaries only have
    # the older `mip_gap` key, which held the same model.MIPGap value.
    gap = _num(extra.get("mip_gap_achieved"))
    if gap is None:
        gap = _num(extra.get("mip_gap"))
    row["mip_gap_achieved"] = gap

    row["gurobi_runtime_s"] = _num(extra.get("gurobi_runtime"))
    if row["gurobi_runtime_s"] is None:
        row["gurobi_runtime_s"] = _num(runtime.get("solver_runtime_seconds"))

    row["wall_time_s"] = _num(runtime.get("wall_seconds"))
    row["node_count"] = _num(extra.get("node_count"))
    row["threads"] = _int(extra.get("threads_resolved"))
    row["peak_rss_mb"] = _num(runtime.get("peak_memory_mb"))
    row["open_hubs"] = _int(final.get("open_hubs_count"))
    row["closed_hubs"] = _int(final.get("closed_hubs_count"))

    # Both conditions required. A status of OPTIMAL alone is not enough: it is
    # reported once the achieved gap is within the REQUESTED MIPGap, so a run
    # with --mip-gap 0.001 says OPTIMAL at a 0.1% bracket that is not proven.
    row["proven_optimal"] = bool(
        row["status"] == "OPTIMAL" and gap is not None and gap <= OPTIMAL_GAP_TOL
    )

    # Flag the case the memory column silently depends on.
    if runtime.get("memory_accounting_version") != 2:
        row["_mem_v1"] = True
    return row


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "money":
        return f"{float(value):,.2f}"
    if kind == "gap":
        return "0" if float(value) == 0.0 else f"{float(value):.3e}"
    if kind == "sec":
        return f"{float(value):,.1f}"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "mb":
        return f"{float(value):,.1f}"
    if kind == "bool":
        return "yes" if value else "no"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    header = [
        ("instance", "instance", 22, "str"),
        ("n_hubs", "hubs", 5, "int"),
        ("status", "status", 14, "str"),
        ("total_cost", "total_cost", 16, "money"),
        ("obj_bound", "obj_bound", 16, "money"),
        ("mip_gap_achieved", "gap", 11, "gap"),
        ("proven_optimal", "proven", 6, "bool"),
        ("gurobi_runtime_s", "grb_s", 10, "sec"),
        ("wall_time_s", "wall_s", 10, "sec"),
        ("node_count", "nodes", 12, "int"),
        ("threads", "thr", 4, "int"),
        ("peak_rss_mb", "rss_mb", 10, "mb"),
        ("open_hubs", "open", 5, "int"),
        ("closed_hubs", "closed", 6, "int"),
    ]
    line = "  ".join(f"{label:>{width}}" for _key, label, width, _kind in header)
    print(line)
    print("  ".join("-" * width for _k, _l, width, _kd in header))
    for row in rows:
        cells = []
        for key, _label, width, kind in header:
            text = fmt(row.get(key), kind)
            if len(text) > width:
                text = text[: width - 1] + "…"
            cells.append(f"{text:>{width}}")
        print("  ".join(cells))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-root",
        default="results/gurobi_optima",
        help="Folder holding the per-size run folders. Must match OUTDIR in gurobi_optima_solve.sh.",
    )
    p.add_argument(
        "--output",
        default="",
        help="CSV path. Default: <run-root>/gurobi_optima.csv",
    )
    args = p.parse_args(argv)

    run_root = Path(args.run_root).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else run_root / "gurobi_optima.csv"

    rows = [read_row(n_hubs, run_name, run_root) for n_hubs, run_name in LADDER]

    print(f"run root: {run_root}")
    print()
    print_table(rows)
    print()

    missing = [r for r in rows if str(r["status"]).startswith(("MISSING", "UNREADABLE"))]
    unproven = [r for r in rows if not r["proven_optimal"] and r not in missing]
    mem_v1 = [r for r in rows if r.pop("_mem_v1", False)]

    if missing:
        print(f"NOTE: {len(missing)} of {len(rows)} runs have no readable summary "
              f"({', '.join(str(r['instance']) for r in missing)}). "
              f"Check logs/gurobi_optima_*.err.")
    if unproven:
        for r in unproven:
            bracket = ""
            if r["obj_bound"] is not None and r["total_cost"] is not None:
                bracket = (f" optimum is in [{float(r['obj_bound']):,.2f}, "
                           f"{float(r['total_cost']):,.2f}]")
            print(f"NOTE: {r['instance']} is NOT proven optimal (status={r['status']}, "
                  f"gap={fmt(r['mip_gap_achieved'], 'gap')}).{bracket}")
    if mem_v1:
        print(f"NOTE: {len(mem_v1)} row(s) predate memory_accounting_version 2; their "
              f"peak_rss_mb is the old tracemalloc-blended number and understates "
              f"Gurobi's C-level usage. Do not compare those against annealing.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in COLUMNS})
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
