#!/usr/bin/env python3
"""Three-way comparison: Gurobi optimum vs Vector Annealing vs greedy heuristic.

WHY THREE AND NOT TWO
---------------------
Gurobi alone gives VA a gap but no yardstick. "VA is 0.46% above optimal" is a
strong claim or a weak one depending entirely on what a cheap classical method
does on the same instance, and until that number exists the comparison cannot be
read. The greedy column is the yardstick.

Every cost in this table is produced by the SAME compute_solution_cost() in
run_va_fsl_solver.py, so differences between columns are differences in solution
quality and never in accounting.

    python3 compare_solvers.py
    python3 compare_solvers.py --out results/solver_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BAR = "=" * 100

GUROBI_CSV = Path("results/gurobi_optima/gurobi_optima.csv")
GREEDY_CSV = Path("results/greedy/greedy_ladder.csv")
VA_GLOB = "results/va_parallel/*/va/summary.json"


def load_gurobi() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not GUROBI_CSV.is_file():
        return out
    with open(GUROBI_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            # Only proven optima. An incumbent from a timed-out solve is not a
            # valid denominator; a "gap" against it can come out negative.
            if str(r.get("proven_optimal", "")).strip().lower() == "true":
                out[r["instance"]] = {
                    "cost": float(r["total_cost"]),
                    "seconds": float(r["wall_time_s"]),
                    "open_hubs": int(r["open_hubs"]),
                }
    return out


def load_va() -> dict[str, list[dict]]:
    """Every VA run, grouped by instance. Several seeds per instance is normal."""
    out: dict[str, list[dict]] = {}
    for p in sorted(Path().glob(VA_GLOB)):
        try:
            with open(p, encoding="utf-8") as fh:
                s = json.load(fh)
            out.setdefault(s["dataset"]["dataset_name"], []).append({
                "run": p.parts[2],
                "seed": s["extra"]["va"].get("seed"),
                "cost": float(s["final_solution"]["cost"]["total_cost"]),
                "seconds": float(s["runtime"]["wall_seconds"]),
                "open_hubs": int(s["final_solution"]["open_hubs_count"]),
                "violations": int(
                    s["final_solution"]["audit"]["total_structural_violations"]),
            })
        except (KeyError, OSError, ValueError) as exc:
            print(f"  skipped {p}: {type(exc).__name__}: {exc}")
    return out


def load_greedy() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not GREEDY_CSV.is_file():
        return out
    with open(GREEDY_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["dataset_name"], []).append({
                "variant": r["variant"],
                "cost": float(r["total_cost"]),
                "seconds": float(r["seconds"]),
                "open_hubs": int(r["open_hubs"]),
                "violations": int(r["structural_violations"]),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/solver_comparison.csv")
    a = ap.parse_args()

    gur, va, greedy = load_gurobi(), load_va(), load_greedy()
    order = ["instances_10hubs", "instances_20hubs",
             "instances_50hubs", "instances_100hubs"]
    instances = [i for i in order if i in gur] + \
                [i for i in sorted(set(gur) | set(va) | set(greedy)) if i not in order]

    rows: list[dict] = []
    print(BAR)
    print("SOLVER COMPARISON -- all costs from the same compute_solution_cost()")
    print(BAR)
    head = (f"  {'instance':18} {'method':22} {'total cost':>17} {'gap%':>8} "
            f"{'hubs':>5} {'viol':>5} {'seconds':>11}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    for inst in instances:
        opt = gur.get(inst, {}).get("cost")
        entries: list[tuple[str, dict]] = []
        if inst in gur:
            entries.append(("gurobi (proven opt)", {**gur[inst], "violations": 0}))
        for r in sorted(va.get(inst, []), key=lambda r: (r["seed"] is None, r["seed"])):
            entries.append((f"VA seed {r['seed']}", r))
        for r in greedy.get(inst, []):
            entries.append((f"greedy {r['variant']}", r))

        for label, r in entries:
            gap = 100.0 * (r["cost"] - opt) / opt if opt else None
            print(f"  {inst:18} {label:22} {r['cost']:17,.2f} "
                  f"{(f'{gap:8.3f}' if gap is not None else f'{chr(45):>8}')} "
                  f"{r.get('open_hubs', 0):5} {r.get('violations', 0):5} "
                  f"{r['seconds']:11,.2f}")
            rows.append({
                "instance": inst, "method": label, "total_cost": r["cost"],
                "gurobi_optimum": opt, "gap_pct": gap,
                "open_hubs": r.get("open_hubs"), "violations": r.get("violations"),
                "seconds": r["seconds"],
            })
        print("  " + "-" * (len(head) - 2))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["instance", "method", "total_cost",
                                           "gurobi_optimum", "gap_pct", "open_hubs",
                                           "violations", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
