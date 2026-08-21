#!/usr/bin/env python3
"""
Read the summary.json files a VA ladder run produced and print them as a table.

    python3 va_results.py                      # newest run under results/va_ladder
    python3 va_results.py --run-root results/va_ladder/va_run_20260821_101500
    python3 va_results.py --all                # every run root found
    python3 va_results.py --json               # machine-readable

WHAT TO LOOK AT

  raw_*   is what the annealer actually returned.
  final_* is after the repair/trim/hub-prune post-pass.

Judge VA on raw_*. The post-pass repairs violations to zero by construction, so
final_violations is almost always 0 and says nothing about sampler quality.

instances_10hubs is a special case worth checking first: every ZIP in it has
exactly one eligible hub, so C1 pins Z, C2 pins Y and C3 pins X and the whole
solution is forced. raw_violations there MUST be 0. Anything else means the
sampler or the evaluation path is broken, not that the problem was hard.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any


def find_run_roots(base: str) -> list[Path]:
    """Run roots newest-first, by mtime."""
    roots = [Path(p) for p in glob.glob(os.path.join(base, "va_run_*")) if os.path.isdir(p)]
    return sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True)


def load_summaries(run_root: Path) -> list[tuple[str, dict[str, Any]]]:
    """(instance, summary) for every summary.json under this run root."""
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(run_root.glob("*/va/summary.json")):
        instance = path.parent.parent.name
        try:
            out.append((instance, json.loads(path.read_text())))
        except Exception as exc:
            print(f"  WARNING: could not read {path}: {exc}", file=sys.stderr)
    return out


def g(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Nested .get() so a schema change prints '-' instead of raising."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(v: Any, spec: str = ",.0f", dash: str = "-") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
        if f != f:
            return dash
        return format(f, spec)
    except (TypeError, ValueError):
        return str(v)


def report(run_root: Path) -> int:
    """Print one run root. Returns the number of instances with a problem."""
    rows = load_summaries(run_root)
    print("=" * 118)
    print(f"VA LADDER RESULTS  --  {run_root}")
    print("=" * 118)
    if not rows:
        print("  No summary.json found. Either the job has not finished an instance yet,")
        print("  or it failed before writing one -- check logs/va_solve_<jobid>.out")
        return 0

    # --- what the annealer returned, before any repair ---------------------
    print("\n1. SOLUTION QUALITY  (raw_* = what VA returned; judge the annealer on this)")
    h = (f"  {'instance':<20} {'raw cost':>16} {'final cost':>16} {'raw viol':>9} "
         f"{'C1':>7} {'C2':>7} {'C3':>7} {'feasible reads':>15} {'batches feas':>13}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    problems = 0
    for name, s in rows:
        p = g(s, "extra", "performance", default={}) or {}
        reads = p.get("total_reads") or 0
        feas = p.get("structurally_feasible_reads")
        share = f"{feas}/{reads}" if reads else "-"
        print(f"  {name:<20} {fmt(p.get('raw_cost'), ',.2f'):>16} "
              f"{fmt(g(s, 'final_solution', 'cost', 'total_cost'), ',.2f'):>16} "
              f"{fmt(p.get('raw_structural_violations')):>9} "
              f"{fmt(p.get('raw_c1_violations')):>7} {fmt(p.get('raw_c2_violations')):>7} "
              f"{fmt(p.get('raw_c3_violations')):>7} {share:>15} "
              f"{fmt(p.get('batches_feasible_from_sampler')):>7}/"
              f"{fmt(p.get('batches')):<5}")

        # The forced instance must come back clean.
        if name == "instances_10hubs":
            rv = p.get("raw_structural_violations")
            if rv not in (0, None):
                print(f"       ^^ WARNING: instances_10hubs is fully forced (one eligible hub "
                      f"per ZIP) yet returned {rv} raw violations. Investigate before "
                      f"trusting any other row.")
                problems += 1
        if g(s, "extra", "stopped_due_to_time_limit"):
            print(f"       ^^ WARNING: hit --qubo-time-limit; "
                  f"{p.get('batches')} of {g(s, 'extra', 'total_batches')} batches done. "
                  f"Costs are for a PARTIAL solution.")
            problems += 1

    # --- feasibility of the delivered solution -----------------------------
    print("\n2. FINAL SOLUTION  (after repair/trim/hub-prune)")
    h = (f"  {'instance':<20} {'open hubs':>10} {'stocked':>10} {'assignments':>12} "
         f"{'struct viol':>12} {'SLA viol':>9} {'overflow units':>15}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for name, s in rows:
        a = g(s, "final_solution", "audit", default={}) or {}
        print(f"  {name:<20} {fmt(g(s, 'final_solution', 'open_hubs_count')):>10} "
              f"{fmt(g(s, 'final_solution', 'stocked_pairs_count')):>10} "
              f"{fmt(g(s, 'final_solution', 'assignments_count')):>12} "
              f"{fmt(a.get('total_structural_violations')):>12} "
              f"{fmt(a.get('sla_distance_violations')):>9} "
              f"{fmt(a.get('c4_total_overflow_units')):>15}")

    # --- where the time went ----------------------------------------------
    print("\n3. SIZE AND TIME")
    h = (f"  {'instance':<20} {'binary vars':>12} {'batches':>8} {'construct s':>12} "
         f"{'anneal s':>10} {'eval s':>9} {'wall s':>10} {'anneal %':>9} {'peak RSS MB':>12}")
    print(h)
    print("  " + "-" * (len(h) - 2))
    for name, s in rows:
        p = g(s, "extra", "performance", default={}) or {}
        wall = p.get("total_wall_seconds") or 0
        anneal = p.get("annealing_seconds") or 0
        pct = (anneal / wall) if wall else None
        print(f"  {name:<20} {fmt(p.get('binary_variables')):>12} {fmt(p.get('batches')):>8} "
              f"{fmt(p.get('qubo_construction_seconds'), ',.2f'):>12} "
              f"{fmt(anneal, ',.2f'):>10} "
              f"{fmt(p.get('evaluation_seconds'), ',.2f'):>9} "
              f"{fmt(wall, ',.2f'):>10} {fmt(pct, '.1%'):>9} "
              f"{fmt(p.get('rss_peak_mb'), ',.1f'):>12}")

    # --- proof it ran on the card, plus the V3 settings actually used ------
    print("\n4. PROVENANCE AND SETTINGS")
    for name, s in rows:
        v = g(s, "extra", "va", default={}) or {}
        pa = g(s, "extra", "precision_audit", default={}) or {}
        print(f"  {name}")
        print(f"      execution      {v.get('execution_mode', '-')} on {v.get('hostname', '-')}"
              f"  |  cards {v.get('ve_card_count', '-')}"
              f"  |  devices {', '.join(v.get('ve_devices_visible') or []) or '-'}")
        print(f"      sampler        vector_mode={v.get('vector_mode', '-')}"
              f"  precision={v.get('precision', '-')}"
              f"  reads={v.get('num_reads_base', '-')}"
              f"  sweeps={v.get('num_sweeps', '-')}"
              f"  repeats={v.get('repeats', '-')}")
        print(f"      seed           seeded={v.get('seeded', '-')} seed={v.get('seed', '-')}")
        print(f"      penalties      objective_scale={v.get('objective_scale_enabled', '-')}"
              f"  min_penalty={fmt(v.get('min_penalty'), ',.0f')}"
              f"  adaptive={v.get('adaptive_penalty_mode', '-')}"
              f"  exits={v.get('adaptive_exit_reasons', '-')}")
        if pa:
            print(f"      fp32 audit     max|rel_diff|={pa.get('max_abs_rel_diff', '-')}"
                  f"  median={pa.get('median_abs_rel_diff', '-')}"
                  f"  over {fmt(pa.get('reads_audited'))} reads")
        if v.get("ve_card_count") in (0, None) or not v.get("ve_devices_visible"):
            print("      ^^ WARNING: no VE device recorded -- did this really run on the card?")
            problems += 1
        if v.get("execution_mode") != "local_ve_card":
            print(f"      ^^ WARNING: execution_mode is {v.get('execution_mode')!r}, "
                  f"expected 'local_ve_card'")
            problems += 1

    print()
    if problems:
        print(f">>> {problems} thing(s) flagged above. Read those before using these numbers.")
    else:
        print(">>> Nothing flagged. raw_* is the column that describes the annealer.")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="results/va_ladder",
                    help="Directory holding the va_run_* folders.")
    ap.add_argument("--run-root", default="",
                    help="Report this run root specifically instead of the newest.")
    ap.add_argument("--all", action="store_true", help="Report every run root found.")
    ap.add_argument("--json", action="store_true",
                    help="Emit the collected summaries as JSON instead of a table.")
    args = ap.parse_args(argv)

    if args.run_root:
        roots = [Path(args.run_root)]
    else:
        roots = find_run_roots(args.base)
        if not roots:
            print(f"No va_run_* folders under {args.base}.", file=sys.stderr)
            print("If the job is still queued or running, check:", file=sys.stderr)
            print("    squeue -u $USER", file=sys.stderr)
            print("    tail -f \"$(ls -t logs/va_solve_*.out | head -1)\"", file=sys.stderr)
            return 1
        if not args.all:
            roots = roots[:1]

    if args.json:
        blob = {str(r): {n: s for n, s in load_summaries(r)} for r in roots}
        print(json.dumps(blob, indent=2))
        return 0

    problems = 0
    for root in roots:
        problems += report(root)
        print()
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
