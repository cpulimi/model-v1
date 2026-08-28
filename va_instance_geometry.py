#!/usr/bin/env python3
"""Is the open-hub outcome FORCED by instance geometry, or CHOSEN by the solver?

A degenerate instance makes a solver look good or bad for reasons that have
nothing to do with the solver. If one hub can reach essentially all demand, then
opening one hub is simply correct and no annealer deserves credit for finding it.
If instead many rows have exactly one reachable hub, then those hubs are forced
open by geometry and no annealer could have closed them. Either way the result
is a property of the instance, not of the optimiser.

This measures that directly, with no solver and no card:

  1. Per demand row, how many hubs are reachable.
  2. Per hub, what fraction of demand it could serve.
  3. The smallest hub set covering all demand (greedy -- an UPPER bound), and
     the count of hubs that are the sole option for some row (an EXACT lower
     bound on how many must be open).

TWO RADII, because the model has two and they answer different questions:

  * service radius (`max_service_miles`) -- HARD feasibility. `zip_to_hubs` is
    built from it, so it defines what the solver may choose at all.
  * SLA radius (`penalty_start_miles`) -- the SOFT threshold. Serving beyond it
    is allowed but counts as an SLA violation in the audit.

Both are reported. The SLA radius is the headline because that is the promise
the network is supposed to keep; the service radius is what the solver is
actually constrained by.

Uses the solver's own load_problem_data(), so every number here is directly
comparable to what a run produced.

    python3 va_instance_geometry.py --dataset-dir instances_10hubs \\
                                    --dataset-dir instances_20hubs
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_va_fsl_solver as solver  # noqa: E402

BAR = "=" * 84


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def candidates_within(
    data: dict, radius: float
) -> dict[str, set[str]]:
    """zip_id -> hubs within `radius` miles. Subset of the loader's zip_to_hubs."""
    out: dict[str, set[str]] = {}
    for zip_id, hubs in data["zip_to_hubs"].items():
        out[str(zip_id)] = {str(j) for j, d in hubs if float(d) <= radius}
    return out


def greedy_set_cover(
    rows: list[tuple[str, str]], cand: dict[str, set[str]]
) -> tuple[list[str], int]:
    """Fewest hubs covering every row. GREEDY: an upper bound, not the optimum.

    Set cover is NP-hard; greedy is within a ln(n) factor and is the standard
    quick answer. Reported as "at most N", never as "the minimum".
    Returns (chosen hubs in pick order, rows left uncovered).
    """
    uncovered = {r for r in rows if cand.get(r[0])}
    unreachable = len(rows) - len(uncovered)
    by_hub: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in uncovered:
        for j in cand[row[0]]:
            by_hub[j].add(row)

    chosen: list[str] = []
    while uncovered:
        best, gain = None, 0
        for j, covered in by_hub.items():
            g = len(covered & uncovered)
            if g > gain or (g == gain and best is not None and g > 0 and j < best):
                best, gain = j, g
        if not best or gain == 0:
            break
        chosen.append(best)
        uncovered -= by_hub[best]
    return chosen, len(uncovered) + unreachable


def analyse_radius(
    label: str, radius: float, rows: list[tuple[str, str]], data: dict
) -> dict:
    cand = candidates_within(data, radius)
    counts = [len(cand.get(i, ())) for i, _ in rows]
    n = len(rows)
    all_hubs = [str(j) for j in data["J"]]

    # Per-hub reach.
    reach: dict[str, int] = {j: 0 for j in all_hubs}
    for i, _ in rows:
        for j in cand.get(i, ()):
            reach[j] += 1
    top = sorted(reach.items(), key=lambda kv: (-kv[1], kv[0]))

    # Hubs that are the ONLY option for some row must be open. Exact, not greedy.
    forced: set[str] = set()
    for i, _ in rows:
        c = cand.get(i, set())
        if len(c) == 1:
            forced |= c

    chosen, uncovered = greedy_set_cover(rows, cand)

    print(f"\n  {label} = {radius:,.1f} miles")
    print("  " + "-" * 80)
    zero = sum(1 for c in counts if c == 0)
    one = sum(1 for c in counts if c == 1)
    two_or_fewer = sum(1 for c in counts if 1 <= c <= 2)
    print(f"    reachable hubs per demand row: min={min(counts)} "
          f"median={statistics.median(counts):.0f} max={max(counts)} "
          f"mean={statistics.mean(counts):.2f}")
    print(f"      rows with 0 reachable hubs:  {zero:>7,} ({pct(zero / n)})"
          + ("   <-- UNSERVABLE at this radius" if zero else ""))
    print(f"      rows with exactly 1:         {one:>7,} ({pct(one / n)})")
    print(f"      rows with 2 or fewer:        {two_or_fewer:>7,} ({pct(two_or_fewer / n)})")

    print(f"    top 5 hubs by demand coverage (of {n:,} rows):")
    for j, c in top[:5]:
        print(f"      {j:<22} {c:>7,} rows  {pct(c / n):>7}")

    print(f"    hubs FORCED open (sole option for >=1 row): {len(forced)} of {len(all_hubs)}"
          + (f"  -> {', '.join(sorted(forced)[:6])}" if forced else ""))
    print(f"    greedy cover (UPPER bound, not exact): {len(chosen)} hub(s)"
          + (f", {uncovered:,} row(s) still uncovered" if uncovered else ", covers all rows"))
    if chosen:
        print(f"      pick order: {', '.join(chosen[:8])}"
              + (" ..." if len(chosen) > 8 else ""))
    return {
        "radius": radius, "n_rows": n, "top_share": (top[0][1] / n) if top else 0.0,
        "top_hub": top[0][0] if top else None, "forced": len(forced),
        "n_hubs": len(all_hubs), "greedy": len(chosen), "uncovered": uncovered,
        "frac_one": one / n, "frac_zero": zero / n,
    }


def cross_check(result: dict, summary_path: str) -> None:
    """Validate this analysis against what a real run actually reported.

    A row with NO hub inside the SLA radius cannot be served without an SLA
    violation, whatever the solver does. So the count of such rows is a hard
    FLOOR on the run's sla_distance_violations. If the run reports fewer, this
    script and the solver disagree about the geometry and one of them is wrong.
    Anything above the floor is a violation the solver chose.
    """
    try:
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        actual = int(summary["final_solution"]["audit"]["sla_distance_violations"])
        opened = int(summary["final_solution"]["open_hubs_count"])
    except Exception as exc:
        print(f"    cross-check unavailable ({type(exc).__name__}: {exc})")
        return

    floor = int(round(result["sla"]["frac_zero"] * result["n_rows"]))
    forced = result["service"]["forced"]
    print(f"\n  cross-check against {summary_path}")
    print("  " + "-" * 80)
    print(f"    run reported SLA violations:        {actual:,}")
    print(f"    geometric floor (0 hubs in radius): {floor:,}")
    if actual < floor:
        print(f"    *** DISAGREEMENT: run reports FEWER violations than geometry allows.")
    else:
        chosen = actual - floor
        print(f"    unavoidable: {floor:,} ({pct(floor / max(1, actual))} of them) | "
              f"solver-attributable: {chosen:,}")
    print(f"    run opened {opened} hub(s); geometry forces {forced} open"
          + ("  -> consistent, the solver had no choice"
             if opened == forced else f"  -> {opened - forced} were a real choice"))


def run_instance(dataset_dir: str, summary: str | None = None) -> dict:
    data = solver.load_problem_data(
        dataset_dir,
        max_service_miles_override=None,
        penalty_start_miles_override=None,
        top_hubs_per_zip=None,     # keep every hub within the service radius
        max_parts_total=None,
    )
    scalar = data["scalar"]
    active = data["active"]
    rows = [(str(z), str(k)) for z, k in
            zip(active["zip_id"].tolist(), active["part_id"].tolist())]

    print("\n" + BAR)
    print(f"INSTANCE GEOMETRY -- {data['dataset_name']}")
    print(BAR)
    print(f"  hubs {len(data['J']):,} | zips {len(data['zips']):,} | parts {len(data['K']):,} "
          f"| active demand rows {len(rows):,}")
    print(f"  service radius (max_service_miles, HARD): {float(scalar['max_service_miles']):,.1f} mi")
    print(f"  SLA radius (penalty_start_miles, SOFT):   {float(scalar['penalty_start_miles']):,.1f} mi")
    if data.get("top_hubs_per_zip"):
        print(f"  NOTE: top_hubs_per_zip={data['top_hubs_per_zip']} would truncate candidates; "
              f"this report keeps ALL hubs within the service radius.")

    sla = analyse_radius("SLA radius", float(scalar["penalty_start_miles"]), rows, data)
    svc = analyse_radius("service radius", float(scalar["max_service_miles"]), rows, data)
    result = {"name": data["dataset_name"], "sla": sla, "service": svc, "n_rows": len(rows)}
    if summary:
        cross_check(result, summary)
    return result


def verdict(results: list[dict]) -> None:
    print("\n" + BAR)
    print("VERDICT")
    print(BAR)
    for r in results:
        sla, svc = r["sla"], r["service"]
        print(f"\n  {r['name']}  ({r['n_rows']:,} demand rows, {svc['n_hubs']} hubs)")
        print(f"    best single hub reaches {pct(svc['top_share'])} of demand within the "
              f"service radius ({svc['top_hub']})")
        print(f"    best single hub reaches {pct(sla['top_share'])} within the SLA radius")

        one_hub_suffices = svc["top_share"] >= 0.99 and svc["uncovered"] == 0
        if one_hub_suffices:
            print("    => ONE HUB COVERS ESSENTIALLY ALL DEMAND.")
            print("       A single open hub is the CORRECT answer here. This is an")
            print("       INSTANCE-DESIGN issue, not a solver issue: the instance offers")
            print("       no siting decision to get right.")
        else:
            print(f"    => No single hub covers all demand "
                  f"(best is {pct(svc['top_share'])}); at least "
                  f"{svc['forced']} hub(s) are FORCED open because some row can be")
            print(f"       served by that hub alone. Greedy needs <= {svc['greedy']} hub(s).")
            if svc["forced"] >= svc["n_hubs"]:
                print("       EVERY hub is forced open by geometry. The solver had NO")
                print("       siting choice to make -- opening all hubs is the only")
                print("       feasible answer, so 'all hubs open' says nothing about")
                print("       solution quality. INSTANCE-DESIGN issue, not a solver issue.")
            elif svc["forced"] > 0:
                print(f"       {svc['forced']} of {svc['n_hubs']} forced; the remaining "
                      f"{svc['n_hubs'] - svc['forced']} are a genuine solver choice.")
            else:
                print("       No hub is forced. Hub siting is a real decision here and the")
                print("       solver's answer is meaningful. SOLVER result, not geometry.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", action="append", required=True,
                    help="Instance directory. Repeat for several.")
    ap.add_argument("--compare-run", action="append", default=[],
                    help="summary.json from a real run, to validate this analysis "
                         "against it. Paired positionally with --dataset-dir.")
    a = ap.parse_args()
    results = [
        run_instance(d, a.compare_run[i] if i < len(a.compare_run) else None)
        for i, d in enumerate(a.dataset_dir)
    ]
    verdict(results)
    print("\n" + BAR)
    print("Greedy set cover is an UPPER bound on the minimum hub count, not the optimum.")
    print("The FORCED count is exact: a hub that is the only option for some row must open.")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
