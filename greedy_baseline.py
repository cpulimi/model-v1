#!/usr/bin/env python3
"""Classical greedy baseline for the FSL hub-siting problem.

WHY THIS EXISTS
---------------
The ladder so far has two points per instance: Gurobi's proven optimum and VA's
answer. That pair tells you the gap but not whether the gap is *impressive*. A
0.46% gap is a strong result if a naive heuristic lands 8% out, and an
embarrassing one if nearest-hub assignment -- five lines of pandas, 0.2 seconds
-- also lands at 0.46%. Without the third point the comparison cannot be read.

So this is the third point, and it is deliberately built to be STRONG rather
than a strawman. A baseline that has been hobbled flatters the annealer, which
is the one outcome that would make the whole study worthless. Four escalating
variants are provided; report the best one.

  nearest   Assign each (zip,part) row to its closest eligible hub. Ignores cost
            entirely. This is the "what would the warehouse planner do" answer.
  marginal  Assign each row to whichever hub minimises the TRUE marginal cost at
            the moment it is placed -- transport plus, if this is the first time
            the hub stocks the part, the inventory price and new-hub transfer,
            plus the fixed-open charge if the hub is not yet open. Order matters
            because the state moves, so rows are processed largest-demand-first
            (the standard greedy tie-break: commit the expensive decisions while
            the most options are still cheap).
  1opt      marginal, then sweep every row and move it if the move lowers the
            TOTAL cost, crediting stock and hubs that fall out of use. Repeat to
            a fixed point.
  close     1opt, then try closing each open hub outright, reassigning its rows
            to their next-best eligible hub, keeping the closure only if the
            $500,000 saved beats the added transport and duplicated inventory.

COST ACCOUNTING IS NOT REIMPLEMENTED HERE. Every number is produced by the
solver's own load_problem_data() and compute_solution_cost(), the same functions
that score the VA output, so a difference between the two is a difference in
solution quality and never in bookkeeping. global_audit() is run on the result
for the same reason -- a greedy solution that quietly violated C1 would look
like a brilliant lower bound.

    python3 greedy_baseline.py --dataset-dir instances_20hubs
    python3 greedy_baseline.py --all --out results/greedy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_va_fsl_solver as solver  # noqa: E402

BAR = "=" * 86

# Proven optima from results/gurobi_optima/gurobi_optima.csv (all four MIPs
# closed at gap 0.0, so these are exact, not incumbents).
GUROBI_OPTIMA_CSV = Path("results/gurobi_optima/gurobi_optima.csv")

VARIANTS = ("nearest", "marginal", "1opt", "close")


# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------


class Solution:
    """Assignments plus the stock/open sets they imply.

    Only `assign` is a free decision. `stocked` and `open_hubs` are DERIVED --
    C2 forces a hub to stock any part it serves and C3 forces it open if it
    stocks anything -- so they are recomputed from `assign` rather than tracked
    independently, which is how the two fall out of sync.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.assign: dict[tuple[str, str], str] = {}   # (zip, part) -> hub

    # -- derived sets -----------------------------------------------------
    def stocked(self) -> set[tuple[str, str]]:
        return {(j, k) for (_, k), j in self.assign.items()}

    def open_hubs(self) -> set[str]:
        return {j for j in self.assign.values()}

    def triples(self) -> list[tuple[str, str, str]]:
        return sorted((i, j, k) for (i, k), j in self.assign.items())

    # -- scoring ----------------------------------------------------------
    def cost(self) -> dict[str, float]:
        return solver.compute_solution_cost(
            self.triples(), sorted(self.stocked()), sorted(self.open_hubs()), self.data
        )

    def audit(self) -> dict[str, int]:
        return solver.global_audit(
            self.triples(), sorted(self.stocked()), sorted(self.open_hubs()), self.data
        )

    def copy(self) -> "Solution":
        s = Solution(self.data)
        s.assign = dict(self.assign)
        return s


# ---------------------------------------------------------------------------
# Incremental cost bookkeeping
# ---------------------------------------------------------------------------


def transport_of(i: str, j: str, k: str, data: dict[str, Any]) -> float:
    return float(solver.assignment_cost(i, j, k, data)["assignment_cost"])


def stock_charge(j: str, k: str, data: dict[str, Any]) -> float:
    """Inventory price of part k plus the one-off transfer for a NEW hub.

    T_j == 1 means the hub already exists, so no $50 transfer is charged. This
    mirrors the (1 - T_j) * C term in compute_solution_cost exactly.
    """
    scalar = data["scalar"]
    return float(data["P"].get(k, 0.0)) + (1 - int(data["T"].get(j, 0))) * float(scalar["C"])


# ---------------------------------------------------------------------------
# Variant 1: nearest hub
# ---------------------------------------------------------------------------


def greedy_nearest(data: dict[str, Any]) -> Solution:
    """Closest eligible hub, every row, no cost reasoning at all.

    zip_to_hubs is already sorted by (distance, hub_id) in load_problem_data, so
    element 0 is the nearest with a deterministic tie-break.
    """
    sol = Solution(data)
    for i, k in active_rows(data):
        cands = data["zip_to_hubs"].get(i)
        if not cands:
            raise ValueError(f"ZIP {i} has no eligible hub; load_problem_data should have caught this")
        sol.assign[(i, k)] = cands[0][0]
    return sol


# ---------------------------------------------------------------------------
# Variant 2: marginal-cost greedy
# ---------------------------------------------------------------------------


def active_rows(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Active (zip, part) rows, sorted. Sorted so runs are bit-identical."""
    return sorted(
        (str(r.zip_id), str(r.part_id))
        for r in data["active"][["zip_id", "part_id"]].itertuples(index=False)
    )


def greedy_marginal(data: dict[str, Any]) -> Solution:
    """Place rows largest-demand-first at whichever hub is cheapest right now.

    "Right now" is the point: the first row to use a hub pays the $500,000 fixed
    charge and the part's inventory price, and every later row that lands on the
    same (hub, part) pays only transport. So the order of placement changes the
    answer, and descending b_ik puts the rows with the most transport at stake in
    front while the fewest hubs are committed.
    """
    scalar = data["scalar"]
    s_lim = float(scalar["S_lim"])
    s_var = float(scalar["S_var"])
    l_cap = int(scalar["L"])

    rows = active_rows(data)
    # Descending demand, then (zip, part) so equal demand never depends on
    # dict order. b_ik drives the transport term, so this is "expensive first".
    rows.sort(key=lambda p: (-float(data["B"].get(p, 0.0)), p[0], p[1]))

    sol = Solution(data)
    stocked: set[tuple[str, str]] = set()
    open_hubs: set[str] = set()
    stock_count: dict[str, int] = defaultdict(int)

    for i, k in rows:
        best_hub, best_delta = None, math.inf
        for j, _dij in data["zip_to_hubs"][i]:
            delta = transport_of(i, j, k, data)
            if (j, k) not in stocked:
                delta += stock_charge(j, k, data)
                # C4 overflow: only bites once a hub passes L stocked pairs.
                # L = 50,000 against a few thousand pairs, so this is inert on
                # the current instances -- kept so the baseline stays correct if
                # L is ever tightened to make the constraint bind.
                if stock_count[j] + 1 > l_cap:
                    delta += s_var
            if j not in open_hubs:
                delta += s_lim
            if delta < best_delta or (delta == best_delta and best_hub is not None and j < best_hub):
                best_hub, best_delta = j, delta

        assert best_hub is not None
        sol.assign[(i, k)] = best_hub
        if (best_hub, k) not in stocked:
            stocked.add((best_hub, k))
            stock_count[best_hub] += 1
        open_hubs.add(best_hub)

    return sol


# ---------------------------------------------------------------------------
# Variant 3: 1-opt local search
# ---------------------------------------------------------------------------


def _state(sol: Solution) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    """Reference counts: how many rows depend on each (hub,part) and each hub.

    A move is only free to drop a stock pair or close a hub when its count hits
    zero, and recomputing that from scratch per candidate move is what makes a
    naive 1-opt quadratic. These counters make each move O(1).
    """
    stock_refs: dict[tuple[str, str], int] = defaultdict(int)
    hub_refs: dict[str, int] = defaultdict(int)
    for (_, k), j in sol.assign.items():
        stock_refs[(j, k)] += 1
        hub_refs[j] += 1
    return stock_refs, hub_refs


def local_search_1opt(sol: Solution, max_passes: int = 20) -> tuple[Solution, dict[str, int]]:
    """Move single rows while any move lowers the total. Runs to a fixed point.

    The delta is exact, not approximate: leaving a hub refunds the inventory,
    transfer and fixed-open charges if this row was the last user of them, and
    arriving pays them if it is the first. That means 1-opt can close a hub as a
    side effect, which plain marginal greedy can never do.
    """
    data = sol.data
    scalar = data["scalar"]
    s_lim = float(scalar["S_lim"])
    stock_refs, hub_refs = _state(sol)
    stats = {"passes": 0, "moves": 0}

    for _pass in range(max_passes):
        moved = 0
        # Sorted iteration: the improvement sequence must not depend on dict order.
        for (i, k) in sorted(sol.assign):
            cur = sol.assign[(i, k)]
            # What leaving `cur` gives back.
            refund = transport_of(i, cur, k, data)
            if stock_refs[(cur, k)] == 1:
                refund += stock_charge(cur, k, data)
            if hub_refs[cur] == 1:
                refund += s_lim

            best_hub, best_gain = None, 1e-9  # strict improvement only
            for j, _dij in data["zip_to_hubs"][i]:
                if j == cur:
                    continue
                add = transport_of(i, j, k, data)
                if stock_refs[(j, k)] == 0:
                    add += stock_charge(j, k, data)
                if hub_refs[j] == 0:
                    add += s_lim
                gain = refund - add
                if gain > best_gain:
                    best_hub, best_gain = j, gain

            if best_hub is not None:
                stock_refs[(cur, k)] -= 1
                hub_refs[cur] -= 1
                stock_refs[(best_hub, k)] += 1
                hub_refs[best_hub] += 1
                sol.assign[(i, k)] = best_hub
                moved += 1

        stats["passes"] += 1
        stats["moves"] += moved
        if moved == 0:
            break
    return sol, stats


# ---------------------------------------------------------------------------
# Variant 4: hub-closure pass
# ---------------------------------------------------------------------------


def close_hubs(sol: Solution, max_passes: int = 5) -> tuple[Solution, dict[str, int]]:
    """Try emptying each open hub; keep the closure only if the total drops.

    Closing hub j saves $500,000 plus the inventory and transfer charge of every
    (j, part) pair it holds, and costs the extra transport of pushing its rows to
    their next-best hub plus any inventory that now has to be duplicated there.
    Gurobi closes 3 of 100 hubs on the largest instance, so this is a real move
    and not a formality -- without it the baseline cannot even represent the
    shape of the optimal solution.

    A hub holding a row that can reach nowhere else is not closeable at any price.
    """
    data = sol.data
    s_lim = float(data["scalar"]["S_lim"])
    stats = {"passes": 0, "closed": 0}

    for _pass in range(max_passes):
        closed_this_pass = 0
        stock_refs, hub_refs = _state(sol)
        for j in sorted(h for h, n in hub_refs.items() if n > 0):
            rows = sorted((i, k) for (i, k), h in sol.assign.items() if h == j)
            if not rows:
                continue

            # Everything hub j costs today and would stop costing: its fixed
            # charge, every part it stocks, and the transport of its own rows.
            saving = math.fsum(
                [s_lim]
                + [stock_charge(j, k, data) for (jj, k) in sorted(stock_refs)
                   if jj == j and stock_refs[(jj, k)] > 0]
                + [transport_of(i, j, k, data) for (i, k) in rows]
            )

            # Rehome every row in a sandbox, so a rejected closure leaves no
            # trace. `new_stock` / `new_open` track commitments made WITHIN this
            # plan, so two rows landing on the same fresh (hub, part) pay for it
            # once rather than twice.
            added_terms: list[float] = []
            new_stock: set[tuple[str, str]] = set()
            new_open: set[str] = set()
            plan: dict[tuple[str, str], str] = {}
            feasible = True

            for (i, k) in rows:
                best_h, best_c = None, math.inf
                for h, _dij in data["zip_to_hubs"][i]:
                    if h == j:
                        continue
                    c = transport_of(i, h, k, data)
                    if stock_refs.get((h, k), 0) == 0 and (h, k) not in new_stock:
                        c += stock_charge(h, k, data)
                    if hub_refs.get(h, 0) == 0 and h not in new_open:
                        c += s_lim
                    if c < best_c:
                        best_h, best_c = h, c
                if best_h is None:
                    feasible = False  # this row can only be served by j
                    break
                plan[(i, k)] = best_h
                added_terms.append(best_c)
                new_stock.add((best_h, k))
                new_open.add(best_h)

            if not feasible:
                continue

            if saving - math.fsum(added_terms) > 1e-9:
                sol.assign.update(plan)
                stock_refs, hub_refs = _state(sol)
                closed_this_pass += 1

        stats["passes"] += 1
        stats["closed"] += closed_this_pass
        if closed_this_pass == 0:
            break
    return sol, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_optima() -> dict[str, float]:
    out: dict[str, float] = {}
    if not GUROBI_OPTIMA_CSV.is_file():
        return out
    with open(GUROBI_OPTIMA_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("proven_optimal", "")).strip().lower() == "true":
                out[str(r["instance"])] = float(r["total_cost"])
    return out


def run_instance(dataset_dir: str, variants: tuple[str, ...], out_root: Path) -> dict[str, Any]:
    print(BAR)
    print(f"GREEDY BASELINE -- {dataset_dir}")
    print(BAR)

    t_load = time.time()
    data = solver.load_problem_data(
        dataset_dir,
        max_service_miles_override=None,
        penalty_start_miles_override=None,
        top_hubs_per_zip=None,
        max_parts_total=None,
    )
    load_s = time.time() - t_load
    rows = active_rows(data)
    print(f"  hubs {len(data['J'])}  zips {len(data['zips'])}  "
          f"active rows {len(rows):,}  loaded in {load_s:,.1f}s")

    optima = load_optima()
    opt = optima.get(Path(dataset_dir).name)

    results: list[dict[str, Any]] = []

    # nearest -------------------------------------------------------------
    if "nearest" in variants:
        t0 = time.time()
        sol = greedy_nearest(data)
        results.append(score(sol, "nearest", time.time() - t0, opt, {}))

    # marginal, and everything built on it --------------------------------
    need_marginal = any(v in variants for v in ("marginal", "1opt", "close"))
    if need_marginal:
        t0 = time.time()
        marg = greedy_marginal(data)
        marg_s = time.time() - t0
        if "marginal" in variants:
            results.append(score(marg, "marginal", marg_s, opt, {}))

        if "1opt" in variants or "close" in variants:
            t0 = time.time()
            one, st = local_search_1opt(marg.copy())
            one_s = marg_s + (time.time() - t0)
            if "1opt" in variants:
                results.append(score(one, "1opt", one_s, opt, st))

            if "close" in variants:
                t0 = time.time()
                cl, st2 = close_hubs(one.copy())
                # A closure frees rows to move again, so re-run 1-opt after it.
                cl, st3 = local_search_1opt(cl)
                close_s = one_s + (time.time() - t0)
                results.append(score(cl, "close", close_s, opt,
                                     {**st2, "post_close_moves": st3["moves"]}))

    out_dir = out_root / Path(dataset_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "dataset_name": Path(dataset_dir).name,
        "hubs": len(data["J"]),
        "active_rows": len(rows),
        "gurobi_optimum": opt,
        "load_seconds": load_s,
        "variants": results,
    }
    with open(out_dir / "greedy_summary.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print_table(results, opt)
    print(f"  -> {out_dir / 'greedy_summary.json'}")
    return payload


def score(sol: Solution, name: str, seconds: float, opt: float | None,
          extra: dict[str, Any]) -> dict[str, Any]:
    cost = sol.cost()
    audit = sol.audit()
    row: dict[str, Any] = {
        "variant": name,
        "seconds": round(seconds, 3),
        "total_cost": cost["total_cost"],
        "open_hubs": len(sol.open_hubs()),
        "stocked_pairs": len(sol.stocked()),
        "assignments": len(sol.assign),
        "structural_violations": audit["total_structural_violations"],
        "sla_violations": audit["sla_distance_violations"],
        "cost_breakdown": cost,
        **extra,
    }
    if opt:
        row["gap_pct"] = 100.0 * (cost["total_cost"] - opt) / opt
    return row


def print_table(results: list[dict[str, Any]], opt: float | None) -> None:
    print()
    head = f"  {'variant':10} {'total cost':>18} {'gap%':>8} {'hubs':>5} {'stock':>7} {'viol':>5} {'sec':>8}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    if opt:
        print(f"  {'GUROBI':10} {opt:18,.2f} {0.0:8.3f} {'-':>5} {'-':>7} {'-':>5} {'-':>8}")
    for r in results:
        gap = f"{r['gap_pct']:8.3f}" if "gap_pct" in r else f"{'-':>8}"
        print(f"  {r['variant']:10} {r['total_cost']:18,.2f} {gap} {r['open_hubs']:5} "
              f"{r['stocked_pairs']:7,} {r['structural_violations']:5} {r['seconds']:8,.2f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", action="append", default=[],
                    help="Instance directory. Repeat for several.")
    ap.add_argument("--all", action="store_true",
                    help="Run the whole 10/20/50/100-hub ladder.")
    ap.add_argument("--out", default="results/greedy")
    ap.add_argument("--variants", default="nearest,marginal,1opt,close",
                    help=f"Comma-separated subset of {','.join(VARIANTS)}.")
    a = ap.parse_args()

    datasets = list(a.dataset_dir)
    if a.all:
        datasets = ["instances_10hubs", "instances_20hubs",
                    "instances_50hubs", "instances_100hubs"]
    if not datasets:
        datasets = ["instances_20hubs"]

    variants = tuple(v.strip() for v in a.variants.split(",") if v.strip())
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        ap.error(f"unknown variant(s) {bad}; choose from {VARIANTS}")

    out_root = Path(a.out).expanduser().resolve()
    all_payloads = [run_instance(d, variants, out_root) for d in datasets]

    # One tidy CSV across every instance and variant -- this is what the
    # comparison chart is built from.
    csv_path = out_root / "greedy_ladder.csv"
    fields = ["dataset_name", "hubs", "active_rows", "variant", "total_cost",
              "gurobi_optimum", "gap_pct", "open_hubs", "stocked_pairs",
              "assignments", "structural_violations", "sla_violations", "seconds"]
    out_root.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for p in all_payloads:
            for v in p["variants"]:
                w.writerow({"dataset_name": p["dataset_name"], "hubs": p["hubs"],
                            "active_rows": p["active_rows"],
                            "gurobi_optimum": p["gurobi_optimum"], **v})
    print(f"\n  LADDER CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
