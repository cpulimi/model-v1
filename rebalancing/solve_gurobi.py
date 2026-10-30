"""Gurobi implementation of the BASE rebalancing model (constraints (2)-(5) only).

    min  sum C X + sum B T + UD sum U + FC1 sum Y + FC2 sum Z
    (2)  sum_j X[i,j,k]                          <= E[i,k]          for all i, k
    (3)  sum_i X[i,j,k] + sum_m T[m,j,k] + U[j,k] = N[j,k]          for all j, k with N > 0
    (4)  sum_{j,k} X[i,j,k]                      <= M_i Y[i]        M_i = sum_k E[i,k]
    (5)  sum_{i,k} X[i,j,k] + sum_{m,k} T[m,j,k] <= M_j Z[j]        M_j = sum_k N[j,k]

--strong-linking replaces (4)/(5) with the per-arc form
    X[i,j,k] <= min(E[i,k], N[j,k]) Y[i]
    X[i,j,k] <= min(E[i,k], N[j,k]) Z[j]
    T[m,j,k] <= N[j,k] Z[j]
which has the same integer feasible set (X <= E by (2), X <= N and T <= N by (3))
but a tighter LP relaxation.

Usage (from the repo root):
    python -m rebalancing.solve_gurobi --sites 10 --parts 1 --suppliers 5 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rebalancing.data import (  # noqa: E402
    Instance,
    generate_synthetic,
    load_instance,
    purchase_keys,
    ship_arcs,
    write_instance_csv,
)

STATUS_NAMES = {
    getattr(GRB.Status, n): n
    for n in dir(GRB.Status)
    if n.isupper() and isinstance(getattr(GRB.Status, n), int)
}


# ---------------------------------------------------------------------------
# Memory / provenance (same VmHWM quantity as run_aligned_fsl_comparison.py)
# ---------------------------------------------------------------------------
def vm_hwm_mb() -> float:
    """Kernel peak RSS from /proc/self/status VmHWM, in MB. Linux only; 0.0 elsewhere."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def ru_maxrss_mb() -> float:
    """getrusage peak RSS in MB (bytes on macOS, kB on Linux). Fallback only."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0


def git_info() -> dict:
    root = Path(__file__).resolve().parent.parent

    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "rebalancing_dirty": bool(run("status", "--porcelain", "--", "rebalancing")),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(inst: Instance, strong_linking: bool = False, env=None, free_purchase: str = "keep"):
    """Build the sparse model. Returns (model, vars, counts).

    Variables exist only where they can be nonzero:
        X  on ship_arcs(inst)          (E > 0, N > 0, i != j)
        T  on purchase_keys(inst)      (N > 0, every supplier; with
                                        free_purchase="block", minus keys whose B is 0)
        U  on N's keys                 (N > 0)
        Y  for sites with >= 1 outgoing arc
        Z  for sites with >= 1 incoming arc or purchase key
    A site with no Y (no Z) cannot ship (receive), so (4) ((5)) is vacuous there.
    Objective coefficients go in through addVars(obj=...), so no giant objective
    expression is ever built.
    """
    if free_purchase not in ("keep", "block"):
        raise ValueError(f"free_purchase must be 'keep' or 'block', got {free_purchase!r}")
    x_keys = ship_arcs(inst)
    t_keys = purchase_keys(inst)
    n_blocked = 0
    if free_purchase == "block":
        kept = [t for t in t_keys if inst.B[t] != 0]
        n_blocked = len(t_keys) - len(kept)
        t_keys = kept
    u_keys = list(inst.N)

    # Index lists, one pass over each key set.
    x_out_ik, x_in_jk = defaultdict(list), defaultdict(list)
    x_out_i, x_in_j = defaultdict(list), defaultdict(list)
    for a in x_keys:
        i, j, k = a
        x_out_ik[(i, k)].append(a)
        x_in_jk[(j, k)].append(a)
        x_out_i[i].append(a)
        x_in_j[j].append(a)
    t_in_jk, t_in_j = defaultdict(list), defaultdict(list)
    for t in t_keys:
        _, j, k = t
        t_in_jk[(j, k)].append(t)
        t_in_j[j].append(t)
    y_keys = [s for s in inst.sites if s in x_out_i]
    z_keys = [s for s in inst.sites if s in x_in_j or s in t_in_j]

    M_out = defaultdict(float)
    for (i, k), v in inst.E.items():
        M_out[i] += v
    M_in = defaultdict(float)
    for (j, k), v in inst.N.items():
        M_in[j] += v

    model = gp.Model("rebalancing_base", env=env)
    model.ModelSense = GRB.MINIMIZE
    X = model.addVars(
        x_keys, lb=0.0, vtype=GRB.INTEGER, obj=[inst.ship_cost(*a) for a in x_keys], name="X"
    )
    T = model.addVars(t_keys, lb=0.0, vtype=GRB.INTEGER, obj=[inst.B[t] for t in t_keys], name="T")
    U = model.addVars(u_keys, lb=0.0, vtype=GRB.INTEGER, obj=inst.UD, name="U")
    Y = model.addVars(y_keys, vtype=GRB.BINARY, obj=inst.FC1, name="Y")
    Z = model.addVars(z_keys, vtype=GRB.BINARY, obj=inst.FC2, name="Z")

    cons = {}
    # (2) excess -- only (i,k) with an outgoing arc; otherwise it reads 0 <= E.
    cons["c2_excess"] = model.addConstrs(
        (gp.quicksum(X[a] for a in x_out_ik[i, k]) <= inst.E[i, k] for (i, k) in x_out_ik),
        name="c2_excess",
    )
    # (3) need balance -- every (j,k) with N > 0.
    cons["c3_need"] = model.addConstrs(
        (
            gp.quicksum(X[a] for a in x_in_jk.get((j, k), ()))
            + gp.quicksum(T[t] for t in t_in_jk[j, k])
            + U[j, k]
            == inst.N[j, k]
            for (j, k) in u_keys
        ),
        name="c3_need",
    )
    if not strong_linking:
        # (4) / (5) aggregated big-M with the tight per-site M.
        cons["c4_ship_link"] = model.addConstrs(
            (gp.quicksum(X[a] for a in x_out_i[i]) <= M_out[i] * Y[i] for i in y_keys),
            name="c4_ship_link",
        )
        cons["c5_recv_link"] = model.addConstrs(
            (
                gp.quicksum(X[a] for a in x_in_j.get(j, ()))
                + gp.quicksum(T[t] for t in t_in_j.get(j, ()))
                <= M_in[j] * Z[j]
                for j in z_keys
            ),
            name="c5_recv_link",
        )
    else:
        cap = {a: min(inst.E[a[0], a[2]], inst.N[a[1], a[2]]) for a in x_keys}
        cons["c4s_ship_link_arc"] = model.addConstrs(
            (X[i, j, k] <= cap[i, j, k] * Y[i] for (i, j, k) in x_keys), name="c4s_ship_link_arc"
        )
        cons["c5s_recv_link_arc"] = model.addConstrs(
            (X[i, j, k] <= cap[i, j, k] * Z[j] for (i, j, k) in x_keys), name="c5s_recv_link_arc"
        )
        cons["c5s_recv_link_buy"] = model.addConstrs(
            (T[m, j, k] <= inst.N[j, k] * Z[j] for (m, j, k) in t_keys), name="c5s_recv_link_buy"
        )
    model.update()

    vars_ = {"X": X, "T": T, "U": U, "Y": Y, "Z": Z}
    counts = {
        "vars": {name: len(v) for name, v in vars_.items()},
        "constraints": {name: len(c) for name, c in cons.items()},
    }
    counts["vars"]["total"] = model.NumVars
    counts["vars"]["integer_nonbinary"] = len(X) + len(T) + len(U)
    counts["vars"]["binary"] = model.NumBinVars
    counts["vars"]["T_free_blocked"] = n_blocked
    counts["constraints"]["total"] = model.NumConstrs
    counts["nonzeros"] = model.NumNZs
    return model, vars_, counts


def extract_solution(model, vars_: dict) -> tuple:
    """Nonzero, integer-rounded solution plus the max deviation from integrality seen."""
    sol, max_frac = {}, 0.0
    for name, v in vars_.items():
        if len(v) == 0:
            sol[name] = {}
            continue
        vals = model.getAttr("X", v)
        out = {}
        for key, x in vals.items():
            r = round(x)
            max_frac = max(max_frac, abs(x - r))
            if r != 0:
                out[key] = int(r)
        sol[name] = out
    return sol, max_frac


def cost_breakdown(inst: Instance, sol: dict) -> dict:
    """The five objective terms evaluated on the rounded solution."""
    terms = {
        "ship": sum(inst.ship_cost(*a) * v for a, v in sol["X"].items()),
        "purchase": sum(inst.B[t] * v for t, v in sol["T"].items()),
        "unmet": inst.UD * sum(sol["U"].values()),
        "fixed_ship_FC1": inst.FC1 * sum(sol["Y"].values()),
        "fixed_receive_FC2": inst.FC2 * sum(sol["Z"].values()),
    }
    terms["total"] = sum(terms.values())
    return terms


def solve(
    inst: Instance,
    time_limit: float | None = None,
    mip_gap: float = 1e-4,
    threads: int | None = None,
    strong_linking: bool = False,
    log_file: str | None = None,
    mps_path: str | None = None,
    quiet: bool = False,
    free_purchase: str = "keep",
) -> dict:
    """Build, optionally write MPS, solve. Returns a result dict (no file output besides log/MPS)."""
    env = gp.Env(empty=True)
    if quiet:
        env.setParam("OutputFlag", 0)
    env.start()

    t0 = time.perf_counter()
    model, vars_, counts = build_model(
        inst, strong_linking=strong_linking, env=env, free_purchase=free_purchase
    )
    build_s = time.perf_counter() - t0
    hwm_after_build = vm_hwm_mb()

    if mps_path:
        model.write(str(mps_path))
    if log_file:
        model.Params.LogFile = str(log_file)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    if threads is not None:
        model.Params.Threads = threads

    t1 = time.perf_counter()
    model.optimize()
    solve_s = time.perf_counter() - t1

    status = STATUS_NAMES.get(model.Status, str(model.Status))
    has_sol = model.SolCount > 0
    try:
        obj_bound = model.ObjBound
    except gp.GurobiError:
        obj_bound = None
    res = {
        "status": status,
        "status_code": model.Status,
        "SolCount": model.SolCount,
        "ObjVal": model.ObjVal if has_sol else None,
        "ObjBound": obj_bound,
        "MIPGap": model.MIPGap if has_sol else None,
        "Runtime": model.Runtime,
        "NodeCount": model.NodeCount,
        "build_seconds": build_s,
        "solve_seconds": solve_s,
        "counts": counts,
        "vm_hwm_mb_after_build": hwm_after_build,
    }
    if has_sol:
        sol, max_frac = extract_solution(model, vars_)
        res["solution"] = sol
        res["max_integrality_deviation"] = max_frac
        res["cost_breakdown"] = cost_breakdown(inst, sol)
    else:
        res["solution"] = None
    model.dispose()
    env.dispose()
    return res


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_solution_csvs(run_dir: Path, sol: dict, breakdown: dict) -> None:
    specs = {
        "X": ["i", "j", "k"],
        "T": ["m", "j", "k"],
        "U": ["j", "k"],
        "Y": ["i"],
        "Z": ["j"],
    }
    for name, cols in specs.items():
        with open(run_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(cols + ["value"])
            for key, v in sorted((sol or {}).get(name, {}).items()):
                key = key if isinstance(key, tuple) else (key,)
                w.writerow([*key, v])
    with open(run_dir / "cost_breakdown.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["term", "value"])
        for term, v in (breakdown or {}).items():
            w.writerow([term, v])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default=None,
                   help="old-model workbook <city>_<n>_hubs/<seed>.xlsx instead of the synthetic generator")
    p.add_argument("--sites", type=int, default=None)
    p.add_argument("--parts", type=int, default=None)
    p.add_argument("--suppliers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--free-purchase", choices=("keep", "block"), default="keep",
                   help="block: drop T variables whose purchase cost is 0")
    p.add_argument("--fc1", type=float, default=100.0, help="fixed cost per shipping site (paper '102' read as 10^2)")
    p.add_argument("--fc2", type=float, default=1000.0, help="fixed cost per receiving site (paper '103' read as 10^3)")
    p.add_argument("--ud", type=float, default=10000.0, help="unmet cost per unit (paper '104' read as 10^4)")
    p.add_argument("--ship-cost-by-part", action="store_true", help="draw C[i,j,k] instead of C[i,j]")
    p.add_argument("--time-limit", type=float, default=None, help="seconds; default none")
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--strong-linking", action="store_true", help="per-arc linking instead of (4)/(5)")
    p.add_argument("--out-dir", default="results/rebalancing")
    p.add_argument("--run-name", default=None, help="default: derived from sizes/seed/flags")
    p.add_argument("--write-mps", action="store_true", help="also write model.mps (for CPLEX)")
    p.add_argument("--quiet", action="store_true", help="no Gurobi console log (log file still written)")
    args = p.parse_args(argv)
    sizes = (args.sites, args.parts, args.suppliers)
    if args.instance is None and None in sizes:
        p.error("give --instance PATH, or all of --sites --parts --suppliers")
    if args.instance is not None and any(s is not None for s in sizes):
        p.error("--instance and --sites/--parts/--suppliers are alternatives; give one")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    wall0 = time.perf_counter()

    t0 = time.perf_counter()
    if args.instance:
        inst = load_instance(args.instance, ud=args.ud, fc1=args.fc1, fc2=args.fc2)
    else:
        inst = generate_synthetic(
            args.sites,
            args.parts,
            args.suppliers,
            args.seed,
            fc1=args.fc1,
            fc2=args.fc2,
            ud=args.ud,
            ship_cost_by_part=args.ship_cost_by_part,
        )
    data_s = time.perf_counter() - t0

    suffix = ("_blockfree" if args.free_purchase == "block" else "") + ("_strong" if args.strong_linking else "")
    run_name = args.run_name or (inst.name + suffix)
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_instance_csv(inst, run_dir / "instance")

    res = solve(
        inst,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        strong_linking=args.strong_linking,
        log_file=run_dir / "gurobi.log",
        mps_path=(run_dir / "model.mps") if args.write_mps else None,
        quiet=args.quiet,
        free_purchase=args.free_purchase,
    )
    sol = res.pop("solution")
    write_solution_csvs(run_dir, sol, res.get("cost_breakdown"))

    hwm = vm_hwm_mb()
    summary = {
        **{k: res[k] for k in ("status", "status_code", "SolCount", "ObjVal", "ObjBound", "MIPGap", "Runtime", "NodeCount")},
        "num_vars": res["counts"]["vars"],
        "num_constraints": res["counts"]["constraints"],
        "num_nonzeros": res["counts"]["nonzeros"],
        "timing_seconds": {
            "data_load" if args.instance else "data_generation": data_s,
            "model_build": res["build_seconds"],
            "solve": res["solve_seconds"],
            "gurobi_runtime": res["Runtime"],
            "wall_total": time.perf_counter() - wall0,
        },
        "memory": {
            "peak_memory_mb": hwm if hwm > 0 else None,
            "peak_memory_source": "proc_self_status_vmhwm" if hwm > 0 else "unavailable_not_linux",
            "vm_hwm_mb_after_build": res["vm_hwm_mb_after_build"] or None,
            "ru_maxrss_mb_fallback": ru_maxrss_mb(),
        },
        "solution_stats": None,
        "cost_breakdown": res.get("cost_breakdown"),
        "max_integrality_deviation": res.get("max_integrality_deviation"),
        "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        "git": git_info(),
        "params": {
            **vars(args),
            "run_name": run_name,
            "MIPGap": args.mip_gap,
            "linking": "strong_per_arc" if args.strong_linking else "aggregated_per_site_tight_M",
        },
        "instance": {"name": inst.name, **inst.meta, "UD": inst.UD, "FC1": inst.FC1, "FC2": inst.FC2},
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    if sol is not None:
        summary["solution_stats"] = {
            "units_shipped": sum(sol["X"].values()),
            "units_bought": sum(sol["T"].values()),
            "units_bought_free": sum(v for t, v in sol["T"].items() if inst.B[t] == 0),
            "units_unmet": sum(sol["U"].values()),
            "total_need": sum(inst.N.values()),
            "total_excess": sum(inst.E.values()),
            "sites_shipping": len(sol["Y"]),
            "sites_receiving": len(sol["Z"]),
            "ship_arcs_used": len(sol["X"]),
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[{run_name}] {summary['status']}  ObjVal={summary['ObjVal']}  ObjBound={summary['ObjBound']}"
          f"  gap={summary['MIPGap']}  -> {run_dir}")
    return 0 if sol is not None else 1


if __name__ == "__main__":
    sys.exit(main())
