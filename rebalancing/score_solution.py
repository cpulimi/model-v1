"""Independent check of a solve_gurobi run -- no Gurobi import.

Reads the run's instance/ directory and the solution CSVs (X, T, U, Y, Z),
recomputes the objective term by term and checks every constraint of the base
model, plus the per-arc linking rows if the run used --strong-linking.

    python -m rebalancing.score_solution --run-dir results/rebalancing/<run>

Exit code 0 when there are no violations and the recomputed objective matches
summary.json's ObjVal to within --rtol (default 1e-6 relative); 1 otherwise.
Writes <run>/score.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rebalancing.data import Instance, read_instance_csv  # noqa: E402

TOL = 1e-6


def read_solution(run_dir: Path) -> dict:
    """{name: {key_tuple_or_id: value}} from the five solution CSVs."""
    sol = {}
    for name, nkey in (("X", 3), ("T", 3), ("U", 2), ("Y", 1), ("Z", 1)):
        out = {}
        with open(run_dir / f"{name}.csv", newline="") as fh:
            r = csv.reader(fh)
            next(r)
            for row in r:
                key = tuple(row[:nkey]) if nkey > 1 else row[0]
                if key in out:
                    raise ValueError(f"{name}.csv: duplicate key {key}")
                out[key] = float(row[nkey])
        sol[name] = out
    return sol


def score(inst: Instance, sol: dict, strong_linking: bool = False) -> dict:
    """Recompute the objective and list every violated constraint."""
    X, T, U, Y, Z = (sol[n] for n in ("X", "T", "U", "Y", "Z"))
    E, N = inst.E, inst.N
    viol = []

    # Domains: integer, nonnegative, binaries in {0,1}, and every variable on a
    # key the model actually creates.
    for name, d in sol.items():
        for key, v in d.items():
            if v < -TOL or abs(v - round(v)) > TOL:
                viol.append(f"domain: {name}{key} = {v} not a nonnegative integer")
            if name in ("Y", "Z") and v > 1 + TOL:
                viol.append(f"domain: {name}[{key}] = {v} > 1")
    for (i, j, k), v in X.items():
        if i == j or E.get((i, k), 0) <= 0 or N.get((j, k), 0) <= 0:
            viol.append(f"sparsity: X[{i},{j},{k}] = {v} is not an arc (needs E[i,k]>0, N[j,k]>0, i!=j)")
    for (m, j, k), v in T.items():
        if N.get((j, k), 0) <= 0 or m not in inst.suppliers:
            viol.append(f"sparsity: T[{m},{j},{k}] = {v} has no need or unknown supplier")
    for (j, k), v in U.items():
        if N.get((j, k), 0) <= 0:
            viol.append(f"sparsity: U[{j},{k}] = {v} has no need")

    out_ik, in_jk, out_i, in_j = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    for (i, j, k), v in X.items():
        out_ik[i, k] += v
        in_jk[j, k] += v
        out_i[i] += v
        in_j[j] += v
    buy_jk, buy_j = defaultdict(float), defaultdict(float)
    for (m, j, k), v in T.items():
        buy_jk[j, k] += v
        buy_j[j] += v

    # (2) sum_j X[i,j,k] <= E[i,k]
    for (i, k), s in out_ik.items():
        if s > E.get((i, k), 0) + TOL:
            viol.append(f"(2) excess: site {i} part {k} ships {s} > E = {E.get((i, k), 0)}")
    # (3) sum_i X + sum_m T + U = N for N > 0
    for (j, k), n in N.items():
        lhs = in_jk[j, k] + buy_jk[j, k] + U.get((j, k), 0.0)
        if abs(lhs - n) > TOL:
            viol.append(f"(3) need: site {j} part {k}: in {in_jk[j, k]} + buy {buy_jk[j, k]} "
                        f"+ unmet {U.get((j, k), 0.0)} = {lhs} != N = {n}")
    # (4) sum X out of i <= M_i Y[i],  M_i = sum_k E[i,k]
    M_out, M_in = defaultdict(float), defaultdict(float)
    for (i, k), v in E.items():
        M_out[i] += v
    for (j, k), v in N.items():
        M_in[j] += v
    for i, s in out_i.items():
        if s > M_out[i] * Y.get(i, 0.0) + TOL:
            viol.append(f"(4) ship link: site {i} ships {s} with Y = {Y.get(i, 0.0)} (M_i = {M_out[i]})")
    # (5) sum X into j + sum T into j <= M_j Z[j],  M_j = sum_k N[j,k]
    for j in set(in_j) | set(buy_j):
        s = in_j[j] + buy_j[j]
        if s > M_in[j] * Z.get(j, 0.0) + TOL:
            viol.append(f"(5) recv link: site {j} receives {s} with Z = {Z.get(j, 0.0)} (M_j = {M_in[j]})")
    if strong_linking:
        for (i, j, k), v in X.items():
            cap = min(E.get((i, k), 0), N.get((j, k), 0))
            if v > cap * Y.get(i, 0.0) + TOL or v > cap * Z.get(j, 0.0) + TOL:
                viol.append(f"(4s/5s) arc link: X[{i},{j},{k}] = {v}, cap {cap}, "
                            f"Y = {Y.get(i, 0.0)}, Z = {Z.get(j, 0.0)}")
        for (m, j, k), v in T.items():
            if v > N.get((j, k), 0) * Z.get(j, 0.0) + TOL:
                viol.append(f"(5s) buy link: T[{m},{j},{k}] = {v}, Z = {Z.get(j, 0.0)}")

    # Objective, term by term.
    terms = {
        "ship": sum(inst.ship_cost(i, j, k) * v for (i, j, k), v in X.items()),
        "purchase": sum(inst.B[t] * v for t, v in T.items()),
        "unmet": inst.UD * sum(U.values()),
        "fixed_ship_FC1": inst.FC1 * sum(Y.values()),
        "fixed_receive_FC2": inst.FC2 * sum(Z.values()),
    }
    terms["total"] = sum(terms.values())

    # Informational: binaries switched on with no flow are feasible but wasteful.
    idle = [f"Y[{i}]" for i, v in Y.items() if v > 0.5 and out_i.get(i, 0) <= TOL]
    idle += [f"Z[{j}]" for j, v in Z.items() if v > 0.5 and in_j.get(j, 0) + buy_j.get(j, 0) <= TOL]

    return {"objective": terms, "violations": viol, "idle_binaries": idle}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--instance-dir", default=None, help="default: <run-dir>/instance")
    p.add_argument("--rtol", type=float, default=1e-6)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    inst = read_instance_csv(args.instance_dir or run_dir / "instance")
    summary = json.loads((run_dir / "summary.json").read_text())
    strong = summary.get("params", {}).get("strong_linking", False)
    sol = read_solution(run_dir)
    res = score(inst, sol, strong_linking=strong)

    obj = res["objective"]["total"]
    ref = summary.get("ObjVal")
    rel = None if ref is None else abs(obj - ref) / max(1.0, abs(ref))
    match = rel is not None and rel <= args.rtol
    ok = match and not res["violations"]

    print(f"run: {run_dir}")
    for term, v in res["objective"].items():
        print(f"  {term:<18} {v:>16,.4f}")
    print(f"  Gurobi ObjVal      {ref if ref is None else f'{ref:>16,.4f}'}")
    print(f"  relative diff      {rel}   (rtol {args.rtol})  -> {'MATCH' if match else 'MISMATCH'}")
    if res["violations"]:
        print(f"  {len(res['violations'])} VIOLATIONS:")
        for v in res["violations"]:
            print("    " + v)
    else:
        print("  constraints (2)-(5)" + (" + per-arc linking" if strong else "") + ": all satisfied")
    if res["idle_binaries"]:
        print(f"  note: binaries on with no flow: {', '.join(res['idle_binaries'])}")
    print("PASS" if ok else "FAIL")

    (run_dir / "score.json").write_text(json.dumps({
        "pass": ok,
        "objective_recomputed": res["objective"],
        "objval_gurobi": ref,
        "relative_diff": rel,
        "rtol": args.rtol,
        "violations": res["violations"],
        "idle_binaries": res["idle_binaries"],
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
