"""Run all of Aarav's old-model datasets through the base model and check the optima.

For every <city>_<n>_hubs/<seed>.xlsx in the OldModelDatasets repo, and for both
--free-purchase keep and block, this solves with --mip-gap 0, FC1=100, FC2=1000,
UD=10000, runs score_solution on the result, and compares ObjVal with the
reference optima (column highs_obj; variant as_given = keep,
no_free_purchase = block -- block is the same model as the reference's
"price at 175", because a cost-175 purchase option is never the cheapest).

    python -m rebalancing.run_old_datasets
    python -m rebalancing.run_old_datasets --data-root ../OldModelDatasets --threads 4

Per-run folders go to --out-dir (not committed). One summary CSV with every
run, the reference values, the match flags and per-instance data checks goes to
--summary-csv. .mps files for --mps-seeds (keep variant) are also copied to
<out-dir>/mps/ for solving the identical models in CPLEX.

Exit code 0 only if every ObjVal matches its reference exactly and every run
passes score_solution.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import shutil
import statistics
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rebalancing import score_solution, solve_gurobi  # noqa: E402
from rebalancing.data import load_instance, purchase_keys  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VARIANT_REF = {"keep": "as_given", "block": "no_free_purchase"}
FC1, FC2, UD = 100.0, 1000.0, 10000.0


def data_checks(inst) -> dict:
    """Per-instance facts about the workbook, for spotting anything unexpected."""
    hubs = inst.sites
    e_hubs = {i for i, _ in inst.E}
    n_hubs = {j for j, _ in inst.N}
    b_nonzero = [v for v in inst.B.values() if v != 0]
    c_vals = list(inst.C.values())
    asym = sum(
        1 for (i, j, k), v in inst.C.items()
        if int(i) < int(j) and inst.C.get((j, i, k)) not in (None, v)
    )
    free_needy = sorted({j for (m, j, k) in purchase_keys(inst) if inst.B[m, j, k] == 0}, key=int)
    return {
        "hubs": len(hubs),
        "hubs_label": inst.meta.get("hubs_label"),
        "hubs_excess": len(e_hubs),
        "hubs_need": len(n_hubs),
        "hubs_both": len(e_hubs & n_hubs),
        "hubs_neither": len(set(hubs) - e_hubs - n_hubs),
        "total_excess": sum(inst.E.values()),
        "total_need": sum(inst.N.values()),
        "E_min": min(inst.E.values(), default=None),
        "E_max": max(inst.E.values(), default=None),
        "N_min": min(inst.N.values(), default=None),
        "N_max": max(inst.N.values(), default=None),
        "B_nonzero_min": min(b_nonzero, default=None),
        "B_nonzero_max": max(b_nonzero, default=None),
        "B_zero_entries": inst.meta["B_zero_entries"],
        "B_zero_on_needy_hubs": len(free_needy),
        "free_needy_hubs": " ".join(free_needy),
        "C_min": min(c_vals),
        "C_max": max(c_vals),
        "C_zero_offdiag": sum(1 for v in c_vals if v == 0),
        "C_diag_nonzero": inst.meta["C_diagonal_nonzero"],
        "C_asymmetric_pairs": asym,
    }


def read_reference(path: Path) -> dict:
    with open(path, newline="") as fh:
        return {(r["city"], int(r["seed"]), r["variant"]): r for r in csv.DictReader(fh)}


def run_one(xlsx: Path, variant: str, out_dir: Path, write_mps: bool, threads, time_limit) -> Path:
    argv = [
        "--instance", str(xlsx), "--free-purchase", variant,
        "--fc1", str(FC1), "--fc2", str(FC2), "--ud", str(UD),
        "--mip-gap", "0", "--out-dir", str(out_dir), "--quiet",
    ]
    if write_mps:
        argv.append("--write-mps")
    if threads is not None:
        argv += ["--threads", str(threads)]
    if time_limit is not None:
        argv += ["--time-limit", str(time_limit)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solve_gurobi.main(argv)
    line = [s for s in buf.getvalue().splitlines() if "-> " in s][-1]
    run_dir = Path(line.split("-> ", 1)[1].strip())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        score_solution.main(["--run-dir", str(run_dir)])
    (run_dir / "score.txt").write_text(buf.getvalue())
    return run_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(REPO.parent / "OldModelDatasets"))
    p.add_argument("--out-dir", default="results/rebalancing/old_datasets")
    p.add_argument("--reference", default=str(REPO / "rebalancing/reference/old_datasets_reference_optima.csv"))
    p.add_argument("--summary-csv", default=str(REPO / "rebalancing/results/old_datasets_summary.csv"))
    p.add_argument("--mps-seeds", type=int, nargs="*", default=[1001, 2001, 3001, 4001, 5001])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--time-limit", type=float, default=None)
    args = p.parse_args(argv)

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    workbooks = sorted(root.glob("*_*_hubs/*.xlsx"), key=lambda q: (int(q.parent.name.split("_")[1]), q.stem))
    if not workbooks:
        print(f"no <city>_<n>_hubs/<seed>.xlsx under {root}", file=sys.stderr)
        return 2
    ref = read_reference(Path(args.reference))
    print(f"{len(workbooks)} workbooks under {root}; reference {args.reference}")

    rows = []
    t_all = time.perf_counter()
    for xlsx in workbooks:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*purchase costs on needed.*")
            checks = data_checks(load_instance(xlsx, ud=UD, fc1=FC1, fc2=FC2))
            for variant in ("keep", "block"):
                write_mps = variant == "keep" and int(xlsx.stem) in args.mps_seeds
                run_dir = run_one(xlsx, variant, out_dir, write_mps, args.threads, args.time_limit)
                s = json.loads((run_dir / "summary.json").read_text())
                sc = json.loads((run_dir / "score.json").read_text())
                st = s.get("solution_stats") or {}
                city = s["instance"]["city"]
                seed = int(s["instance"]["seed"])
                r = ref.get((city, seed, VARIANT_REF[variant]))
                ref_obj = float(r["highs_obj"]) if r else None
                obj = s["ObjVal"]
                match = (
                    obj is not None and ref_obj is not None
                    and abs(obj - round(obj)) < 1e-9 and round(obj) == ref_obj
                )
                if write_mps:
                    (out_dir / "mps").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(run_dir / "model.mps", out_dir / "mps" / f"{city}_{seed}.mps")
                row = {
                    "city": city,
                    "hubs": checks["hubs"],
                    "seed": seed,
                    "variant": variant,
                    "ref_variant": VARIANT_REF[variant],
                    "status": s["status"],
                    "obj": obj,
                    "ref_highs_obj": ref_obj,
                    "obj_match": match,
                    "bound": s["ObjBound"],
                    "gap": s["MIPGap"],
                    "runtime_s": round(s["Runtime"], 4),
                    "build_s": round(s["timing_seconds"]["model_build"], 4),
                    "load_s": round(s["timing_seconds"]["data_load"], 4),
                    "nodes": int(s["NodeCount"]),
                    "vars": s["num_vars"]["total"],
                    "T_free_blocked": s["num_vars"]["T_free_blocked"],
                    "cons": s["num_constraints"]["total"],
                    "score_pass": sc["pass"],
                    "shipped": st.get("units_shipped"),
                    "bought": st.get("units_bought"),
                    "bought_free": st.get("units_bought_free"),
                    "unmet": st.get("units_unmet"),
                    "Y_on": st.get("sites_shipping"),
                    "Z_on": st.get("sites_receiving"),
                    "ref_shipped": int(r["shipped"]) if r else None,
                    "ref_bought": int(r["bought"]) if r else None,
                    "ref_bought_free": int(r["bought_free"]) if r else None,
                    "ref_unmet": int(r["unmet"]) if r else None,
                    "flows_same_as_ref": (
                        r is not None
                        and (st.get("units_shipped"), st.get("units_bought"),
                             st.get("units_bought_free"), st.get("units_unmet"))
                        == (int(r["shipped"]), int(r["bought"]), int(r["bought_free"]), int(r["unmet"]))
                    ),
                    "data_same_as_ref": (
                        r is not None
                        and (checks["hubs_excess"], checks["hubs_need"],
                             checks["total_excess"], checks["total_need"])
                        == (int(r["hubs_excess"]), int(r["hubs_need"]),
                            int(r["total_excess"]), int(r["total_need"]))
                        and (variant == "block"
                             or checks["B_zero_on_needy_hubs"] == int(r["free_purchase_needy_hubs"]))
                    ),
                    "mps": write_mps,
                    "run_dir": str(run_dir),
                    **{f"data_{k}": v for k, v in checks.items() if k != "hubs"},
                }
                rows.append(row)
                print(f"  {city:<7} {seed} {variant:<5} {s['status']:<8} obj {obj:>12,.1f}  ref {ref_obj:>12,.1f}  "
                      f"{'MATCH' if match else 'MISMATCH':<8} score {'PASS' if sc['pass'] else 'FAIL'}  "
                      f"{s['Runtime']:.3f}s")

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if not (r["obj_match"] and r["score_pass"])]
    missing_ref = {(c, s, v) for (c, s, v) in ref} - {(r["city"], r["seed"], r["ref_variant"]) for r in rows}
    print(f"\n{len(rows)} runs in {time.perf_counter() - t_all:.1f}s -> {summary_path}")
    print(f"ObjVal matches reference: {sum(r['obj_match'] for r in rows)}/{len(rows)};  "
          f"score_solution PASS: {sum(r['score_pass'] for r in rows)}/{len(rows)}")
    for r in bad:
        print(f"  !! {r['city']} {r['seed']} {r['variant']}: obj {r['obj']} ref {r['ref_highs_obj']} "
              f"match={r['obj_match']} score={r['score_pass']}")
    if missing_ref:
        print(f"  !! reference rows with no run: {sorted(missing_ref)}")

    print("\nsolve time by size (Gurobi Runtime, seconds)")
    print(f"  {'hubs':>4} {'variant':<7} {'n':>2} {'mean':>8} {'median':>8} {'max':>8} {'max nodes':>9}")
    groups = defaultdict(list)
    for r in rows:
        groups[r["hubs"], r["variant"]].append(r)
    for (h, v), g in sorted(groups.items()):
        t = [r["runtime_s"] for r in g]
        print(f"  {h:>4} {v:<7} {len(g):>2} {statistics.mean(t):>8.4f} {statistics.median(t):>8.4f} "
              f"{max(t):>8.4f} {max(r['nodes'] for r in g):>9}")
    return 0 if not bad and not missing_ref else 1


if __name__ == "__main__":
    sys.exit(main())
