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

--param-set cplex (FC1 102, FC2 103, UD 104, BIGM 10000 -- Aarav's CPLEX model)
solves the keep variant only, with --mip-gap 0, into
results/rebalancing/old_datasets_cplex/ and writes
rebalancing/results/old_datasets_summary_cplex.csv. There is no highs_obj
comparison (that reference was solved at 100/1000/10000). Instead every
instance is also re-solved with the tight per-site M, and bigm_equal records
whether the two ObjVals are exactly equal. --cplex-results PATH (CSV with
city, seed, cplex_obj) fills the cplex_obj / cplex_match columns (exact
equality); --fill-only does that on the existing summary CSV without re-solving.

    python -m rebalancing.run_old_datasets --param-set cplex
    python -m rebalancing.run_old_datasets --param-set cplex --cplex-results aarav.csv --fill-only

Exit code 0 for cplex only if every run is OPTIMAL, passes score_solution,
and has bigm_equal; a mismatching cplex_match also gives 1.
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


CPLEX_COLUMNS = [
    "city", "hubs", "seed", "status", "obj", "bound", "gap", "runtime_s",
    "cost_ship", "cost_purchase", "cost_unmet", "cost_FC1", "cost_FC2",
    "shipped", "bought", "bought_free", "unmet", "Y_on", "Z_on",
    "score_pass", "bigm_equal", "cplex_obj", "cplex_match",
    # supporting columns
    "tight_m_obj", "bigm_below_tight_M_rows", "total_need", "unmet_share", "mps", "run_dir",
]


def run_one_cplex(xlsx: Path, out_dir: Path, write_mps: bool, threads, time_limit) -> Path:
    """One keep-variant solve with --param-set cplex (BIGM 10000), then score_solution."""
    argv = [
        "--instance", str(xlsx), "--free-purchase", "keep", "--param-set", "cplex",
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


def read_cplex_results(path: Path) -> dict:
    """{(city, seed): cplex_obj} from a CSV with columns city, seed, cplex_obj."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = {"city", "seed", "cplex_obj"}
    if rows and not need <= set(rows[0]):
        raise SystemExit(f"{path}: needs columns {sorted(need)}, has {sorted(rows[0])}")
    out = {}
    for r in rows:
        if str(r["cplex_obj"]).strip() == "":
            continue
        key = (r["city"].strip().lower(), int(r["seed"]))
        if key in out:
            raise SystemExit(f"{path}: duplicate row for {key}")
        out[key] = float(r["cplex_obj"])
    return out


def fill_cplex(rows: list, cplex: dict) -> tuple:
    """Set cplex_obj / cplex_match (exact equality) on rows. Returns (filled, unknown keys)."""
    filled = 0
    for r in rows:
        v = cplex.get((r["city"], int(r["seed"])))
        if v is None:
            r["cplex_obj"], r["cplex_match"] = "", ""
            continue
        r["cplex_obj"] = v
        r["cplex_match"] = r["obj"] not in ("", None) and float(r["obj"]) == v
        filled += 1
    unknown = set(cplex) - {(r["city"], int(r["seed"])) for r in rows}
    return filled, sorted(unknown)


def write_rows(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def report_cplex_fill(rows: list, cplex_path) -> list:
    """Print the CPLEX comparison; return the rows that do not match."""
    have = [r for r in rows if r["cplex_obj"] not in ("", None)]
    bad = [r for r in have if str(r["cplex_match"]) != "True"]
    print(f"CPLEX objectives from {cplex_path}: {len(have)}/{len(rows)} instances; "
          f"exact match {len(have) - len(bad)}/{len(have)}")
    for r in bad:
        print(f"  !! {r['city']} {r['seed']}: gurobi {r['obj']} cplex {r['cplex_obj']}")
    return bad


def main_cplex(args) -> int:
    root = Path(args.data_root)
    out_dir = Path(args.out_dir or "results/rebalancing/old_datasets_cplex")
    summary_path = Path(args.summary_csv or REPO / "rebalancing/results/old_datasets_summary_cplex.csv")
    cplex = read_cplex_results(Path(args.cplex_results)) if args.cplex_results else None

    if args.fill_only:
        if cplex is None:
            raise SystemExit("--fill-only needs --cplex-results")
        with open(summary_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        filled, unknown = fill_cplex(rows, cplex)
        write_rows(summary_path, rows, CPLEX_COLUMNS)
        print(f"filled {filled} rows of {summary_path}")
        if unknown:
            print(f"  !! CPLEX rows with no instance in the summary: {unknown}")
        return 1 if report_cplex_fill(rows, args.cplex_results) or unknown else 0

    workbooks = sorted(root.glob("*_*_hubs/*.xlsx"), key=lambda q: (int(q.parent.name.split("_")[1]), q.stem))
    if not workbooks:
        print(f"no <city>_<n>_hubs/<seed>.xlsx under {root}", file=sys.stderr)
        return 2
    pset = solve_gurobi.PARAM_SETS["cplex"]
    print(f"{len(workbooks)} workbooks under {root}; param set cplex "
          f"(FC1 {pset['fc1']:g}, FC2 {pset['fc2']:g}, UD {pset['ud']:g}, BIGM {pset['bigm']:g}), keep only")

    rows = []
    t_all = time.perf_counter()
    for xlsx in workbooks:
        write_mps = int(xlsx.stem) in args.mps_seeds
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*purchase costs on needed.*")
                inst = load_instance(xlsx, ud=pset["ud"], fc1=pset["fc1"], fc2=pset["fc2"])
                run_dir = run_one_cplex(xlsx, out_dir, write_mps, args.threads, args.time_limit)
                # Same model with the tight per-site M: the BIGM choice must not move the optimum.
                tight = solve_gurobi.solve(inst, mip_gap=0.0, quiet=True, bigm=None,
                                           threads=args.threads, time_limit=args.time_limit)
        except solve_gurobi.gp.GurobiError as e:
            # Most likely a size-limited license. Stop here, never skip.
            print(f"\n!! STOPPED at {xlsx}: GurobiError {e.errno}: {e}", file=sys.stderr)
            print(f"   {len(rows)} instances finished before it; nothing after it was run.", file=sys.stderr)
            return 3
        s = json.loads((run_dir / "summary.json").read_text())
        sc = json.loads((run_dir / "score.json").read_text())
        st = s.get("solution_stats") or {}
        cb = s.get("cost_breakdown") or {}
        obj = s["ObjVal"]
        bigm_equal = (
            s["status"] == "OPTIMAL" and tight["status"] == "OPTIMAL"
            and obj is not None and obj == tight["ObjVal"]
        )
        (run_dir / "bigm_check.json").write_text(json.dumps({
            "bigm": s["params"]["bigm"], "obj_bigm": obj, "status_bigm": s["status"],
            "obj_tight_M": tight["ObjVal"], "status_tight_M": tight["status"],
            "bound_tight_M": tight["ObjBound"], "runtime_tight_M": tight["Runtime"],
            "equal": bigm_equal,
        }, indent=2))
        if write_mps:
            (out_dir / "mps").mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / "model.mps", out_dir / "mps" / f"{s['instance']['city']}_{s['instance']['seed']}_cplex.mps")
        need = st.get("total_need") or 0
        row = {
            "city": s["instance"]["city"],
            "hubs": len(inst.sites),
            "seed": int(s["instance"]["seed"]),
            "status": s["status"],
            "obj": obj,
            "bound": s["ObjBound"],
            "gap": s["MIPGap"],
            "runtime_s": round(s["Runtime"], 4),
            "cost_ship": cb.get("ship"),
            "cost_purchase": cb.get("purchase"),
            "cost_unmet": cb.get("unmet"),
            "cost_FC1": cb.get("fixed_ship_FC1"),
            "cost_FC2": cb.get("fixed_receive_FC2"),
            "shipped": st.get("units_shipped"),
            "bought": st.get("units_bought"),
            "bought_free": st.get("units_bought_free"),
            "unmet": st.get("units_unmet"),
            "Y_on": st.get("sites_shipping"),
            "Z_on": st.get("sites_receiving"),
            "score_pass": sc["pass"],
            "bigm_equal": bigm_equal,
            "cplex_obj": "",
            "cplex_match": "",
            "tight_m_obj": tight["ObjVal"],
            "bigm_below_tight_M_rows": s["params"]["bigm_below_tight_M_rows"],
            "total_need": need,
            "unmet_share": round(st.get("units_unmet", 0) / need, 6) if need else None,
            "mps": write_mps,
            "run_dir": str(run_dir),
        }
        rows.append(row)
        print(f"  {row['city']:<7} {row['seed']} {s['status']:<8} obj {obj:>10,.1f}  tightM {tight['ObjVal']:>10,.1f}  "
              f"{'EQUAL' if bigm_equal else 'DIFF':<5} unmet {row['unmet']:>5}/{need:<5} "
              f"score {'PASS' if sc['pass'] else 'FAIL'}  {s['Runtime']:.3f}s")

    if cplex is not None:
        _, unknown = fill_cplex(rows, cplex)
        if unknown:
            print(f"  !! CPLEX rows with no instance: {unknown}")
    write_rows(summary_path, rows, CPLEX_COLUMNS)

    bad = [r for r in rows if not (r["status"] == "OPTIMAL" and r["score_pass"] and r["bigm_equal"])]
    print(f"\n{len(rows)} instances in {time.perf_counter() - t_all:.1f}s -> {summary_path}")
    print(f"OPTIMAL {sum(r['status'] == 'OPTIMAL' for r in rows)}/{len(rows)};  "
          f"score_solution PASS {sum(r['score_pass'] for r in rows)}/{len(rows)};  "
          f"BIGM 10000 == tight M {sum(r['bigm_equal'] for r in rows)}/{len(rows)}")
    for r in bad:
        print(f"  !! {r['city']} {r['seed']}: status {r['status']} obj {r['obj']} tight {r['tight_m_obj']} "
              f"score={r['score_pass']} bigm_equal={r['bigm_equal']}")
    cplex_bad = report_cplex_fill(rows, args.cplex_results) if cplex is not None else []
    if cplex is None:
        print("cplex_obj / cplex_match left empty: no --cplex-results given (CPLEX match NOT verified)")
    return 0 if not bad and not cplex_bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(REPO.parent / "OldModelDatasets"))
    p.add_argument("--param-set", choices=("pow10", "cplex"), default="pow10")
    p.add_argument("--out-dir", default=None,
                   help="default results/rebalancing/old_datasets (pow10), .../old_datasets_cplex (cplex)")
    p.add_argument("--reference", default=str(REPO / "rebalancing/reference/old_datasets_reference_optima.csv"))
    p.add_argument("--summary-csv", default=None,
                   help="default rebalancing/results/old_datasets_summary.csv (pow10), "
                        "old_datasets_summary_cplex.csv (cplex)")
    p.add_argument("--mps-seeds", type=int, nargs="*", default=[1001, 2001, 3001, 4001, 5001])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--cplex-results", default=None, help="cplex only: CSV with city, seed, cplex_obj")
    p.add_argument("--fill-only", action="store_true",
                   help="cplex only: fill cplex_obj/cplex_match in the existing summary CSV, no solving")
    args = p.parse_args(argv)
    if args.param_set == "cplex":
        return main_cplex(args)
    if args.cplex_results or args.fill_only:
        p.error("--cplex-results / --fill-only go with --param-set cplex")
    args.out_dir = args.out_dir or "results/rebalancing/old_datasets"
    args.summary_csv = args.summary_csv or str(REPO / "rebalancing/results/old_datasets_summary.csv")

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
