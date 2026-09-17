#!/usr/bin/env python3
"""Determinism study: separate ANNEALER variation from harness noise.

Two arms, everything else identical:

  Arm A -- 10 runs at ONE fixed --va-seed. The annealer is pinned, so anything
           that varies here is nondeterminism OUTSIDE the annealer. Arm A is a
           GATE: if it is not clean, Arm B cannot be interpreted, because you
           cannot tell annealer variation from harness variation.
  Arm B -- 10 runs at --va-seed 1..10. Variation here is the annealer.

PYTHONHASHSEED=0 is forced for every run, so set/dict iteration order is pinned
and cannot contribute.

Fire-and-forget: each run's row is appended to the CSV the moment it finishes, so
a crash at run 17 keeps the first 16. Re-running skips completed runs, so it
resumes. --analyze-only rebuilds the summary from the CSV without running.

One VE card, so runs are SEQUENTIAL by necessity.

    # on the VE node, or via sbatch_scripts/va_determinism.sh
    python3 va_determinism_study.py --out results/determinism
    python3 va_determinism_study.py --out results/determinism --analyze-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BAR = "=" * 86
FIELDS = [
    "arm", "run_index", "seed", "rc", "wall_seconds",
    "n_open_hubs", "open_hubs", "total_cost", "gap_pct",
    "peak_host_rss_mb", "stocked_pairs", "assignments",
    "structural_violations", "c1", "c2", "c3", "c4_hubs_over_L",
    "sla_violations", "max_service_violations",
    "python_hash_seed", "hash_order_deterministic", "memory_accounting_version",
    "run_dir",
]
# Fields that MUST be identical across Arm A. Wall time and RSS are excluded:
# they are machine noise, not solver output, and would flag every clean run.
INVARIANT = [
    "n_open_hubs", "open_hubs", "total_cost", "stocked_pairs", "assignments",
    "structural_violations", "c1", "c2", "c3", "c4_hubs_over_L",
    "sla_violations", "max_service_violations",
]


def build_plan(n: int, fixed_seed: int, arms: str = "AB",
               seeds: list[int] | None = None) -> list[tuple[str, int, int]]:
    """(arm, run_index, seed). Arm A pins the seed; Arm B varies it.

    `arms` selects which to run. Running B alone is legitimate when the card is
    scarce -- B is the arm that carries the scaling/seed-sensitivity result --
    but without A there is no gate, so any variation B shows cannot be
    attributed to the annealer rather than the harness. The summary says so.

    `seeds` overrides Arm B's default 1..n. The default is a terrible sample for
    the question actually being asked: 1..10 are all tiny and all adjacent, so
    they cannot distinguish "the seed does nothing" from "the seed matters but
    only across orders of magnitude". Pass a list spanning several decades
    instead -- see SEED_LADDER.
    """
    plan: list[tuple[str, int, int]] = []
    if "A" in arms.upper():
        plan += [("A", i, fixed_seed) for i in range(1, n + 1)]
    if "B" in arms.upper():
        chosen = list(seeds) if seeds else list(range(1, n + 1))
        plan += [("B", i, s) for i, s in enumerate(chosen, start=1)]
    return plan


# A seed sample designed to falsify "bigger seed -> better cost". Spans six
# orders of magnitude, is NOT sorted by magnitude in run order, and includes the
# three already-run values (42, 1000, 2026) so those runs are reused rather than
# repeated -- each one is ~27 minutes of a single shared card.
SEED_LADDER = [42, 7, 1000, 123456, 3, 2026, 99991, 500, 8675309, 17, 31337, 2]


def read_result(run_root: Path) -> dict | None:
    """Harvest one finished run. None if it did not produce a summary."""
    summary_path = run_root / "va" / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as fh:
            s = json.load(fh)
    except Exception:
        return None
    fs, rt = s["final_solution"], s["runtime"]
    audit = fs["audit"]

    hubs: list[str] = []
    hubs_csv = run_root / "va" / "open_hubs.csv"
    if hubs_csv.is_file():
        with open(hubs_csv, newline="", encoding="utf-8") as fh:
            hubs = sorted(r["hub_id"] for r in csv.DictReader(fh) if r.get("hub_id"))
    return {
        "n_open_hubs": int(fs["open_hubs_count"]),
        # Sorted and joined so the SET is compared, not the discovery order.
        "open_hubs": ";".join(hubs),
        "total_cost": repr(float(fs["cost"]["total_cost"])),
        "peak_host_rss_mb": round(float(rt.get("peak_memory_mb") or 0.0), 3),
        "stocked_pairs": int(fs["stocked_pairs_count"]),
        "assignments": int(fs["assignments_count"]),
        "structural_violations": int(audit.get("total_structural_violations", -1)),
        "c1": int(audit.get("c1_assignment_violations", -1)),
        "c2": int(audit.get("c2_assignment_without_stock", -1)),
        "c3": int(audit.get("c3_stock_without_open_hub", -1)),
        "c4_hubs_over_L": int(audit.get("c4_hubs_over_L", -1)),
        "sla_violations": int(audit.get("sla_distance_violations", -1)),
        "max_service_violations": int(audit.get("max_service_distance_violations", -1)),
        "python_hash_seed": str(rt.get("python_hash_seed", "<unrecorded>")),
        "hash_order_deterministic": bool(rt.get("hash_order_deterministic", False)),
        "memory_accounting_version": int(rt.get("memory_accounting_version", 0)),
    }


GUROBI_OPTIMA_CSV = Path("results/gurobi_optima/gurobi_optima.csv")


def gurobi_optimum(dataset_dir: str) -> float | None:
    """The proven optimum for this instance, or None if it was never solved.

    Only rows Gurobi closed at gap 0.0 are returned. An incumbent from a run that
    hit a time limit is not an optimum and must not be used as a denominator --
    a "gap" measured against it can come out negative and would be nonsense.
    """
    path = Path(GUROBI_OPTIMA_CSV)
    if not path.is_file():
        return None
    want = Path(dataset_dir).name
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("instance", "")) == want and \
                    str(r.get("proven_optimal", "")).strip().lower() == "true":
                return float(r["total_cost"])
    return None


def add_gap(row: dict, opt: float | None) -> dict:
    """Percent above the proven optimum. Blank when no optimum is known."""
    row["gap_pct"] = ""
    if opt and row.get("total_cost"):
        try:
            row["gap_pct"] = round(100.0 * (float(row["total_cost"]) - opt) / opt, 6)
        except (TypeError, ValueError):
            pass
    return row


def import_external_runs(sources: list[str], out: Path, dataset_dir: str) -> int:
    """Fold already-finished solver runs into the study CSV as Arm B rows.

    The three 20-hub seed runs (42, 1000, 2026) were produced by hand through
    run_va_parallel_batches.py before this harness existed. They are perfectly
    good Arm B samples and each cost ~27 minutes on the one shared VE card, so
    they are imported rather than repeated. The seed is read from the run's own
    summary.json, never inferred from the directory name.
    """
    csv_path = out / "determinism_runs.csv"
    have = {(r["arm"], str(r["seed"])) for r in load_rows(csv_path)}
    opt = gurobi_optimum(dataset_dir)
    n = 0
    for src_dir in sources:
        run_root = Path(src_dir).expanduser().resolve()
        got = read_result(run_root)
        if got is None:
            print(f"  IMPORT SKIP {run_root} -- no va/summary.json")
            continue
        try:
            with open(run_root / "va" / "summary.json", encoding="utf-8") as fh:
                seed = json.load(fh)["extra"]["va"]["seed"]
        except (KeyError, OSError, ValueError) as exc:
            print(f"  IMPORT SKIP {run_root} -- seed unreadable ({exc})")
            continue
        if ("B", str(seed)) in have:
            print(f"  IMPORT SKIP seed={seed} -- already in CSV")
            continue
        row = {"arm": "B", "run_index": 900 + n, "seed": int(seed), "rc": 0,
               "wall_seconds": "", "run_dir": str(run_root), **got}
        append_row(csv_path, add_gap(row, opt))
        have.add(("B", str(seed)))
        print(f"  IMPORTED seed={seed}  cost={float(row['total_cost']):,.2f}  from {run_root.name}")
        n += 1
    return n


def run_one(a: argparse.Namespace, arm: str, idx: int, seed: int, out: Path) -> dict:
    run_root = out / "runs" / f"{arm}_{idx:02d}_seed{seed}"
    existing = read_result(run_root)
    if existing and not a.force:
        print(f"  [{arm}{idx:02d}] seed={seed}  already complete, reusing", flush=True)
        return {"arm": arm, "run_index": idx, "seed": seed, "rc": 0,
                "wall_seconds": "", "run_dir": str(run_root), **existing}

    run_root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "run_va_fsl_solver.py",
           "--dataset-dir", a.dataset_dir,
           "--run-root", str(run_root),
           "--va-seed", str(seed)] + a.solver_flag
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"      # the whole point; set before the child starts

    log = out / "logs" / f"{arm}_{idx:02d}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{arm}{idx:02d}] seed={seed}  running -> {log.name}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0

    row = {"arm": arm, "run_index": idx, "seed": seed, "rc": int(proc.returncode),
           "wall_seconds": round(wall, 2), "run_dir": str(run_root)}
    got = read_result(run_root)
    if got is None:
        print(f"  [{arm}{idx:02d}] FAILED rc={proc.returncode} after {wall:,.0f}s "
              f"-- no summary.json. See {log}", flush=True)
        row.update({f: "" for f in FIELDS if f not in row})
        row["rc"] = int(proc.returncode) or -1
    else:
        row.update(got)
        add_gap(row, gurobi_optimum(a.dataset_dir))
        print(f"  [{arm}{idx:02d}] ok in {wall:,.0f}s | hubs={row['n_open_hubs']} "
              f"cost={float(row['total_cost']):,.2f} rss={row['peak_host_rss_mb']:,.0f}MB "
              f"viol={row['structural_violations']}", flush=True)
    return row


def append_row(csv_path: Path, row: dict) -> None:
    """Append one run, migrating the header first if FIELDS has grown.

    Appending a wider row to a CSV written under an older FIELDS list silently
    shifts every value one column right from that point on, and the file still
    parses -- so the corruption only shows up as nonsense in the summary. When
    the on-disk header differs, the existing rows are rewritten under the new
    header with blanks for the columns they never had.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            header = next(rdr, None)
        if header is not None and header != list(FIELDS):
            old_rows = load_rows(csv_path)
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in old_rows:
                    w.writerow({f: r.get(f, "") for f in FIELDS})
            print(f"  (migrated {csv_path.name} to the current column set)")
    new = not csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def spread(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {
        "n": n, "mean": mean, "stdev": sd,
        "cv": (sd / mean) if mean else 0.0,
        "min": min(values), "max": max(values), "range": max(values) - min(values),
        "median": statistics.median(values),
        "distinct": len({repr(v) for v in values}),
    }


def _ranks(values: list[float]) -> list[float]:
    """Ranks with ties averaged, which is what Spearman requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(math.fsum((a - mx) ** 2 for a in x))
    dy = math.sqrt(math.fsum((b - my) ** 2 for b in y))
    return 0.0 if dx == 0 or dy == 0 else num / (dx * dy)


def seed_trend_test(seeds: list[int], costs: list[float]) -> dict:
    """Does a larger seed give a lower cost? Spearman rho + an exact p-value.

    THE CLAIM UNDER TEST. Three runs -- seeds 42, 1000 and 2026 -- came back in
    descending cost order, which looks like "bigger seed, better answer". With
    three points that ordering has probability 1/6 = 0.167 by chance alone, so
    the observation is indistinguishable from a coin landing heads twice.

    A seed feeds a pseudo-random number generator. Its MAGNITUDE carries no
    information: 8,675,309 is not a "more thorough" starting point than 3. If a
    real monotone trend survives a dozen seeds spanning six orders of magnitude,
    the finding is not "use big seeds", it is "this RNG is not seeding properly",
    which is a bug report about the annealer.

    The p-value is the exact permutation test -- every ordering of the observed
    costs, counted -- while n! is tractable, and a fixed-seed Monte Carlo
    approximation above that. It is one-sided: the alternative is specifically
    rho < 0 (cost falls as the seed rises), because that is the direction that
    was eyeballed. Testing two-sided after seeing the direction would be
    double-dipping.
    """
    n = len(seeds)
    if n < 3:
        return {"n": n, "verdict": "too few runs to test"}

    # Every seed returning the same cost is not a weak trend, it is no trend at
    # all -- and on a geometrically forced instance it is the expected answer.
    # Left to the general path it would come out rho=0 and be described as
    # "the sign is wrong", which reads as a finding where there is none.
    if len({repr(c) for c in costs}) == 1:
        return {"n": n, "rho": 0.0, "p_one_sided": 1.0,
                "method": "not applicable -- zero variance",
                "verdict": ("NO VARIATION TO EXPLAIN. Every seed returned exactly "
                            f"{costs[0]:,.4f}. There is no trend because there is no "
                            "spread; check whether the instance leaves the solver any "
                            "freedom at all before reading this as stability.")}

    sr, cr = _ranks([float(s) for s in seeds]), _ranks(costs)
    rho = _pearson(sr, cr)

    import itertools
    import random

    exact = n <= 8
    if exact:
        total = hits = 0
        for perm in itertools.permutations(cr):
            total += 1
            if _pearson(sr, list(perm)) <= rho + 1e-12:
                hits += 1
        p = hits / total
        method = f"exact permutation ({total:,} orderings)"
    else:
        rng = random.Random(20260917)   # fixed: the p-value must reproduce
        trials, hits = 20000, 0
        shuf = list(cr)
        for _ in range(trials):
            rng.shuffle(shuf)
            if _pearson(sr, shuf) <= rho + 1e-12:
                hits += 1
        p = (hits + 1) / (trials + 1)   # add-one: never report p = 0
        method = f"Monte Carlo ({trials:,} shuffles, fixed rng seed)"

    if p < 0.05 and rho < 0:
        verdict = ("SUPPORTED at p<0.05 -- and that is a RED FLAG, not a result. "
                   "A PRNG seed should carry no ordering information; a real "
                   "monotone trend means the seed is not being consumed as a "
                   "seed. Investigate the annealer's seeding before reporting "
                   "this as a tuning knob.")
    elif rho < 0:
        verdict = ("NOT SUPPORTED. The downward slope is within what random "
                   "ordering produces. Report the seed spread as noise around a "
                   "central value, not as a trend.")
    else:
        verdict = ("NOT SUPPORTED, and the sign is wrong -- cost rises with seed "
                   "in this sample. Noise.")

    return {"n": n, "rho": rho, "p_one_sided": p, "method": method, "verdict": verdict}


def geometry_caveat(dataset_dir: str) -> str:
    """Is the open-hub set even free to vary on this instance?

    If every hub is the sole reachable option for some demand row, all hubs are
    forced open and 'the hub set never varied' is a statement about the instance,
    not about the solver. Reporting stability without this is misleading.
    """
    try:
        import va_instance_geometry as geo
        import run_va_fsl_solver as solver
        data = solver.load_problem_data(
            dataset_dir, max_service_miles_override=None,
            penalty_start_miles_override=None, top_hubs_per_zip=None,
            max_parts_total=None)
        cand = geo.candidates_within(data, float(data["scalar"]["max_service_miles"]))
        rows = [str(z) for z in data["active"]["zip_id"].tolist()]
        forced = {next(iter(cand[i])) for i in rows if len(cand.get(i, ())) == 1}
        n_hubs = len(data["J"])
        only_one = sum(1 for i in rows if len(cand.get(i, ())) == 1)
        if len(forced) >= n_hubs:
            return (f"ALL {n_hubs} hubs are FORCED open by geometry ({only_one:,} of "
                    f"{len(rows):,} demand rows have exactly one reachable hub). The open-hub "
                    f"set CANNOT vary on this instance, so its stability below is a property "
                    f"of the instance, not evidence about the solver. "
                    f"See va_instance_geometry.py.")
        return (f"{len(forced)} of {n_hubs} hubs are forced open by geometry; the other "
                f"{n_hubs - len(forced)} are a genuine choice, so hub-set variation is "
                f"meaningful here.")
    except Exception as exc:
        return f"geometry caveat unavailable ({type(exc).__name__}: {exc})"


def summarise(rows: list[dict], dataset_dir: str) -> None:
    ok = [r for r in rows if str(r.get("rc")) == "0" and r.get("total_cost")]
    A = [r for r in ok if r["arm"] == "A"]
    B = [r for r in ok if r["arm"] == "B"]
    failed = [r for r in rows if str(r.get("rc")) != "0"]

    print("\n" + BAR)
    print("DETERMINISM STUDY SUMMARY")
    print(BAR)
    print(f"  instance: {dataset_dir}")
    print(f"  runs: {len(ok)} usable ({len(A)} arm A, {len(B)} arm B)"
          + (f", {len(failed)} FAILED" if failed else ""))
    seeds = {r.get("python_hash_seed") for r in ok}
    print(f"  PYTHONHASHSEED recorded in runs: {sorted(seeds)}")
    if seeds - {"0"}:
        print("  *** WARNING: not every run was hash-pinned. Arm A cannot be trusted.")

    # ---- Arm A: the gate ------------------------------------------------
    print("\n  " + "-" * 82)
    print("  ARM A -- identical seed. Anything varying here is NOT the annealer.")
    print("  " + "-" * 82)
    if not A:
        print("    NOT RUN. Without the fixed-seed control there is no gate: any variation")
        print("    Arm B shows cannot be attributed to the annealer rather than the harness.")
        print("    Re-run with --arms A to add it (the runs are independent and resumable).")
        varied_a = []
    elif len(A) < 2:
        print("    too few runs to judge")
        varied_a = []
    else:
        varied_a = []
        for f in INVARIANT:
            vals = {r.get(f, "") for r in A}
            if len(vals) > 1:
                varied_a.append((f, vals))
        seed_used = {r["seed"] for r in A}
        print(f"    {len(A)} runs at --va-seed {sorted(seed_used)}")
        if not varied_a:
            print("    RESULT: NOTHING VARIED. Every solution field is identical across all")
            print("            runs. No nondeterminism outside the annealer. Arm B is safe")
            print("            to interpret.")
        else:
            print(f"    RESULT: {len(varied_a)} FIELD(S) VARIED. There is nondeterminism")
            print("            OUTSIDE the annealer. Do NOT interpret Arm B until this is")
            print("            resolved -- annealer and harness variation are confounded.")
            for f, vals in varied_a:
                shown = sorted(vals)[:4]
                print(f"      * {f}: {len(vals)} distinct values, e.g. {shown}")
        wl = [float(r["wall_seconds"]) for r in A if r.get("wall_seconds")]
        if wl:
            s = spread(wl)
            print(f"    (wall time varied {s['min']:,.0f}-{s['max']:,.0f}s, mean {s['mean']:,.0f}s "
                  f"-- machine noise, excluded from the invariance test)")

    # ---- Arm B: the annealer -------------------------------------------
    print("\n  " + "-" * 82)
    print("  ARM B -- varying seed. Variation here IS the annealer.")
    print("  " + "-" * 82)
    if len(B) < 2:
        print("    too few runs to judge")
    else:
        hub_sets = {r["open_hubs"] for r in B}
        costs = [float(r["total_cost"]) for r in B]
        cost_vals = {r["total_cost"] for r in B}
        print(f"    {len(B)} runs at seeds {sorted(int(r['seed']) for r in B)}")
        print(f"    distinct open-hub sets: {len(hub_sets)}")
        print(f"    distinct total costs:   {len(cost_vals)}")

        s = spread(costs)
        print(f"\n    cost across arm B:")
        print(f"      mean    {s['mean']:>18,.4f}")
        print(f"      stdev   {s['stdev']:>18,.4f}   (CV {s['cv']:.3e})")
        print(f"      median  {s['median']:>18,.4f}")
        print(f"      min     {s['min']:>18,.4f}")
        print(f"      max     {s['max']:>18,.4f}")
        print(f"      range   {s['range']:>18,.4f}")

        print("\n    INTERPRETATION:")
        if len(hub_sets) == 1 and len(cost_vals) == 1:
            print("      Neither the hub set nor the cost varied. The annealer returned the")
            print("      same solution at every seed.")
        elif len(hub_sets) == 1 and len(cost_vals) > 1:
            print("      Hub set identical, cost VARIED -> the variation is in assignment or")
            print("      stocking, not siting.")
        elif len(hub_sets) > 1 and len(cost_vals) == 1:
            print("      Hub sets DIFFER but cost is IDENTICAL -> these are TIES, not")
            print("      instability. The annealer is finding equally-good alternative optima.")
        else:
            # Mixed: separate genuine ties from genuine spread.
            by_cost: dict[str, set[str]] = {}
            for r in B:
                by_cost.setdefault(r["total_cost"], set()).add(r["open_hubs"])
            ties = {c: h for c, h in by_cost.items() if len(h) > 1}
            print(f"      Both varied: {len(hub_sets)} hub sets across {len(cost_vals)} costs.")
            if ties:
                print(f"      {len(ties)} cost value(s) are reached by MORE THAN ONE hub set --")
                print("      those are ties, not instability.")
            print(f"      Genuine cost spread: {s['range']:,.4f} "
                  f"({100.0 * s['cv']:.4f}% CV) across seeds.")

        # ---- the seed-magnitude question --------------------------------
        seeds = [int(r["seed"]) for r in B]
        opt = gurobi_optimum(dataset_dir)
        print("\n    per-seed results (ascending seed):")
        width = "      {:>10}  {:>18}" + ("  {:>8}" if opt else "{}")
        print(width.format("seed", "total cost", "gap%" if opt else ""))
        for r in sorted(B, key=lambda r: int(r["seed"])):
            c = float(r["total_cost"])
            gap = f"{100.0 * (c - opt) / opt:8.4f}" if opt else ""
            print(f"      {int(r['seed']):>10,}  {c:>18,.2f}  {gap}")
        if opt:
            print(f"      {'GUROBI':>10}  {opt:>18,.2f}  {0.0:8.4f}   (proven optimal)")

        t = seed_trend_test(seeds, costs)
        print("\n    TREND TEST -- does a larger seed give a lower cost?")
        if t.get("rho") is None:
            print(f"      {t['verdict']}")
        else:
            print(f"      n={t['n']}  Spearman rho={t['rho']:+.4f}  "
                  f"p={t['p_one_sided']:.4f} (one-sided, rho<0)")
            print(f"      method: {t['method']}")
            for line in _wrap(t["verdict"], 74):
                print(f"      {line}")

    print("\n  " + "-" * 82)
    print("  INSTANCE CAVEAT")
    print("  " + "-" * 82)
    for line in _wrap(geometry_caveat(dataset_dir), 80):
        print(f"    {line}")
    print(BAR)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="instances_10hubs")
    ap.add_argument("--out", default="results/determinism")
    ap.add_argument("--runs-per-arm", type=int, default=10)
    ap.add_argument("--fixed-seed", type=int, default=42,
                    help="Arm A's single seed (default 42).")
    ap.add_argument("--solver-flag", action="append", default=[],
                    help="Extra flag passed to run_va_fsl_solver.py, verbatim. "
                         "Repeat. Applied identically to EVERY run in both arms.")
    ap.add_argument("--arms", default="AB", choices=["AB", "A", "B"],
                    help="Which arms to run. AB (default) runs both. B alone skips the "
                         "fixed-seed control -- half the card time, but no gate, so "
                         "variation cannot be attributed to the annealer.")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Rebuild the summary from the existing CSV; run nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even runs that already have a summary.json.")
    ap.add_argument("--seeds", default="",
                    help="Comma-separated Arm B seeds, replacing the default 1..n. "
                         "Use a range spanning several orders of magnitude.")
    ap.add_argument("--seed-ladder", action="store_true",
                    help=f"Arm B over the built-in falsification ladder: {SEED_LADDER}. "
                         "Twelve seeds across six decades, deliberately not in "
                         "magnitude order, including the three already run.")
    ap.add_argument("--import-run", action="append", default=[],
                    help="Fold an existing finished run directory in as an Arm B "
                         "sample instead of spending card time repeating it. Repeat.")
    a = ap.parse_args()

    seeds: list[int] | None = None
    if a.seed_ladder:
        seeds = list(SEED_LADDER)
    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if seeds and len(set(seeds)) != len(seeds):
        ap.error("duplicate seeds: a repeated seed is an Arm A control, not an Arm B sample")

    out = Path(a.out).expanduser().resolve()
    csv_path = out / "determinism_runs.csv"

    if a.import_run:
        out.mkdir(parents=True, exist_ok=True)
        n = import_external_runs(a.import_run, out, a.dataset_dir)
        print(f"  imported {n} run(s)")

    if a.analyze_only:
        rows = load_rows(csv_path)
        if not rows:
            print(f"No rows in {csv_path}")
            return 1
        summarise(rows, a.dataset_dir)
        print(f"\n  CSV: {csv_path}")
        return 0

    plan = build_plan(a.runs_per_arm, a.fixed_seed, a.arms, seeds)
    print(BAR)
    print("DETERMINISM STUDY")
    print(BAR)
    print(f"  instance     {a.dataset_dir}")
    if "A" in a.arms:
        print(f"  arm A        {a.runs_per_arm} runs, --va-seed {a.fixed_seed} (fixed)")
    else:
        print("  arm A        SKIPPED -- no control arm, so Arm B has no gate")
    if "B" in a.arms:
        b_seeds = [s for arm, _, s in plan if arm == "B"]
        print(f"  arm B        {len(b_seeds)} runs, --va-seed {b_seeds}")
    print(f"  extra flags  {a.solver_flag or '<none>'}  (identical in both arms)")
    print(f"  PYTHONHASHSEED=0 forced for every run")
    print(f"  output       {out}")
    print(f"  NOTE: one VE card, so runs are sequential. {len(plan)} runs total.")
    print(BAR)

    done = {(r["arm"], r["run_index"], r["seed"]) for r in load_rows(csv_path)}
    for arm, idx, seed in plan:
        if not a.force and (arm, str(idx), str(seed)) in done:
            print(f"  [{arm}{idx:02d}] seed={seed}  already in CSV, skipping")
            continue
        append_row(csv_path, run_one(a, arm, idx, seed, out))

    summarise(load_rows(csv_path), a.dataset_dir)
    print(f"\n  CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
