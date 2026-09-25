#!/usr/bin/env python3
"""Score externally produced FSL solutions with the project's own cost and audit code.

An outside solver (an AI run, a colleague's heuristic, ...) hands back three files
per instance: open_hubs.csv, stocked_pairs.csv and assignments.csv. This script
loads each instance with run_aligned_fsl_comparison.load_problem_data() using the
same data settings as that instance's Gurobi reference run, then scores the
solution with the same compute_solution_cost() and global_audit(). Nothing about
the accounting is reimplemented here, so a number in this table is directly
comparable to the Gurobi, greedy, SA and VA numbers next to it.

Reference costs are put on the same footing. Wherever a reference method's
solution files exist they are re-scored here instead of trusting that run's
summary.json, and every reference is marked "rescored" or "from_summary".

    python3 score_external_solution.py --instances-root . \\
        --solutions-root outputs/ai_runs/Opus5.5_code/FSL_optimization_results \\
        --label opus55_high_code

    # Sanity check first: score the Gurobi reference solution for one instance and
    # compare against that run's summary.json. Writes nothing.
    python3 score_external_solution.py --instances-root . --check-gurobi instances_20hubs
"""
from __future__ import annotations

import os
import sys

# compute_solution_cost() sums over Python sets, whose iteration order over strings
# depends on PYTHONHASHSEED. Pin it so re-running this script reproduces its own
# numbers bit for bit. (The project's parallel runs were also made with seed 0.)
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# The only file this script writes is --out, so keep imports from writing .pyc files.
sys.dont_write_bytecode = True

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_aligned_fsl_comparison import (  # noqa: E402
    REQUIRED_FILES,
    compute_solution_cost,
    global_audit,
    load_problem_data,
)

BAR = "=" * 100
METHODS = ("gurobi", "greedy", "sa", "va")
RUN_FOLDER_METHOD = {"gurobi": "gurobi", "qubo": "sa", "va": "va"}
OPTIMAL_GAP_TOL = 1e-9  # same tolerance as collect_gurobi_optima.py
# A few ulps of a float64 sum. Anything larger is a real accounting difference.
FP_REL_TOL = 1e-12

COST_FIELDS = [
    "total_cost",
    "inventory_cost",
    "fixed_open_hub_cost",
    "overflow_storage_cost",
    "new_hub_transfer_cost",
    "assignment_transport_cost",
    "overflow_units",
]
AUDIT_FIELDS = [
    "c1_assignment_violations",
    "c1_missing_assignments",
    "c1_multiple_assignments",
    "c1_invalid_hub_assignments",
    "c1_extra_assignments",
    "c2_assignment_without_stock",
    "c3_stock_without_open_hub",
    "c4_hubs_over_L",
    "c4_total_overflow_units",
    "max_service_distance_violations",
    "sla_distance_violations",
    "total_structural_violations",
]

COLUMNS = (
    [
        "label", "instance", "status", "solution_dir", "notes",
        "max_service_miles", "penalty_start_miles", "top_hubs_per_zip", "max_parts_total",
        "settings_source",
        "hubs_total", "open_hubs", "closed_hubs", "stocked_pairs", "assignment_rows",
        "duplicate_assignment_rows", "active_pairs", "missing_active_pairs",
        "assignments_to_ineligible_hub", "unknown_id_rows",
    ]
    + COST_FIELDS
    + ["audit_" + f for f in AUDIT_FIELDS]
    + ["feasible", "claimed_total_cost", "claimed_minus_true"]
    + [
        c
        for m in METHODS
        for c in (f"{m}_cost", f"{m}_gap_pct", f"{m}_source", f"{m}_structural_violations",
                  f"{m}_file", f"{m}_match")
    ]
    + ["gurobi_status", "gurobi_proven_optimal", "gurobi_mip_gap", "gurobi_obj_bound",
       "gurobi_bound_gap_pct", "greedy_variant", "va_seed"]
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def natural_key(name: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def norm_limit(value: Any) -> int | None:
    """top_hubs_per_zip / max_parts_total: None or <=0 both mean 'no limit'."""
    if value is None:
        return None
    v = int(float(value))
    return v if v > 0 else None


def num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gap_pct(value: float | None, ref: float | None) -> float | None:
    if value is None or ref is None or ref == 0:
        return None
    return 100.0 * (value - ref) / ref


def money(x: float | None) -> str:
    return "-" if x is None else f"{x:,.2f}"


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:+.4f}%"


# ---------------------------------------------------------------------------
# Instance loading with the Gurobi reference settings
# ---------------------------------------------------------------------------


def discover_instances(instances_root: Path) -> list[Path]:
    found = [
        d for d in instances_root.iterdir()
        if d.is_dir() and all((d / f).is_file() for f in REQUIRED_FILES)
    ]
    return sorted(found, key=lambda d: natural_key(d.name))


def config_settings(cfg: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """The four data-loading knobs exactly as run_aligned passes them to load_problem_data()."""
    msm = cfg.get("max_service_miles")
    psm = cfg.get("penalty_start_miles")
    return (
        None if msm is None else float(msm),
        None if psm is None else float(psm),
        norm_limit(cfg.get("top_hubs_per_zip", -1)),
        norm_limit(cfg.get("max_parts_total", -1)),
    )


def gurobi_load_settings(inst: str, runs: list[dict[str, Any]]) -> tuple[tuple[Any, ...], str, list[str]]:
    """Pick load settings from this instance's Gurobi run_config.json files."""
    lines: list[str] = []
    by_settings: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in runs:
        if r["method"] != "gurobi" or r["dataset_name"] != inst:
            continue
        cfg_path = r["run_dir"].parent / "run_config.json"
        if not cfg_path.is_file():
            lines.append(f"    {rel(cfg_path)}: missing")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        s = config_settings(cfg)
        lines.append(
            f"    {rel(cfg_path)}: max_service_miles={cfg.get('max_service_miles')} "
            f"penalty_start_miles={cfg.get('penalty_start_miles')} "
            f"top_hubs_per_zip={cfg.get('top_hubs_per_zip')} max_parts_total={cfg.get('max_parts_total')} "
            f"mip_gap={cfg.get('mip_gap')}"
        )
        by_settings.setdefault(s, []).append(r)
    if not by_settings:
        return (None, None, None, None), "no Gurobi run_config.json found; parameters.csv defaults", lines
    if len(by_settings) > 1:
        # Prefer the settings a proven-optimal run used, then the most common.
        def rank(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[int, int]:
            return (-sum(1 for r in item[1] if r["proven_optimal"]), -len(item[1]))
        chosen = sorted(by_settings.items(), key=rank)[0][0]
        return chosen, f"Gurobi run_config.json files DISAGREE; using {chosen}", lines
    return next(iter(by_settings)), "Gurobi run_config.json", lines


def load_instance(inst_dir: Path, settings: tuple[Any, ...]) -> dict[str, Any]:
    msm, psm, top, mpt = settings
    return load_problem_data(
        inst_dir,
        max_service_miles_override=msm,
        penalty_start_miles_override=psm,
        top_hubs_per_zip=top,
        max_parts_total=mpt,
    )


def resolved_settings(data: dict[str, Any]) -> tuple[float, float, int | None, int | None]:
    return (
        float(data["scalar"]["max_service_miles"]),
        float(data["scalar"]["penalty_start_miles"]),
        norm_limit(data["top_hubs_per_zip"]),
        norm_limit(data["max_parts_total"]),
    )


# ---------------------------------------------------------------------------
# Reading and scoring a solution folder
# ---------------------------------------------------------------------------

SOLUTION_FILES = {
    "open_hubs": ("open_hubs.csv",),
    "stocked_pairs": ("stocked_pairs.csv", "stocked_hub_part_pairs.csv"),
    # Gurobi/SA/VA run folders name the assignments file hub_zip_part_pairings.csv.
    "assignments": ("assignments.csv", "hub_zip_part_pairings.csv"),
}
SOLUTION_COLUMNS = {
    "open_hubs": ["hub_id"],
    "stocked_pairs": ["hub_id", "part_id"],
    "assignments": ["zip_id", "hub_id", "part_id"],
}


def find_solution_files(sol_dir: Path) -> tuple[dict[str, Path], list[str]]:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for key, names in SOLUTION_FILES.items():
        for name in names:
            if (sol_dir / name).is_file():
                found[key] = sol_dir / name
                break
        else:
            missing.append(names[0])
    return found, missing


def read_id_rows(path: Path, columns: list[str]) -> list[tuple[str, ...]]:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return []
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; has {list(df.columns)}")
    df = df[columns].apply(lambda s: s.str.strip())
    return [tuple(r) for r in df.itertuples(index=False, name=None)]


def score_solution(files: dict[str, Path], data: dict[str, Any]) -> dict[str, Any]:
    open_hubs = [r[0] for r in read_id_rows(files["open_hubs"], SOLUTION_COLUMNS["open_hubs"])]
    stocked = read_id_rows(files["stocked_pairs"], SOLUTION_COLUMNS["stocked_pairs"])
    assignments = read_id_rows(files["assignments"], SOLUTION_COLUMNS["assignments"])

    cost = compute_solution_cost(assignments, stocked, open_hubs, data)
    audit = global_audit(assignments, stocked, open_hubs, data)

    hubs, zips, parts = set(data["J"]), set(data["zips"]["zip_id"]), set(data["K"])
    candidates = {i: {j for j, _ in hs} for i, hs in data["zip_to_hubs"].items()}
    active = set(zip(data["active"]["zip_id"], data["active"]["part_id"]))
    assigned = {(i, k) for i, _, k in assignments}
    unknown = (
        sum(1 for j in open_hubs if j not in hubs)
        + sum(1 for j, k in stocked if j not in hubs or k not in parts)
        + sum(1 for i, j, k in assignments if i not in zips or j not in hubs or k not in parts)
    )
    open_set = set(open_hubs)
    return {
        "cost": cost,
        "audit": audit,
        "hubs_total": len(hubs),
        "open_hubs": len(open_set),
        "closed_hubs": len(hubs - open_set),
        "stocked_pairs": len(set(stocked)),
        "assignment_rows": len(assignments),
        "duplicate_assignment_rows": sum(c - 1 for c in Counter(assignments).values() if c > 1),
        "active_pairs": len(active),
        "missing_active_pairs": len(active - assigned),
        "assignments_to_ineligible_hub": sum(
            1 for i, j, _ in assignments if j not in candidates.get(i, set())
        ),
        "unknown_id_rows": unknown,
    }


def is_feasible(s: dict[str, Any]) -> bool:
    return s["audit"]["total_structural_violations"] == 0 and s["missing_active_pairs"] == 0


# ---------------------------------------------------------------------------
# Reference runs (Gurobi / SA / VA folders with summary.json, plus greedy CSV)
# ---------------------------------------------------------------------------


def discover_runs(runs_roots: list[Path], exclude: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for root in runs_roots:
        if not root.is_dir():
            continue
        # Skip the solutions being scored, unless they sit above the runs root.
        skip = None if (exclude == root or exclude in root.parents) else exclude
        for sp in sorted(root.rglob("summary.json")):
            method = RUN_FOLDER_METHOD.get(sp.parent.name)
            if method is None or (skip is not None and skip in sp.resolve().parents):
                continue
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
                ds = summary["dataset"]
                final = summary["final_solution"]
            except (OSError, ValueError, KeyError):
                continue
            extra = summary.get("extra") or {}
            cfg_path = sp.parent.parent / "run_config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            gap = num(extra.get("mip_gap_achieved"))
            if gap is None:
                gap = num(extra.get("mip_gap"))
            status = extra.get("gurobi_status")
            runs.append({
                "method": method,
                "run_dir": sp.parent,
                "summary": summary,
                "config": cfg,
                "dataset_name": str(ds.get("dataset_name")),
                "dataset_sig": (ds.get("hubs"), ds.get("parts"), ds.get("zips"), ds.get("active_demand_pairs")),
                "settings": (
                    num(ds.get("max_service_miles")),
                    num(ds.get("penalty_start_miles")),
                    norm_limit(ds.get("top_hubs_per_zip")),
                    norm_limit(ds.get("max_parts_total")),
                ),
                "summary_cost": num((final.get("cost") or {}).get("total_cost")),
                "gurobi_status": status,
                "mip_gap": gap,
                "proven_optimal": bool(status == "OPTIMAL" and gap is not None and gap <= OPTIMAL_GAP_TOL),
            })
    return runs


def gurobi_obj_bound(run: dict[str, Any]) -> float | None:
    extra = run["summary"].get("extra") or {}
    b = num(extra.get("obj_bound"))
    if b is not None:
        return b
    log = run["run_dir"] / "gurobi.log"
    if log.is_file():
        m = re.findall(r"best bound ([-+0-9.eE]+)", log.read_text(encoding="utf-8", errors="replace"))
        if m:
            return float(m[-1])
    return None


def reject_reason(run: dict[str, Any], want: tuple[Any, ...]) -> str | None:
    """Why a same-instance run cannot serve as a reference, or None if it can."""
    if run["settings"] != want:
        names = ("max_service_miles", "penalty_start_miles", "top_hubs_per_zip", "max_parts_total")
        diffs = [f"{n} {a} vs {b}" for n, a, b in zip(names, run["settings"], want) if a != b]
        return "different data settings: " + ", ".join(diffs)
    extra = run["summary"].get("extra") or {}
    if run["method"] == "gurobi" and run["summary_cost"] is None:
        return f"no incumbent (status {run['gurobi_status']})"
    if run["method"] in ("sa", "va"):
        if extra.get("full_batch_coverage") is False:
            return "partial batch coverage (time limit)"
        if not (extra.get("postprocess") or {}).get("hub_prune_enabled", False):
            return "hub prune disabled (not post-prune)"
    if run["method"] == "sa":
        sampler = str(run["config"].get("sampler", "")).lower()
        if sampler == "sqa" or any("sqa" in p.lower() for p in run["run_dir"].parts):
            return "SQA sampler, not SA"
    return None


def select_run_reference(
    method: str, inst: str, data: dict[str, Any], runs: list[dict[str, Any]], log: list[str]
) -> dict[str, Any]:
    want = resolved_settings(data)
    sig = (len(data["J"]), len(data["K"]), len(data["zips"]), len(data["active"]))
    usable: list[dict[str, Any]] = []
    for r in runs:
        if r["method"] != method:
            continue
        if r["dataset_name"] != inst:
            if r["dataset_sig"] == sig:
                log.append(f"      skip  {rel(r['run_dir'])}: same size but different instance folder "
                           f"'{r['dataset_name']}'")
            continue
        why = reject_reason(r, want)
        if why:
            log.append(f"      skip  {rel(r['run_dir'])}: {why}")
            continue
        files, missing = find_solution_files(r["run_dir"])
        if missing:
            cost, source, viol = r["summary_cost"], "from_summary", \
                r["summary"]["final_solution"].get("audit", {}).get("total_structural_violations")
            detail = f"summary {money(cost)} (no {', '.join(missing)})"
        else:
            s = score_solution(files, data)
            cost, source, viol = s["cost"]["total_cost"], "rescored", s["audit"]["total_structural_violations"]
            diff = cost - r["summary_cost"] if r["summary_cost"] is not None else None
            detail = (f"rescored {money(cost)}  summary {money(r['summary_cost'])}  "
                      f"diff {'-' if diff is None else f'{diff:.3e}'}  viol {viol}")
        tag = ""
        if method == "gurobi":
            tag = f"  [{r['gurobi_status']} gap={r['mip_gap']} proven={r['proven_optimal']}]"
        if method == "va":
            tag = f"  [seed {((r['summary'].get('extra') or {}).get('va') or {}).get('seed')}]"
        log.append(f"      cand  {rel(r['run_dir'])}: {detail}{tag}")
        if cost is not None and (viol in (None, 0) or source == "from_summary"):
            usable.append({**r, "cost": cost, "source": source, "violations": viol})
        elif cost is not None:
            log.append(f"      skip  {rel(r['run_dir'])}: rescored solution has {viol} structural violations")

    if not usable:
        return {"cost": None, "source": "", "file": "", "match": "no matching run"}
    # Gurobi: a proven optimum beats any incumbent. Everyone else: best known cost.
    key = (lambda r: (not r["proven_optimal"], r["cost"])) if method == "gurobi" else (lambda r: r["cost"])
    best = sorted(usable, key=key)[0]
    match = f"dataset_name=={inst}, settings {want} match"
    if len(usable) > 1:
        match += f"; lowest-cost of {len(usable)} usable runs"
    if method == "gurobi":
        match += "; proven optimum" if best["proven_optimal"] else "; NOT proven optimal (incumbent)"
    out = {
        "cost": best["cost"],
        "source": best["source"],
        "violations": best["violations"],
        "file": rel(best["run_dir"]),
        "match": match,
        "run": best,
    }
    log.append(f"      USE   {out['file']}  ({out['source']}, {money(out['cost'])})")
    return out


def select_greedy_reference(
    inst: str, settings: tuple[Any, ...], greedy_csv: Path, data: dict[str, Any], log: list[str]
) -> dict[str, Any]:
    blank = {"cost": None, "source": "", "file": rel(greedy_csv), "match": ""}
    if not greedy_csv.is_file():
        return {**blank, "match": "greedy CSV not found"}
    # greedy_baseline.py hard-codes load_problem_data(None, None, None, None).
    if settings != (None, None, None, None):
        return {**blank, "match": f"greedy used default settings; instance loaded with {settings}"}
    with open(greedy_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["dataset_name"] == inst]
    if not rows:
        log.append(f"      no rows for dataset_name=={inst} in {rel(greedy_csv)}")
        return {**blank, "match": "no greedy row for this instance"}
    for r in rows:
        log.append(f"      cand  {r['variant']:8} {money(float(r['total_cost']))}  viol {r['structural_violations']}")
    best = min(rows, key=lambda r: float(r["total_cost"]))
    greedy_dir = greedy_csv.parent / inst
    files, missing = find_solution_files(greedy_dir)
    if not missing:
        s = score_solution(files, data)
        cost, source, viol = s["cost"]["total_cost"], "rescored", s["audit"]["total_structural_violations"]
        file = rel(greedy_dir)
    else:
        cost, source, viol = float(best["total_cost"]), "from_summary", int(best["structural_violations"])
        file = rel(greedy_csv)
        log.append(f"      no solution files in {rel(greedy_dir)} -> from_summary")
    log.append(f"      USE   {file} variant={best['variant']} ({source}, {money(cost)})")
    return {
        "cost": cost,
        "source": source,
        "violations": viol,
        "file": file,
        "match": f"dataset_name=={inst}; best of {len(rows)} variants; greedy_baseline.py loads with defaults",
        "variant": best["variant"],
    }


# ---------------------------------------------------------------------------
# AI claimed cost
# ---------------------------------------------------------------------------

CLAIM_PATTERNS = [
    re.compile(r"\|\s*\*\*Total\*\*\s*\|\s*\*\*\$?([0-9][0-9,]*(?:\.[0-9]+)?)\*\*"),
    re.compile(r"\|\s*Total(?: cost)?\s*\|\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)\s*\|", re.I),
    re.compile(r"total cost[^0-9\n$]{0,20}\$?([0-9][0-9,]*(?:\.[0-9]+)?)", re.I),
]


def parse_claimed_cost(report: Path) -> float | None:
    if not report.is_file():
        return None
    text = report.read_text(encoding="utf-8", errors="replace")
    for pat in CLAIM_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


# ---------------------------------------------------------------------------
# Main flows
# ---------------------------------------------------------------------------


def resolve_solution_dir(solutions_root: Path, inst: str) -> Path:
    nested = solutions_root / "outputs" / inst
    return nested if nested.is_dir() else solutions_root / inst


def check_gurobi(inst: str, instances_root: Path, runs: list[dict[str, Any]]) -> int:
    print(BAR)
    print(f"SCORER CHECK -- Gurobi reference solution for {inst}")
    print(BAR)
    settings, source, cfg_lines = gurobi_load_settings(inst, runs)
    print("  Gurobi run_config.json files:")
    print("\n".join(cfg_lines) or "    (none)")
    print(f"  load settings: {settings}  ({source})")
    data = load_instance(instances_root / inst, settings)
    log: list[str] = []
    ref = select_run_reference("gurobi", inst, data, runs, log)
    print("\n".join(log))
    if ref["cost"] is None:
        print("  no usable Gurobi run")
        return 1
    run = ref["run"]
    files, missing = find_solution_files(run["run_dir"])
    if missing:
        print(f"  Gurobi run has no {missing}; cannot re-score")
        return 1
    print("  files scored:")
    for k, p in files.items():
        print(f"    {k:14} {rel(p)}")
    s = score_solution(files, data)
    summary = run["summary"]["final_solution"]
    ok = True          # every count and audit field identical, costs within rounding
    bit_exact = True   # additionally, every cost field identical to the last bit
    worst_rel = 0.0
    print(f"\n  {'field':34} {'scorer':>24} {'summary.json':>24} {'diff':>11}  exact")
    for f in COST_FIELDS:
        a, b = s["cost"][f], float(summary["cost"][f])
        rel_diff = abs(a - b) / max(1.0, abs(b))
        worst_rel = max(worst_rel, rel_diff)
        bit_exact &= a == b
        ok &= rel_diff <= FP_REL_TOL
        print(f"  {f:34} {a!r:>24} {b!r:>24} {a - b:11.3e}  {a == b}")
    for f in AUDIT_FIELDS:
        a, b = s["audit"][f], int(summary["audit"][f])
        ok &= a == b
        print(f"  {'audit.' + f:34} {a:22d} {b:22d} {a - b:11d}  {a == b}")
    for f, sf in (("open_hubs", "open_hubs_count"), ("closed_hubs", "closed_hubs_count"),
                  ("stocked_pairs", "stocked_pairs_count"), ("assignment_rows", "assignments_count")):
        a, b = s[f], int(summary[sf])
        ok &= a == b
        print(f"  {f:34} {a:22d} {b:22d} {a - b:11d}  {a == b}")
    print(f"\n  missing active pairs {s['missing_active_pairs']}, ineligible-hub assignments "
          f"{s['assignments_to_ineligible_hub']}, unknown ids {s['unknown_id_rows']}, feasible {is_feasible(s)}")
    if ok and bit_exact:
        verdict = "PASS -- every field matches summary.json bit for bit"
    elif ok:
        verdict = (f"PASS -- counts and audit identical; costs agree to {worst_rel:.1e} relative. The last-bit "
                   "difference is summation order: compute_solution_cost() sums over a Python set, whose "
                   "order depends on PYTHONHASHSEED, so bit-exact agreement with another process is not "
                   "guaranteed")
    else:
        verdict = f"MISMATCH (worst relative cost difference {worst_rel:.3e})"
    print(f"\n  RESULT: {verdict}")
    return 0 if ok else 1


def score_instance(
    inst_dir: Path, args: argparse.Namespace, runs: list[dict[str, Any]], solutions_root: Path
) -> dict[str, Any]:
    inst = inst_dir.name
    print(BAR)
    print(f"{inst}")
    print(BAR)
    settings, settings_source, cfg_lines = gurobi_load_settings(inst, runs)
    print("  Gurobi run_config.json files:")
    print("\n".join(cfg_lines) or "    (none)")
    data = load_instance(inst_dir, settings)
    msm, psm, top, mpt = resolved_settings(data)
    print(f"  loaded with max_service_miles={msm} penalty_start_miles={psm} "
          f"top_hubs_per_zip={top} max_parts_total={mpt}  ({settings_source})")

    row: dict[str, Any] = {c: "" for c in COLUMNS}
    row.update({
        "label": args.label, "instance": inst,
        "max_service_miles": msm, "penalty_start_miles": psm,
        "top_hubs_per_zip": "" if top is None else top, "max_parts_total": "" if mpt is None else mpt,
        "settings_source": settings_source,
    })
    notes: list[str] = []

    sol_dir = resolve_solution_dir(solutions_root, inst)
    row["solution_dir"] = rel(sol_dir)
    files, missing = find_solution_files(sol_dir) if sol_dir.is_dir() else ({}, ["folder"])
    s: dict[str, Any] | None = None
    if missing:
        row["status"] = "no_solution"
        notes.append(f"missing {', '.join(missing)} in {rel(sol_dir)}")
        print(f"  solution: NONE ({notes[-1]})")
    else:
        row["status"] = "scored"
        print("  solution files:")
        for k, p in files.items():
            print(f"    {k:14} {rel(p)}")
        s = score_solution(files, data)
        for k in ("hubs_total", "open_hubs", "closed_hubs", "stocked_pairs", "assignment_rows",
                  "duplicate_assignment_rows", "active_pairs", "missing_active_pairs",
                  "assignments_to_ineligible_hub", "unknown_id_rows"):
            row[k] = s[k]
        row.update(s["cost"])
        row.update({"audit_" + f: s["audit"][f] for f in AUDIT_FIELDS})
        row["feasible"] = is_feasible(s)
        if s["assignments_to_ineligible_hub"]:
            notes.append("ineligible-hub assignments are costed at d_ij=0 by compute_solution_cost, "
                         "so transport cost is understated")
        if s["unknown_id_rows"]:
            notes.append(f"{s['unknown_id_rows']} rows reference ids not in the instance")

    claimed = parse_claimed_cost(sol_dir / "report.md")
    row["claimed_total_cost"] = "" if claimed is None else claimed
    true_total = s["cost"]["total_cost"] if s else None
    if claimed is not None and true_total is not None:
        row["claimed_minus_true"] = claimed - true_total

    print("  references:")
    refs: dict[str, dict[str, Any]] = {}
    for m in METHODS:
        log: list[str] = []
        print(f"    {m}:")
        if m == "greedy":
            refs[m] = select_greedy_reference(inst, settings, args.greedy_csv, data, log)
        else:
            refs[m] = select_run_reference(m, inst, data, runs, log)
        print("\n".join(log) if log else "      (no candidate runs)")
        if refs[m]["cost"] is None:
            print(f"      -> blank: {refs[m]['match']}")
        ref = refs[m]
        row[f"{m}_cost"] = "" if ref["cost"] is None else ref["cost"]
        g = gap_pct(true_total, ref["cost"])
        row[f"{m}_gap_pct"] = "" if g is None else g
        row[f"{m}_source"] = ref["source"]
        row[f"{m}_structural_violations"] = "" if ref["cost"] is None else ref.get("violations", "")
        row[f"{m}_file"] = ref["file"] if ref["cost"] is not None else ""
        row[f"{m}_match"] = ref["match"]

    if refs["gurobi"]["cost"] is not None:
        run = refs["gurobi"]["run"]
        bound = gurobi_obj_bound(run)
        row.update({
            "gurobi_status": run["gurobi_status"],
            "gurobi_proven_optimal": run["proven_optimal"],
            "gurobi_mip_gap": "" if run["mip_gap"] is None else run["mip_gap"],
            "gurobi_obj_bound": "" if bound is None else bound,
        })
        g = gap_pct(true_total, bound)
        row["gurobi_bound_gap_pct"] = "" if g is None else g
    row["greedy_variant"] = refs["greedy"].get("variant", "")
    if refs["va"]["cost"] is not None:
        row["va_seed"] = ((refs["va"]["run"]["summary"].get("extra") or {}).get("va") or {}).get("seed", "")

    row["notes"] = "; ".join(notes)
    if s:
        c = s["cost"]
        print(f"  AI total {money(c['total_cost'])}  (inventory {money(c['inventory_cost'])}, fixed "
              f"{money(c['fixed_open_hub_cost'])}, overflow {money(c['overflow_storage_cost'])}, transfer "
              f"{money(c['new_hub_transfer_cost'])}, transport {money(c['assignment_transport_cost'])})")
        print(f"  structural violations {s['audit']['total_structural_violations']}, missing pairs "
              f"{s['missing_active_pairs']}, feasible {row['feasible']}")
    if claimed is not None:
        diff = row["claimed_minus_true"]
        print(f"  claimed total {money(claimed)}  claimed - true {'-' if diff == '' else f'{diff:.4f}'}")
    return row


def append_rows(out: Path, rows: list[dict[str, Any]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.is_file() and out.stat().st_size > 0
    if exists:
        with open(out, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if header != COLUMNS:
            raise SystemExit(f"{out} has a different header; refusing to append misaligned rows")
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    def f(v: Any, kind: str) -> str:
        if v == "" or v is None:
            return "-"
        return money(float(v)) if kind == "money" else pct(float(v))

    print("\n" + BAR)
    print(f"SUMMARY -- label {rows[0]['label'] if rows else ''}")
    print(BAR)
    head = (f"  {'instance':18} {'status':11} {'feas':5} {'AI total':>17} {'claim-true':>11} "
            f"{'vs Gurobi':>10} {'vs greedy':>10} {'vs SA':>10} {'vs VA':>10} {'hubs':>5}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        star = "" if r["gurobi_proven_optimal"] in ("", True) else "*"
        cm = r["claimed_minus_true"]
        print(f"  {r['instance']:18} {r['status']:11} {str(r['feasible']) if r['feasible'] != '' else '-':5} "
              f"{f(r['total_cost'], 'money'):>17} {('-' if cm == '' else f'{cm:.2e}'):>11} "
              f"{f(r['gurobi_gap_pct'], 'pct') + star:>10} {f(r['greedy_gap_pct'], 'pct'):>10} "
              f"{f(r['sa_gap_pct'], 'pct'):>10} {f(r['va_gap_pct'], 'pct'):>10} "
              f"{r['open_hubs'] if r['open_hubs'] != '' else '-':>5}")
    print("  gap = 100*(AI - reference)/reference; negative means the AI is cheaper.")
    if any(r["gurobi_proven_optimal"] is False for r in rows):
        print("  * Gurobi reference is an incumbent, not a proven optimum (see gurobi_mip_gap / gurobi_obj_bound).")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances-root", required=True, type=Path,
                    help="Folder with one subfolder per instance (six CSVs each).")
    ap.add_argument("--solutions-root", type=Path,
                    help="Folder with outputs/<instance>/ (or <instance>/) holding the three solution CSVs.")
    ap.add_argument("--label", default="", help="Written into every output row, e.g. opus55_high_code.")
    ap.add_argument("--out", default=ROOT / "results/ai_baseline/ai_scores.csv", type=Path,
                    help="Output CSV; rows are appended.")
    ap.add_argument("--instance", action="append", default=[],
                    help="Score only this instance name. Repeat for several.")
    ap.add_argument("--runs-roots", nargs="+", type=Path, default=[ROOT / "outputs", ROOT / "results"],
                    help="Where to look for Gurobi/SA/VA run folders.")
    ap.add_argument("--greedy-csv", type=Path, default=ROOT / "results/greedy/greedy_ladder.csv")
    ap.add_argument("--check-gurobi", metavar="INSTANCE", default="",
                    help="Score the Gurobi reference solution for INSTANCE against its summary.json and exit.")
    args = ap.parse_args()
    if not args.check_gurobi and (args.solutions_root is None or not args.label):
        ap.error("--solutions-root and --label are required unless --check-gurobi is given")
    return args


def main() -> int:
    args = parse_args()
    instances_root = args.instances_root.expanduser().resolve()
    solutions_root = (args.solutions_root or instances_root).expanduser().resolve()
    runs = discover_runs([p.expanduser().resolve() for p in args.runs_roots], exclude=solutions_root)

    if args.check_gurobi:
        return check_gurobi(args.check_gurobi, instances_root, runs)

    instances = discover_instances(instances_root)
    if args.instance:
        wanted = set(args.instance)
        instances = [d for d in instances if d.name in wanted]
        unknown = wanted - {d.name for d in instances}
        if unknown:
            raise SystemExit(f"not found under {instances_root}: {sorted(unknown)}")
    if not instances:
        raise SystemExit(f"no instance folders with {list(REQUIRED_FILES)} under {instances_root}")

    rows = [score_instance(d, args, runs, solutions_root) for d in instances]
    append_rows(args.out, rows)
    print_table(rows)
    print(f"\n  appended {len(rows)} rows to {rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
