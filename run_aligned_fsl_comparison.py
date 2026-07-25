#!/usr/bin/env python3
"""
Aligned one-command runner for the Dell FSL risk optimization instance.

This file consolidates the two workflows into one command:

    python run_aligned_fsl_comparison.py --dataset-dir instances_low --solver both

What it aligns:
- Both Gurobi and QUBO use the same transport/risk cost basis:
    lambda_1 * h_s * b_ik
  + lambda_2 * h_d * b_ik * max(0, d_ij - base_miles)
  + lambda_3 * b_ik * max(0, d_ij - penalty_start_miles)
- Both solvers report structural feasibility separately from SLA distance
  violations. SLA violations are assignments with d_ij > penalty_start_miles.
- The parent process launches each solver in a separate child process. That keeps
  memory accounting cleaner while still requiring only this one Python file.

Required packages:
    pip install pandas psutil openjij gurobipy

OpenJij is required only for --solver qubo or --solver both.
Gurobi/gurobipy is required only for --solver gurobi or --solver both.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

try:
    import psutil  # type: ignore
except ModuleNotFoundError:
    psutil = None


SCALAR_COLUMNS = (
    "C",
    "h_s",
    "h_d",
    "d_s",
    "L",
    "S_lim",
    "S_var",
    "lambda_1",
    "lambda_2",
    "lambda_3",
    "base_miles",
    "penalty_start_miles",
    "max_service_miles",
)

REQUIRED_FILES = (
    "demand.csv",
    "distances.csv",
    "hubs.csv",
    "parameters.csv",
    "parts.csv",
    "zips.csv",
)

ID_COLUMNS = {"hub_id", "zip_id", "part_id", "anchor_id", "region_code"}


@dataclass(frozen=True)
class BatchSpec:
    batch_id: int
    row_indices: list[int]
    estimated_z_vars: int
    note: str = ""


@dataclass
class BatchResult:
    batch_id: int
    num_rows: int
    num_parts: int
    num_z: int
    num_y: int
    num_x: int
    interactions: int
    build_seconds: float
    sample_seconds: float
    eval_seconds: float
    total_seconds: float
    stage_used: int
    suggested_reads: int
    energy: float
    reconstructed_cost: float
    c1_violations: int
    c2_violations: int
    c3_violations: int
    c4_violations: int
    penalty_c1: float
    penalty_c2: float
    penalty_c3: float
    open_hubs: list[str]
    stocked_pairs: list[tuple[str, str]]
    assignments: list[tuple[str, str, str]]
    adaptive_iterations_used: int = 0
    adaptive_was_feasible: bool = False
    final_penalty_multipliers: dict[str, float] = field(
        default_factory=lambda: {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    )
    adaptive_iteration_log: list[dict] = field(default_factory=list)
    adaptive_exit_reason: str = ""


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def money(x: float | int | None) -> str:
    if x is None:
        return "n/a"
    return f"${float(x):,.2f}"


def fmt_number(x: float | int | None, decimals: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{float(x):,.{decimals}f}"


def peak_or_current_rss_mb() -> float:
    """Best-effort peak or current RSS in MB, cross-platform."""
    if psutil is not None:
        try:
            proc = psutil.Process()
            if hasattr(proc, "memory_full_info"):
                info = proc.memory_full_info()
                peak = getattr(info, "peak_wset", None)  # Windows peak working set
                if peak:
                    return float(peak) / (1024.0 * 1024.0)
            return float(proc.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            pass

    try:
        import resource  # type: ignore

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:
        return 0.0


def memory_report_mb() -> tuple[float, float]:
    current_mb = 0.0
    peak_mb = 0.0
    try:
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024.0 * 1024.0)
        peak_mb = peak / (1024.0 * 1024.0)
    except Exception:
        pass

    rss = peak_or_current_rss_mb()
    current_mb = max(current_mb, rss)
    peak_mb = max(peak_mb, current_mb)
    return peak_mb, current_mb


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in ID_COLUMNS or col.endswith("_id"):
            out[col] = out[col].astype(str)
    return out


def read_csv_required(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return normalize_id_columns(pd.read_csv(path))


def require_columns(df: pd.DataFrame, filename: str, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")


def load_problem_data(
    dataset_dir: Path | str,
    *,
    max_service_miles_override: float | None,
    penalty_start_miles_override: float | None,
    top_hubs_per_zip: int | None,
    max_parts_total: int | None,
) -> dict[str, Any]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    missing_files = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Dataset directory is missing required files: {missing_files}")

    params = read_csv_required(root, "parameters.csv")
    if params.empty:
        raise ValueError(f"Empty parameters.csv in {root}")
    row = params.iloc[0].to_dict()
    missing_scalars = [c for c in SCALAR_COLUMNS if c not in row]
    if missing_scalars:
        raise ValueError(f"parameters.csv missing columns: {missing_scalars}")

    scalar = {c: float(row[c]) for c in SCALAR_COLUMNS}
    scalar["L"] = int(float(scalar["L"]))
    if max_service_miles_override is not None and float(max_service_miles_override) > 0:
        scalar["max_service_miles"] = float(max_service_miles_override)
    if penalty_start_miles_override is not None and float(penalty_start_miles_override) >= 0:
        scalar["penalty_start_miles"] = float(penalty_start_miles_override)

    hubs = read_csv_required(root, "hubs.csv")
    parts = read_csv_required(root, "parts.csv")
    zips = read_csv_required(root, "zips.csv")
    demand = read_csv_required(root, "demand.csv")
    distances = read_csv_required(root, "distances.csv")

    require_columns(hubs, "hubs.csv", ["hub_id", "T_j"])
    require_columns(parts, "parts.csv", ["part_id"])
    require_columns(zips, "zips.csv", ["zip_id"])
    require_columns(demand, "demand.csv", ["zip_id", "part_id", "Q_ik", "b_ik"])
    require_columns(distances, "distances.csv", ["zip_id", "hub_id", "d_ij"])

    price_col = "P_k" if "P_k" in parts.columns else "price"
    if price_col not in parts.columns:
        raise ValueError("parts.csv must contain P_k or price")
    if price_col != "P_k":
        parts = parts.rename(columns={price_col: "P_k"})

    hubs = hubs.copy()
    parts = parts.copy()
    zips = zips.copy()
    demand = demand.copy()
    distances = distances.copy()

    hubs["T_j"] = pd.to_numeric(hubs["T_j"], errors="raise").astype(int)
    if "B_j" in hubs.columns:
        hubs["B_j"] = pd.to_numeric(hubs["B_j"], errors="coerce").fillna(0.0)
    else:
        hubs["B_j"] = 0.0
    parts["P_k"] = pd.to_numeric(parts["P_k"], errors="raise")
    demand["Q_ik"] = pd.to_numeric(demand["Q_ik"], errors="raise")
    demand["b_ik"] = pd.to_numeric(demand["b_ik"], errors="raise")
    distances["d_ij"] = pd.to_numeric(distances["d_ij"], errors="raise")

    if hubs.duplicated("hub_id").any():
        raise ValueError("hubs.csv has duplicate hub_id values")
    if parts.duplicated("part_id").any():
        raise ValueError("parts.csv has duplicate part_id values")

    selected_parts = sorted(parts["part_id"].unique().tolist())
    if max_parts_total is not None and int(max_parts_total) > 0:
        selected_parts = selected_parts[: int(max_parts_total)]
    selected_parts_set = set(selected_parts)
    parts = parts[parts["part_id"].isin(selected_parts_set)].copy()
    demand = demand[demand["part_id"].isin(selected_parts_set)].copy()

    missing_parts = sorted(set(demand.loc[demand["Q_ik"] > 0, "part_id"]) - set(parts["part_id"]))
    if missing_parts:
        raise ValueError(f"parts.csv is missing demanded parts. Examples: {missing_parts[:10]}")
    missing_zips = sorted(set(demand.loc[demand["Q_ik"] > 0, "zip_id"]) - set(zips["zip_id"]))
    if missing_zips:
        raise ValueError(f"zips.csv is missing demanded ZIPs. Examples: {missing_zips[:10]}")

    active = demand[demand["Q_ik"] > 0][["zip_id", "part_id", "b_ik"]].copy()
    active = (
        active.groupby(["zip_id", "part_id"], as_index=False, sort=False)["b_ik"]
        .sum()
        .reset_index(drop=True)
    )
    if active.empty:
        raise ValueError("No active demand rows after filtering")

    max_service_miles = float(scalar["max_service_miles"])
    dist_f = distances[distances["d_ij"] <= max_service_miles].copy()
    if dist_f.empty:
        raise ValueError(f"No distance rows with d_ij <= max_service_miles={max_service_miles}")

    dist_f = dist_f.sort_values(["zip_id", "d_ij", "hub_id"])
    if top_hubs_per_zip is not None and int(top_hubs_per_zip) > 0:
        dist_f = dist_f.groupby("zip_id", sort=False).head(int(top_hubs_per_zip)).copy()

    zip_to_hubs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    d_map: dict[tuple[str, str], float] = {}
    for r in dist_f[["zip_id", "hub_id", "d_ij"]].itertuples(index=False):
        i, j, dij = str(r.zip_id), str(r.hub_id), float(r.d_ij)
        zip_to_hubs[i].append((j, dij))
        d_map[(i, j)] = dij
    for i in list(zip_to_hubs):
        zip_to_hubs[i].sort(key=lambda t: (t[1], t[0]))

    no_candidate = sorted(set(active["zip_id"]) - set(zip_to_hubs))
    if no_candidate:
        examples = active[active["zip_id"].isin(no_candidate)][["zip_id", "part_id"]].head(10)
        raise ValueError(
            "Some active ZIP-part pairs have no eligible hub within max_service_miles. Examples:\n"
            + examples.to_string(index=False)
        )

    active_parts = set(active["part_id"].unique().tolist())
    part_order = [p for p in selected_parts if p in active_parts]

    p_map = {str(r.part_id): float(r.P_k) for r in parts[["part_id", "P_k"]].itertuples(index=False)}
    t_map = {str(r.hub_id): int(r.T_j) for r in hubs[["hub_id", "T_j"]].itertuples(index=False)}
    b_cap_map = {str(r.hub_id): float(r.B_j) for r in hubs[["hub_id", "B_j"]].itertuples(index=False)}
    b_demand = {(str(r.zip_id), str(r.part_id)): float(r.b_ik) for r in active.itertuples(index=False)}

    baseline_path = root / "optional_baseline_part_homes.csv"
    parameter_key_path = root / "optional_parameter_key.csv"

    return {
        "dataset_dir": str(root),
        "dataset_name": root.name,
        "scalar": scalar,
        "hubs": hubs,
        "parts": parts,
        "zips": zips,
        "demand": demand,
        "distances_filtered": dist_f,
        "active": active,
        "part_order": part_order,
        "P": p_map,
        "T": t_map,
        "B_j": b_cap_map,
        "B": b_demand,
        "D": d_map,
        "zip_to_hubs": dict(zip_to_hubs),
        "J": sorted(hubs["hub_id"].unique().tolist()),
        "K": selected_parts,
        "max_service_miles": max_service_miles,
        "top_hubs_per_zip": top_hubs_per_zip,
        "max_parts_total": max_parts_total,
        "baseline_part_homes_rows": len(pd.read_csv(baseline_path)) if baseline_path.is_file() else 0,
        "parameter_key_rows": len(pd.read_csv(parameter_key_path)) if parameter_key_path.is_file() else 0,
    }


def assignment_cost_by_values(b: float, dij: float, scalar: dict[str, Any]) -> dict[str, float]:
    miles_after_base = max(0.0, float(dij) - float(scalar["base_miles"]))
    miles_after_penalty_start = max(0.0, float(dij) - float(scalar["penalty_start_miles"]))
    linehaul = float(scalar["lambda_1"]) * float(scalar["h_s"]) * float(b)
    distance = float(scalar["lambda_2"]) * float(scalar["h_d"]) * float(b) * miles_after_base
    penalty = float(scalar["lambda_3"]) * float(b) * miles_after_penalty_start
    return {
        "b_ik": float(b),
        "d_ij": float(dij),
        "miles_after_base": float(miles_after_base),
        "miles_after_penalty_start": float(miles_after_penalty_start),
        "linehaul_cost": float(linehaul),
        "distance_cost": float(distance),
        "distance_penalty_cost": float(penalty),
        "assignment_cost": float(linehaul + distance + penalty),
        "sla_violation": bool(float(dij) > float(scalar["penalty_start_miles"])),
    }


def assignment_cost(zip_id: str, hub_id: str, part_id: str, data: dict[str, Any]) -> dict[str, float]:
    b = float(data["B"].get((str(zip_id), str(part_id)), 0.0))
    dij = float(data["D"].get((str(zip_id), str(hub_id)), 0.0))
    return assignment_cost_by_values(b, dij, data["scalar"])


def compute_solution_cost(
    assignments: Iterable[tuple[str, str, str]],
    stocked_pairs: Iterable[tuple[str, str]],
    open_hubs: Iterable[str],
    data: dict[str, Any],
) -> dict[str, float]:
    scalar = data["scalar"]
    assignment_set = set((str(i), str(j), str(k)) for i, j, k in assignments)
    stocked_set = set((str(j), str(k)) for j, k in stocked_pairs)
    open_set = set(str(j) for j in open_hubs)

    inventory_cost = sum(float(data["P"].get(k, 0.0)) for _, k in stocked_set)
    fixed_open_cost = float(scalar["S_lim"]) * float(len(open_set))
    transfer_cost = sum((1 - int(data["T"].get(j, 0))) * float(scalar["C"]) for j, _ in stocked_set)

    by_hub: dict[str, int] = defaultdict(int)
    for j, _ in stocked_set:
        by_hub[j] += 1
    overflow_units = sum(max(0, cnt - int(scalar["L"])) for cnt in by_hub.values())
    overflow_cost = float(scalar["S_var"]) * float(overflow_units)

    transport_cost = 0.0
    for i, j, k in assignment_set:
        transport_cost += assignment_cost(i, j, k, data)["assignment_cost"]

    total = inventory_cost + fixed_open_cost + overflow_cost + transfer_cost + transport_cost
    return {
        "total_cost": float(total),
        "inventory_cost": float(inventory_cost),
        "fixed_open_hub_cost": float(fixed_open_cost),
        "overflow_storage_cost": float(overflow_cost),
        "new_hub_transfer_cost": float(transfer_cost),
        "assignment_transport_cost": float(transport_cost),
        "overflow_units": float(overflow_units),
    }


def global_audit(
    assignments: Iterable[tuple[str, str, str]],
    stocked_pairs: Iterable[tuple[str, str]],
    open_hubs: Iterable[str],
    data: dict[str, Any],
) -> dict[str, int]:
    assignment_list = [(str(i), str(j), str(k)) for i, j, k in assignments]
    stocked_set = set((str(j), str(k)) for j, k in stocked_pairs)
    open_set = set(str(j) for j in open_hubs)

    active_pairs = set(
        (str(r.zip_id), str(r.part_id))
        for r in data["active"][["zip_id", "part_id"]].itertuples(index=False)
    )
    candidate_by_zip = {i: {j for j, _ in hubs} for i, hubs in data["zip_to_hubs"].items()}

    assignment_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    invalid_hub = 0
    max_service_violations = 0
    sla_distance_violations = 0
    for i, j, k in assignment_list:
        assignment_map[(i, k)].append(j)
        if j not in candidate_by_zip.get(i, set()):
            invalid_hub += 1
        dij = float(data["D"].get((i, j), math.inf))
        if not math.isfinite(dij) or dij > float(data["scalar"]["max_service_miles"]):
            max_service_violations += 1
        if math.isfinite(dij) and dij > float(data["scalar"]["penalty_start_miles"]):
            sla_distance_violations += 1

    missing = 0
    multiple = 0
    for pair in active_pairs:
        count = len(assignment_map.get(pair, []))
        if count == 0:
            missing += 1
        elif count != 1:
            multiple += 1
    extra = sum(1 for pair in assignment_map if pair not in active_pairs)

    c1 = missing + multiple + invalid_hub + extra
    c2 = sum(1 for _, j, k in assignment_list if (j, k) not in stocked_set)
    c3 = sum(1 for j, _ in stocked_set if j not in open_set)

    by_hub: dict[str, int] = defaultdict(int)
    for j, _ in stocked_set:
        by_hub[j] += 1
    l_cap = int(data["scalar"]["L"])
    c4_hubs_over_l = sum(1 for c in by_hub.values() if c > l_cap)
    c4_overflow_units = sum(max(0, c - l_cap) for c in by_hub.values())

    structural = c1 + c2 + c3
    return {
        "c1_assignment_violations": int(c1),
        "c1_missing_assignments": int(missing),
        "c1_multiple_assignments": int(multiple),
        "c1_invalid_hub_assignments": int(invalid_hub),
        "c1_extra_assignments": int(extra),
        "c2_assignment_without_stock": int(c2),
        "c3_stock_without_open_hub": int(c3),
        "c4_hubs_over_L": int(c4_hubs_over_l),
        "c4_total_overflow_units": int(c4_overflow_units),
        "max_service_distance_violations": int(max_service_violations),
        "sla_distance_violations": int(sla_distance_violations),
        "total_structural_violations": int(structural),
    }


def assignment_rows_dataframe(
    assignments: Iterable[tuple[str, str, str]],
    data: dict[str, Any],
    assignment_sources: dict[tuple[str, str, str], str] | None = None,
) -> pd.DataFrame:
    rows = []
    sources = assignment_sources or {}
    for i, j, k in sorted((str(i), str(j), str(k)) for i, j, k in assignments):
        c = assignment_cost(i, j, k, data)
        rows.append(
            {
                "zip_id": i,
                "hub_id": j,
                "part_id": k,
                "b_ik": c["b_ik"],
                "d_ij": c["d_ij"],
                "miles_after_base": c["miles_after_base"],
                "miles_after_penalty_start": c["miles_after_penalty_start"],
                "sla_violation": int(bool(c["sla_violation"])),
                "linehaul_cost": c["linehaul_cost"],
                "distance_cost": c["distance_cost"],
                "distance_penalty_cost": c["distance_penalty_cost"],
                "assignment_cost": c["assignment_cost"],
                "source": sources.get((i, j, k), "raw_or_solver"),
            }
        )
    return pd.DataFrame(rows)


def stocked_pairs_dataframe(stocked_pairs: Iterable[tuple[str, str]], data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for j, k in sorted((str(j), str(k)) for j, k in stocked_pairs):
        rows.append(
            {
                "hub_id": j,
                "part_id": k,
                "part_cost_P_k": float(data["P"].get(k, 0.0)),
                "hub_existing_T_j": int(data["T"].get(j, 0)),
                "new_hub_transfer_cost": (1 - int(data["T"].get(j, 0))) * float(data["scalar"]["C"]),
            }
        )
    return pd.DataFrame(rows)


def hub_status_dataframe(
    open_hubs: Iterable[str],
    stocked_pairs: Iterable[tuple[str, str]],
    assignments: Iterable[tuple[str, str, str]],
    data: dict[str, Any],
) -> pd.DataFrame:
    open_set = set(str(j) for j in open_hubs)
    stock_count: dict[str, int] = defaultdict(int)
    assign_count: dict[str, int] = defaultdict(int)
    for j, _ in stocked_pairs:
        stock_count[str(j)] += 1
    for _, j, _ in assignments:
        assign_count[str(j)] += 1

    rows = []
    for j in data["J"]:
        opened = j in open_set
        existing = int(data["T"].get(j, 0))
        rows.append(
            {
                "hub_id": j,
                "status": "OPEN" if opened else "CLOSED",
                "opened": int(opened),
                "closed": int(not opened),
                "existing_T_j": existing,
                "opened_new_hub": int(opened and existing == 0),
                "stocked_part_count": int(stock_count.get(j, 0)),
                "assignment_count": int(assign_count.get(j, 0)),
                "fixed_open_cost_if_open": float(data["scalar"]["S_lim"]) if opened else 0.0,
            }
        )
    return pd.DataFrame(rows)


def final_results_block(summary: dict[str, Any], title: str) -> str:
    sol = summary["final_solution"]
    cost = sol["cost"]
    audit = sol["audit"]
    rt = summary["runtime"]
    lines = [
        "=" * 76,
        title,
        "=" * 76,
        f"  total cost:                 {money(cost['total_cost'])}",
        f"  inventory cost:             {money(cost['inventory_cost'])}",
        f"  fixed open-hub cost:        {money(cost['fixed_open_hub_cost'])}",
        f"  overflow storage cost:      {money(cost['overflow_storage_cost'])}",
        f"  new-hub transfer cost:      {money(cost['new_hub_transfer_cost'])}",
        f"  assignment transport cost:  {money(cost['assignment_transport_cost'])}",
        f"  open hubs:                  {sol['open_hubs_count']:,}",
        f"  closed hubs:                {sol['closed_hubs_count']:,}",
        f"  stocked hub-part pairs:     {sol['stocked_pairs_count']:,}",
        f"  hub-zip-part pairings:      {sol['assignments_count']:,}",
        f"  structural violations:      {audit['total_structural_violations']:,}",
        f"  SLA distance violations:    {audit['sla_distance_violations']:,}",
        f"  wall time:                  {rt['wall_seconds']:,.2f}s",
        f"  peak/current memory:        {rt['peak_memory_mb']:,.1f}/{rt['current_memory_mb']:,.1f} MB",
        f"  output folder:              {summary['output_dir']}",
        "=" * 76,
    ]
    return "\n".join(lines)


def write_solution_outputs(
    run_dir: Path,
    *,
    solver_name: str,
    data: dict[str, Any],
    assignments: list[tuple[str, str, str]],
    stocked_pairs: list[tuple[str, str]],
    open_hubs: list[str],
    runtime: dict[str, Any],
    extra: dict[str, Any] | None = None,
    assignment_sources: dict[tuple[str, str, str], str] | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    closed_count = int(len(data["J"]) - len(set(open_hubs)))
    cost = compute_solution_cost(assignments, stocked_pairs, open_hubs, data)
    audit = global_audit(assignments, stocked_pairs, open_hubs, data)

    assignment_rows_dataframe(assignments, data, assignment_sources).to_csv(
        run_dir / "hub_zip_part_pairings.csv", index=False
    )
    stocked_pairs_dataframe(stocked_pairs, data).to_csv(run_dir / "stocked_hub_part_pairs.csv", index=False)
    stocked_pairs_dataframe(stocked_pairs, data).to_csv(run_dir / "stocked_pairs.csv", index=False)
    hub_status_dataframe(open_hubs, stocked_pairs, assignments, data).to_csv(run_dir / "hubs_open_closed.csv", index=False)
    pd.DataFrame({"hub_id": sorted(set(open_hubs))}).to_csv(run_dir / "open_hubs.csv", index=False)
    pd.DataFrame({"hub_id": sorted(set(data["J"]) - set(open_hubs))}).to_csv(run_dir / "closed_hubs.csv", index=False)

    summary = {
        "solver": solver_name,
        "dataset": {
            "dataset_name": data["dataset_name"],
            "dataset_dir": data["dataset_dir"],
            "hubs": int(len(data["J"])),
            "parts": int(len(data["K"])),
            "zips": int(len(data["zips"])),
            "active_demand_pairs": int(len(data["active"])),
            "max_service_miles": float(data["scalar"]["max_service_miles"]),
            "base_miles": float(data["scalar"]["base_miles"]),
            "penalty_start_miles": float(data["scalar"]["penalty_start_miles"]),
            "top_hubs_per_zip": data["top_hubs_per_zip"],
            "max_parts_total": data["max_parts_total"],
        },
        "final_solution": {
            "open_hubs_count": int(len(set(open_hubs))),
            "closed_hubs_count": closed_count,
            "stocked_pairs_count": int(len(set(stocked_pairs))),
            "assignments_count": int(len(set(assignments))),
            "cost": cost,
            "audit": audit,
        },
        "runtime": runtime,
        "output_dir": str(run_dir),
        "extra": extra or {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.txt").write_text(final_results_block(summary, f"{solver_name.upper()} FINAL RESULTS") + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Gurobi solver
# ---------------------------------------------------------------------------


def _resolve_gurobi_license(explicit: str | None) -> None:
    if not explicit:
        return
    path = Path(str(explicit)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gurobi license file not found: {path}")
    os.environ["GRB_LICENSE_FILE"] = str(path)


def gurobi_status_name(GRB: Any, status_code: int) -> str:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_map.get(status_code, f"STATUS_{status_code}")


def run_gurobi_solver(args: argparse.Namespace) -> dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    _resolve_gurobi_license(getattr(args, "gurobi_license_file", ""))

    try:
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing package: gurobipy. Install with: pip install gurobipy") from exc

    data = load_problem_data(
        args.dataset_dir,
        max_service_miles_override=args.max_service_miles,
        penalty_start_miles_override=args.penalty_start_miles,
        top_hubs_per_zip=None if int(args.top_hubs_per_zip) < 0 else int(args.top_hubs_per_zip),
        max_parts_total=None if int(args.max_parts_total) < 0 else int(args.max_parts_total),
    )
    run_dir = Path(args.run_root).expanduser().resolve() / "gurobi"
    run_dir.mkdir(parents=True, exist_ok=True)

    scalar = data["scalar"]
    J = data["J"]
    active_df = data["active"]
    active_pairs = [(str(r.zip_id), str(r.part_id)) for r in active_df[["zip_id", "part_id"]].itertuples(index=False)]
    big_m_parts = max(1, len(set(k for _, k in active_pairs)))

    z_keys: list[tuple[str, str, str]] = []
    z_by_pair: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    y_keys_set: set[tuple[str, str]] = set()
    for i, k in active_pairs:
        for j, _ in data["zip_to_hubs"].get(i, []):
            key = (i, j, k)
            z_keys.append(key)
            z_by_pair[(i, k)].append(key)
            y_keys_set.add((j, k))
    y_keys = sorted(y_keys_set)
    y_by_hub: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for j, k in y_keys:
        y_by_hub[j].append((j, k))

    print("\n" + "=" * 76, flush=True)
    print("GUROBI MODEL - ALIGNED COST BASIS", flush=True)
    print("=" * 76, flush=True)
    print(f"  dataset:                  {data['dataset_name']}", flush=True)
    print(f"  active demand pairs:      {len(active_pairs):,}", flush=True)
    print(f"  candidate Z variables:    {len(z_keys):,}", flush=True)
    print(f"  stocked Y variables:      {len(y_keys):,}", flush=True)
    print(f"  hub X variables:          {len(J):,}", flush=True)
    print(f"  base miles:               {scalar['base_miles']}", flush=True)
    print(f"  penalty start miles:      {scalar['penalty_start_miles']}", flush=True)
    print(f"  max service miles:        {scalar['max_service_miles']}", flush=True)
    print(f"  output folder:            {run_dir}", flush=True)
    print("=" * 76, flush=True)

    model = gp.Model("aligned_fsl_risk_optimization")
    model.Params.LogToConsole = 1 if bool(args.console_log) else 0
    model.Params.LogFile = str(run_dir / "gurobi.log")
    model.Params.MIPGap = max(0.0, float(args.mip_gap))
    if args.gurobi_time_limit is not None and float(args.gurobi_time_limit) > 0:
        model.Params.TimeLimit = float(args.gurobi_time_limit)
    if args.threads is not None and int(args.threads) > 0:
        model.Params.Threads = int(args.threads)
    if args.seed is not None and int(args.seed) >= 0:
        model.Params.Seed = int(args.seed)

    print("Creating variables...", flush=True)
    Z = model.addVars(z_keys, vtype=GRB.BINARY, name="Z")
    Y = model.addVars(y_keys, vtype=GRB.BINARY, name="Y")
    X = model.addVars(J, vtype=GRB.BINARY, name="X")
    W = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="W")

    print("Building objective...", flush=True)
    term_inventory = gp.quicksum(float(data["P"].get(k, 0.0)) * Y[j, k] for j, k in y_keys)
    term_open = gp.quicksum(float(scalar["S_lim"]) * X[j] for j in J)
    term_overflow = gp.quicksum(float(scalar["S_var"]) * W[j] for j in J)
    term_transfer = gp.quicksum(
        (1 - int(data["T"].get(j, 0))) * float(scalar["C"]) * Y[j, k] for j, k in y_keys
    )
    term_assignment = gp.quicksum(
        assignment_cost(i, j, k, data)["assignment_cost"] * Z[i, j, k] for i, j, k in z_keys
    )
    model.setObjective(term_inventory + term_open + term_overflow + term_transfer + term_assignment, GRB.MINIMIZE)

    print("Adding assignment constraints...", flush=True)
    for pair, keys in z_by_pair.items():
        if not keys:
            raise ValueError(f"No candidate hubs for active pair {pair}")
        model.addConstr(gp.quicksum(Z[key] for key in keys) == 1, name=f"assign_{pair[0]}_{pair[1]}")

    print("Adding stock/open constraints...", flush=True)
    for i, j, k in z_keys:
        model.addConstr(Z[i, j, k] <= Y[j, k], name=f"stock_{i}_{j}_{k}")
    for j, k in y_keys:
        model.addConstr(Y[j, k] <= X[j], name=f"stock_requires_open_{j}_{k}")

    print("Adding activation and overflow constraints...", flush=True)
    for j in J:
        y_at_j = y_by_hub.get(j, [])
        sum_y = gp.quicksum(Y[jj, kk] for jj, kk in y_at_j)
        if bool(args.enforce_hub_capacity):
            cap = float(data["B_j"].get(j, big_m_parts))
            if cap <= 0:
                cap = float(big_m_parts)
        else:
            cap = float(big_m_parts)
        model.addConstr(sum_y <= cap * X[j], name=f"activate_hub_{j}")
        model.addConstr(X[j] <= sum_y, name=f"no_empty_open_hub_{j}")
        model.addConstr(sum_y - float(scalar["L"]) <= W[j], name=f"over_storage_limit_{j}")

    model.update()
    print(
        f"Starting Gurobi optimize: vars={model.NumVars:,}, constraints={model.NumConstrs:,}, mip_gap={float(args.mip_gap):g}",
        flush=True,
    )
    model.optimize()
    status_name = gurobi_status_name(GRB, model.Status)
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi ended with status {status_name} and found no feasible solution")

    assignments = sorted([(i, j, k) for i, j, k in z_keys if Z[i, j, k].X > 0.5])
    stocked_pairs = sorted([(j, k) for j, k in y_keys if Y[j, k].X > 0.5])
    open_hubs = sorted([j for j in J if X[j].X > 0.5])
    overflow_rows = sorted([(j, float(W[j].X)) for j in J if W[j].X > 1e-9])
    pd.DataFrame(overflow_rows, columns=["hub_id", "W_j"]).to_csv(run_dir / "overflow.csv", index=False)

    wall = time.perf_counter() - start
    peak_mb, current_mb = memory_report_mb()
    runtime = {
        "wall_seconds": float(wall),
        "solver_runtime_seconds": float(model.Runtime),
        "peak_memory_mb": float(peak_mb),
        "current_memory_mb": float(current_mb),
    }
    extra = {
        "gurobi_status": status_name,
        "gurobi_status_code": int(model.Status),
        "gurobi_objective": float(model.ObjVal),
        "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
        "num_variables": int(model.NumVars),
        "num_constraints": int(model.NumConstrs),
        "has_incumbent_solution": bool(model.SolCount > 0),
    }
    summary = write_solution_outputs(
        run_dir,
        solver_name="gurobi",
        data=data,
        assignments=assignments,
        stocked_pairs=stocked_pairs,
        open_hubs=open_hubs,
        runtime=runtime,
        extra=extra,
    )
    print(final_results_block(summary, "GUROBI FINAL RESULTS"), flush=True)
    model.dispose()
    gc.collect()
    return summary


# ---------------------------------------------------------------------------
# QUBO solver
# ---------------------------------------------------------------------------


def build_batches(
    active: pd.DataFrame,
    part_order: list[str],
    zip_to_hubs: dict[str, list[tuple[str, float]]],
    *,
    part_batch_size: int,
    max_z_vars_per_batch: int,
) -> list[BatchSpec]:
    if part_batch_size <= 0:
        raise ValueError("--part-batch-size must be > 0")
    if max_z_vars_per_batch <= 0:
        raise ValueError("--max-z-vars-per-batch must be > 0")

    df = active.reset_index(drop=True).copy()
    row_z = df["zip_id"].map(lambda z: len(zip_to_hubs.get(str(z), ())))
    if (row_z <= 0).any():
        bad = df.loc[row_z <= 0, ["zip_id", "part_id"]].head(5).to_dict("records")
        raise ValueError(f"Active demand rows without candidate hubs: {bad}")
    if (row_z > max_z_vars_per_batch).any():
        bad = df.loc[row_z > max_z_vars_per_batch, ["zip_id", "part_id"]].head(5).to_dict("records")
        raise ValueError(
            "A single demand row has more candidate hubs than --max-z-vars-per-batch. "
            f"Examples: {bad}. Increase the cap or reduce --top-hubs-per-zip."
        )

    rows_by_part: dict[str, list[int]] = defaultdict(list)
    for idx, part_id in enumerate(df["part_id"].tolist()):
        rows_by_part[str(part_id)].append(idx)

    batches: list[BatchSpec] = []
    current: list[int] = []
    current_parts: set[str] = set()
    current_z = 0

    def flush(note: str = "") -> None:
        nonlocal current, current_parts, current_z
        if current:
            batches.append(BatchSpec(len(batches) + 1, list(current), int(current_z), note))
        current = []
        current_parts = set()
        current_z = 0

    for part_id in part_order:
        idxs = rows_by_part.get(str(part_id), [])
        if not idxs:
            continue
        part_z = int(row_z.iloc[idxs].sum())

        if part_z > max_z_vars_per_batch:
            flush()
            chunk: list[int] = []
            chunk_z = 0
            for idx in idxs:
                rz = int(row_z.iloc[idx])
                if chunk and chunk_z + rz > max_z_vars_per_batch:
                    batches.append(BatchSpec(len(batches) + 1, list(chunk), int(chunk_z), f"split_large_part:{part_id}"))
                    chunk = []
                    chunk_z = 0
                chunk.append(idx)
                chunk_z += rz
            if chunk:
                batches.append(BatchSpec(len(batches) + 1, list(chunk), int(chunk_z), f"split_large_part:{part_id}"))
            continue

        if current and (current_z + part_z > max_z_vars_per_batch or len(current_parts) >= part_batch_size):
            flush()
        current.extend(idxs)
        current_parts.add(str(part_id))
        current_z += part_z

    flush()
    return batches


def var_x(hub_id: str) -> str:
    return f"X|{hub_id}"


def var_y(hub_id: str, part_id: str) -> str:
    return f"Y|{hub_id}|{part_id}"


def var_z(zip_id: str, hub_id: str, part_id: str) -> str:
    return f"Z|{zip_id}|{hub_id}|{part_id}"


def add_qubo(Q: dict[tuple[str, str], float], u: str, v: str, coeff: float) -> None:
    if abs(coeff) <= 1e-12:
        return
    if u != v and v < u:
        u, v = v, u
    key = (u, v)
    new_val = Q.get(key, 0.0) + float(coeff)
    if abs(new_val) <= 1e-12:
        Q.pop(key, None)
    else:
        Q[key] = new_val


def estimate_objective_scale(batch_df: pd.DataFrame, data: dict[str, Any]) -> float:
    if batch_df.empty:
        return 1.0
    scalar = data["scalar"]
    part_ids = batch_df["part_id"].unique().tolist()
    max_part_cost = max((float(data["P"].get(k, 0.0)) for k in part_ids), default=0.0)
    max_b = float(batch_df["b_ik"].max()) if "b_ik" in batch_df.columns else 0.0
    max_d = max((float(v) for v in data["D"].values()), default=0.0)
    max_transport = (
        float(scalar["lambda_1"]) * float(scalar["h_s"]) * max_b
        + float(scalar["lambda_2"]) * float(scalar["h_d"]) * max_b * max(0.0, max_d - float(scalar["base_miles"]))
        + float(scalar["lambda_3"]) * max_b * max(0.0, max_d - float(scalar["penalty_start_miles"]))
    )
    return max(1.0, max_part_cost + float(scalar["C"]) + float(scalar["S_lim"]) + max_transport)


def penalty_weights(
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    multipliers: dict[str, float] | None = None,
    return_diagnostics: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, dict[str, Any]]]:
    scale = estimate_objective_scale(batch_df, data)
    out: dict[str, float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for c in ("c1", "c2", "c3", "c4"):
        override_min = float(getattr(args, f"min_penalty_{c}", -1.0))
        override_mult = float(getattr(args, f"constraint_multiplier_{c}", -1.0))
        min_pen = override_min if override_min >= 0 else float(args.min_penalty)
        mult = override_mult if override_mult >= 0 else float(args.constraint_multiplier)
        scaled_value = float(scale) * float(mult)
        if args.penalty_mode == "fixed":
            out[c] = float(min_pen)
        else:
            out[c] = max(float(min_pen), scaled_value)

        if return_diagnostics:
            # "floor" means the min_penalty floor won; "scaled" means scale*mult won.
            # In fixed mode the floor is used unconditionally.
            if args.penalty_mode == "fixed" or float(min_pen) >= scaled_value:
                binding_branch = "floor"
            else:
                binding_branch = "scaled"
            adaptive_mult = float(multipliers[c]) if (multipliers is not None and c in multipliers) else 1.0
            diagnostics[c] = {
                "objective_scale": float(scale),
                "min_pen": float(min_pen),
                "mult": float(mult),
                "scaled_value": float(scaled_value),
                "chosen_value": float(out[c]),
                "binding_branch": binding_branch,
                "adaptive_mult": float(adaptive_mult),
                "final_value": float(out[c]) * float(adaptive_mult),
            }

    # Adaptive multipliers (within-batch adaptive penalty). When None, the
    # returned penalties are bit-identical to the static behavior.
    if multipliers is not None:
        for c in ("c1", "c2", "c3", "c4"):
            if c in multipliers:
                out[c] = out[c] * float(multipliers[c])

    if return_diagnostics:
        return out, diagnostics
    return out


def build_qubo_for_batch(
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    scalar = data["scalar"]
    zip_to_hubs = data["zip_to_hubs"]
    penalties = penalty_weights(batch_df, data, args, multipliers=multipliers)

    Q: dict[tuple[str, str], float] = {}
    z_name: dict[tuple[str, str, str], str] = {}
    y_name: dict[tuple[str, str], str] = {}
    x_name: dict[str, str] = {}
    demand_groups: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for r in batch_df[["zip_id", "part_id", "b_ik"]].itertuples(index=False):
        i = str(r.zip_id)
        k = str(r.part_id)
        b_val = float(r.b_ik)
        group_entries: list[tuple[str, str]] = []
        for j, dij in zip_to_hubs[i]:
            zn = var_z(i, j, k)
            yn = var_y(j, k)
            xn = var_x(j)
            z_name[(i, j, k)] = zn
            y_name[(j, k)] = yn
            x_name[j] = xn
            group_entries.append((j, zn))
            transport = assignment_cost_by_values(b_val, float(dij), scalar)["assignment_cost"]
            add_qubo(Q, zn, zn, transport)
        demand_groups[(i, k)] = group_entries

    for (j, k), yn in y_name.items():
        add_qubo(Q, yn, yn, float(data["P"].get(k, 0.0)))
        add_qubo(Q, yn, yn, (1 - int(data["T"].get(j, 0))) * float(scalar["C"]))

    for j, xn in x_name.items():
        add_qubo(Q, xn, xn, float(scalar["S_lim"]))

    # C1: exact one assignment per active (zip, part). Constant term omitted.
    lam1 = float(penalties["c1"])
    for entries in demand_groups.values():
        zvars = [zn for _, zn in entries]
        for zn in zvars:
            add_qubo(Q, zn, zn, -lam1)
        for a in range(len(zvars)):
            za = zvars[a]
            for b in range(a + 1, len(zvars)):
                add_qubo(Q, za, zvars[b], 2.0 * lam1)

    # C2: Z_ijk <= Y_jk, encoded as penalty * z * (1 - y).
    lam2 = float(penalties["c2"])
    for (i, j, k), zn in z_name.items():
        yn = y_name[(j, k)]
        add_qubo(Q, zn, zn, lam2)
        add_qubo(Q, zn, yn, -lam2)

    # C3: Y_jk <= X_j, encoded as penalty * y * (1 - x).
    lam3 = float(penalties["c3"])
    for (j, k), yn in y_name.items():
        xn = x_name[j]
        add_qubo(Q, yn, yn, lam3)
        add_qubo(Q, yn, xn, -lam3)

    # Tier 3.8: approximate X<=sum(Y) penalty. Pure QUBO can't express OR(Y),
    # so we add lam_xy on each X-diagonal and refund -lam_xy/n_k per (X,Y) pair.
    # When X=1 with all parts stocked, the refund cancels the bias; with no Y
    # stocked, the full lam_xy penalizes empty hubs. Default factor is 0.
    x_empty_factor = float(getattr(args, "x_empty_penalty_factor", 0.0))
    if x_empty_factor > 0:
        lam_xy = float(scalar["S_lim"]) * x_empty_factor
        parts_per_hub: dict[str, list[str]] = defaultdict(list)
        for (j, k) in y_name:
            parts_per_hub[j].append(k)
        for j, xn in x_name.items():
            parts = parts_per_hub.get(j, [])
            if not parts:
                continue
            n_k = len(parts)
            add_qubo(Q, xn, xn, lam_xy)
            per_pair_refund = lam_xy / float(n_k)
            for k in parts:
                yn = y_name[(j, k)]
                add_qubo(Q, xn, yn, -per_pair_refund)

    # Tier 3.9: linear S_var-style overflow penalty per stocked Y. This is a
    # rough proxy because true overflow is max(0, sum(Y) - L), which needs
    # slack vars to encode in QUBO. Default factor is 0 (off); inactive on
    # instances_low since L=50000 dwarfs the parts count.
    y_overflow_factor = float(getattr(args, "y_overflow_penalty_factor", 0.0))
    if y_overflow_factor > 0:
        s_var_coeff = float(scalar["S_var"]) * y_overflow_factor
        for (j, k), yn in y_name.items():
            add_qubo(Q, yn, yn, s_var_coeff)

    c4_note = "inactive_or_soft_cost_only"
    max_stock_per_hub = max((sum(1 for jj, _ in y_name if jj == j) for j in x_name), default=0)
    if max_stock_per_hub > int(scalar["L"]):
        c4_note = "capacity_can_bind; QUBO does not hard-encode C4, final cost includes overflow"
        if args.c4_mode == "on":
            raise NotImplementedError(
                "C4 can bind for this batch. This runner does not use the old incorrect equality slack encoding. "
                "Use --c4-mode auto/off or implement a bounded inequality slack formulation."
            )

    return {
        "Q": Q,
        "z_name": z_name,
        "y_name": y_name,
        "x_name": x_name,
        "demand_groups": demand_groups,
        "penalties": penalties,
        "c4_note": c4_note,
    }


def sample_is_one(value: Any) -> bool:
    try:
        return int(round(float(value))) == 1
    except Exception:
        return bool(value)


def iter_openjij_samples(response: Any) -> Iterator[tuple[dict[str, int], float]]:
    if hasattr(response, "data"):
        for kwargs in ({"fields": ["sample", "energy"]}, {}):
            try:
                iterator = response.data(**kwargs) if kwargs else response.data(["sample", "energy"])
                for item in iterator:
                    sample = getattr(item, "sample", None)
                    energy = getattr(item, "energy", None)
                    if sample is None and isinstance(item, tuple) and len(item) >= 2:
                        sample, energy = item[0], item[1]
                    if sample is not None and energy is not None:
                        yield {str(k): int(round(float(v))) for k, v in dict(sample).items()}, float(energy)
                return
            except Exception:
                pass

    if hasattr(response, "states") and hasattr(response, "energies"):
        variables = None
        for attr in ("variables", "indices"):
            if hasattr(response, attr):
                variables = list(getattr(response, attr))
                break
        if variables is None:
            raise RuntimeError("OpenJij response has states/energies but no variables/indices labels")
        for state, energy in zip(response.states, response.energies):
            yield {str(variables[i]): int(round(float(v))) for i, v in enumerate(state)}, float(energy)
        return

    if hasattr(response, "record") and hasattr(response, "variables"):
        variables = list(response.variables)
        for rec in response.record:
            sample_arr = getattr(rec, "sample", rec[0])
            energy = getattr(rec, "energy", rec[1])
            yield {str(variables[i]): int(round(float(v))) for i, v in enumerate(sample_arr)}, float(energy)
        return

    raise RuntimeError("Could not iterate samples from OpenJij response. Unknown response format.")


def evaluate_sample(sample: dict[str, int], energy: float, qubo_meta: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    demand_groups = qubo_meta["demand_groups"]
    z_name = qubo_meta["z_name"]
    y_name = qubo_meta["y_name"]
    x_name = qubo_meta["x_name"]

    c1 = 0
    for entries in demand_groups.values():
        selected = sum(1 for _, zn in entries if sample_is_one(sample.get(zn, 0)))
        if selected != 1:
            c1 += 1

    c2 = 0
    for (i, j, k), zn in z_name.items():
        if sample_is_one(sample.get(zn, 0)) and not sample_is_one(sample.get(y_name[(j, k)], 0)):
            c2 += 1

    c3 = 0
    for (j, k), yn in y_name.items():
        if sample_is_one(sample.get(yn, 0)) and not sample_is_one(sample.get(x_name[j], 0)):
            c3 += 1

    selected_assignments = sorted([(i, j, k) for (i, j, k), zn in z_name.items() if sample_is_one(sample.get(zn, 0))])
    stocked_pairs = sorted([(j, k) for (j, k), yn in y_name.items() if sample_is_one(sample.get(yn, 0))])
    open_hubs = sorted([j for j, xn in x_name.items() if sample_is_one(sample.get(xn, 0))])
    cost = compute_solution_cost(selected_assignments, stocked_pairs, open_hubs, data)["total_cost"]

    return {
        "sample": sample,
        "energy": float(energy),
        "cost": float(cost),
        "c1": int(c1),
        "c2": int(c2),
        "c3": int(c3),
        "c4": 0,
        "total_violations": int(c1 + c2 + c3),
        "assignments": selected_assignments,
        "stocked_pairs": stocked_pairs,
        "open_hubs": open_hubs,
    }


class _SqaAdapter:
    """Wrap SQASampler so beta/trotter/gamma are injected into every sample_qubo call.

    gamma is the transverse-field strength, the parameter that makes SQA
    quantum-inspired rather than a trotterized SA. When gamma is None the
    OpenJij library default is used (unchanged behavior); pass --sqa-gamma to
    set it explicitly so it is reproducible and reportable.
    """

    def __init__(self, sampler: Any, beta: float, trotter: int, gamma: float | None = None) -> None:
        self._s, self._beta, self._trotter, self._gamma = sampler, beta, trotter, gamma

    def sample_qubo(self, Q: Any, **kw: Any) -> Any:
        kw.setdefault("beta", self._beta)
        kw.setdefault("trotter", self._trotter)
        if self._gamma is not None:
            kw.setdefault("gamma", self._gamma)
        return self._s.sample_qubo(Q, **kw)


def build_sampler(args: argparse.Namespace) -> Any:
    import openjij  # type: ignore

    if getattr(args, "sampler", "sa") == "sqa":
        # Negative sentinel (-1.0) means "use OpenJij's default gamma".
        gamma_arg = float(getattr(args, "sqa_gamma", -1.0))
        gamma = gamma_arg if gamma_arg >= 0 else None
        return _SqaAdapter(openjij.SQASampler(), float(args.sqa_beta), int(args.sqa_trotter), gamma)
    return openjij.SASampler()


def suggested_num_reads(num_z: int, args: argparse.Namespace) -> int:
    if num_z <= 0:
        return max(1, int(args.num_reads))
    scale = max(1.0, math.sqrt(float(num_z) / 5000.0))
    return max(1, int(math.ceil(float(args.num_reads) * scale)))


def run_adaptive_penalty_loop(
    batch: BatchSpec,
    batch_df: pd.DataFrame,
    data: dict[str, Any],
    args: argparse.Namespace,
    sampler: Any,
    base_reads: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, float], int, bool, list[dict], str]:
    """Within-batch adaptive penalty: iteratively grow penalties for violated
    constraints, rebuild the QUBO, resample. Returns (best_eval, qubo_meta_final,
    multipliers_final, adaptive_iterations_used, was_feasible, iteration_log,
    exit_reason).
    """
    multipliers = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    growth = float(args.adaptive_penalty_growth)
    max_iter = int(args.adaptive_penalty_iterations)
    stagnation_patience = int(getattr(args, "adaptive_penalty_stagnation_patience", 0))
    best_eval: dict[str, Any] | None = None
    qubo_meta_final: dict[str, Any] | None = None
    adaptive_iter_used = 0
    was_feasible = False
    iteration_log: list[dict] = []
    exit_reason = "max_iterations"
    best_total_violations: int | None = None
    no_improve_count = 0

    for iteration in range(1, max_iter + 1):
        adaptive_iter_used = iteration

        # Rebuild the QUBO with the current per-constraint multipliers.
        qubo_meta = build_qubo_for_batch(batch_df, data, args, multipliers=multipliers)
        qubo_meta_final = qubo_meta
        Q = qubo_meta["Q"]

        # Deterministic recompute of the penalty resolution for this iteration's
        # multipliers (no sampling; cannot affect solution values). This doubles as
        # explicit confirmation that the QUBO was rebuilt with these penalties.
        _, diag = penalty_weights(batch_df, data, args, multipliers=multipliers, return_diagnostics=True)
        objective_scale = float(diag["c1"]["objective_scale"])
        num_vars = len(qubo_meta["z_name"]) + len(qubo_meta["y_name"]) + len(qubo_meta["x_name"])
        num_interactions = len(Q)

        # Single sampling pass at base reads (no retry-reads escalation here).
        sample_kwargs: dict[str, Any] = {"num_reads": base_reads}
        if args.seed is not None and int(args.seed) >= 0:
            sample_kwargs["seed"] = int(args.seed) + batch.batch_id * 1000 + iteration
        if int(args.num_sweeps or 0) > 0:
            sample_kwargs["num_sweeps"] = int(args.num_sweeps)
        seed_used = sample_kwargs.get("seed", None)

        binding_summary = ",".join(f"{c}={diag[c]['binding_branch']}" for c in ("c1", "c2", "c3", "c4"))
        print(
            f"    adaptive iter {iteration}/{max_iter} | scale={objective_scale:,.1f} | "
            f"multipliers c1={multipliers['c1']:.2f} c2={multipliers['c2']:.2f} "
            f"c3={multipliers['c3']:.2f} c4={multipliers['c4']:.2f} | binding[{binding_summary}]",
            flush=True,
        )
        response = sampler.sample_qubo(Q, **sample_kwargs)

        iter_best: dict[str, Any] | None = None
        for sample, energy in iter_openjij_samples(response):
            ev = evaluate_sample(sample, energy, qubo_meta, data)
            key = (ev["total_violations"], ev["c1"], ev["c2"], ev["c3"], ev["cost"], ev["energy"])
            if iter_best is None:
                iter_best = ev
            else:
                old_key = (
                    iter_best["total_violations"],
                    iter_best["c1"],
                    iter_best["c2"],
                    iter_best["c3"],
                    iter_best["cost"],
                    iter_best["energy"],
                )
                if key < old_key:
                    iter_best = ev

        if iter_best is None:
            raise RuntimeError("OpenJij returned no samples during adaptive penalty loop.")

        if best_eval is None:
            best_eval = iter_best
        else:
            old_key = (
                best_eval["total_violations"],
                best_eval["c1"],
                best_eval["c2"],
                best_eval["c3"],
                best_eval["cost"],
                best_eval["energy"],
            )
            new_key = (
                iter_best["total_violations"],
                iter_best["c1"],
                iter_best["c2"],
                iter_best["c3"],
                iter_best["cost"],
                iter_best["energy"],
            )
            if new_key < old_key:
                best_eval = iter_best

        print(
            f"    adaptive iter {iteration} | violations C1={iter_best['c1']} "
            f"C2={iter_best['c2']} C3={iter_best['c3']} C4={iter_best.get('c4', 0)} | "
            f"cost={iter_best['cost']:.2f}",
            flush=True,
        )

        # Record this iteration. exit_reason is filled on the final iteration only.
        log_row: dict[str, Any] = {
            "batch_id": int(batch.batch_id),
            "iteration": int(iteration),
            "objective_scale": objective_scale,
            "total_violations": int(iter_best["total_violations"]),
            "cost": float(iter_best["cost"]),
            "energy": float(iter_best["energy"]),
            "num_vars": int(num_vars),
            "num_interactions": int(num_interactions),
            "seed_used": seed_used,
            "num_reads": int(base_reads),
            "exit_reason": "",
        }
        for c in ("c1", "c2", "c3", "c4"):
            log_row[f"min_pen_{c}"] = float(diag[c]["min_pen"])
            log_row[f"scaled_{c}"] = float(diag[c]["scaled_value"])
            log_row[f"chosen_{c}"] = float(diag[c]["chosen_value"])
            log_row[f"binding_branch_{c}"] = str(diag[c]["binding_branch"])
            log_row[f"mult_{c}"] = float(multipliers[c])
            log_row[f"viol_{c}"] = int(iter_best.get(c, 0))
        iteration_log.append(log_row)

        if int(iter_best["total_violations"]) == 0:
            was_feasible = True
            exit_reason = "feasible"
            iteration_log[-1]["exit_reason"] = exit_reason
            print(f"    adaptive feasible at iter {iteration}", flush=True)
            break

        # Stagnation exit (opt-in, default OFF). Placed AFTER the feasibility break
        # so feasibility always wins first; with patience=0 this is unreachable.
        if best_total_violations is None or int(iter_best["total_violations"]) < best_total_violations:
            best_total_violations = int(iter_best["total_violations"])
            no_improve_count = 0
        else:
            no_improve_count += 1
        if stagnation_patience > 0 and no_improve_count >= stagnation_patience:
            exit_reason = "stagnated"
            iteration_log[-1]["exit_reason"] = exit_reason
            print(
                f"    adaptive stagnated at iter {iteration} "
                f"(no strict improvement for {no_improve_count} iters)",
                flush=True,
            )
            break

        # Grow multipliers for violated constraints only.
        if int(iter_best["c1"]) > 0:
            multipliers["c1"] *= growth
        if int(iter_best["c2"]) > 0:
            multipliers["c2"] *= growth
        if int(iter_best["c3"]) > 0:
            multipliers["c3"] *= growth
        if int(iter_best.get("c4", 0)) > 0:
            multipliers["c4"] *= growth

    # Loop exhausted without an early break -> max_iterations.
    if exit_reason == "max_iterations" and iteration_log:
        iteration_log[-1]["exit_reason"] = exit_reason

    return best_eval, qubo_meta_final, multipliers, adaptive_iter_used, was_feasible, iteration_log, exit_reason


def solve_qubo_batch(batch: BatchSpec, active: pd.DataFrame, data: dict[str, Any], args: argparse.Namespace) -> BatchResult:
    try:
        import openjij  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing package: openjij. Install with: pip install openjij") from exc

    batch_start = time.time()
    batch_df = active.iloc[batch.row_indices].reset_index(drop=True)
    num_rows = len(batch_df)
    num_parts = batch_df["part_id"].nunique()

    print(
        f"\n=== QUBO Batch {batch.batch_id} | {num_parts} parts | {num_rows:,} demand rows | estimated Z={batch.estimated_z_vars:,} ===",
        flush=True,
    )
    if batch.note:
        print(f"  note: {batch.note}", flush=True)

    print("  [1/3] Building aligned manual QUBO dictionary...", flush=True)
    t0 = time.time()
    qubo_meta = build_qubo_for_batch(batch_df, data, args)
    build_seconds = time.time() - t0
    Q = qubo_meta["Q"]
    num_z = len(qubo_meta["z_name"])
    num_y = len(qubo_meta["y_name"])
    num_x = len(qubo_meta["x_name"])
    interactions = len(Q)
    print(
        f"    built in {build_seconds:.2f}s | Z={num_z:,} Y={num_y:,} X={num_x:,} interactions={interactions:,}",
        flush=True,
    )
    print(
        "    penalties: "
        f"C1={qubo_meta['penalties']['c1']:,.2f} "
        f"C2={qubo_meta['penalties']['c2']:,.2f} "
        f"C3={qubo_meta['penalties']['c3']:,.2f}",
        flush=True,
    )

    sampler = build_sampler(args)
    base_reads = suggested_num_reads(num_z, args)
    best_eval: dict[str, Any] | None = None
    sample_seconds = 0.0
    eval_seconds = 0.0
    stage_used = 0
    adaptive_iters_used = 0
    adaptive_was_feasible = False
    final_multipliers = {"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0}
    adaptive_iteration_log: list[dict] = []
    adaptive_exit_reason = ""

    # Within-batch adaptive penalty phase (runs before the retry-reads loop).
    if args.adaptive_penalty_mode == "within-batch":
        print("  [2a/3] Adaptive penalty phase...", flush=True)
        t_adapt = time.time()
        (
            best_eval,
            qubo_meta,
            final_multipliers,
            adaptive_iters_used,
            adaptive_was_feasible,
            adaptive_iteration_log,
            adaptive_exit_reason,
        ) = run_adaptive_penalty_loop(batch, batch_df, data, args, sampler, base_reads)
        sample_seconds += time.time() - t_adapt
        Q = qubo_meta["Q"]
        if adaptive_was_feasible:
            print("  [2b/3] Adaptive succeeded - skipping retry-reads fallback", flush=True)
        else:
            print(
                f"  [2b/3] Adaptive exhausted ({adaptive_iters_used} iters) - "
                f"running retry-reads fallback with final multipliers",
                flush=True,
            )

    # Existing retry-reads loop. Runs always when adaptive is off, or as the
    # fallback when adaptive did not reach feasibility.
    if args.adaptive_penalty_mode == "off" or not adaptive_was_feasible:
        if args.adaptive_penalty_mode == "within-batch":
            # Rebuild the QUBO with the final adapted multipliers for the fallback.
            qubo_meta = build_qubo_for_batch(batch_df, data, args, multipliers=final_multipliers)
            Q = qubo_meta["Q"]

        print("  [2/3] Sampling QUBO with OpenJij...", flush=True)
        for stage in range(1, int(args.max_stages) + 1):
            stage_used = stage
            stage_reads = max(1, int(math.ceil(base_reads * (float(args.retry_reads_boost) ** (stage - 1)))))
            sample_kwargs: dict[str, Any] = {"num_reads": stage_reads}
            if args.seed is not None and int(args.seed) >= 0:
                sample_kwargs["seed"] = int(args.seed) + batch.batch_id * 1000 + stage
            if int(args.num_sweeps or 0) > 0:
                sample_kwargs["num_sweeps"] = int(args.num_sweeps)

            print(f"    stage {stage}/{args.max_stages}: sampling {stage_reads} reads...", flush=True)
            t_sample = time.time()
            response = sampler.sample_qubo(Q, **sample_kwargs)
            stage_sample_seconds = time.time() - t_sample
            sample_seconds += stage_sample_seconds

            print(f"    stage {stage}/{args.max_stages}: evaluating samples...", flush=True)
            t_eval = time.time()
            stage_best: dict[str, Any] | None = None
            sample_count = 0
            for sample, energy in iter_openjij_samples(response):
                sample_count += 1
                ev = evaluate_sample(sample, energy, qubo_meta, data)
                key = (ev["total_violations"], ev["c1"], ev["c2"], ev["c3"], ev["cost"], ev["energy"])
                if stage_best is None:
                    stage_best = ev
                else:
                    old_key = (
                        stage_best["total_violations"],
                        stage_best["c1"],
                        stage_best["c2"],
                        stage_best["c3"],
                        stage_best["cost"],
                        stage_best["energy"],
                    )
                    if key < old_key:
                        stage_best = ev
            stage_eval_seconds = time.time() - t_eval
            eval_seconds += stage_eval_seconds
            if stage_best is None:
                raise RuntimeError("OpenJij returned no samples")

            if best_eval is None:
                best_eval = stage_best
            else:
                key = (stage_best["total_violations"], stage_best["c1"], stage_best["c2"], stage_best["c3"], stage_best["cost"], stage_best["energy"])
                old_key = (best_eval["total_violations"], best_eval["c1"], best_eval["c2"], best_eval["c3"], best_eval["cost"], best_eval["energy"])
                if key < old_key:
                    best_eval = stage_best

            print(
                f"    stage {stage} done | samples={sample_count} sample_time={stage_sample_seconds:.2f}s "
                f"eval_time={stage_eval_seconds:.2f}s best_structural_violations={stage_best['total_violations']} "
                f"(C1={stage_best['c1']}, C2={stage_best['c2']}, C3={stage_best['c3']}) "
                f"energy={stage_best['energy']:.2f}",
                flush=True,
            )
            if int(stage_best["total_violations"]) == 0:
                print("    early stop: feasible batch sample found", flush=True)
                break

    if best_eval is None:
        raise RuntimeError("No QUBO sample was selected")

    total_seconds = time.time() - batch_start
    status = "OK" if best_eval["total_violations"] == 0 else f"{best_eval['total_violations']} structural violations"
    print(
        f"  [3/3] Batch selected | {status} | aligned cost={money(best_eval['cost'])} | total_time={total_seconds:.2f}s",
        flush=True,
    )

    return BatchResult(
        batch_id=batch.batch_id,
        num_rows=num_rows,
        num_parts=int(num_parts),
        num_z=num_z,
        num_y=num_y,
        num_x=num_x,
        interactions=interactions,
        build_seconds=build_seconds,
        sample_seconds=sample_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total_seconds,
        stage_used=stage_used,
        suggested_reads=base_reads,
        energy=float(best_eval["energy"]),
        reconstructed_cost=float(best_eval["cost"]),
        c1_violations=int(best_eval["c1"]),
        c2_violations=int(best_eval["c2"]),
        c3_violations=int(best_eval["c3"]),
        c4_violations=int(best_eval["c4"]),
        penalty_c1=float(qubo_meta["penalties"]["c1"]),
        penalty_c2=float(qubo_meta["penalties"]["c2"]),
        penalty_c3=float(qubo_meta["penalties"]["c3"]),
        open_hubs=list(best_eval["open_hubs"]),
        stocked_pairs=list(best_eval["stocked_pairs"]),
        assignments=list(best_eval["assignments"]),
        adaptive_iterations_used=int(adaptive_iters_used),
        adaptive_was_feasible=bool(adaptive_was_feasible),
        final_penalty_multipliers=dict(final_multipliers),
        adaptive_iteration_log=list(adaptive_iteration_log),
        adaptive_exit_reason=str(adaptive_exit_reason),
    )


def batch_summary_dataframe(results: list[BatchResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "batch_id": r.batch_id,
                "num_rows": r.num_rows,
                "num_parts": r.num_parts,
                "num_z": r.num_z,
                "num_y": r.num_y,
                "num_x": r.num_x,
                "interactions": r.interactions,
                "build_seconds": r.build_seconds,
                "sample_seconds": r.sample_seconds,
                "eval_seconds": r.eval_seconds,
                "total_seconds": r.total_seconds,
                "stage_used": r.stage_used,
                "suggested_reads": r.suggested_reads,
                "energy": r.energy,
                "reconstructed_cost": r.reconstructed_cost,
                "c1_violations": r.c1_violations,
                "c2_violations": r.c2_violations,
                "c3_violations": r.c3_violations,
                "c4_violations": r.c4_violations,
                "penalty_c1": r.penalty_c1,
                "penalty_c2": r.penalty_c2,
                "penalty_c3": r.penalty_c3,
                "open_hubs_count": len(r.open_hubs),
                "stocked_pairs_count": len(r.stocked_pairs),
                "assignments_count": len(r.assignments),
            }
        )
    return pd.DataFrame(rows)


def batch_adaptive_summary_dataframe(results: list[BatchResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        mult = r.final_penalty_multipliers
        rows.append(
            {
                "batch_id": r.batch_id,
                "adaptive_iterations_used": int(r.adaptive_iterations_used),
                "adaptive_was_feasible": bool(r.adaptive_was_feasible),
                "adaptive_exit_reason": str(r.adaptive_exit_reason),
                "final_mult_c1": float(mult.get("c1", 1.0)),
                "final_mult_c2": float(mult.get("c2", 1.0)),
                "final_mult_c3": float(mult.get("c3", 1.0)),
                "final_mult_c4": float(mult.get("c4", 1.0)),
            }
        )
    return pd.DataFrame(rows)


def adaptive_iteration_log_dataframe(results: list[BatchResult]) -> pd.DataFrame:
    """Flatten every batch's per-iteration adaptive log into one DataFrame."""
    return pd.DataFrame([row for r in results for row in r.adaptive_iteration_log])


def aggregate_raw_results(results: list[BatchResult]) -> dict[str, Any]:
    return {
        "open_hubs": sorted({hub for r in results for hub in r.open_hubs}),
        "stocked_pairs": sorted({pair for r in results for pair in r.stocked_pairs}),
        "assignments": sorted({asg for r in results for asg in r.assignments}),
    }


def incremental_assignment_score(
    zip_id: str,
    hub_id: str,
    part_id: str,
    data: dict[str, Any],
    current_open: set[str],
    current_stocked: set[tuple[str, str]],
) -> float:
    scalar = data["scalar"]
    score = assignment_cost(zip_id, hub_id, part_id, data)["assignment_cost"]
    if (hub_id, part_id) not in current_stocked:
        score += float(data["P"].get(part_id, 0.0))
        score += (1 - int(data["T"].get(hub_id, 0))) * float(scalar["C"])
    if hub_id not in current_open:
        score += float(scalar["S_lim"])
    return float(score)


def hub_prune_pass(
    assignments: list[tuple[str, str, str]],
    stocked_pairs: list[tuple[str, str]],
    open_hubs: list[str],
    data: dict[str, Any],
    *,
    max_iterations: int = 10,
) -> dict[str, Any]:
    """Tier 1.2: close open hubs whose assignments can be cheaply rerouted.

    For each open hub j (lowest-traffic first), compute the marginal cost of moving
    each (zip, j, part) assignment to the cheapest already-open alternative within
    max_service_miles. If S_lim plus j's transport and stocking costs exceed the
    sum of relocation costs, close j and apply the moves. Iterate until no hub
    closes in a full pass or max_iterations is hit.
    """
    scalar = data["scalar"]
    s_lim = float(scalar["S_lim"])
    transfer_C = float(scalar["C"])
    zip_to_hubs = data["zip_to_hubs"]
    P_map = data["P"]
    T_map = data["T"]

    asg = list((str(i), str(j), str(k)) for i, j, k in assignments)
    stocked: set[tuple[str, str]] = set((str(j), str(k)) for j, k in stocked_pairs)
    opened: set[str] = set(str(j) for j in open_hubs)

    candidate_hubs_by_zip: dict[str, set[str]] = {
        str(i): {str(jj) for jj, _ in zip_to_hubs.get(str(i), [])}
        for i in {a[0] for a in asg}
    }

    closures = 0
    relocations = 0

    for _ in range(max_iterations):
        hub_assignment_indices: dict[str, list[int]] = defaultdict(list)
        for idx, (i, j, k) in enumerate(asg):
            hub_assignment_indices[j].append(idx)

        # Try low-traffic hubs first; they're cheapest to dismantle.
        candidates = sorted(opened, key=lambda j: len(hub_assignment_indices.get(j, [])))
        closed_this_round = False

        for j in candidates:
            indices = hub_assignment_indices.get(j, [])

            # Cost we save by closing j: S_lim + transport at j + stocking at j.
            transport_at_j = sum(
                assignment_cost(asg[idx][0], j, asg[idx][2], data)["assignment_cost"]
                for idx in indices
            )
            stocked_at_j = {(jj, k) for jj, k in stocked if jj == j}
            stocking_at_j = sum(
                float(P_map.get(k, 0.0)) + (1 - int(T_map.get(j, 0))) * transfer_C
                for _, k in stocked_at_j
            )
            current_cost = s_lim + transport_at_j + stocking_at_j

            # Try to relocate every assignment at j to another already-open hub.
            relocate_plan: list[tuple[int, str]] = []
            new_pairs: set[tuple[str, str]] = set()
            relocate_cost = 0.0
            feasible = True

            for idx in indices:
                i, _, k = asg[idx]
                alternatives = [
                    (jj, dij) for jj, dij in zip_to_hubs.get(i, [])
                    if jj != j and jj in opened
                ]
                if not alternatives:
                    feasible = False
                    break

                best_jj: str | None = None
                best_marginal = math.inf
                for jj, _dij in alternatives:
                    transport = assignment_cost(i, jj, k, data)["assignment_cost"]
                    stock_pair_exists = (jj, k) in stocked or (jj, k) in new_pairs
                    stock_marginal = (
                        0.0 if stock_pair_exists
                        else float(P_map.get(k, 0.0)) + (1 - int(T_map.get(jj, 0))) * transfer_C
                    )
                    total = transport + stock_marginal
                    if total < best_marginal:
                        best_marginal = total
                        best_jj = jj

                if best_jj is None or not math.isfinite(best_marginal):
                    feasible = False
                    break

                relocate_plan.append((idx, best_jj))
                relocate_cost += best_marginal
                if (best_jj, k) not in stocked:
                    new_pairs.add((best_jj, k))

            if not feasible:
                continue

            if relocate_cost < current_cost - 1e-9:
                # Apply the closure.
                for idx, new_j in relocate_plan:
                    i, _, k = asg[idx]
                    asg[idx] = (i, new_j, k)
                    stocked.add((new_j, k))
                stocked = {(jj, k) for jj, k in stocked if jj != j}
                opened.discard(j)
                closures += 1
                relocations += len(relocate_plan)
                closed_this_round = True
                break  # restart with refreshed traffic counts

        if not closed_this_round:
            break

    return {
        "assignments": sorted(asg),
        "stocked_pairs": sorted(stocked),
        "open_hubs": sorted(opened),
        "closures": int(closures),
        "relocations": int(relocations),
    }


def postprocess_qubo_solution(
    raw_assignments: list[tuple[str, str, str]],
    raw_stocked_pairs: list[tuple[str, str]],
    raw_open_hubs: list[str],
    data: dict[str, Any],
    *,
    repair_assignments: bool,
    trim_unused: bool,
    hub_prune: bool = True,
    hub_prune_max_iterations: int = 10,
) -> dict[str, Any]:
    active = data["active"]
    zip_to_hubs = data["zip_to_hubs"]

    raw_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i, j, k in raw_assignments:
        raw_by_pair[(str(i), str(k))].append(str(j))

    current_open = set(str(j) for j in raw_open_hubs)
    current_stocked = set((str(j), str(k)) for j, k in raw_stocked_pairs)

    final_assignments: list[tuple[str, str, str]] = []
    assignment_sources: dict[tuple[str, str, str], str] = {}
    missing_unrepaired = 0

    for r in active[["zip_id", "part_id"]].itertuples(index=False):
        i = str(r.zip_id)
        k = str(r.part_id)
        candidates = zip_to_hubs.get(i, [])
        candidate_hubs = {j for j, _ in candidates}
        raw_selected = [j for j in raw_by_pair.get((i, k), []) if j in candidate_hubs]

        chosen: str | None = None
        source = ""
        if len(raw_selected) == 1:
            chosen = raw_selected[0]
            source = "raw_unique"
        elif len(raw_selected) > 1:
            chosen = min(raw_selected, key=lambda j: incremental_assignment_score(i, j, k, data, current_open, current_stocked))
            source = "raw_multiple_repaired"
        elif repair_assignments:
            if not candidates:
                missing_unrepaired += 1
                continue
            chosen = min([j for j, _ in candidates], key=lambda j: incremental_assignment_score(i, j, k, data, current_open, current_stocked))
            source = "missing_repaired"
        else:
            missing_unrepaired += 1
            continue

        row = (i, chosen, k)
        final_assignments.append(row)
        assignment_sources[row] = source
        current_open.add(chosen)
        current_stocked.add((chosen, k))

    if trim_unused:
        final_stocked = sorted({(j, k) for _, j, k in final_assignments})
        final_open = sorted({j for _, j, _ in final_assignments})
    else:
        final_stocked = sorted(current_stocked)
        final_open = sorted(current_open)

    prune_stats = {"closures": 0, "relocations": 0}
    if hub_prune:
        pruned = hub_prune_pass(
            sorted(final_assignments),
            final_stocked,
            final_open,
            data,
            max_iterations=int(hub_prune_max_iterations),
        )
        # Reassignments from pruning invalidate the recorded source labels for moved rows;
        # re-tag those rows so the audit CSV stays accurate.
        new_assignments = pruned["assignments"]
        original_lookup = {(i, k): j for i, j, k in final_assignments}
        relocated_sources: dict[tuple[str, str, str], str] = {}
        for i, j, k in new_assignments:
            orig_j = original_lookup.get((i, k))
            if orig_j is not None and orig_j != j:
                relocated_sources[(i, j, k)] = "hub_prune_relocated"
            else:
                relocated_sources[(i, j, k)] = assignment_sources.get((i, orig_j or j, k), "raw_or_solver")
        assignment_sources = relocated_sources
        final_assignments = new_assignments
        final_stocked = pruned["stocked_pairs"]
        final_open = pruned["open_hubs"]
        prune_stats = {"closures": pruned["closures"], "relocations": pruned["relocations"]}

    return {
        "assignments": sorted(final_assignments),
        "stocked_pairs": final_stocked,
        "open_hubs": final_open,
        "assignment_sources": assignment_sources,
        "missing_unrepaired": int(missing_unrepaired),
        "hub_prune_stats": prune_stats,
    }


def postprocess_qubo_solution_timed(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """Additive diagnostic wrapper around postprocess_qubo_solution.

    Runs the EXACT same postprocess (identical result), but temporarily swaps the
    module-global hub_prune_pass for a timing shim so the hub-prune sub-portion can
    be measured separately from the rest of the postprocess. The original function
    is restored in a finally block, so this changes no optimization logic.

    Returns (final_dict, {"postprocess_seconds": <full call>, "hub_prune_seconds": <prune subset>}).

    CAVEAT (for review): this rebinds this module's global `hub_prune_pass`, so the
    timing only interposes when postprocess_qubo_solution resolves `hub_prune_pass`
    via this module's globals (the normal case). If a future caller instead does
    `from run_aligned_fsl_comparison import hub_prune_pass` and calls that bound
    name, the shim is bypassed and hub_prune_seconds stays 0.0 (silently, no error).
    It is diagnostic-only and never alters the optimization result.
    """
    timings = {"postprocess_seconds": 0.0, "hub_prune_seconds": 0.0}
    original_hub_prune = globals()["hub_prune_pass"]

    def _timed_hub_prune(*a: Any, **k: Any) -> Any:
        t = time.perf_counter()
        try:
            return original_hub_prune(*a, **k)
        finally:
            timings["hub_prune_seconds"] += time.perf_counter() - t

    globals()["hub_prune_pass"] = _timed_hub_prune
    t0 = time.perf_counter()
    try:
        result = postprocess_qubo_solution(*args, **kwargs)
    finally:
        globals()["hub_prune_pass"] = original_hub_prune
        timings["postprocess_seconds"] = time.perf_counter() - t0
    return result, timings


def print_qubo_header(data: dict[str, Any], batches: list[BatchSpec], args: argparse.Namespace, run_dir: Path) -> None:
    avg_hubs = sum(len(v) for v in data["zip_to_hubs"].values()) / max(1, len(data["zip_to_hubs"]))
    print("\n" + "=" * 76, flush=True)
    print("QUBO MODEL - ALIGNED COST BASIS", flush=True)
    print("=" * 76, flush=True)
    print(f"  dataset:                  {data['dataset_name']}", flush=True)
    print(f"  active demand pairs:      {len(data['active']):,}", flush=True)
    print(f"  parts loaded:             {len(data['K']):,}", flush=True)
    print(f"  hubs loaded:              {len(data['J']):,}", flush=True)
    print(f"  candidate hubs/zip:       {'all' if data['top_hubs_per_zip'] is None else data['top_hubs_per_zip']} (avg {avg_hubs:.2f})", flush=True)
    print(f"  total QUBO batches:       {len(batches):,}", flush=True)
    print(f"  max Z vars/batch:         {args.max_z_vars_per_batch:,}", flush=True)
    print(f"  sampler:                  {args.sampler}", flush=True)
    if args.sampler == "sqa":
        print(f"    sqa beta:               {args.sqa_beta}", flush=True)
        print(f"    sqa trotter:            {args.sqa_trotter}", flush=True)
        _g = float(getattr(args, "sqa_gamma", -1.0))
        print(f"    sqa gamma:              {'library-default' if _g < 0 else _g}", flush=True)
    print(f"  base miles:               {data['scalar']['base_miles']}", flush=True)
    print(f"  penalty start miles:      {data['scalar']['penalty_start_miles']}", flush=True)
    print(f"  max service miles:        {data['scalar']['max_service_miles']}", flush=True)
    if args.adaptive_penalty_mode != "off":
        print(f"  adaptive penalty:         {args.adaptive_penalty_mode}", flush=True)
        print(f"    max iterations:         {args.adaptive_penalty_iterations}", flush=True)
        print(f"    growth factor:          {args.adaptive_penalty_growth}", flush=True)
    print(f"  output folder:            {run_dir}", flush=True)
    print("=" * 76, flush=True)


def run_qubo_solver(args: argparse.Namespace) -> dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    if args.seed is not None and int(args.seed) >= 0:
        random.seed(int(args.seed))

    data = load_problem_data(
        args.dataset_dir,
        max_service_miles_override=args.max_service_miles,
        penalty_start_miles_override=args.penalty_start_miles,
        top_hubs_per_zip=None if int(args.top_hubs_per_zip) < 0 else int(args.top_hubs_per_zip),
        max_parts_total=None if int(args.max_parts_total) < 0 else int(args.max_parts_total),
    )
    run_dir = Path(args.run_root).expanduser().resolve() / "qubo"
    run_dir.mkdir(parents=True, exist_ok=True)

    batches = build_batches(
        data["active"],
        data["part_order"],
        data["zip_to_hubs"],
        part_batch_size=int(args.part_batch_size),
        max_z_vars_per_batch=int(args.max_z_vars_per_batch),
    )
    print_qubo_header(data, batches, args, run_dir)

    deadline = None
    if args.qubo_time_limit is not None and float(args.qubo_time_limit) > 0:
        deadline = time.perf_counter() + float(args.qubo_time_limit)

    results: list[BatchResult] = []
    stopped_time_limit = False
    for batch in batches:
        if deadline is not None and time.perf_counter() >= deadline:
            print(
                f"\nQUBO wall-time limit reached before next batch. Completed {len(results)}/{len(batches)} batches.",
                flush=True,
            )
            stopped_time_limit = True
            break
        result = solve_qubo_batch(batch, data["active"], data, args)
        results.append(result)

        checkpoint_raw = aggregate_raw_results(results)
        pd.DataFrame(checkpoint_raw["assignments"], columns=["zip_id", "hub_id", "part_id"]).to_csv(
            run_dir / "checkpoint_raw_assignments.csv", index=False
        )
        pd.DataFrame({"hub_id": checkpoint_raw["open_hubs"]}).to_csv(run_dir / "checkpoint_raw_open_hubs.csv", index=False)
        batch_summary_dataframe(results).to_csv(run_dir / "batch_summary_checkpoint.csv", index=False)

    if not results:
        raise RuntimeError("No QUBO batches completed; increase --qubo-time-limit or check the instance")

    raw = aggregate_raw_results(results)
    final = postprocess_qubo_solution(
        raw["assignments"],
        raw["stocked_pairs"],
        raw["open_hubs"],
        data,
        repair_assignments=not bool(args.no_repair_assignments),
        trim_unused=not bool(args.no_trim_unused),
        hub_prune=not bool(getattr(args, "no_hub_prune", False)),
        hub_prune_max_iterations=int(getattr(args, "hub_prune_max_iterations", 10)),
    )

    batch_summary_dataframe(results).to_csv(run_dir / "batch_summary.csv", index=False)

    # Per-batch adaptive metadata: only emitted in within-batch mode so off-mode
    # produces zero new artifacts and batch_summary.csv stays byte-identical.
    if args.adaptive_penalty_mode == "within-batch":
        batch_adaptive_summary_dataframe(results).to_csv(
            run_dir / "batch_adaptive_summary.csv", index=False
        )
        adaptive_iteration_log_dataframe(results).to_csv(
            run_dir / "adaptive_iteration_log.csv", index=False
        )

    assignment_rows_dataframe(raw["assignments"], data).to_csv(run_dir / "raw_qubo_hub_zip_part_pairings.csv", index=False)
    pd.DataFrame({"hub_id": raw["open_hubs"]}).to_csv(run_dir / "raw_qubo_open_hubs.csv", index=False)
    pd.DataFrame(raw["stocked_pairs"], columns=["hub_id", "part_id"]).to_csv(run_dir / "raw_qubo_stocked_pairs.csv", index=False)

    wall = time.perf_counter() - start
    peak_mb, current_mb = memory_report_mb()
    runtime = {
        "wall_seconds": float(wall),
        "qubo_build_seconds": float(sum(r.build_seconds for r in results)),
        "qubo_sample_seconds": float(sum(r.sample_seconds for r in results)),
        "sample_eval_seconds": float(sum(r.eval_seconds for r in results)),
        "batch_total_seconds": float(sum(r.total_seconds for r in results)),
        "peak_memory_mb": float(peak_mb),
        "current_memory_mb": float(current_mb),
    }
    raw_cost = compute_solution_cost(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    raw_audit = global_audit(raw["assignments"], raw["stocked_pairs"], raw["open_hubs"], data)
    extra = {
        "completed_batches": int(len(results)),
        "total_batches": int(len(batches)),
        "full_batch_coverage": bool(len(results) == len(batches) and not stopped_time_limit),
        "stopped_due_to_time_limit": bool(stopped_time_limit),
        "raw_solution": {
            "open_hubs_count": int(len(raw["open_hubs"])),
            "stocked_pairs_count": int(len(raw["stocked_pairs"])),
            "assignments_count": int(len(raw["assignments"])),
            "cost": raw_cost,
            "audit": raw_audit,
        },
        "postprocess": {
            "repair_assignments": bool(not args.no_repair_assignments),
            "trim_unused_open_stock": bool(not args.no_trim_unused),
            "missing_unrepaired": int(final.get("missing_unrepaired", 0)),
            "hub_prune_enabled": bool(not getattr(args, "no_hub_prune", False)),
            "hub_prune_closures": int(final.get("hub_prune_stats", {}).get("closures", 0)),
            "hub_prune_relocations": int(final.get("hub_prune_stats", {}).get("relocations", 0)),
        },
        "adaptive_penalty_mode": args.adaptive_penalty_mode,
        "adaptive_penalty_iterations_max": int(args.adaptive_penalty_iterations),
        "adaptive_penalty_growth": float(args.adaptive_penalty_growth),
    }
    summary = write_solution_outputs(
        run_dir,
        solver_name="qubo",
        data=data,
        assignments=final["assignments"],
        stocked_pairs=final["stocked_pairs"],
        open_hubs=final["open_hubs"],
        runtime=runtime,
        extra=extra,
        assignment_sources=final.get("assignment_sources", {}),
    )
    print(final_results_block(summary, "QUBO FINAL RESULTS"), flush=True)
    gc.collect()
    return summary


# ---------------------------------------------------------------------------
# Parent orchestration and comparison
# ---------------------------------------------------------------------------


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summary_to_row(summary: dict[str, Any]) -> dict[str, Any]:
    sol = summary["final_solution"]
    cost = sol["cost"]
    audit = sol["audit"]
    rt = summary["runtime"]
    return {
        "solver": summary["solver"],
        "total_cost": cost["total_cost"],
        "open_hubs": sol["open_hubs_count"],
        "closed_hubs": sol["closed_hubs_count"],
        "stocked_hub_part_pairs": sol["stocked_pairs_count"],
        "hub_zip_part_pairings": sol["assignments_count"],
        "structural_violations": audit["total_structural_violations"],
        "sla_distance_violations": audit["sla_distance_violations"],
        "max_service_distance_violations": audit["max_service_distance_violations"],
        "wall_seconds": rt.get("wall_seconds"),
        "peak_memory_mb": rt.get("peak_memory_mb"),
        "current_memory_mb": rt.get("current_memory_mb"),
        "output_dir": summary.get("output_dir"),
    }


def read_assignment_set(run_dir: Path) -> set[tuple[str, str, str]]:
    path = run_dir / "hub_zip_part_pairings.csv"
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set((str(r.zip_id), str(r.hub_id), str(r.part_id)) for r in df[["zip_id", "hub_id", "part_id"]].itertuples(index=False))


def read_open_hub_set(run_dir: Path) -> set[str]:
    path = run_dir / "open_hubs.csv"
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(str(x) for x in df["hub_id"].tolist())


def read_stocked_set(run_dir: Path) -> set[tuple[str, str]]:
    path = run_dir / "stocked_pairs.csv"
    if not path.is_file():
        path = run_dir / "stocked_hub_part_pairs.csv"
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set((str(r.hub_id), str(r.part_id)) for r in df[["hub_id", "part_id"]].itertuples(index=False))


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def build_combined_outputs(run_root: Path, solvers: list[str]) -> dict[str, Any]:
    summaries = []
    for solver in solvers:
        summary_path = run_root / solver / "summary.json"
        if summary_path.is_file():
            summaries.append(load_summary(summary_path))

    rows = [summary_to_row(s) for s in summaries]
    df = pd.DataFrame(rows)
    df.to_csv(run_root / "combined_summary.csv", index=False)

    combined: dict[str, Any] = {
        "run_root": str(run_root),
        "solvers": [s["solver"] for s in summaries],
        "rows": rows,
    }

    by_solver = {s["solver"]: s for s in summaries}
    if "gurobi" in by_solver and "qubo" in by_solver:
        g = by_solver["gurobi"]
        q = by_solver["qubo"]
        g_cost = float(g["final_solution"]["cost"]["total_cost"])
        q_cost = float(q["final_solution"]["cost"]["total_cost"])
        gap = q_cost - g_cost
        gap_pct = (gap / abs(g_cost) * 100.0) if abs(g_cost) > 1e-12 else None

        g_dir = run_root / "gurobi"
        q_dir = run_root / "qubo"
        g_asg = read_assignment_set(g_dir)
        q_asg = read_assignment_set(q_dir)
        g_open = read_open_hub_set(g_dir)
        q_open = read_open_hub_set(q_dir)
        g_stock = read_stocked_set(g_dir)
        q_stock = read_stocked_set(q_dir)
        g_map = {(i, k): j for i, j, k in g_asg}
        q_map = {(i, k): j for i, j, k in q_asg}
        shared_pairs = set(g_map) & set(q_map)
        same_hub = sum(1 for pair in shared_pairs if g_map[pair] == q_map[pair])
        assignment_hub_agreement = same_hub / len(shared_pairs) if shared_pairs else 1.0

        combined["comparison"] = {
            "qubo_minus_gurobi_cost": float(gap),
            "qubo_minus_gurobi_cost_pct": float(gap_pct) if gap_pct is not None else None,
            "open_hubs_delta_qubo_minus_gurobi": int(q["final_solution"]["open_hubs_count"] - g["final_solution"]["open_hubs_count"]),
            "stocked_pairs_delta_qubo_minus_gurobi": int(q["final_solution"]["stocked_pairs_count"] - g["final_solution"]["stocked_pairs_count"]),
            "assignments_delta_qubo_minus_gurobi": int(q["final_solution"]["assignments_count"] - g["final_solution"]["assignments_count"]),
            "open_hubs_jaccard": float(jaccard(g_open, q_open)),
            "stocked_pairs_jaccard": float(jaccard(g_stock, q_stock)),
            "assignments_triple_jaccard": float(jaccard(g_asg, q_asg)),
            "shared_zip_part_count": int(len(shared_pairs)),
            "assignment_hub_agreement_rate": float(assignment_hub_agreement),
            "assignment_hub_disagreement_count": int(len(shared_pairs) - same_hub),
        }

    (run_root / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    write_combined_text(run_root, combined)
    return combined


def write_combined_text(run_root: Path, combined: dict[str, Any]) -> None:
    rows = combined.get("rows", [])
    lines = [
        "=" * 92,
        "COMBINED FINAL COMPARISON",
        "=" * 92,
        f"Run folder: {run_root}",
        "",
        "Solver      Total Cost        Open  Closed  Stocked Pairs  Assignments  Struct Viol  SLA Viol  Wall(s)  Peak MB",
        "----------  ----------------  -----  ------  -------------  -----------  -----------  --------  -------  -------",
    ]
    for r in rows:
        lines.append(
            f"{r['solver']:<10}  {money(r['total_cost']):>16}  "
            f"{int(r['open_hubs']):>5,}  {int(r['closed_hubs']):>6,}  "
            f"{int(r['stocked_hub_part_pairs']):>13,}  {int(r['hub_zip_part_pairings']):>11,}  "
            f"{int(r['structural_violations']):>11,}  {int(r['sla_distance_violations']):>8,}  "
            f"{float(r['wall_seconds'] or 0):>7.2f}  {float(r['peak_memory_mb'] or 0):>7.1f}"
        )

    comp = combined.get("comparison")
    if comp:
        lines.extend(
            [
                "",
                "QUBO vs Gurobi",
                f"  cost gap:                       {money(comp['qubo_minus_gurobi_cost'])} ({comp['qubo_minus_gurobi_cost_pct']:.2f}%)",
                f"  open hubs delta:                {comp['open_hubs_delta_qubo_minus_gurobi']:+,}",
                f"  stocked pairs delta:            {comp['stocked_pairs_delta_qubo_minus_gurobi']:+,}",
                f"  assignments delta:              {comp['assignments_delta_qubo_minus_gurobi']:+,}",
                f"  open-hub Jaccard:               {comp['open_hubs_jaccard']:.4f}",
                f"  stocked-pair Jaccard:           {comp['stocked_pairs_jaccard']:.4f}",
                f"  assignment-triple Jaccard:      {comp['assignments_triple_jaccard']:.4f}",
                f"  assignment hub agreement:       {comp['assignment_hub_agreement_rate']:.2%}",
                f"  assignment hub disagreements:   {comp['assignment_hub_disagreement_count']:,}",
            ]
        )
    lines.extend(
        [
            "",
            "Files written:",
            "  combined_summary.csv",
            "  combined_summary.json",
            "  combined_summary.txt",
            "  gurobi/summary.txt and qubo/summary.txt when those solvers were run",
            "=" * 92,
        ]
    )
    text = "\n".join(lines)
    (run_root / "combined_summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text, flush=True)


def run_child_solver(solver: str, config_path: Path) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--internal-solver", solver, "--config-json", str(config_path)]
    print("\n" + "#" * 92, flush=True)
    print(f"Launching {solver.upper()} child process", flush=True)
    print("#" * 92, flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{solver} child process failed with exit code {rc}")


def parent_run(args: argparse.Namespace) -> None:
    run_name = args.run_name or f"aligned_fsl_{now_stamp()}"
    run_root = Path(args.output_dir).expanduser().resolve() / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = vars(args).copy()
    cfg["run_root"] = str(run_root)
    cfg.pop("internal_solver", None)
    cfg.pop("config_json", None)
    config_path = run_root / "run_config.json"
    config_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    solvers = ["gurobi", "qubo"] if args.solver == "both" else [args.solver]
    print("\n" + "=" * 92, flush=True)
    print("ALIGNED FSL RUNNER", flush=True)
    print("=" * 92, flush=True)
    print(f"  dataset dir:      {Path(args.dataset_dir).expanduser().resolve()}", flush=True)
    print(f"  solvers:          {', '.join(solvers)}", flush=True)
    print(f"  run folder:       {run_root}", flush=True)
    print(f"  config file:      {config_path}", flush=True)
    print("=" * 92, flush=True)

    completed: list[str] = []
    for solver in solvers:
        run_child_solver(solver, config_path)
        completed.append(solver)

    build_combined_outputs(run_root, completed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run aligned Gurobi/QUBO FSL comparison from one Python file.")
    p.add_argument("--internal-solver", choices=["gurobi", "qubo"], default="", help=argparse.SUPPRESS)
    p.add_argument("--config-json", default="", help=argparse.SUPPRESS)

    p.add_argument("--solver", choices=["both", "gurobi", "qubo"], default="both")
    p.add_argument("--dataset-dir", default="instances_low", help="Folder with demand/distances/hubs/parameters/parts/zips CSV files.")
    p.add_argument("--output-dir", default="outputs/aligned_fsl", help="Root folder for run outputs.")
    p.add_argument("--run-name", default="", help="Optional run folder name. Default uses timestamp.")
    p.add_argument("--run-root", default="", help=argparse.SUPPRESS)

    p.add_argument("--max-service-miles", "--max-distance", dest="max_service_miles", type=float, default=None, help="Eligibility cutoff. Default uses parameters.csv max_service_miles.")
    p.add_argument("--penalty-start-miles", type=float, default=None, help="Override parameters.csv penalty_start_miles.")
    p.add_argument("--top-hubs-per-zip", type=int, default=-1, help="Nearest eligible hubs per ZIP. -1 keeps all within max service miles.")
    p.add_argument("--max-parts-total", type=int, default=-1, help="Limit parts for testing. -1 keeps all.")
    p.add_argument("--seed", type=int, default=42)

    # Gurobi controls.
    p.add_argument("--gurobi-license-file", default="", help="Optional path assigned to GRB_LICENSE_FILE before importing gurobipy.")
    p.add_argument("--gurobi-time-limit", type=float, default=-1.0, help="Gurobi time limit in seconds. <=0 means no limit.")
    p.add_argument("--mip-gap", type=float, default=0.001, help="Relative MIP gap for Gurobi.")
    p.add_argument("--threads", type=int, default=0, help="Gurobi thread count. 0 lets Gurobi choose.")
    p.add_argument("--console-log", action="store_true", help="Show Gurobi log in console. It is always written to gurobi.log.")
    p.add_argument("--enforce-hub-capacity", action="store_true", help="Treat hubs.csv B_j as hard max stocked parts at each hub.")

    # QUBO controls. Defaults bumped for Tier 1+2+3 changes (see hub-prune post-pass below).
    p.add_argument("--part-batch-size", type=int, default=1000, help="Soft part count per batch. Bumped from 200 to consolidate batches and reduce per-batch X duplication.")
    p.add_argument("--max-z-vars-per-batch", type=int, default=50000, help="Hard Z cap. Bumped from 20000 to reduce batch count.")
    p.add_argument("--sampler", choices=["sa", "sqa"], default="sa",
                   help="Sampler backend: sa=OpenJij SASampler (default), sqa=OpenJij SQASampler.")
    p.add_argument("--sqa-beta", type=float, default=5.0,
                   help="SQA fixed inverse temperature (only used when --sampler sqa).")
    p.add_argument("--sqa-trotter", type=int, default=8,
                   help="SQA Trotter slices (only used when --sampler sqa). "
                        "Note: each slice replicates the spin system, so trotter=N is ~N x "
                        "the memory of SA; size --mem/-t accordingly at high scale.")
    p.add_argument("--sqa-gamma", type=float, default=-1.0,
                   help="SQA transverse-field strength (only used when --sampler sqa). "
                        "Default -1.0 means use OpenJij's library default (unchanged behavior); "
                        "set a value >= 0 to fix it explicitly for reproducible/reportable runs.")
    p.add_argument("--num-reads", type=int, default=100, help="Base OpenJij reads. Bumped from 10 for Tier 1.3.")
    p.add_argument("--num-sweeps", type=int, default=3000, help="OpenJij sweeps per read. Bumped from default for Tier 1.3.")
    p.add_argument("--max-stages", type=int, default=3, help="SA retry stages. Bumped from 2 for Tier 1.3.")
    p.add_argument("--retry-reads-boost", type=float, default=2.0)
    p.add_argument("--penalty-mode", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--min-penalty", type=float, default=50000.0, help="Penalty floor. Bumped from 10000 (Tier 2.5) to dominate hub fixed costs.")
    p.add_argument("--constraint-multiplier", type=float, default=5.0)
    p.add_argument("--disable-objective-scale", action="store_true",
                   help="Force objective_scale=1.0 so penalties equal the base (min-penalty) "
                        "floor instead of being lifted to ~scale*multiplier (millions). "
                        "Diagnostic: isolates the objective-scale normalization.")
    for c in ("c1", "c2", "c3", "c4"):
        p.add_argument(f"--min-penalty-{c}", type=float, default=-1.0)
        p.add_argument(f"--constraint-multiplier-{c}", type=float, default=-1.0)
    p.add_argument("--c4-mode", choices=["off", "auto", "on"], default="auto")
    p.add_argument("--qubo-time-limit", type=float, default=5400.0, help="QUBO wall time budget in seconds, checked between batches. 0 disables. Bumped from 2700 to fit larger SA budget.")
    p.add_argument("--no-repair-assignments", action="store_true", help="Do not repair missing/multiple QUBO assignments in final output.")
    p.add_argument("--no-trim-unused", action="store_true", help="Do not trim unused QUBO open hubs / stocked pairs after final assignments.")
    # Tier 1.2 / Tier 3 controls.
    p.add_argument("--no-hub-prune", action="store_true", help="Disable the global hub-pruning post-pass (Tier 1.2). Pruning closes hubs whose assignments can be cheaply rerouted to other already-open hubs.")
    p.add_argument("--hub-prune-max-iterations", type=int, default=10, help="Max passes over open hubs in the hub-prune post-pass. Each pass closes at most one hub before re-evaluating.")
    p.add_argument("--x-empty-penalty-factor", type=float, default=0.0, help="Tier 3.8: extra X-diagonal penalty as a multiple of S_lim, partially refunded per stocked Y. Default 0 (disabled). Approximates X<=sum(Y) without slack vars.")
    p.add_argument("--y-overflow-penalty-factor", type=float, default=0.0, help="Tier 3.9: linear S_var-style penalty per stocked Y as a multiple of S_var. Default 0 (disabled). Inactive for instances_low since L=50000 is never approached.")
    p.add_argument("--adaptive-penalty-mode", type=str, default="off",
                   choices=["off", "within-batch"],
                   help="Adaptive penalty strategy. 'off'=current static behavior. "
                        "'within-batch'=iteratively scale violated constraint penalties.")
    p.add_argument("--adaptive-penalty-iterations", type=int, default=5,
                   help="Max adaptive penalty iterations per batch (only used when mode != off).")
    p.add_argument("--adaptive-penalty-growth", type=float, default=1.5,
                   help="Multiplicative growth factor for violated constraint penalties. "
                        "Typical range 1.5-2.0. Lemonge & Barbosa (2004) default ~1.5.")
    p.add_argument("--adaptive-penalty-initial-source", type=str, default="batch-scale",
                   choices=["batch-scale", "external"],
                   help="Where initial penalties come from. 'batch-scale'=current "
                        "penalty_weights() behavior. 'external'=hook for cross-batch "
                        "(not used in v1).")
    p.add_argument("--adaptive-penalty-stagnation-patience", type=int, default=0,
                   help="Break the adaptive loop if total_violations has not strictly "
                        "improved for N consecutive iterations. 0 = disabled (current behavior).")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.config_json:
        config_json_path = str(Path(args.config_json).resolve())
        cfg = json.loads(Path(config_json_path).read_text(encoding="utf-8"))
        internal_solver = args.internal_solver
        args = argparse.Namespace(**cfg)
        args.internal_solver = internal_solver
        args.config_json = config_json_path

    try:
        if args.internal_solver == "gurobi":
            run_gurobi_solver(args)
        elif args.internal_solver == "qubo":
            run_qubo_solver(args)
        else:
            parent_run(args)
        return 0
    except Exception as exc:
        peak_mb, current_mb = memory_report_mb()
        print("ERROR: run failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(f"peak/current memory before exit: {peak_mb:,.1f}/{current_mb:,.1f} MB", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
